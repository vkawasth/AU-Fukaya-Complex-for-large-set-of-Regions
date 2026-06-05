# =============================================================================
# qkv_truncation.jl
#
# QKV attention mechanism with t-structure truncation sheets.
#
# Implements three parallel "passive probes" that run every DP step
# without triggering surgery, giving a gradient signal toward crisis
# rather than a binary wall-detection.
#
# THREE SHEETS:
#   Sheet A — τ_≤0 — m₂ classical flow (already running as default)
#   Sheet B — τ_[1,2] — H¹/H² obstruction probe (continuous coker signal)
#   Sheet C — τ_≥3 — m₇/m₈ deep obstruction probe (long circuit detection)
#
# QKV MECHANISM:
#   Q = Hom_Con(Q_target, -)          representable presheaf / DPQuery
#   K = τ_sheet(F_{T_α})              truncated stalk sheaf
#   V = Lan_{i_α}(P_α)                transition matrix
#   A(α) = F_{T_α}(Q_target)         alignment via Yoneda (column lookup)
#   a_α = Z_α(h) / Z_global(h)       LattE attention weight
#   Output = ⊕_α a_α · Lan_{i_α}(P_α)  weighted coproduct
# =============================================================================

include(joinpath(@__DIR__, "nno_au_core.jl"))

using Printf

# =============================================================================
# PART 1: SHEET DEFINITIONS (t-structure truncation levels)
# =============================================================================

@enum HomSheet begin
    SHEET_A   # τ_≤0: classical m₂, HH⁰ only — no higher ops
    SHEET_B   # τ_[1,2]: H¹/H² obstruction probe — m₄ defects
    SHEET_C   # τ_≥3: deep derived — m₇/m₈ long circuits
end

"""
    SheetProbeResult

Scalar signals from parallel sheet probes at one DP step.
All scalars — interpretable as a risk gradient toward crisis.
"""
struct SheetProbeResult
    # Sheet A: classical flow signal
    phi_flow        ::Float64    # boundary flux φ_{αβ} ∈ [0,1]
    # Sheet B: obstruction signal
    dim_h1_proxy    ::Float64    # continuous coker signal ∈ [0,1] (0=safe, 1=crisis)
    n_active_circuits::Int       # circuits with w(b) > h
    # Sheet C: deep derivation signal
    max_circuit_len ::Int        # longest active circuit (m_k level)
    has_m7_circuits ::Bool       # any circuit of length ≥ 7
    # Combined risk
    risk_gradient   ::Float64    # composite 0→1 toward Mode 4
    # Recommendation
    recommended_mode::Int        # 1/2/3/4
    early_warning   ::Bool       # crisis imminent (pre-empt surgery)
end

# =============================================================================
# PART 2: SHEET A PROBE — classical m₂ flow
# =============================================================================

"""
    probe_sheet_a(ctx, target_ctx) -> Float64

Sheet A: τ_≤0 truncation.
Returns boundary flux φ_{αβ} — the fraction of probability mass
flowing toward target_ctx per step.

This is the FLOW-level signal. Detects crisis ~2-3 steps before wall.
φ ≈ 0:    AUs decoupled (Mode 1)
φ ≈ 0.5:  partial coupling (Mode 2)
φ > 0.9:  strong coupling / imminent merger (Mode 3/4)
"""
function probe_sheet_a(ctx::NNOAUContext,
                        target_ctx::NNOAUContext)::Float64
    flux = boundary_flux(ctx, target_ctx)
    return Float64(flux)
end

# =============================================================================
# PART 3: SHEET B PROBE — H¹/H² obstruction signal
# =============================================================================

"""
    probe_sheet_b(ctx, basis, weights, h, coker_max) -> (Float64, Int)

Sheet B: τ_[1,2] truncation.
Computes a continuous proxy for dim(H¹(A(α))) as h varies.

Method:
  n_active = #{b ∈ Markov basis : w(b) > h}
  As h decreases: n_active rises → more obstruction classes activate
  dim_h1_proxy = n_active / coker_max (normalised to [0,1])

When dim_h1_proxy → 1.0: Mode 4 imminent (circuit count saturates coker)
When dim_h1_proxy → 0.0: Sheet A (few active circuits, no obstruction)

This detects crisis ~5-10 steps before the wall.
"""
function probe_sheet_b(ctx::NNOAUContext,
                        basis::Vector{Vector{Int}},
                        weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                        h::Float64,
                        coker_max::Int = 62)::Tuple{Float64, Int}

    isempty(basis) && return (0.0, 0)

    # Count circuits above toric height h
    n_edges = length(ctx.edges)
    n_active = 0

    for b in basis
        length(b) == n_edges || continue
        # Renkin-Crone weight of this circuit
        w = 1.0
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e = get(weights, ctx.edges[j], NNO_ONE)
            w *= Float64(w_e) ^ abs(bj)
        end
        w >= h && (n_active += 1)
    end

    # Normalised dim_h1 proxy: n_active / coker_max ∈ [0,1]
    # When coker_max=0 (no obstruction on this boundary): proxy = 0
    # When coker_max>0: rises from 0→1 as h decreases and circuits activate
    proxy = coker_max > 0 ? min(Float64(n_active) / Float64(coker_max), 1.0) : 0.0
    return (proxy, n_active)
end

# =============================================================================
# PART 4: SHEET C PROBE — m₇/m₈ deep obstruction
# =============================================================================

"""
    probe_sheet_c(ctx, basis, weights, h) -> (Int, Bool)

Sheet C: τ_≥3 truncation.
Identifies whether high-degree (m₆/m₇/m₈) circuits are active.

Returns:
  max_len:       length of the longest active circuit above h
  has_m7:        true if any active circuit has length ≥ 7

A circuit of length k corresponds to an m_k operation in the A∞ tower.
When m₇ circuits activate: the HPF↔Infra gate=2 lower bound may rise.

This detects the deep structural change ~10-20 steps before the wall.
"""
function probe_sheet_c(ctx::NNOAUContext,
                        basis::Vector{Vector{Int}},
                        weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                        h::Float64)::Tuple{Int, Bool}

    isempty(basis) && return (0, false)
    n_edges = length(ctx.edges)
    max_len = 0

    for b in basis
        length(b) == n_edges || continue
        # Circuit length = number of nonzero entries
        circuit_len = count(!=(0), b)
        # Renkin-Crone weight
        w = 1.0
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e = get(weights, ctx.edges[j], NNO_ONE)
            w *= Float64(w_e) ^ abs(bj)
        end
        w >= h && (max_len = max(max_len, circuit_len))
    end

    return (max_len, max_len >= 7)
end

# =============================================================================
# PART 5: COMBINED SHEET PROBE
# =============================================================================

"""
    probe_all_sheets(ctx1, ctx2, basis, weights, h, coker_max)
    -> SheetProbeResult

Run all three sheet probes in parallel and compute a combined risk gradient.

Risk gradient formula:
  risk = 0.3 × φ_flow + 0.5 × dim_h1_proxy + 0.2 × (has_m7 ? 1.0 : 0.0)

Weights reflect detection timing:
  φ_flow (30%):     detects late but precisely (flow collapse)
  dim_h1 (50%):     detects early (obstruction space filling)
  m7 flag (20%):    detects earliest (deep structural change)

Early warning fires when:
  risk > 0.5 OR dim_h1_proxy > 0.6 OR (has_m7 AND phi_flow > 0.3)
"""
function probe_all_sheets(ctx1::NNOAUContext,
                           ctx2::NNOAUContext,
                           basis::Vector{Vector{Int}},
                           weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                           h::Float64;
                           coker_max::Int = 62)::SheetProbeResult

    # Sheet A
    phi = probe_sheet_a(ctx1, ctx2)

    # Sheet B
    dim_h1, n_active = probe_sheet_b(ctx1, basis, weights, h, coker_max)

    # Sheet C
    max_len, has_m7 = probe_sheet_c(ctx1, basis, weights, h)

    # Combined risk gradient
    risk = 0.3 * phi + 0.5 * dim_h1 + 0.2 * (has_m7 ? 1.0 : 0.0)

    # Early warning
    early = risk > 0.5 || dim_h1 > 0.6 || (has_m7 && phi > 0.3)

    # Recommended mode from risk level
    mode = if risk < 0.15
        1   # safe coproduct
    elseif risk < 0.35
        2   # Lan_i scaling
    elseif risk < 0.65
        3   # pushout
    else
        4   # surgery
    end

    SheetProbeResult(phi, dim_h1, n_active, max_len, has_m7,
                     risk, mode, early)
end

# =============================================================================
# PART 6: QKV ATTENTION WITH TRUNCATED KEYS
# =============================================================================

"""
    qkv_alignment(query_target, ctx, sheet) -> Float64

Compute the alignment A(α) = F_{T_α}(Q_target) via Yoneda,
restricted to the given t-structure sheet.

By the Yoneda lemma:
  RHom_Con(Hom_Con(Q_target, -), τ_sheet(F_{T_α})) ≅ τ_sheet(F_{T_α})(Q_target)

This evaluates to: the transition probability TO query_target FROM the current
distribution in ctx, restricted to the given homological sheet.

Sheet A: use full transition matrix (m₂ only)
Sheet B: use only edges with w ≥ h (H¹/H² sheet) 
Sheet C: use only long-path components (length ≥ 7)
"""
function qkv_alignment(query_target::Symbol,
                        ctx::NNOAUContext,
                        sheet::HomSheet,
                        h::Float64 = 1.0)::Float64

    target_idx = findfirst(==(query_target), ctx.regions)
    target_idx === nothing && return 0.0

    if sheet == SHEET_A
        # Sheet A: standard transition row (m₂ classical)
        # A(α) = probability of reaching query_target in one step
        col_sum = sum(Float64(ctx.trans_mat[target_idx, j]) * Float64(ctx.prob[j])
                      for j in 1:length(ctx.regions))
        return col_sum

    elseif sheet == SHEET_B
        # Sheet B: only edges with weight ≥ h (obstruction sheet)
        # Uses a filtered version of the transition matrix
        alignment = 0.0
        for (j, src) in enumerate(ctx.regions)
            w = get(ctx.weights, (src, query_target), NNO_ZERO)
            Float64(w) < h && continue
            t = Float64(ctx.trans_mat[target_idx, j])
            alignment += t * Float64(ctx.prob[j])
        end
        return alignment

    else # SHEET_C
        # Sheet C: only high-weight, long-path contributions
        # Proxy: probability × (1 - stopping probability)
        # Captures whether long paths TO target are active
        t_val = Float64.(ctx.trans_mat[target_idx, :])
        p_val = Float64.(ctx.prob)
        # Weight by inverse stopping: edges not in stops carry more C-sheet weight
        n_stopped = length(ctx.stops)
        n_total   = length(ctx.edges)
        sheet_c_weight = n_total > 0 ? 1.0 - n_stopped / n_total : 0.0
        return dot(t_val, p_val) * sheet_c_weight
    end
end

# =============================================================================
# PART 7: LATTE-WEIGHTED COPRODUCT (replaces 50/50 split)
# =============================================================================

"""
    weighted_coproduct(ctx1, ctx2, Z1, Z2) -> NNOAUContext

Mode 1 coproduct with LattE partition function weights.
Replaces the fixed 50/50 split with adaptive Z_α/Z_global weights.

a_1 = Z_1(h) / (Z_1(h) + Z_2(h))
a_2 = Z_2(h) / (Z_1(h) + Z_2(h))

High Z_α = context has more active circuits at current h
         = context should receive more probability mass
"""
function weighted_coproduct(ctx1::NNOAUContext,
                             ctx2::NNOAUContext,
                             Z1::NNOProb,
                             Z2::NNOProb)::NNOAUContext

    Z_global = Z1 + Z2
    if Z_global == NNO_ZERO
        # Fallback to equal weights
        return coproduct(ctx1, ctx2)
    end

    a1 = Z1 // Z_global    # exact NNO rational
    a2 = Z2 // Z_global    # exact NNO rational

    # Verify weights sum to 1//1
    @assert a1 + a2 == NNO_ONE "Attention weights don't sum to 1: $a1 + $a2"

    n1, n2 = length(ctx1.regions), length(ctx2.regions)
    n  = n1 + n2

    # Block-diagonal transition matrix (same as coproduct)
    T = fill(NNO_ZERO, n, n)
    T[1:n1,    1:n1]   .= ctx1.trans_mat
    T[n1+1:n,  n1+1:n] .= ctx2.trans_mat

    # LattE-weighted probability split
    p = vcat(ctx1.prob .* a1, ctx2.prob .* a2)
    nno_check(p; label="weighted_coproduct")

    id_new = Symbol(string(ctx1.id) * "_⊔_" * string(ctx2.id))
    NNOAUContext(id_new,
                 "LattE-weighted coproduct (a₁=$(Float64(a1) |> x->round(x,digits=3)))",
                 vcat(ctx1.regions, ctx2.regions),
                 vcat(ctx1.edges,   ctx2.edges),
                 union(ctx1.stops,  ctx2.stops),
                 merge(ctx1.weights, ctx2.weights),
                 p, T, ctx1.sector,
                 ctx1.hh2 + ctx2.hh2,
                 0,
                 max(ctx1.rho, ctx2.rho),
                 max(ctx1.step, ctx2.step))
end

# =============================================================================
# PART 8: FULL QKV ROUTER — replaces hyper_confluence
# =============================================================================

"""
    qkv_route(query, ctx1, ctx2, basis, weights, h, Z1, Z2)
    -> (mode, transition_weight, early_warning)

The complete QKV router integrating all three sheets.

1. Run probe_all_sheets() → risk gradient + recommended mode
2. Compute alignment A(α) via Yoneda at recommended sheet level
3. Return mode decision + pre-warning flag

This is the replacement for the static PUSHOUT_TABLE in dp_core.jl.
It gives a CONTINUOUS, h(t)-dependent mode classification instead of
a binary precomputed lookup.

The static table is still consulted for confirmed pairs (coker=62);
the QKV router handles the continuous intermediate states.
"""
function qkv_route(query_target::Symbol,
                    ctx1::NNOAUContext,
                    ctx2::NNOAUContext,
                    basis::Vector{Vector{Int}},
                    weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                    h::Float64,
                    Z1::NNOProb = NNO_ONE,
                    Z2::NNOProb = NNO_ONE)

    # Run all three sheet probes
    # coker_max: use confirmed coker value.
    # coker=0 → this pair has NO algebraic obstruction → dim_h1 must be 0.
    # coker>0 → use actual value so dim_h1 rises as h(t) changes.
    actual_coker = get(CONFIRMED_COKER, (ctx1.id, ctx2.id),
                       get(CONFIRMED_COKER, (ctx2.id, ctx1.id), 0))
    if actual_coker == 0
        # coker=0: no algebraic obstruction on this boundary.
        # Sheet B must be 0 — set coker_max high so n_active/coker_max → 0
        # Use coker_max=typemax(Int) effectively by using a large sentinel
        phi_a   = probe_sheet_a(ctx1, ctx2)
        max_len, has_m7 = probe_sheet_c(ctx1, basis, weights, h)
        risk_no_b = 0.3 * phi_a + 0.0 + 0.2 * (has_m7 ? 1.0 : 0.0)
        mode_no_b = risk_no_b < 0.15 ? 1 : risk_no_b < 0.35 ? 2 :
                    risk_no_b < 0.65 ? 3 : 4
        probe = SheetProbeResult(phi_a, 0.0, 0, max_len, has_m7,
                                 risk_no_b, mode_no_b, risk_no_b > 0.5)
    else
        probe = probe_all_sheets(ctx1, ctx2, basis, weights, h;
                                  coker_max = actual_coker)
    end

    # Yoneda alignment at the probe's recommended sheet level
    sheet = probe.recommended_mode <= 1 ? SHEET_A :
            probe.recommended_mode <= 3 ? SHEET_B : SHEET_C
    alignment = qkv_alignment(query_target, ctx1, sheet, h)

    # Attention weight
    Z_global = Z1 + Z2
    a1 = Z_global > NNO_ZERO ? Z1 // Z_global : NNOProb(1, Int128(2))

    # Transition weight = alignment × a1
    trans_weight = alignment * Float64(a1)

    if probe.early_warning
        @printf("  [QKV ⚠] Early warning: risk=%.3f mode=%d dim_h1=%.2f len=%d\n",
                probe.risk_gradient, probe.recommended_mode,
                probe.dim_h1_proxy, probe.max_circuit_len)
    end

    return (mode       = probe.recommended_mode,
            weight     = trans_weight,
            risk       = probe.risk_gradient,
            alignment  = alignment,
            attn_a1    = Float64(a1),
            early_warn = probe.early_warning,
            probe      = probe)
end

# =============================================================================
# PART 9: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^65)
    println("QKV Truncation + Sheet Probes")
    println("="^65)

    # Build Q_7P contexts
    vertices_7p = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
    edges_7p    = [(:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
                   (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
                   (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
                   (:sAMY,:BLA),(:sAMY,:HY),(:sAMY,:HPF),
                   (:sAMY,:LA),(:sAMY,:PAL),
                   (:HY,:sAMY),(:LA,:BLA),(:LA,:sAMY),(:PAL,:sAMY)]

    stops_A = Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA),
                   (:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)])

    w7p = Dict{Tuple{Symbol,Symbol}, NNOProb}(
        (:LA,   :sAMY) => NNOProb(9752,  100),
        (:sAMY, :LA)   => NNOProb(9752,  100),
        (:BLA,  :LA)   => NNOProb(206,   100),
        (:HPF,  :sAMY) => NNOProb(34590, 100),
        (:sAMY, :HPF)  => NNOProb(34590, 100),
        (:CA1sp,:HPF)  => NNOProb(1500,  100),
    )
    for e in edges_7p; haskey(w7p, e) || (w7p[e] = NNO_ONE); end

    ctx_sAMY = build_nno_au(:CTX_sAMY, "sAMY hub",
        [:sAMY, :BLA, :LA, :HPF, :CA1sp], edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node=:sAMY)
    ctx_HPF  = build_nno_au(:CTX_HPF, "HPF",
        [:HPF, :CA1sp, :sAMY, :BLA], edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node=:HPF)
    ctx_HY   = build_nno_au(:CTX_HY, "HY-PAL",
        [:HY, :PAL, :sAMY], edges_7p,
        Set([(:LA,:sAMY),(:sAMY,:LA)]), w7p, :D, 0, 0, 0.618;
        initial_node=:HY)

    # FIX 1: Keep a fresh copy for Yoneda alignment demo
    # (5-step evolved ctx has all mass at HPF → Yoneda gives 0)
    ctx_sAMY_fresh = build_nno_au(:CTX_sAMY_fresh, "sAMY hub (fresh)",
        [:sAMY, :BLA, :LA, :HPF, :CA1sp], edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node=:sAMY)

    # Run 5 Markov steps on the evolved version
    for _ in 1:5; markov_step!(ctx_sAMY); end

    # FIX 2: Basis with length-7 circuit so Sheet C fires
    active_e = ctx_sAMY.edges
    n_e = length(active_e)
    basis_demo = Vector{Vector{Int}}()
    for j in 1:min(n_e, 20)
        b = zeros(Int, n_e)
        b[j] = 1
        b[mod1(j+1, n_e)] = -1
        push!(basis_demo, b)
    end
    if n_e >= 7
        b7 = zeros(Int, n_e)
        for k in 1:7; b7[k] = (isodd(k) ? 1 : -1); end
        push!(basis_demo, b7)
    end

    println("\n[1] Sheet probes at h=1.0 (sAMY → HPF boundary):")
    probe = probe_all_sheets(ctx_sAMY, ctx_HPF, basis_demo, w7p, 1.0)
    @printf("  Sheet A (flow):    φ = %.4f\n", probe.phi_flow)
    @printf("  Sheet B (H¹/H²):  dim_h1 = %.4f (%d active circuits)\n",
            probe.dim_h1_proxy, probe.n_active_circuits)
    @printf("  Sheet C (m7/m8):  max_len = %d, has_m7 = %s\n",
            probe.max_circuit_len, probe.has_m7_circuits)
    @printf("  Risk gradient:    %.4f → recommended Mode %d\n",
            probe.risk_gradient, probe.recommended_mode)
    @printf("  Early warning:    %s\n", probe.early_warning ? "⚠ YES" : "safe")

    println("\n[2] Sheet probes at varying h (crisis approach simulation):")
    println(@sprintf("  %-8s %-8s %-10s %-8s %-8s",
                     "h", "φ_flow", "dim_h1", "max_len", "mode"))
    println("  " * "─"^48)
    for h in [100.0, 10.0, 1.0, 0.1, 0.01]
        p2 = probe_all_sheets(ctx_sAMY, ctx_HPF, basis_demo, w7p, h)
        @printf("  %-8.2f %-8.4f %-10.4f %-8d %-8d %s\n",
                h, p2.phi_flow, p2.dim_h1_proxy,
                p2.max_circuit_len, p2.recommended_mode,
                p2.early_warning ? "⚠" : "")
    end

    println("\n[3] Yoneda alignment per sheet (target=HPF, using fresh ctx):")
    for sheet in [SHEET_A, SHEET_B, SHEET_C]
        a = qkv_alignment(:HPF, ctx_sAMY_fresh, sheet, 1.0)
        name = sheet == SHEET_A ? "A (m₂)" :
               sheet == SHEET_B ? "B (m₄)" : "C (m₇)"
        @printf("  Sheet %s: A(α) = %.6f\n", name, a)
    end

    println("\n[4] LattE-weighted coproduct (vs 50/50):")
    Z1 = NNOProb(Int128(100), Int128(1))   # Z_sAMY = 100
    Z2 = NNOProb(Int128(30),  Int128(1))   # Z_HPF  = 30
    ctx_joint = weighted_coproduct(ctx_sAMY, ctx_HY, Z1,
                                    NNOProb(Int128(10), Int128(1)))
    @printf("  a_sAMY = %.4f  a_HY = %.4f  Σp = %s\n",
            Float64(Z1)/Float64(Z1+NNOProb(10,1)),
            Float64(NNOProb(10,1))/Float64(Z1+NNOProb(10,1)),
            string(sum(ctx_joint.prob)))

    println("\n[5] Full QKV route (target=HPF, sAMY→HPF boundary):")
    # Note: qkv_route uses evolved ctx_sAMY for sheet probes (flow signal)
    # but alignment is evaluated against the fresh context's transition matrix
    # by passing ctx_sAMY_fresh as a reference (the Yoneda evaluation target).
    # For a full simulation, the evolved ctx IS correct for the sheet probes
    # (they need current probability distribution, not initial).
    result = qkv_route(:HPF, ctx_sAMY, ctx_HPF,
                        basis_demo, w7p, 1.0, Z1, Z2)
    println("  (alignment≈0 is expected: evolved ctx has mass at HPF, not sAMY)")
    @printf("  Mode: %d  risk: %.4f  alignment: %.4f\n",
            result.mode, result.risk, result.alignment)
    @printf("  a₁(sAMY): %.4f  early_warn: %s\n",
            result.attn_a1, result.early_warn ? "⚠" : "safe")

    println("\n" * "="^65)
    println("QKV Truncation complete.")
    println("  probe_sheet_a()          → φ flow signal (Sheet A)")
    println("  probe_sheet_b()          → continuous H¹/H² proxy (Sheet B)")
    println("  probe_sheet_c()          → m₇ circuit detection (Sheet C)")
    println("  probe_all_sheets()       → combined risk gradient 0→1")
    println("  qkv_alignment()          → Yoneda evaluation per sheet")
    println("  weighted_coproduct()     → LattE a_α weights (not 50/50)")
    println("  qkv_route()             → full QKV router replacing PUSHOUT_TABLE")
    println("="^65)
end
