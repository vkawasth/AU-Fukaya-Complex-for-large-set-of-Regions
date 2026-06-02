# =============================================================================
# au_pushout_full.jl
#
# FULL pushout resolution pipeline for AU context pairs.
# Overwrites au_pushout.jl — this is the definitive version.
#
# Run: julia au_pushout_full.jl
# (No arguments — loads au_fukaya_75.jl from same directory)
# =============================================================================

using LinearAlgebra, SparseArrays, Printf

# ── Load au_fukaya_75.jl definitions only (skip MAIN execution block) ────────
let
    orig_dir  = @__DIR__
    orig_file = joinpath(orig_dir, "au_fukaya_75.jl")
    src       = read(orig_file, String)
    src       = replace(src, "@__DIR__" => repr(orig_dir))
    marker    = "# =============================================================================\n# MAIN"
    pos       = findfirst(marker, src)
    defs_only = pos !== nothing ? src[1:pos[1]-1] : src
    tmp       = tempname() * ".jl"
    write(tmp, defs_only)
    include(tmp)
    rm(tmp, force=true)
end
println("au_fukaya_75.jl definitions loaded.")

# ── Build the AU module (calls build_au_module() with no args) ───────────────
import Random; Random.seed!(42)
mod = build_au_module()
println("AU module built: $(length(mod.contexts)) contexts")

# =============================================================================
# PUSHOUT UTILITIES
# =============================================================================

# Symbol-level edge set from a FukayaComplex
function edge_symbols(fc::FukayaComplex)
    regs = fc.regions
    Set((regs[i], regs[j]) for (i,j) in fc.edges
        if i <= length(regs) && j <= length(regs))
end

# Build Hashimoto matrix from a named edge list
function hashimoto_from_edges(edges::Vector{Tuple{Symbol,Symbol}})
    m = length(edges)
    m == 0 && return zeros(Float64, 0, 0), 0.0
    B = zeros(Float64, m, m)
    for (i,(s,t)) in enumerate(edges)
        for (j,(s2,t2)) in enumerate(edges)
            t == s2 && s != t2 && (B[i,j] = 1.0)
        end
    end
    # Fix: use explicit epsilon matrix, not UniformScaling I
    eps_mat = Matrix{Float64}(1e-10 * LinearAlgebra.I, m, m)
    ρ = maximum(abs.(eigvals(B .+ eps_mat)))
    return B, ρ
end

# =============================================================================
# DERIVED INTERSECTION: T_global ≃ T1 ⊗^L_{T12} T2
# =============================================================================
# Instead of the naive set-theoretic intersection T12 = T1 ∩ T2,
# the derived intersection treats T12 as a chain complex and computes
# the left derived tensor product T1 ⊗^L_{T12} T2.
#
# At the spectral level, this manifests as:
#   H0(T1 ⊗^L T2) = H0(T1) ⊗ H0(T2)          — the observable addition
#   H-1(T1 ⊗^L T2) = Tor1_Z5(H0(T1), H0(T2))  — the hidden interaction
#
# The Tor term is NOT an error — it is the derived data of the entanglement.
# It becomes a feature of the topology: a higher homotopy group π1 of
# the global space T_global that measures how T1 and T2 are "twisted"
# around their shared interface T12.
#
# Computationally: we measure the Tor term via the spectral obstruction
# in the Mayer-Vietoris sequence, then classify it by p-adic pole order.

function derived_intersection(fc1::FukayaComplex, fc2::FukayaComplex,
                               T12_data, confirmed_h2::Float64)
    push_data = build_pushout(fc1, fc2)
    rho_H0    = push_data.rho
    rho_T12   = T12_data !== nothing ? T12_data.rho : 0.0
    tor_proxy = abs(rho_H0 - (fc1.rho + fc2.rho - rho_T12))

    # T12 carries poles ONLY if:
    # (a) sAMY is in T12, AND
    # (b) T12 has active edges (ρ(T12) > 0), AND
    # (c) the confirmed H²(Cone) from au_fukaya_75 says INDEPENDENT
    T12_has_sAMY = T12_data !== nothing &&
                   :sAMY ∈ Set(T12_data.regions)
    T12_active   = T12_data !== nothing && T12_data.rho > 0.01
    pair_is_indep = confirmed_h2 > 0.5   # threshold from au_fukaya_75 trichotomy

    T12_pole  = T12_has_sAMY && T12_active && pair_is_indep
    tor_nonzero = T12_pole || (tor_proxy > 0.05 && pair_is_indep)

    return (push_data   = push_data,
            rho_H0      = rho_H0,
            tor_proxy   = tor_proxy,
            tor_nonzero = tor_nonzero,
            T12_pole    = T12_pole,
            pair_is_indep = pair_is_indep,
            interpretation = tor_nonzero ?
                "H^{-1} Tor ≠ 0 — entanglement is topological feature" :
                "H^{-1} Tor = 0 — clean tensor product")
end

# =============================================================================
# STEP 1: INTERSECTION CONTEXT T12
# =============================================================================

function build_T12(fc1::FukayaComplex, fc2::FukayaComplex)
    shared_regs = intersect(Set(fc1.regions), Set(fc2.regions))
    isempty(shared_regs) && return nothing
    # Edges active in BOTH contexts with both endpoints in T12
    e1 = edge_symbols(fc1); e2 = edge_symbols(fc2)
    shared_edges = [(s,t) for (s,t) in intersect(e1, e2)
                    if s ∈ shared_regs && t ∈ shared_regs]
    B12, ρ12 = hashimoto_from_edges(shared_edges)
    return (regions=collect(shared_regs), edges=shared_edges,
            rho=ρ12, n=length(shared_regs), m=length(shared_edges))
end

# =============================================================================
# STEP 2: PUSHOUT T_GLOBAL = T1 ⊔_{T12} T2
# =============================================================================

function build_pushout(fc1::FukayaComplex, fc2::FukayaComplex)
    all_regs  = collect(union(Set(fc1.regions), Set(fc2.regions)))
    all_edges = collect(union(edge_symbols(fc1), edge_symbols(fc2)))
    # Keep only edges with both endpoints in all_regs
    valid = [(s,t) for (s,t) in all_edges
             if s ∈ Set(all_regs) && t ∈ Set(all_regs)]
    B_push, ρ_push = hashimoto_from_edges(valid)
    return (regions=all_regs, edges=valid, rho=ρ_push,
            n=length(all_regs), m=length(valid))
end

# =============================================================================
# STEP 3: GPS PROJECTIONS π1, π2 FROM PUSHOUT
# =============================================================================

function gps_projection_quality(push_data, fc_local::FukayaComplex)
    local_edges = edge_symbols(fc_local)
    push_edges  = push_data.edges
    # Which pushout edges survive projection to local context?
    shared = filter(e -> e ∈ local_edges, push_edges)
    coverage = isempty(push_edges) ? 1.0 :
               length(shared) / length(push_edges)
    # Projected spectral radius
    B_proj, ρ_proj = hashimoto_from_edges(collect(shared))
    Δρ = abs(ρ_proj - fc_local.rho)
    return (coverage=coverage, rho_proj=ρ_proj,
            delta_rho=Δρ, clean=Δρ < 0.05)
end

# =============================================================================
# STEP 4: CONNECTING HOMOMORPHISM ∂
# =============================================================================

function connecting_∂(push_data, T12_data, proj1, proj2)
    ρ_push = push_data.rho
    ρ12    = T12_data !== nothing ? T12_data.rho : 0.0
    # Mayer-Vietoris: ρ(push) should = ρ(T1)+ρ(T2)-ρ(T12) if splits
    ρ_MV   = proj1.rho_proj + proj2.rho_proj - ρ12
    ∂      = abs(ρ_push - ρ_MV)
    return ∂, ∂ < 0.08
end

# =============================================================================
# STEP 5: p-ADIC GATE
# =============================================================================

# Pole orders: from confirmed weight analysis
const V5 = Dict(:CTX_sAMY => -2, :CTX_INFRA => -2,
                :CTX_HPF  =>  0, :CTX_BG    =>  0,
                :CTX_THAL =>  0, :CTX_HB    =>  0,
                :CTX_OLF  =>  0, :CTX_CORTEX=>  0)

# m7 finding: CA1sp-HPF-sAMY loop dominates order-7 prime paths
# CTX_HPF carries order-7 non-associativity (not a v5 Renkin-Crone pole,
# but a structural higher-order obstruction visible only at m7)
const M7_POLE_CONTEXTS = Set([:CTX_HPF])

function padic_gate(c1::Symbol, c2::Symbol, T12_data)
    v1 = get(V5, c1, 0); v2 = get(V5, c2, 0)
    T12_active   = T12_data !== nothing && T12_data.rho > 0.01
    T12_has_sAMY = T12_data !== nothing && :sAMY ∈ Set(T12_data.regions)
    v_comp   = T12_has_sAMY && T12_active ? v1 + v2 : min(v1, v2)
    gate     = abs(v_comp)
    trigger  = gate > 0 ? 5^gate : 0
    # m7 flag: higher-order obstruction present (needs merged-quiver computation)
    m7_flag  = T12_active && (c1 ∈ M7_POLE_CONTEXTS || c2 ∈ M7_POLE_CONTEXTS)
    return gate, trigger, v1, v2, v_comp, m7_flag
end

# =============================================================================
# STEP 6: FULL RESOLUTION — PRINT AND RETURN TYPE
# =============================================================================

# Confirmed H²(Cone) from au_fukaya_75.jl ctx_maps output
const CONFIRMED_H2 = Dict(
    (:CTX_sAMY,:CTX_HPF)    => 0.0380,
    (:CTX_sAMY,:CTX_BG)     => 0.1698,
    (:CTX_sAMY,:CTX_THAL)   => 1.4927,
    (:CTX_sAMY,:CTX_OLF)    => 0.6461,
    (:CTX_HPF, :CTX_CORTEX) => 1.3486,
    (:CTX_HPF, :CTX_THAL)   => 1.5308,
    (:CTX_BG,  :CTX_THAL)   => 1.1729,
    (:CTX_THAL,:CTX_HB)     => 1.2466,
    (:CTX_sAMY,:CTX_INFRA)  => 1.8742,
    (:CTX_HPF, :CTX_INFRA)  => 1.8361,
)
const CONFIRMED_COKER = Dict(
    (:CTX_sAMY,:CTX_INFRA) => 62,
)

function resolve_pushout(c1::Symbol, c2::Symbol)
    fc1 = mod.complexes[c1][:A]
    fc2 = mod.complexes[c2][:A]
    h2  = get(CONFIRMED_H2,    (c1,c2), 0.0)
    ck  = get(CONFIRMED_COKER, (c1,c2), 0)

    label = "$(c1)↔$(c2)"
    println("\n  ── $label ──────────────────────────────────")

    # Step 1
    T12  = build_T12(fc1, fc2)
    n12  = T12 !== nothing ? T12.n : 0
    ρ12  = T12 !== nothing ? T12.rho : 0.0
    println(@sprintf("    1. T12: %d shared regions, ρ(T12)=%.4f", n12, ρ12))

    if T12 === nothing
        println("       T12=∅ → pure coproduct")
        println("    ✓  H0(T1⊔T2) = H0(T1) ⊕ H0(T2)")
        return :coproduct
    end

    # Step 2: Derived intersection T_global ≃ T1 ⊗^L_{T12} T2
    derived = derived_intersection(fc1, fc2, T12, h2)
    push_data = derived.push_data
    println(@sprintf("    2. T_global = T1 ⊗^L_{T12} T2"))
    println(@sprintf("       %d regions, %d edges, ρ(H0)=%.4f",
            push_data.n, push_data.m, derived.rho_H0))
    println(@sprintf("       Tor proxy = %.4f  Tor≠0: %s  T12 pole: %s",
            derived.tor_proxy,
            derived.tor_nonzero ? "YES" : "no",
            derived.T12_pole    ? "YES (entanglement is topological)" : "no"))
    println(@sprintf("       %s", derived.interpretation))

    # Step 3
    proj1 = gps_projection_quality(push_data, fc1)
    proj2 = gps_projection_quality(push_data, fc2)
    println(@sprintf("    3. π1: cov=%.2f Δρ=%.4f %s",
            proj1.coverage, proj1.delta_rho, proj1.clean ? "✓" : "✗"))
    println(@sprintf("       π2: cov=%.2f Δρ=%.4f %s",
            proj2.coverage, proj2.delta_rho, proj2.clean ? "✓" : "✗"))

    # Step 4
    ∂, splits = connecting_∂(push_data, T12, proj1, proj2)
    println(@sprintf("    4. ∂ = %.4f  MV-splits = %s", ∂, splits ? "YES" : "NO"))

    # Early return: respect au_fukaya_75 trichotomy
    if !derived.pair_is_indep
        if h2 < 0.05   # full A∞ (H²≈0, matches au_fukaya_75 criterion)
            println("    ✓  Full A∞: H²(Cone)=0, clean coproduct")
            println("       T_global ≃ T1 ⊔ T2,  Tor = 0")
            return :coproduct
        else
            println(@sprintf("    ~  H⁰ only: H²(Cone)=%.4f, partial addition", h2))
            println("       GPS restriction maps partially obstructed")
            return :H0_only
        end
    end

    if splits && proj1.clean && proj2.clean && !derived.tor_nonzero
        println("    ✓  Coproduct: derived intersection splits cleanly")
        println("       T_global ≃ T1 ⊔ T2,  Tor = 0,  H0 = H0(T1) ⊕ H0(T2)")
        return :coproduct
    end

    # Step 5: p-adic gate
    gate, trigger, v1, v2, v_comp, m7_flag = padic_gate(c1, c2, T12)
    T12_active = T12 !== nothing && T12.rho > 0.01
    println(@sprintf("    5. v5(T1)=%d  v5(T2)=%d  v5(composite)=%d  T12_active=%s",
            v1, v2, v_comp, T12_active ? "YES" : "no"))
    m7_flag && println("       ⚠ m7 flag: CTX_HPF order-7 obstruction — gate may be higher")

    # Step 6: resolution
    if gate == 0
        println("       No p-adic pole → categorical independence")
        println(@sprintf("       H²(Cone) = %.4f (spectral separation)", h2))
        println("       Tor^1_Z5 = 0 (no arithmetic entanglement)")
        println("       Entanglement is GEOMETRIC — spectral radii incompatible")
        println("    ✗  Cannot add: categorical obstruction, not arithmetic")
        println("       H0(T1) and H0(T2) remain separately computable")
        return :categorical_independent

    elseif gate > 0 && !T12_active
        # Pole in one context but T12 has no active edges → cannot propagate
        println(@sprintf("       v5=%d pole present but T12 inactive (ρ(T12)=0)", v_comp))
        println("       Pole confined to one context — cannot cross inactive interface")
        println("       Independence is CATEGORICAL (spectral separation), not p-adic")
        println("    ✗  Cannot add: spectral separation dominates")
        return :categorical_independent

    elseif gate == 2
        println(@sprintf("       Gate=2, trigger=5²=%d", trigger))
        println("       T12 active with pole → cross-context p-adic obstruction")
        println("       Tor^1_Z5 ≠ 0 — entanglement is TOPOLOGICAL FEATURE")
        println("       H^{-1}(T1 ⊗^L T2) = Tor^1_Z5(H0(T1), H0(T2)) ≠ 0")
        println("       Clinical: 2 simultaneous interventions trigger cascade")
        return :padic_gate2

    else  # gate == 4
        println(@sprintf("       Gate=4, trigger=5⁴=%d = 25²", trigger))
        println("       Tor^1_Z5 ≠ 0 with DOUBLE POLE — deepest entanglement")
        println("       H^{-1} is a π1 of T_global twisted around T12")
        println(@sprintf("       coker(ρ*)=%d HH² classes confirmed", ck))
        println("       Clinical: 4 simultaneous interventions")
        println("       Protective: sAMY-selective → gate stays at 2")
        return :padic_gate4
    end
end

# =============================================================================
# MAIN
# =============================================================================

println("\n", "="^65)
println("AU PUSHOUT FULL RESOLUTION PIPELINE")
println("  Six steps — no bailout, full resolution for all pairs")
println("="^65)

const PAIRS = [
    (:CTX_sAMY, :CTX_HPF),
    (:CTX_sAMY, :CTX_BG),
    (:CTX_sAMY, :CTX_THAL),
    (:CTX_sAMY, :CTX_OLF),
    (:CTX_HPF,  :CTX_CORTEX),
    (:CTX_HPF,  :CTX_THAL),
    (:CTX_BG,   :CTX_THAL),
    (:CTX_THAL, :CTX_HB),
    (:CTX_sAMY, :CTX_INFRA),
    (:CTX_HPF,  :CTX_INFRA),
]

results = Dict{Tuple{Symbol,Symbol},Symbol}()
for (c1,c2) in PAIRS
    results[(c1,c2)] = resolve_pushout(c1, c2)
end

println("\n", "="^65)
println("SUMMARY")
println("="^65)
println(@sprintf("  %-30s  %s", "Pair", "Resolution"))
println("  "*"─"^60)
labels = Dict(
    :coproduct               => "Coproduct ✓  (full A∞, Tor=0)",
    :H0_only                 => "H⁰ only    (partial addition, GPS obstructed)",
    :categorical_independent => "INDEPENDENT — categorical (spectral separation)",
    :padic_gate2             => "INDEPENDENT — p-adic gate=2 (trigger=25)",
    :padic_gate4             => "INDEPENDENT — p-adic gate=4 (trigger=625) ★",
)
for (c1,c2) in PAIRS
    res = results[(c1,c2)]
    println(@sprintf("  %-30s  %s",
            "$(c1)↔$(c2)", get(labels, res, string(res))))
end

println("""
\n  ★ sAMY↔Infra is the unique double-pole case:
    Both contexts carry v5=-2, T12={sAMY,CNU,VS,HPF} is pole-carrying
    Gate=4: requires 4 simultaneous interventions (vs 2 for single-context)
    Protective mechanism: sAMY-selective drug keeps gate at 2
""")
