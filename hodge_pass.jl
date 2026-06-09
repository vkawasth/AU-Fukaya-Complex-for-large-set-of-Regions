# =============================================================================
# hodge_pass.jl
#
# Pass 0: Discrete Hodge Decomposition
#
# Replaces three separate heuristic computations with one unified algebraic pass:
#
#   BEFORE (three separate passes):
#     Pass 2 routing score   ← uses raw ω(s,h,m)  (gradient + harmonic mixed)
#     Pass 3 stability score ← finite difference of ω   (curl proxy)
#     Pass 5 HMM bracket     ← FK expectation of ω      (harmonic proxy)
#
#   AFTER (one Hodge pass feeds all three):
#     SpectralProjection → f_gradient  → routing score  (conservative flow)
#     SpectralProjection → f_harmonic  → bracket bounds (loop corrections)
#     SpectralProjection → f_curl      → stability flag (surplus deviation)
#
# Dead-zone / hallucination detection (three-valued, not binary):
#   |f_harmonic| ≈ 0  → DEAD ZONE    (no loop flows: topologically isolated)
#   |f_curl|/|f| > 0.5 → UNSTABLE   (curl dominates: volatile placement)
#   otherwise          → LIVE         (gradient + harmonic: serve normally)
#
# Mapping to the document's LLM framing:
#   gradient  = valid forward flow (grounded inference)
#   harmonic  = topological dead zone (trapped off-manifold)
#   curl      = tautological loop (cyclic reasoning)
# =============================================================================

using LinearAlgebra, Printf

# =============================================================================
# SECTION 1: BUILD THE INCIDENCE MATRIX
# =============================================================================

"""
    build_incidence(stations, edges) -> Matrix{Float64}

Build the boundary operator ∂₁ ∈ ℝ^{|V|×|E|}.
  d1[v, e] = -1  if v is the tail of edge e
  d1[v, e] = +1  if v is the head of edge e

With this orientation:
  div(f) = d1 * f ∈ ℝ^|V|     (vertex divergence of edge flow)
  L      = d1 * d1'            (graph Laplacian |V|×|V|)
  grad(φ)= d1' * φ ∈ ℝ^|E|    (edge gradient of node potential)
"""
function build_incidence(stations::Vector{String},
                          edges::Vector{Tuple{Int,Int}})::Matrix{Float64}
    nV = length(stations)
    nE = length(edges)
    d1 = zeros(nV, nE)
    for (k, (s1, s2)) in enumerate(edges)
        d1[s1, k] = -1.0   # tail
        d1[s2, k] = +1.0   # head
    end
    return d1
end

# =============================================================================
# SECTION 2: HODGE DECOMPOSITION OF A SINGLE EDGE FLOW VECTOR
# =============================================================================

"""
    hodge_decompose(f, d1, L_pinv) -> (f_grad, f_harm, f_curl)

Decompose edge flow f ∈ ℝ^|E| into three orthogonal Hodge components.

Arguments:
  f      : edge flow vector (|E|,)
  d1     : incidence matrix |V|×|E|  (from build_incidence)
  L_pinv : precomputed pseudoinverse of L = d1*d1', shape |V|×|V|

Returns:
  f_grad : gradient component  (exact, conservative)
  f_harm : harmonic component  (closed, loop flows)
  f_curl : curl component      (coexact, surplus)  = 0 for plain graphs
"""
function hodge_decompose(f      ::Vector{Float64},
                          d1     ::Matrix{Float64},
                          L_pinv ::Matrix{Float64})
    div_f  = d1 * f                    # vertex divergence  |V|
    x      = L_pinv * div_f            # node potentials    |V|
    f_grad = d1' * x                   # gradient component |E|
    f_harm = f .- f_grad               # harmonic component |E|
    f_curl = zeros(length(f))          # β₂=0 for plain graph
    return f_grad, f_harm, f_curl
end

# =============================================================================
# SECTION 3: HODGE GATE — THREE-VALUED CLASSIFICATION
# =============================================================================

const HODGE_DEAD     = 0   # harmonic ≈ 0: topologically isolated
const HODGE_UNSTABLE = 1   # curl dominates: volatile
const HODGE_LIVE     = 2   # gradient + harmonic: serve normally

"""
    hodge_gate(f_grad, f_harm, f_curl, omega_local;
               flow_floor=1e-3, curl_ratio_ceil=0.4) -> Int

Classify a station/slot into one of three Hodge states.

The MTR graph has β₁ = 9 independent cycles (|E|-|V|+1 = 89-81+1).
Most stations lie on the spanning TREE, not on a cycle.
A tree station has harm_ratio ≈ 0 but is NOT a dead zone —
it has nonzero ridership and serves ads normally.

Correct dead-zone criterion: |f_total| ≈ 0 OR omega_local ≈ 0.
This matches W₀(u) ≈ 0 (classical superpotential near zero).

  HODGE_DEAD     (0): omega_local < flow_floor
                       No ridership: truly isolated (Lei Tung, Ocean Park off-peak)
                       ↔ W₀(u) ≈ 0, k-inv ≈ 0
                       LLM analogue: harmonic = dead zone (zero flow, off-manifold)

  HODGE_UNSTABLE (1): |f_curl| / |f| > curl_ratio_ceil
                       Curl dominates — surplus deviates far from tree baseline.
                       ↔ m₃ stability < 0.5, high temporal sensitivity
                       LLM analogue: curl = tautological loop (cyclic, no grounding)

  HODGE_LIVE     (2): omega_local > flow_floor AND curl small
                       Station has ridership and is stable.
                       ↔ k-inv > 0, stability > 0.7
                       LLM analogue: gradient = valid forward flow

Note: harm_ratio distinguishes LOOP stations (on one of β₁=9 cycles)
from TREE stations (on spanning tree). Both are LIVE if omega > 0.
Loop stations get stronger HMM bracket signals; tree stations get
pure gradient routing. Both serve ads.
"""
function hodge_gate(f_grad       ::Vector{Float64},
                    f_harm       ::Vector{Float64},
                    f_curl       ::Vector{Float64},
                    omega_local  ::Float64;
                    flow_floor      ::Float64 = 0.02,
                    curl_ratio_ceil ::Float64 = 0.40)::Int

    # Dead zone: omega below absolute threshold (caller multiplies by omega_max)
    omega_local < flow_floor && return HODGE_DEAD

    nf = norm(f_grad .+ f_harm .+ f_curl)
    nf < 1e-12 && return HODGE_DEAD

    n_curl = norm(f_curl)
    n_curl / nf > curl_ratio_ceil && return HODGE_UNSTABLE
    return HODGE_LIVE
end

# =============================================================================
# SECTION 4: PASS 0 — COMPILE HODGE TABLE
# =============================================================================

"""
    HodgeSlot

Per-(station, hour, month) Hodge classification and component norms.
Replaces the separate stab_table and k_inv scalar with a unified object.
"""
struct HodgeSlot
    station_idx ::Int
    hour        ::Int
    month       ::Int
    gate        ::Int         # HODGE_DEAD / HODGE_UNSTABLE / HODGE_LIVE
    norm_total  ::Float64
    norm_grad   ::Float64     # gradient component norm (routing signal)
    norm_harm   ::Float64     # harmonic component norm (loop / bracket signal)
    norm_curl   ::Float64     # curl component norm (volatility signal)
    grad_ratio  ::Float64     # norm_grad / norm_total
    harm_ratio  ::Float64     # norm_harm / norm_total (≈ k-inv proxy)
    curl_ratio  ::Float64     # norm_curl / norm_total (≈ 1 - stability)
end

"""
    compile_hodge_table(stations, edges, omega;
                        harm_floor, curl_ratio_ceil) -> Matrix{HodgeSlot}

Pass 0: compute Hodge decomposition for every (station, hour, month) slot.
Returns a 3D array indexed [station, hour, month].

Each slot's edge flow vector f[e] is built as the average ω at each
edge endpoint — the same signal that ω(s,h,m) embeds into the routing score.

This single pass replaces:
  • compile_stability_table  (was: finite difference of ω → curl proxy)
  • HMM bracket bounds       (was: FK expectation → harmonic proxy)
  • Dead-zone detection      (was: k-inv < 0.01 → now: harm_ratio < threshold)
"""
function compile_hodge_table(stations        ::Vector{String},
                              edges           ::Vector{Tuple{Int,Int}},
                              omega           ::Array{Float64,3};
                              flow_floor      ::Float64 = 0.02,
                              curl_ratio_ceil ::Float64 = 0.40)

    nS, nH, nM = size(omega)
    nE = length(edges)

    # Build incidence matrix once
    d1     = build_incidence(stations, edges)
    L      = d1 * d1'
    L_pinv = pinv(L)

    # omega_max used for relative dead-zone threshold
    omega_max = maximum(omega)
    omega_max < 1e-10 && (omega_max = 1.0)

    # Result array
    table = Array{HodgeSlot}(undef, nS, nH, nM)

    println("  [Hodge Pass 0] decomposing $nS stations × $nH hours × $nM months...")
    println("  [Hodge Pass 0] omega_max=$(round(omega_max,digits=4)), dead-zone threshold=$(round(flow_floor*omega_max,digits=4))")

    for m in 1:nM, h in 1:nH
        # Build edge flow vector for this (hour, month):
        # f[e] = mean of omega at the two endpoint stations
        f = zeros(nE)
        for (k, (s1, s2)) in enumerate(edges)
            f[k] = (omega[s1, h, m] + omega[s2, h, m]) / 2.0
        end

        # Hodge decompose this single flow vector
        f_grad, f_harm, f_curl = hodge_decompose(f, d1, L_pinv)

        nf    = norm(f);      nf < 1e-12 && (nf = 1.0)
        ng    = norm(f_grad)
        nh    = norm(f_harm)
        nc    = norm(f_curl)

        # Gate classification uses global flow norms (for per-station use below)
        nf_global = norm(f)
        nc_global = norm(f_curl)
        gate_global_curl = nf_global > 1e-12 && nc_global/nf_global > curl_ratio_ceil

        # Per-station slot: project the station's contribution
        for s in 1:nS
            # Station s contributes to all edges incident on it
            incident = [k for (k,(s1,s2)) in enumerate(edges) if s1==s || s2==s]
            if isempty(incident)
                table[s,h,m] = HodgeSlot(s,h,m,HODGE_DEAD,0.,0.,0.,0.,0.,0.,0.)
                continue
            end

            # Local norms: only the edges incident on this station
            f_s    = f[incident]
            fg_s   = f_grad[incident]
            fh_s   = f_harm[incident]
            fc_s   = f_curl[incident]

            nf_s   = norm(f_s);  nf_s < 1e-12 && (nf_s = 1.0)
            ng_s   = norm(fg_s)
            nh_s   = norm(fh_s)
            nc_s   = norm(fc_s)

            # Per-station gate: dead zone = low omega relative to network peak
            # omega_max precomputed outside the loop
            gate_s = hodge_gate(fg_s, fh_s, fc_s, omega[s,h,m];
                                  flow_floor=flow_floor * omega_max,
                                  curl_ratio_ceil=curl_ratio_ceil)

            table[s,h,m] = HodgeSlot(
                s, h, m,
                gate_s,
                nf_s,
                ng_s,
                nh_s,
                nc_s,
                clamp(ng_s / nf_s, 0.0, 1.0),   # grad_ratio clamped (pinv overshoot)
                clamp(nh_s / nf_s, 0.0, 1.0),   # harm_ratio
                clamp(nc_s / nf_s, 0.0, 1.0),   # curl_ratio
            )
        end
    end

    println("  [Hodge Pass 0] done.")
    return table
end

# =============================================================================
# SECTION 5: HODGE-AWARE SERVE_AD
# =============================================================================

"""
    hodge_serve_ad(ctx, hodge_table, station_idx, hour, month) -> (AdRoute|Nothing, Int)

Drop-in replacement for serve_ad that uses Hodge gate instead of
separate stability/k-inv checks.

Returns (route, gate) where gate ∈ {HODGE_DEAD, HODGE_UNSTABLE, HODGE_LIVE}.

Gate behaviour:
  HODGE_DEAD     → return nothing (no ad, regardless of routing table rank)
  HODGE_UNSTABLE → serve only if rank 1 and harm_ratio > 0.1 (weakened gate)
  HODGE_LIVE     → normal serve (pass all routing table candidates)
"""
function hodge_serve_ad(ctx          ::RuntimeContext,
                         hodge_table  ::Array{HodgeSlot},
                         station_idx  ::Int,
                         hour         ::Int,
                         month        ::Int;
                         feedback_floor::Float64 = -0.5,
                         verbose       ::Bool = false)

    slot = hodge_table[station_idx, hour, month]

    # Gate 1: dead zone (harmonic component vanishes)
    if slot.gate == HODGE_DEAD
        verbose && @printf("    [Hodge] DEAD ZONE s=%d h=%d m=%d harm_ratio=%.3f\n",
                           station_idx, hour, month, slot.harm_ratio)
        return nothing, HODGE_DEAD
    end

    candidates = get(ctx.routes, (station_idx, hour, month), AdRoute[])
    isempty(candidates) && return nothing, slot.gate

    # Gate 2: unstable (curl dominates) — only serve rank 1
    rank_floor = slot.gate == HODGE_UNSTABLE ? 1 : typemax(Int)

    for route in candidates
        route.rank > rank_floor && continue
        !get(ctx.inventory, route.product_idx, true) && continue
        fb = feedback_signal(ctx, station_idx, route.product_idx)
        fb < feedback_floor && continue

        verbose && @printf("    [Hodge] %s s=%d serve rank=%d %s score=%.4f harm=%.3f curl=%.3f\n",
                           slot.gate == HODGE_UNSTABLE ? "UNSTABLE" : "LIVE",
                           station_idx, route.rank, route.product,
                           route.score, slot.harm_ratio, slot.curl_ratio)
        return route, slot.gate
    end

    return nothing, slot.gate
end

# =============================================================================
# SECTION 6: BRIDGE — DERIVE OLD TABLES FROM HODGE TABLE
# =============================================================================
# If you want to keep using serve_ad unchanged, derive the scalar tables
# from the Hodge table instead of recomputing them from scratch.

"""
    hodge_to_stability(hodge_table) -> Array{Float64,3}

Derive the scalar stability table from the Hodge table.
  stability[s,h,m] = 1 - curl_ratio[s,h,m]
  (matches the existing compile_stability_table output)
"""
function hodge_to_stability(hodge_table::Array{HodgeSlot})::Array{Float64,3}
    nS, nH, nM = size(hodge_table)
    stab = ones(Float64, nS, nH, nM)
    for s in 1:nS, h in 1:nH, m in 1:nM
        stab[s,h,m] = 1.0 - hodge_table[s,h,m].curl_ratio
    end
    return stab
end

"""
    hodge_to_kinv(hodge_table) -> Array{Float64,3}

Derive a k-invariant proxy from the Hodge table.
  k_inv[s,h,m] = harm_ratio[s,h,m]
  (≈ k-inv from HMM brackets: 0 at dead zones, 1 at stable hubs)
"""
function hodge_to_kinv(hodge_table::Array{HodgeSlot})::Array{Float64,3}
    nS, nH, nM = size(hodge_table)
    kinv = zeros(Float64, nS, nH, nM)
    for s in 1:nS, h in 1:nH, m in 1:nM
        kinv[s,h,m] = hodge_table[s,h,m].harm_ratio
    end
    return kinv
end

# =============================================================================
# SECTION 7: REPORTING
# =============================================================================

"""
    print_hodge_report(hodge_table, stations; hours=[9,18], months=[2,7])

Print a summary of Hodge classifications across stations.
"""
function print_hodge_report(hodge_table::Array{HodgeSlot},
                             stations   ::Vector{String};
                             hours  ::Vector{Int} = [9, 18],
                             months ::Vector{Int} = [2, 7])

    println("\n" * "="^70)
    println("HODGE PASS 0 — SPECTRAL DECOMPOSITION REPORT")
    println("  grad=routing signal  harm=loop/bracket  curl=volatility")
    println("="^70)

    gate_labels = Dict(HODGE_DEAD=>"DEAD    ", HODGE_UNSTABLE=>"UNSTABLE",
                       HODGE_LIVE=>"LIVE    ")
    gate_counts = Dict(HODGE_DEAD=>0, HODGE_UNSTABLE=>0, HODGE_LIVE=>0)

    @printf("  %-22s  %s  %6s  %6s  %6s  %s\n",
            "Station", "Gate    ", "grad", "harm", "curl", "topology")
    println("  " * "─"^72)

    for (s, sname) in enumerate(stations)
        for h in hours, m in months
            slot = hodge_table[s, h, m]
            gate_counts[slot.gate] += 1
            # Only print notable slots: dead zones, unstable, or well-known hubs
            notable = slot.gate != HODGE_LIVE ||
                      sname in ("Admiralty","Central","Mong Kok","Tsim Sha Tsui",
                                "Lei Tung","Ocean Park","Tung Chung","Sunny Bay")
            notable || continue

            topo = slot.harm_ratio > 0.05 ? "LOOP" : "tree"
            @printf("  %-22s  %s  %6.3f  %6.3f  %6.3f  %s  h=%02d m=%d\n",
                    sname, gate_labels[slot.gate],
                    slot.grad_ratio, slot.harm_ratio, slot.curl_ratio,
                    topo, h, m)
        end
    end

    total = sum(values(gate_counts))
    println("\n  Summary:")
    @printf("  LIVE     : %4d / %d  (%.1f%%)\n",
            gate_counts[HODGE_LIVE], total,
            100*gate_counts[HODGE_LIVE]/total)
    @printf("  UNSTABLE : %4d / %d  (%.1f%%)\n",
            gate_counts[HODGE_UNSTABLE], total,
            100*gate_counts[HODGE_UNSTABLE]/total)
    @printf("  DEAD     : %4d / %d  (%.1f%%)\n",
            gate_counts[HODGE_DEAD], total,
            100*gate_counts[HODGE_DEAD]/total)

    println("\n  Dead zones (harmonic ≈ 0 = topologically isolated):")
    seen = Set{String}()
    for s in 1:length(stations), h in hours, m in months
        slot = hodge_table[s,h,m]
        if slot.gate == HODGE_DEAD && stations[s] ∉ seen
            @printf("    %s  harm_ratio=%.4f\n", stations[s], slot.harm_ratio)
            push!(seen, stations[s])
        end
    end

    println("\n  Most volatile (curl dominates):")
    volatile = [(hodge_table[s,h,m].curl_ratio, stations[s], h, m)
                for s in 1:length(stations), h in hours, m in months
                if hodge_table[s,h,m].gate == HODGE_UNSTABLE]
    sort!(volatile; rev=true)
    for (cr, sn, h, m) in volatile[1:min(5,end)]
        @printf("    %-22s  curl_ratio=%.3f  h=%02d m=%d\n", sn, cr, h, m)
    end
    println()
end

