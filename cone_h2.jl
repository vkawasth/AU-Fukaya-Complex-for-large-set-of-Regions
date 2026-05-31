# =============================================================================
# cone_h2_v3.jl  —  H²(Cone(ρ)) via Hashimoto chain complexes
#
# The correct object: the GPS Fukaya complex W_•(S) is the HASHIMOTO COMPLEX,
# not the full bar complex of the path algebra.
#
# W_k(S) = span of non-backtracking walks of length k whose rows are active
#         = the k-th power of the Hashimoto matrix B_S
# 
# The chain complex structure comes from the Waldhausen S_•-construction:
#   W_2(S) →^{d_2} W_1(S) →^{d_1} W_0(S)
# where:
#   d_1 = boundary map: edge (i→j) ↦ j - i  (target minus source)
#   d_2 = composition:  walk (i→j→k) ↦ (i→k) - (i→j) + (j→k)  ... BUT
#         only when the composed edge (i→k) exists in the quiver
#
# The RESTRICTION MAP ρ_{S1→S2}: W_k(S1) → W_k(S2) is the inclusion
# of active rows (S1 has more stops = fewer active rows ⊂ S2 active rows)
#
# H²(Cone(ρ)) measures whether this inclusion is a quasi-isomorphism:
#   H²=0  ↔  ρ is quasi-iso  ↔  :full_Ainf
#   H²≠0  ↔  obstruction     ↔  :independent (crisis)
#
# SIMPLER EQUIVALENT: 
#   H²(Cone(ρ: W→V)) ≅ H²(V/W) (relative homology for injection W↪V)
#   = ker(d_2^V restricted to V_2/W_2) / im(d_3^V restricted to V_3/W_3)
# =============================================================================

using LinearAlgebra, Printf

# =============================================================================
# N=7 quiver data (from fukaya_gps_sectors.jl)
# =============================================================================

const VERTS = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
const N     = length(VERTS)
const VIDX  = Dict(v=>i for (i,v) in enumerate(VERTS))

const EDGES = [  # all 18 directed edges
    (:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
    (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
    (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
    (:sAMY,:BLA),(:sAMY,:HY),(:sAMY,:HPF),
    (:sAMY,:LA),(:sAMY,:PAL),
    (:HY,:sAMY),
    (:LA,:BLA),(:LA,:sAMY),
    (:PAL,:sAMY),
]
const EDGE_IDX = Dict(e=>i for (i,e) in enumerate(EDGES))

const STOPS = Dict(
    :A => Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA),
               (:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)]),
    :B => Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA)]),
    :C => Set([(:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)]),
    :D => Set([(:LA,:sAMY),(:sAMY,:LA)]),
)

# =============================================================================
# HASHIMOTO COMPLEX
# Basis of W_k(S) = active rows of B^{k-1} = non-backtracking walks of length k
# whose FIRST edge is not stop-blocked (row active in Hashimoto)
#
# W_1(S) = active directed edges (rows of B_S that are non-zero)
# W_2(S) = composable pairs of active edges (non-backtracking 2-walks)
# W_3(S) = composable triples ...
# =============================================================================

function active_rows(sec::Symbol)
    # Active rows = edges (i→j) where (VERTS[i],VERTS[j]) ∉ STOPS[sec]
    # (matching fukaya_gps_sectors.jl Hashimoto semantics)
    s = STOPS[sec]
    [e for e in EDGES if e ∉ s]
end

function hashimoto_walks(sec::Symbol, k::Int)
    # All non-backtracking walks of length k whose first edge is an active row
    act = Set(active_rows(sec))
    # Build adjacency for all 18 edges (stopped edges can be SOURCES)
    adj = Dict{Tuple{Symbol,Symbol}, Vector{Tuple{Symbol,Symbol}}}()
    for e1 in EDGES
        adj[e1] = Tuple{Symbol,Symbol}[]
        (_, t1) = e1
        for e2 in EDGES
            (s2, t2) = e2
            if s2 == t1 && t2 != e1[1]  # composable, non-backtracking
                push!(adj[e1], e2)
            end
        end
    end
    
    if k == 1
        return [[e] for e in EDGES if e ∈ act]
    end
    
    # Build walks of length k: first edge must be active row
    walks = [[e] for e in EDGES if e ∈ act]
    for _ in 2:k
        new_walks = Vector{Tuple{Symbol,Symbol}}[]
        for w in walks
            last_e = w[end]
            for next_e in adj[last_e]
                push!(new_walks, vcat(w, [next_e]))
            end
        end
        walks = new_walks
    end
    return walks
end

# =============================================================================
# BOUNDARY MAP d_k: W_k → W_{k-1}
# For the Hashimoto complex, d_1(e) = tgt(e) - src(e) in vertex space
# For d_k (k≥2): remove first or last edge of the walk
# d_k(w_1,...,w_k) = (w_2,...,w_k) - (w_1,...,w_{k-1})  [Waldhausen-type]
# This gives the 2-term differential: d²=0 by cancellation
# =============================================================================

function boundary_map(walks_k, walks_km1)
    isempty(walks_k) && return zeros(Float64, max(1,length(walks_km1)),
                                              max(1,length(walks_k)))
    idx = Dict(w => i for (i,w) in enumerate(walks_km1))
    m, n = length(walks_km1), length(walks_k)
    D = zeros(Float64, m, n)
    for (j, walk) in enumerate(walks_k)
        k = length(walk)
        # Remove last edge: walk[1:k-1]
        tail = walk[1:k-1]
        if haskey(idx, tail)
            D[idx[tail], j] += 1.0
        end
        # Remove first edge: walk[2:k]
        head = walk[2:k]
        if haskey(idx, head)
            D[idx[head], j] -= 1.0
        end
    end
    return D
end

# =============================================================================
# VERIFY d²=0
# =============================================================================
println("="^60)
println("VERIFY: Hashimoto complex satisfies d²=0")
println("="^60)

for sec in [:A,:B,:C,:D]
    w1 = hashimoto_walks(sec, 1)
    w2 = hashimoto_walks(sec, 2)
    w3 = hashimoto_walks(sec, 3)
    d1 = boundary_map(w2, w1)
    d2 = boundary_map(w3, w2)
    # d1 ∘ d2 should be zero (when dimensions match)
    if size(d1,2) == size(d2,1)
        err = maximum(abs.(d1 * d2))
        println("  Sector $sec: |W_1|=$(length(w1)) |W_2|=$(length(w2)) |W_3|=$(length(w3))  d²=0: $(err < 1e-10)")
    else
        println("  Sector $sec: |W_1|=$(length(w1)) |W_2|=$(length(w2)) |W_3|=$(length(w3))  (dim mismatch)")
    end
end

# =============================================================================
# SECTOR HOMOLOGY
# H_1 = ker(d_1: W_1→W_0) ... but W_0 = vertex space for length-0 walks
# More useful: H_k = ker(d_k) / im(d_{k+1})
# For the Hashimoto complex W_1→W_2→W_3:
# H_1(W_•) = ker(d_1) [no W_0 boundary]
# H_2(W_•) = ker(d_2) / im(d_1^T) ... 
# Actually use: H_k = nullity(d_k outgoing) - rank(d_{k+1} incoming)
# =============================================================================
println()
println("="^60)
println("SECTOR HOMOLOGY H_k(Hashimoto complex)")
println("="^60)
println(@sprintf("  %-8s %6s %6s", "Sector", "|W_1|", "|W_2|"))
for sec in [:A,:B,:C,:D]
    w1 = hashimoto_walks(sec,1)
    w2 = hashimoto_walks(sec,2)
    w3 = hashimoto_walks(sec,3)
    d1 = boundary_map(w2,w1)
    d2 = boundary_map(w3,w2)
    h1 = size(d1,1) - rank(d1)   # ker(d1: W2→W1) in cochain sense
    h2 = size(d2,1) - rank(d2) - rank(d1)
    println(@sprintf("  %-8s %6d %6d  H₁=%d H₂=%d", sec, length(w1), length(w2), h1, h2))
end

# =============================================================================
# H²(Cone(ρ)) — RELATIVE HOMOLOGY
# For ρ: W_•(S1) ↪ W_•(S2) an injection of chain complexes,
# H²(Cone(ρ)) = H²(W_•(S2), W_•(S1)) = relative homology
# = ker(d_2^{S2} on W_2(S2)/W_2(S1)) / im(d_3^{S2} on W_3(S2)/W_3(S1))
#
# Concretely: identify which S2-walks are NOT in S1 (the cokernel basis),
# restrict d_2^{S2} to those walks, and compute its homology.
# =============================================================================
println()
println("="^60)
println("H²(Cone(ρ)) via relative homology")
println("="^60)

function cone_h2_relative(sec1::Symbol, sec2::Symbol)
    # sec1 = more stops (smaller complex), sec2 = fewer stops (larger)
    w2_s1 = Set(hashimoto_walks(sec1, 2))
    w2_s2 = hashimoto_walks(sec2, 2)
    w3_s2 = hashimoto_walks(sec2, 3)
    w1_s2 = hashimoto_walks(sec2, 1)
    
    # Relative W_2 = walks in S2 that are NOT in S1
    rel_w2 = [w for w in w2_s2 if w ∉ w2_s1]
    isempty(rel_w2) && return 0, true
    
    # d_2 restricted to rel_w2 (mapping to W_1(S2))
    # The relative complex: d maps rel_w2 → rel_w1 (walks in S2 not in S1)
    w1_s1 = Set(hashimoto_walks(sec1, 1))
    rel_w1 = [w for w in w1_s2 if w ∉ w1_s1]
    
    # Build restricted boundary map
    d2_rel = boundary_map(rel_w2, isempty(rel_w1) ? w1_s2 : rel_w1)
    
    # Relative W_3
    w3_s1 = Set(hashimoto_walks(sec1, 3))
    rel_w3 = [w for w in w3_s2 if w ∉ w3_s1]
    
    d3_rel = if isempty(rel_w3) || isempty(rel_w2)
        zeros(Float64, length(rel_w2), 1)
    else
        boundary_map(rel_w3, rel_w2)
    end
    
    ker_d2 = size(d2_rel,2) - rank(d2_rel)
    im_d3  = size(d3_rel,1) > 0 ? rank(d3_rel) : 0
    h2 = max(0, ker_d2 - im_d3)
    return h2, h2 == 0
end

expected = Dict(
    (:A,:B) => "full A∞ (inertia: ρ(A)=ρ(B)=1.2599)",
    (:A,:C) => "INDEPENDENT (crisis: ρ jumps to 1.909)",
    (:A,:D) => "H⁰ only",
    (:B,:D) => "H⁰ only",
    (:C,:D) => "H⁰ only",
)

println(@sprintf("  %-6s %8s  %-12s  %s", "Map","H²(Cone)","Type","Expected"))
println("  " * "─"^62)
for (s1,s2) in [(:A,:B),(:A,:C),(:A,:D),(:B,:D),(:C,:D)]
    h2, is_zero = cone_h2_relative(s1,s2)
    t = is_zero ? "full A∞" : "OBSTRUCTED"
    println(@sprintf("  %s→%s  %8d  %-12s  %s", s1, s2, h2, t,
            get(expected,(s1,s2),"?")))
end

println()
println("  H²=0 ↔ quasi-isomorphism ↔ :full_Ainf")
println("  H²≠0 ↔ obstruction       ↔ :independent (crisis)")
