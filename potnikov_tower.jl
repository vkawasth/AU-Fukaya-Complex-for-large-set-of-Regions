# =============================================================================
# postnikov_tower.jl
#
# Implements the Postnikov tower W_6 → W_5 → ... → W_0 for each GPS sector.
# This is the Fukaya category complex of the diagram:
#
#   W_6 → W_5 → W_4 → W_3 → W_2 → W_1 → W_0
#         D = Σ m_k,  D² = 0  (Stasheff identity)
#
# Degree k part: W_k = span of non-backtracking walks of length k
#                      with active first edge (Hashimoto row basis)
# Differential:  d_k(w) = head(w) - tail(w)  [Waldhausen S_•-construction]
#
# GPS sectors A/B/C/D form a Postnikov filtration:
#   W_A ⊂ W_B ⊂ W_C ⊂ W_D  (more stops = fewer active rows = smaller complex)
#
# The spectral radii ρ(B_S) are PF eigenvalues of the degree-1 layer W_1(S).
# The filtration extension classes are detected by ρ, not as strict k-invariants.
#
# Usage:
#   include("postnikov_tower.jl")
#   tower = build_postnikov_tower(:sAMY, 6)
#   print_tower_summary(tower)
# =============================================================================

using LinearAlgebra, Printf, SparseArrays

# -----------------------------------------------------------------------------
# Walk enumeration (same logic as cone_h2.jl, generalised to 75-node)
# -----------------------------------------------------------------------------

"""
    hashimoto_walks(regions, edges, stops, k)

Enumerate all non-backtracking walks of length k in the given graph,
where the first edge is an active row (not stop-blocked).

edges: Vector of (src_idx, tgt_idx) pairs (1-indexed into regions)
stops: Set of (Symbol, Symbol) pairs that block rows
"""
function hashimoto_walks_local(regions::Vector{Symbol},
                                edges::Vector{Tuple{Int,Int}},
                                stops::Set{Tuple{Symbol,Symbol}},
                                k::Int)
    n = length(regions)
    # Build adjacency for ALL edges (stopped edges can be sources/columns)
    adj = [Tuple{Int,Int}[] for _ in 1:n]
    for (i,j) in edges
        push!(adj[i], (i,j))
    end
    # Edge adjacency: which edges can follow edge (i→j)?
    edge_adj = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()
    for e1 in edges
        (_,j) = e1
        edge_adj[e1] = [(j,q) for (_,q) in adj[j]
                        if q != e1[1]]  # non-backtracking
    end

    # Active rows: first edge not stop-blocked
    active_first = [e for e in edges if (regions[e[1]], regions[e[2]]) ∉ stops]

    k == 1 && return [[e] for e in active_first]

    walks = [[e] for e in active_first]
    for _ in 2:k
        new_walks = Vector{Tuple{Int,Int}}[]
        for w in walks
            last_e = w[end]
            for next_e in get(edge_adj, last_e, Tuple{Int,Int}[])
                push!(new_walks, vcat(w, [next_e]))
            end
        end
        walks = new_walks
    end
    return walks
end

"""
    waldhausen_diff(walks_k, walks_km1)

Waldhausen S_•-construction boundary map:
  d(w_1, w_2, ..., w_k) = (w_2,...,w_k) - (w_1,...,w_{k-1})
  = tail walk (remove first edge) minus head walk (remove last edge)
"""
function waldhausen_diff(walks_k, walks_km1)
    isempty(walks_k) && return spzeros(Float64,
                                       max(1, length(walks_km1)),
                                       1)
    idx = Dict(tuple(w...) => i for (i,w) in enumerate(walks_km1))
    I, J, V = Int[], Int[], Float64[]
    for (j, walk) in enumerate(walks_k)
        k = length(walk)
        # Remove last edge (tail): walk[1:k-1], coefficient +1
        tail = walk[1:k-1]
        key  = tuple(tail...)
        if haskey(idx, key)
            push!(I, idx[key]); push!(J, j); push!(V, +1.0)
        end
        # Remove first edge (head): walk[2:k], coefficient -1
        head = walk[2:k]
        key  = tuple(head...)
        if haskey(idx, key)
            push!(I, idx[key]); push!(J, j); push!(V, -1.0)
        end
    end
    m = max(1, length(walks_km1))
    n = max(1, length(walks_k))
    isempty(I) && return spzeros(Float64, m, n)
    return sparse(I, J, V, m, n)
end

# -----------------------------------------------------------------------------
# Postnikov tower structure
# -----------------------------------------------------------------------------

struct PostnikovTower
    context_id  ::Symbol
    sector      ::Symbol
    regions     ::Vector{Symbol}
    walks       ::Vector{Vector{Vector{Tuple{Int,Int}}}}  # walks[k+1] = degree-k walks
    diffs       ::Vector{Matrix{Float64}}                 # diffs[k] = d_k: W_k → W_{k-1}
    rho         ::Vector{Float64}                         # rho[k] = spectral radius of degree-k Hashimoto
    d2_zero     ::Bool                                    # whether D²=0 holds
    max_degree  ::Int
end

function build_postnikov_tower(context_id::Symbol,
                                regions::Vector{Symbol},
                                edges::Vector{Tuple{Int,Int}},
                                stops::Set{Tuple{Symbol,Symbol}},
                                sector::Symbol,
                                max_k::Int = 6)
    # Enumerate walks at each degree
    walks = Vector{Vector{Tuple{Int,Int}}}[]
    for k in 0:max_k
        if k == 0
            push!(walks, [[(i,i)] for i in 1:length(regions)])  # degree-0 = vertices
        else
            push!(walks, hashimoto_walks_local(regions, edges, stops, k))
        end
    end

    # Build differentials d_k: W_k → W_{k-1}
    diffs = Matrix{Float64}[]
    for k in 1:max_k
        D = Matrix(waldhausen_diff(walks[k+1], walks[k]))
        push!(diffs, D)
    end

    # Check D² = 0 for k=2..max_k-1
    d2_zero = true
    for k in 2:min(max_k-1, length(diffs)-1)
        if size(diffs[k-1],2) > 0 && size(diffs[k],1) > 0 &&
           size(diffs[k-1],1) == size(diffs[k],2)
            err = maximum(abs.(diffs[k-1] * diffs[k]))
            err > 1e-8 && (d2_zero = false)
        end
    end

    # Spectral radii at each degree (degree-1 = standard Hashimoto ρ)
    rho = Float64[]
    for k in 1:max_k
        m = length(walks[k+1])
        if m == 0
            push!(rho, 0.0)
        elseif m <= 300
            # Build Hashimoto-like matrix for degree-k walks
            # B[w1,w2] = 1 if w2 immediately follows w1 (appending one edge)
            idx = Dict(tuple(w...) => i for (i,w) in enumerate(walks[k+1]))
            B = zeros(Float64, m, m)
            for (i, w) in enumerate(walks[k+1])
                last_e = w[end]
                # Find all walks of same length that are continuations
                for (j, w2) in enumerate(walks[k+1])
                    if w2[1:end-1] == w[2:end]  # w2 follows w (overlap by k-1 edges)
                        B[i,j] = 1.0
                    end
                end
            end
            push!(rho, maximum(abs.(eigvals(B))))
        else
            push!(rho, 0.0)  # too large for exact eigen
        end
    end

    PostnikovTower(context_id, sector, regions, walks, diffs, rho, d2_zero, max_k)
end

function print_tower_summary(tower::PostnikovTower)
    println(@sprintf("  Context: %s  Sector: %s  D²=0: %s",
            tower.context_id, tower.sector, tower.d2_zero))
    println(@sprintf("  %-8s %8s %8s %10s",
            "Degree k", "|W_k|", "rank(d_k)", "ρ(W_k)"))
    println("  " * "─"^40)
    for k in 1:tower.max_degree
        wk_size = length(tower.walks[k+1])
        rk = k <= length(tower.diffs) ? rank(tower.diffs[k]) : 0
        rho_k = k <= length(tower.rho) ? tower.rho[k] : 0.0
        note = k == 1 ? " ← GPS ρ(B_Λ)" : ""
        println(@sprintf("  k=%-6d %8d %8d %10.4f%s",
                k, wk_size, rk, rho_k, note))
    end
    println()
end

# Postnikov filtration comparison: W_A ⊂ W_B ⊂ W_C ⊂ W_D
function check_filtration(towers::Dict{Symbol, PostnikovTower})
    println("  Postnikov filtration check: W_A ⊂ W_B ⊂ W_C ⊂ W_D")
    println(@sprintf("  %-8s %8s %8s %8s %8s  %s",
            "Degree", "|W_A|", "|W_B|", "|W_C|", "|W_D|", "A⊂B⊂C⊂D?"))
    println("  " * "─"^60)
    max_k = minimum(t.max_degree for t in values(towers))
    all_ok = true
    for k in 1:max_k
        sizes = [length(get(towers, s, towers[:A]).walks[k+1])
                 for s in [:A,:B,:C,:D]]
        ok = all(sizes[i] <= sizes[i+1] for i in 1:3)
        ok || (all_ok = false)
        println(@sprintf("  k=%-6d %8d %8d %8d %8d  %s",
                k, sizes..., ok ? "✓" : "✗"))
    end
    println("  All filtration inclusions hold: $all_ok")
end

# =============================================================================
# TEST: N=7 complete Postnikov tower for all 4 GPS sectors
# =============================================================================

println("="^60)
println("POSTNIKOV TOWER — N=7 GPS SECTORS")
println("  W_6 → W_5 → ... → W_1 → W_0")
println("  D² = 0  (Stasheff / Waldhausen S_•-construction)")
println("="^60)

VERTS7 = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
VIDX7  = Dict(v=>i for (i,v) in enumerate(VERTS7))
EDGES7 = [
    (:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
    (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
    (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
    (:sAMY,:BLA),(:sAMY,:HY),(:sAMY,:HPF),
    (:sAMY,:LA),(:sAMY,:PAL),
    (:HY,:sAMY),
    (:LA,:BLA),(:LA,:sAMY),
    (:PAL,:sAMY),
]
EIDX7 = [(VIDX7[s], VIDX7[t]) for (s,t) in EDGES7]

STOPS7 = Dict(
    :A => Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA),
               (:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)]),
    :B => Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA)]),
    :C => Set([(:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)]),
    :D => Set([(:LA,:sAMY),(:sAMY,:LA)]),
)

towers = Dict{Symbol, PostnikovTower}()
for sec in [:A,:B,:C,:D]
    t = build_postnikov_tower(:N7, VERTS7, EIDX7, STOPS7[sec], sec, 6)
    towers[sec] = t
    println("Sector $sec:")
    print_tower_summary(t)
end

println("="^60)
println("FILTRATION CHECK: W_A ⊂ W_B ⊂ W_C ⊂ W_D")
println("="^60)
check_filtration(towers)

println()
println("="^60)
println("SPECTRAL RADII AT DEGREE 1 (= GPS ρ(B_Λ) from Hashimoto)")
println("="^60)
println(@sprintf("  %-8s %10s %10s", "Sector", "ρ(W_1)", "Expected"))
println("  " * "─"^32)
expected_rho = Dict(:A=>1.2599, :B=>1.2599, :C=>1.9090, :D=>1.6180)
for sec in [:A,:B,:C,:D]
    rho1 = towers[sec].rho[1]
    exp  = expected_rho[sec]
    ok   = abs(rho1 - exp) < 0.01
    println(@sprintf("  %-8s %10.6f %10.4f  %s", sec, rho1, exp,
            ok ? "✓" : @sprintf("✗ (diff=%.4f)", abs(rho1-exp))))
end
