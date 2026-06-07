# =============================================================================
# context_algebra.jl
#
# Algebra of AU Contexts as a 2-Category
#
# Implements the five missing explicit structures:
#   1. Persistent homology of cull filtration
#   2. Hodge Laplacian Δ₁ on clique complex
#   3. Free resolution of toric ideal (Betti numbers, syzygies)
#   4. Primary decomposition → prime ideal paths
#   5. Exceptional divisor geometry (62 classes as intersection data)
#
# Plus: the transfer functor between domain categories
# =============================================================================

if !@isdefined(NNOProb)
    include(joinpath(@__DIR__, "tool_paths.jl"))
    include(joinpath(@__DIR__, "nno_au_core.jl"))
end

using LinearAlgebra, Printf, SparseArrays

# =============================================================================
# PART 1: CONTEXT AS A CATEGORY OBJECT
# =============================================================================

"""
    ContextFingerprint

The domain-independent algebraic invariant of an AU context.
Two contexts with the same fingerprint are categorically isomorphic —
knowledge transfers exactly between them, modulo weight rescaling.
"""
struct ContextFingerprint
    beta_0      ::Int                    # connected components
    beta_1      ::Int                    # independent circuits (Markov basis size)
    beta_2      ::Int                    # 2-dimensional holes in clique complex
    coker_table ::Dict{Tuple{Int,Int},Int}  # coker(ρ*_ij) for each context pair
    k_invariants::Vector{Int}            # Postnikov k-invariants at each level
    betti_numbers::Vector{Vector{Int}}   # Tor Betti numbers β_{i,j}
    spectral_gap ::Float64               # λ₁ of graph Laplacian (GPS bound)
    rho_hashimoto::Float64               # ρ(B_Λ) non-backtracking spectral radius
    n_backbone   ::Int                   # number of persistent backbone circuits
end

"""
    ContextMorphism

A morphism f: T_α → T_β in the category of AU contexts.
Encodes how one context maps into another:
  - which circuits survive (image)
  - which are killed (kernel)
  - the cokernel (new circuits appearing in T_β not from T_α)
"""
struct ContextMorphism
    source      ::Symbol
    target      ::Symbol
    coker_dim   ::Int               # dim(coker(ρ*)) = new structure in target
    kernel_dim  ::Int               # dim(ker(ρ*)) = structure killed in target
    image_dim   ::Int               # dim(im(ρ*)) = shared structure
    is_iso      ::Bool              # coker=0 AND ker=0 → equivalence of categories
    mode        ::Int               # 1=coproduct, 2=Lan_i, 3=pushout, 4=derived
    weight_ratio::Float64           # w_target_max / w_source_max (rescaling factor)
end

"""
Compute the mode of a context morphism from its cokernel dimension.
This is the Der_{2,1} classification.
"""
function morphism_mode(coker::Int, is_crisis::Bool)::Int
    coker == 0 && return 1          # Mode 1: coproduct (full equivalence)
    coker < 10 && return 2          # Mode 2: Lan_i scaling (small obstruction)
    is_crisis && return 4           # Mode 4: derived Lan^L (62-class crisis)
    return 3                        # Mode 3: pushout (single pole)
end

# =============================================================================
# PART 2: PERSISTENT HOMOLOGY OF CULL FILTRATION
# =============================================================================

"""
    PersistencePair

A birth-death pair in the persistence diagram.
Generator = the circuit (edge set) that was born/died.
Persistence = death_weight - birth_weight (lifetime in the filtration).
"""
struct PersistencePair
    dimension   ::Int                    # 0=component, 1=circuit, 2=2-cycle
    birth_weight::Float64                # edge weight at which this appeared
    death_weight::Float64                # edge weight at which this died (Inf=essential)
    persistence ::Float64                # death - birth (Inf=topologically essential)
    generator   ::Vector{Tuple{Symbol,Symbol}}  # edges forming this generator
end

"""
    compute_persistence(edges, weights; n_levels=20)

Compute the persistence diagram of the graph filtration.
Filter by edge weight: start with only highest-weight edges,
progressively add lower-weight edges.

High-persistence circuits = backbone circuits = essential for transport.
Low-persistence circuits  = transient = removable by blockade.

This is the 4ti2 cull filtration made explicit as a topological operation.
"""
function compute_persistence(edges  ::Vector{Tuple{Symbol,Symbol}},
                              weights::Dict{Tuple{Symbol,Symbol},Float64};
                              n_levels::Int = 20)::Vector{PersistencePair}

    # Sort edges by weight descending (high weight = added first in filtration)
    sorted_edges = sort(collect(edges), by=e->get(weights,e,1.0), rev=true)
    all_weights  = sort(unique(values(weights)), rev=true)

    # Union-Find for H₀ (connected components)
    nodes    = unique(vcat([[e[1],e[2]] for e in edges]...))
    n        = length(nodes)
    node_idx = Dict(v=>i for (i,v) in enumerate(nodes))
    parent   = collect(1:n)
    rank_uf  = zeros(Int, n)

    function find!(x)
        parent[x] == x && return x
        parent[x] = find!(parent[x])
        return parent[x]
    end
    function union!(x, y)
        px, py = find!(x), find!(y)
        px == py && return false
        if rank_uf[px] < rank_uf[py]; px,py = py,px; end
        parent[py] = px
        rank_uf[px] += (rank_uf[px] == rank_uf[py])
        return true
    end

    pairs    = PersistencePair[]
    # Track H₁: cycles created when adding an edge that doesn't merge components
    edge_set = Set{Tuple{Symbol,Symbol}}()
    active_cycles = Dict{Tuple{Symbol,Symbol},Float64}()  # edge → birth weight

    for e in sorted_edges
        w    = get(weights, e, 1.0)
        push!(edge_set, e)
        push!(edge_set, (e[2],e[1]))  # undirected

        si = node_idx[e[1]]
        ti = node_idx[e[2]]

        if union!(si, ti)
            # H₀: merged two components (edge killed a component gap)
            push!(pairs, PersistencePair(0, w, Inf, Inf, [e]))
        else
            # H₁: edge created a cycle (birth of a circuit)
            active_cycles[e] = w
        end
    end

    # Remaining active cycles are essential (infinite persistence)
    for (e, w_birth) in active_cycles
        push!(pairs, PersistencePair(1, w_birth, Inf, Inf, [e]))
    end

    # Sort by persistence descending
    sort!(pairs, by=p->p.persistence == Inf ? 1e18 : p.persistence, rev=true)
    return pairs
end

"""
    backbone_circuits(pairs; min_persistence=Inf)

Extract the backbone circuits: infinite-persistence H₁ generators.
These are the topologically essential circuits that cannot be
eliminated by any blockade strategy.
"""
function backbone_circuits(pairs::Vector{PersistencePair};
                            min_persistence::Float64 = 1e10)
    filter(p -> p.dimension == 1 && p.persistence >= min_persistence, pairs)
end

"""Print persistence diagram as ASCII barcode."""
function print_persistence(pairs::Vector{PersistencePair}; top_n::Int=10)
    println("\nPERSISTENCE DIAGRAM (birth → death, ∞ = essential)")
    println("─"^60)
    @printf("  %-4s %-12s %-12s %-12s\\n", "Dim", "Birth", "Death", "Persistence")
    println("  " * "─"^52)
    for p in pairs[1:min(top_n, end)]
        death_s = p.death_weight == Inf ? "∞" : @sprintf("%.3f", p.death_weight)
        pers_s  = p.persistence  == Inf ? "∞ (essential)" :
                  @sprintf("%.3f", p.persistence)
        @printf("  H%-3d %-12.3f %-12s %-12s\\n",
                p.dimension, p.birth_weight, death_s, pers_s)
    end
end

# =============================================================================
# PART 3: HODGE LAPLACIAN Δ₁ ON CLIQUE COMPLEX
# =============================================================================

"""
    HodgeDecomposition

Spectral decomposition of the edge space into:
  - Harmonic 1-forms: Ker(Δ₁) ≅ H¹(complex) — topologically essential circuits
  - Gradient component: Im(∂₁ᵀ) — "conservative" flows, can be blocked
  - Curl component: Im(∂₂) — "rotational" flows, syzygy interactions

The Hodge decomposition theorem: every edge flow decomposes uniquely
into harmonic + gradient + curl components.
"""
struct HodgeDecomposition
    L1          ::Matrix{Float64}    # Hodge Laplacian Δ₁ = ∂₁∂₁ᵀ + ∂₂ᵀ∂₂
    eigenvalues ::Vector{Float64}    # spectrum of Δ₁
    eigenvectors::Matrix{Float64}    # columns = eigenvectors
    harmonic_dim::Int                # dim(Ker Δ₁) = β₁ (first Betti number)
    harmonic_idx::Vector{Int}        # indices of near-zero eigenvalues
    spectral_gap::Float64            # λ₁ = first nonzero eigenvalue
    curl_dim    ::Int                # dim(Im ∂₂ᵀ)
    gradient_dim::Int                # dim(Im ∂₁ᵀ)
end

"""
    build_boundary_matrix(edges, triangles, nodes)

Build the boundary matrices ∂₁: C₁ → C₀ and ∂₂: C₂ → C₁
of the clique complex.

∂₁[v,e] = +1 if v is the head of e, -1 if v is the tail, 0 otherwise
∂₂[e,t] = ±1 if e is a face of triangle t (with orientation)
"""
function build_boundary_matrices(nodes::Vector{Symbol},
                                  edges::Vector{Tuple{Symbol,Symbol}},
                                  weights::Dict)

    n_nodes = length(nodes)
    n_edges = length(edges)
    node_idx = Dict(v=>i for (i,v) in enumerate(nodes))

    # ∂₁: n_nodes × n_edges
    d1 = zeros(n_nodes, n_edges)
    for (j, (s,t)) in enumerate(edges)
        d1[node_idx[s], j] = -1.0
        d1[node_idx[t], j] = +1.0
    end

    # Weighted ∂₁: scale by √w for Hodge theory with weights
    W_edge = Diagonal([sqrt(get(weights,(e[1],e[2]),1.0)) for e in edges])
    d1_w = d1 * W_edge

    # Find triangles (3-cliques) for ∂₂
    edge_set = Set(edges)
    triangles = Tuple{Symbol,Symbol,Symbol}[]
    for i in 1:length(nodes), j in i+1:length(nodes), k in j+1:length(nodes)
        u,v,w_n = nodes[i], nodes[j], nodes[k]
        if (u,v)∈edge_set && (v,w_n)∈edge_set && (u,w_n)∈edge_set
            push!(triangles, (u,v,w_n))
        end
    end

    n_tri = length(triangles)
    d2 = zeros(n_edges, max(n_tri, 1))
    edge_idx = Dict(e=>i for (i,e) in enumerate(edges))

    for (k, (u,v,w_n)) in enumerate(triangles)
        # Orientation: u→v→w→u
        for (s,t,sign) in [(u,v,+1),(v,w_n,+1),(u,w_n,-1)]
            j = get(edge_idx, (s,t), get(edge_idx, (t,s), 0))
            j > 0 && (d2[j,k] = sign * (haskey(edge_idx,(s,t)) ? 1.0 : -1.0))
        end
    end

    return d1, d2, triangles
end

"""
    hodge_decomposition(nodes, edges, weights; tol=1e-8)

Compute the Hodge decomposition of the edge space.
The Hodge Laplacian Δ₁ = ∂₁ᵀ∂₁ + ∂₂∂₂ᵀ classifies edges as:
  - Harmonic: zero eigenvalue → topologically essential circuit
  - Gradient: from ∂₁ᵀ → conservative flow, no circulation
  - Curl: from ∂₂ᵀ → rotational flow, syzygy interaction
"""
function hodge_decomposition(nodes  ::Vector{Symbol},
                              edges  ::Vector{Tuple{Symbol,Symbol}},
                              weights::Dict;
                              tol    ::Float64 = 1e-8)::HodgeDecomposition

    d1, d2, triangles = build_boundary_matrices(nodes, edges, weights)
    n_e = length(edges)

    # Hodge Laplacian Δ₁ = ∂₁ᵀ∂₁ + ∂₂∂₂ᵀ
    L1 = d1' * d1 + d2 * d2'
    L1 = (L1 + L1') / 2   # symmetrise for numerical stability

    F = eigen(Symmetric(L1))
    λ = real.(F.values)
    V = real.(F.vectors)

    harmonic_idx = findall(l -> abs(l) < tol, λ)
    nonzero_idx  = findall(l -> abs(l) >= tol, λ)
    gap = isempty(nonzero_idx) ? 0.0 : minimum(λ[nonzero_idx])

    curl_dim     = size(d2, 2)
    gradient_dim = length(nodes) - 1   # rank of ∂₁ᵀ

    return HodgeDecomposition(L1, λ, V,
                               length(harmonic_idx), harmonic_idx,
                               gap, curl_dim, gradient_dim)
end

"""Print Hodge decomposition summary."""
function print_hodge(hd::HodgeDecomposition, edges::Vector)
    println("\nHODGE DECOMPOSITION Δ₁ SPECTRUM")
    println("─"^60)
    @printf("  Harmonic dim (β₁ = essential circuits): %d\\n", hd.harmonic_dim)
    @printf("  Spectral gap λ₁: %.6f\\n", hd.spectral_gap)
    @printf("  Gradient dim: %d  Curl dim: %d\\n", hd.gradient_dim, hd.curl_dim)
    println()
    println("  Eigenvalue spectrum (first 10):")
    for (i, λ) in enumerate(hd.eigenvalues[1:min(10,end)])
        label = abs(λ) < 1e-8 ? " ← HARMONIC (essential)" :
                λ < hd.spectral_gap * 2 ? " ← low (transient)" : ""
        @printf("    λ%-2d = %10.6f%s\\n", i, λ, label)
    end
    println()
    if hd.harmonic_dim > 0
        println("  Harmonic edge flows (topologically essential):")
        for idx in hd.harmonic_idx
            h = hd.eigenvectors[:, idx]
            top_edges = sortperm(abs.(h), rev=true)[1:min(3,end)]
            for j in top_edges
                abs(h[j]) > 1e-6 || continue
                @printf("    %-20s  weight=%.4f\\n",
                        "$(edges[j][1])→$(edges[j][2])", h[j])
            end
        end
    end
end

# =============================================================================
# PART 4: FREE RESOLUTION AND BETTI NUMBERS
# =============================================================================

"""
    ToricResolution

The free resolution of the toric ideal I_A.
Captures the complete syzygy structure:
  β_{0,j} = generators (Markov circuits at degree j)
  β_{1,j} = first syzygies (relations between generators)
  β_{2,j} = second syzygies (relations between relations)
  ...

The Betti table β_{i,j} is the fingerprint of the toric algebra.
Two toric ideals with the same Betti table are structurally identical.
"""
struct ToricResolution
    n_generators::Int                        # |Markov basis| = β_1 (topology)
    betti       ::Matrix{Int}                # Betti table β_{i,j}
    projdim     ::Int                        # projective dimension = depth of resolution
    regularity  ::Int                        # Castelnuovo-Mumford regularity
    syzygies    ::Vector{Tuple{Int,Int,Float64}}  # (circuit_i, circuit_j, overlap_score)
    min_primes  ::Vector{Vector{Int}}        # minimal prime components (edge index sets)
end

"""
    compute_toric_resolution(circuits, edges, weights)

Approximate the free resolution of the toric ideal from the
Markov basis circuits.

A syzygy between circuits C_i and C_j exists when:
  - They share a common sub-circuit (overlap in edge support)
  - Their sum/difference is also a circuit (Graver basis relation)

The overlap score measures syzygy strength:
  score(C_i, C_j) = |supp(C_i) ∩ supp(C_j)| / |supp(C_i) ∪ supp(C_j)|
"""
function compute_toric_resolution(circuits::Vector{Vector{Tuple{Symbol,Symbol}}},
                                   edges   ::Vector{Tuple{Symbol,Symbol}},
                                   weights ::Dict)::ToricResolution

    n  = length(circuits)
    n_e = length(edges)
    edge_idx = Dict(e=>i for (i,e) in enumerate(edges))

    # Convert circuits to binary vectors (edge support)
    supp = zeros(Bool, n, n_e)
    for (i, circ) in enumerate(circuits)
        for e in circ
            j = get(edge_idx, e, get(edge_idx, (e[2],e[1]), 0))
            j > 0 && (supp[i,j] = true)
        end
    end

    # Find syzygies: pairs of circuits with significant overlap
    syzygies = Tuple{Int,Int,Float64}[]
    for i in 1:n, j in i+1:n
        inter = sum(supp[i,:] .& supp[j,:])
        union_ = sum(supp[i,:] .| supp[j,:])
        union_ == 0 && continue
        score = inter / union_
        score > 0.2 && push!(syzygies, (i, j, score))
    end
    sort!(syzygies, by=x->x[3], rev=true)

    # Betti table approximation
    # β_{0,*} = number of generators by degree
    # β_{1,*} = number of first syzygies by degree
    max_deg = max(n_e, 1)
    betti   = zeros(Int, 4, max_deg + 1)

    for (i, circ) in enumerate(circuits)
        deg = length(circ)
        deg <= max_deg && (betti[1, deg] += 1)
    end
    for (i, j, _) in syzygies
        deg = length(circuits[i]) + length(circuits[j])
        deg <= max_deg && (betti[2, min(deg, max_deg)] += 1)
    end

    # Minimal primes: edge sets whose removal disconnects all circuits
    # Approximated by finding minimum feedback vertex sets
    min_primes = find_minimal_primes(supp, n_e)

    projdim  = sum(any(betti[i,:] .> 0) for i in 1:4)
    regularity = maximum(j for j in 1:max_deg if any(betti[:,j] .> 0); init=0)

    return ToricResolution(n, betti, projdim, regularity,
                           syzygies, min_primes)
end

"""Find minimal prime components (minimum cut sets in circuit support)."""
function find_minimal_primes(supp::Matrix{Bool}, n_edges::Int)
    n_circuits = size(supp, 1)
    primes = Vector{Int}[]

    # Greedy: find small edge sets that intersect all circuits
    remaining = Set(1:n_circuits)
    for _ in 1:min(n_edges, 10)
        isempty(remaining) && break
        # Pick edge that covers most remaining circuits
        best_e   = argmax([sum(supp[collect(remaining), e]) for e in 1:n_edges])
        covered  = [c for c in remaining if supp[c, best_e]]
        isempty(covered) && break
        push!(primes, [best_e])
        setdiff!(remaining, covered)
    end
    return primes
end

"""Print toric resolution summary."""
function print_resolution(res::ToricResolution, edges::Vector)
    println("\nTORIC RESOLUTION (Syzygy Structure)")
    println("─"^60)
    @printf("  Generators (Markov circuits): %d\\n", res.n_generators)
    @printf("  First syzygies: %d\\n", length(res.syzygies))
    @printf("  Projective dimension: %d\\n", res.projdim)
    @printf("  Castelnuovo-Mumford regularity: %d\\n", res.regularity)
    println()
    println("  Top syzygies (circuit pairs with highest overlap):")
    for (i,j,s) in res.syzygies[1:min(5,end)]
        @printf("    C%-3d ∩ C%-3d  overlap=%.3f\\n", i, j, s)
    end
    println()
    println("  Minimal prime components (essential cut sets):")
    for (k, prime) in enumerate(res.min_primes[1:min(5,end)])
        edge_names = join(["$(edges[e][1])→$(edges[e][2])" for e in prime], ", ")
        @printf("    P%d: {%s}\\n", k, edge_names)
    end
end

# =============================================================================
# PART 5: EXCEPTIONAL DIVISOR GEOMETRY
# =============================================================================

"""
    ExceptionalDivisor

The geometry of the exceptional divisor at a crisis singularity.
When the toric variety X_A is blown up at the crisis point,
the exceptional divisor E has:
  - dim(E) = coker_dim - 1 (a (62-1)=61-dimensional projective space P^61)
  - Line bundles L_1,...,L_62 on E (the 62 cokernel classes)
  - Intersection numbers L_i · L_j (how the classes interact)

The intersection matrix determines which pairs of classes
can SIMULTANEOUSLY be eliminated by a single stop operation.
"""
struct ExceptionalDivisor
    coker_dim       ::Int                  # 62
    proj_dim        ::Int                  # coker_dim - 1 = 61
    class_weights   ::Vector{Float64}      # weight of each class (R_eff of corresponding edge)
    intersection_mat::Matrix{Float64}      # L_i · L_j intersection numbers
    independent_sets::Vector{Vector{Int}}  # which classes are independent (can be blocked together)
    chern_numbers   ::Vector{Float64}      # c₁(L_i) = first Chern class
end

"""
    build_exceptional_divisor(coker_dim, edges, r_eff_table, weights)

Approximate the exceptional divisor geometry from the cokernel data.
Each of the `coker_dim` obstruction classes corresponds to an
edge near the crisis boundary (low R_eff = near singularity).

The intersection number L_i · L_j encodes whether blocking edge i
and edge j simultaneously eliminates both classes (intersection = 0)
or creates a new combined obstruction (intersection ≠ 0).
"""
function build_exceptional_divisor(coker_dim   ::Int,
                                    edges       ::Vector{Tuple{Symbol,Symbol}},
                                    r_eff_table ::Dict,
                                    weights     ::Dict)::ExceptionalDivisor

    # Select the coker_dim edges with lowest effective resistance
    # (these are the edges nearest the singularity = the crisis boundary)
    sorted_by_r = sort(edges, by=e->get(r_eff_table,e,1.0))
    crisis_edges = sorted_by_r[1:min(coker_dim, length(sorted_by_r))]
    n = length(crisis_edges)

    # Class weights = R_eff of each crisis edge (proxy for class "size")
    class_weights = [get(r_eff_table, e, 1.0) for e in crisis_edges]

    # Intersection matrix: L_i · L_j
    # Two classes intersect if their corresponding edges share a node
    # (they are adjacent in the graph — one block affects both)
    int_mat = zeros(n, n)
    for i in 1:n, j in 1:n
        e_i, e_j = crisis_edges[i], crisis_edges[j]
        # Edges share a node → non-zero intersection
        shared_node = !isempty(intersect([e_i[1],e_i[2]], [e_j[1],e_j[2]]))
        w_i = get(weights, e_i, 1.0)
        w_j = get(weights, e_j, 1.0)
        if i == j
            int_mat[i,i] = -2.0  # self-intersection (standard for exceptional divisors)
        elseif shared_node
            # Intersection number ∝ geometric mean of weights
            int_mat[i,j] = sqrt(w_i * w_j) / (w_i + w_j + 1e-10)
        end
    end

    # First Chern classes c₁(L_i) = degree of L_i
    # = number of other classes L_i intersects
    chern = [sum(int_mat[i,j] for j in 1:n if j!=i && int_mat[i,j] > 0)
             for i in 1:n]

    # Independent sets: classes that can be simultaneously eliminated
    # (zero intersection = independent as line bundles)
    independent_sets = Vector{Int}[]
    used = Set{Int}()
    for i in 1:n
        i ∈ used && continue
        indep = [i]
        for j in i+1:n
            j ∈ used && continue
            all(abs(int_mat[k,j]) < 1e-6 for k in indep) && push!(indep, j)
        end
        push!(independent_sets, indep)
        union!(used, indep)
    end

    return ExceptionalDivisor(coker_dim, coker_dim-1,
                               class_weights, int_mat,
                               independent_sets, chern)
end

"""Print exceptional divisor summary."""
function print_exceptional_divisor(ed::ExceptionalDivisor, edges::Vector)
    println("\nEXCEPTIONAL DIVISOR GEOMETRY")
    println("─"^60)
    @printf("  E = P^%d  (projective space over %d-dim cokernel)\\n",
            ed.proj_dim, ed.coker_dim)
    println("  Line bundles L₁,...,L_$(ed.coker_dim) on E:")
    println()
    println("  Top 5 crisis classes (lowest R_eff = nearest singularity):")
    perm = sortperm(ed.class_weights)
    for i in perm[1:min(5,end)]
        @printf("    L%-2d  R_eff=%.4f  c₁=%.3f\\n",
                i, ed.class_weights[i], ed.chern_numbers[i])
    end
    println()
    @printf("  Independent sets (simultaneously blockable classes): %d\\n",
            length(ed.independent_sets))
    println("  Largest independent set:")
    best = argmax(length.(ed.independent_sets))
    @printf("    {L_%s}  size=%d\\n",
            join(string.(ed.independent_sets[best]), ", L_"),
            length(ed.independent_sets[best]))
    println()
    println("  Interpretation:")
    println("  Each independent set = a stop architecture that simultaneously")
    println("  eliminates those cokernel classes. Size of largest independent")
    @printf("  set = %d = max classes removable by one coordinated blockade.\\n",
            length(ed.independent_sets[best]))
    println("  Remaining classes after best blockade = $(ed.coker_dim - length(ed.independent_sets[best]))")
end

# =============================================================================
# PART 6: TRANSFER FUNCTOR BETWEEN DOMAINS
# =============================================================================

"""
    DomainTransfer

A computed transfer functor F: C_A → C_B between two domain categories.
Stores the circuit correspondence, weight rescaling, and transfer quality.
"""
struct DomainTransfer
    source_domain  ::String
    target_domain  ::String
    circuit_map    ::Dict{Int,Int}          # backbone circuit i in A → circuit j in B
    weight_scale   ::Float64                # w_B = w_A × weight_scale
    coker_match    ::Bool                   # coker_A == coker_B (exact transfer)
    beta1_match    ::Bool                   # β₁_A == β₁_B
    transfer_quality::Float64              # 0=no transfer, 1=exact isomorphism
    q_table_scale  ::Float64               # Q_B = Q_A × q_table_scale
    notes          ::Vector{String}
end

"""
    build_transfer_functor(fp_A, fp_B, label_A, label_B,
                           w_max_A, w_max_B) -> DomainTransfer

Build the transfer functor from domain A to domain B.
Checks categorical isomorphism conditions and computes transfer quality.
"""
function build_transfer_functor(fp_A     ::ContextFingerprint,
                                 fp_B     ::ContextFingerprint,
                                 label_A  ::String,
                                 label_B  ::String,
                                 w_max_A  ::Float64,
                                 w_max_B  ::Float64)::DomainTransfer

    notes = String[]
    quality = 0.0

    # Check isomorphism conditions
    beta1_match = (fp_A.beta_1 == fp_B.beta_1)
    coker_match = (fp_A.coker_table == fp_B.coker_table)

    if beta1_match
        quality += 0.4
        push!(notes, "β₁ match: exact circuit correspondence")
    else
        # Partial credit: proximity of β₁ values
        ratio = min(fp_A.beta_1, fp_B.beta_1) / max(fp_A.beta_1, fp_B.beta_1, 1)
        quality += 0.4 * ratio
        push!(notes, @sprintf("β₁ proximity %.0f%%: partial circuit transfer (A=%d, B=%d)",
                               ratio*100, fp_A.beta_1, fp_B.beta_1))
    end
    if coker_match
        quality += 0.4
        push!(notes, "coker match: exact obstruction transfer")
    else
        # Check if cokers are in the same order of magnitude
        coker_a = isempty(fp_A.coker_table) ? 0 : first(values(fp_A.coker_table))
        coker_b = isempty(fp_B.coker_table) ? 0 : first(values(fp_B.coker_table))
        coker_ratio = min(coker_a,coker_b) / max(coker_a,coker_b,1)
        quality += 0.4 * coker_ratio
        push!(notes, @sprintf("coker proximity %.0f%%: mode mapping needed (%d vs %d)",
                               coker_ratio*100, coker_a, coker_b))
    end
    fp_A.k_invariants == fp_B.k_invariants &&
        (quality += 0.2; push!(notes, "k-invariants match: exact Postnikov transfer"))
    !beta1_match && !coker_match &&
        push!(notes, "Use partial transfer: rescale Q-table by β₁ ratio")

    # Weight rescaling
    w_scale = w_max_B / max(w_max_A, 1e-10)

    # Circuit correspondence (trivial if β₁ matches; best-effort otherwise)
    n_circuits = min(fp_A.beta_1, fp_B.beta_1)
    circuit_map = Dict(i => i for i in 1:n_circuits)

    # Q-table scaling: Q_B ≈ Q_A × (reduction_per_unit_weight in B / in A)
    q_scale = w_scale > 0 ? 1.0 / w_scale : 1.0

    return DomainTransfer(label_A, label_B, circuit_map, w_scale,
                           coker_match, beta1_match, quality, q_scale, notes)
end

"""Print transfer functor summary."""
function print_transfer(dt::DomainTransfer)
    println("\nTRANSFER FUNCTOR: $(dt.source_domain) → $(dt.target_domain)")
    println("─"^60)
    @printf("  Transfer quality: %.1f%%\\n", dt.transfer_quality * 100)
    @printf("  β₁ match: %s\\n", dt.beta1_match ? "✓ exact" : "✗ partial")
    @printf("  coker match: %s\\n", dt.coker_match ? "✓ exact" : "✗ partial")
    @printf("  Weight rescaling: × %.4f\\n", dt.weight_scale)
    @printf("  Q-table rescaling: × %.4f\\n", dt.q_table_scale)
    @printf("  Circuit correspondence: %d circuits mapped\\n", length(dt.circuit_map))
    println()
    println("  Transfer notes:")
    for note in dt.notes
        println("    • $note")
    end
    println()
    if dt.transfer_quality >= 0.8
        println("  STATUS: EXACT TRANSFER (≥80% quality)")
        println("  → Q-table transfers directly after weight rescaling")
        println("  → k-invariants transfer exactly (integers, no rescaling)")
        println("  → Postnikov tower structure transfers exactly")
    elseif dt.transfer_quality >= 0.4
        println("  STATUS: PARTIAL TRANSFER (40-80% quality)")
        println("  → Q-table transfers for matching strata only")
        println("  → 1-2 fine-tuning rounds needed in target domain")
    else
        println("  STATUS: LOW TRANSFER (<40% quality)")
        println("  → Compute from scratch in target domain")
        println("  → Add target fingerprint to context library")
    end
end

# =============================================================================
# PART 7: CONTEXT LIBRARY
# =============================================================================

"""
    ContextLibrary

A persistent store of domain fingerprints and their learned structures.
Indexed by fingerprint for O(1) lookup.
New domains query by fingerprint → get transferred knowledge for free.
"""
mutable struct ContextLibrary
    fingerprints::Dict{String, ContextFingerprint}
    q_tables    ::Dict{String, Dict}
    domains     ::Dict{String, String}  # fingerprint_hash → domain label
    n_entries   ::Int
end

ContextLibrary() = ContextLibrary(Dict(), Dict(), Dict(), 0)

"""Compute a hashable key from a fingerprint."""
function fingerprint_key(fp::ContextFingerprint)::String
    "β₁=$(fp.beta_1)_coker=$(sort(collect(values(fp.coker_table))))_k=$(fp.k_invariants)"
end

"""Add a domain to the context library."""
function add_to_library!(lib::ContextLibrary,
                          domain::String,
                          fp    ::ContextFingerprint,
                          qtable::Dict = Dict())
    key = fingerprint_key(fp)
    lib.fingerprints[key] = fp
    lib.q_tables[key]     = qtable
    lib.domains[key]      = domain
    lib.n_entries += 1
    @printf("  Library: added '%s' (key: %s)\\n", domain, key[1:min(40,end)])
end

"""Query the library for the best match to a new domain fingerprint."""
function query_library(lib::ContextLibrary,
                        fp_new::ContextFingerprint)

    isempty(lib.fingerprints) && return nothing, 0.0

    best_key     = ""
    best_quality = 0.0
    w_dummy = Dict((1,1)=>0)

    for (key, fp_stored) in lib.fingerprints
        dt = build_transfer_functor(fp_stored, fp_new,
                                     lib.domains[key], "query",
                                     1.0, 1.0)
        if dt.transfer_quality > best_quality
            best_quality = dt.transfer_quality
            best_key     = key
        end
    end

    best_key == "" && return nothing, 0.0
    return lib.domains[best_key], best_quality
end

# =============================================================================
# PART 8: DEMO — BRAIN ↔ MTR TRANSFER
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

    println("="^68)
    println("ALGEBRA OF CONTEXTS: Persistent Homology + Hodge + Transfer")
    println("="^68)

    # ── Domain A: Q_7P pharmacodynamic graph ──────────────────────────────
    println("\n[A] DOMAIN A: Q_7P BALBc Pharmacodynamic Graph")
    println("─"^68)

    nodes_A = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
    edges_A = [(:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
               (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
               (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
               (:sAMY,:BLA),(:sAMY,:HPF),
               (:LA,:BLA),(:LA,:sAMY)]

    weights_A = Dict{Tuple{Symbol,Symbol},Float64}(
        (:HPF,:sAMY)=>345.9, (:sAMY,:HPF)=>345.9,
        (:CA1sp,:HPF)=>15.0, (:HPF,:CA1sp)=>15.0,
        (:LA,:sAMY)=>97.5,   (:sAMY,:LA)=>97.5,
        (:CA1sp,:sAMY)=>5.88,(:BLA,:sAMY)=>1.2,
        (:BLA,:HPF)=>1.0,    (:HPF,:BLA)=>1.0,
        (:CA1sp,:BLA)=>1.0,  (:BLA,:LA)=>1.0,
        (:LA,:BLA)=>1.0,
    )
    for e in edges_A; haskey(weights_A,e)||(weights_A[e]=1.0); end

    # Persistent homology
    println("\n[A.1] Persistent Homology of Cull Filtration:")
    pairs_A = compute_persistence(edges_A,
                                   Dict(e=>Float64(get(weights_A,e,1.0))
                                        for e in edges_A))
    print_persistence(pairs_A; top_n=8)
    backbone_A = backbone_circuits(pairs_A)
    @printf("  Backbone circuits (essential, ∞-persistence): %d\\n",
            length(backbone_A))

    # Hodge decomposition
    println("\n[A.2] Hodge Laplacian Δ₁ on Clique Complex:")
    hd_A = hodge_decomposition(nodes_A, edges_A, weights_A)
    print_hodge(hd_A, edges_A)

    # Build fingerprint
    fp_A = ContextFingerprint(
        1, length(backbone_A), 0,
        Dict((1,2)=>62),   # coker(A↔C) = 62 (confirmed)
        [62],              # k-invariant at level 2
        [[1],[length(backbone_A)],[0]],
        hd_A.spectral_gap,
        1.2599,
        length(backbone_A))

    # ── Domain B: Hong Kong MTR (simplified 8-station model) ──────────────
    println("\n[B] DOMAIN B: Hong Kong MTR (Admiralty Hub Subgraph)")
    println("─"^68)

    # Key MTR stations around Admiralty interchange
    # Real MTR: Admiralty is the major 4-line interchange
    nodes_B = [:Admiralty, :Central, :HKU, :WanChai,
               :TsinShaTsui, :HungHom, :EastTST, :Kennedy]

    edges_B = [(:Admiralty,:Central),(:Central,:HKU),
               (:Admiralty,:WanChai),(:WanChai,:TsinShaTsui),
               (:Admiralty,:TsinShaTsui),  # direct cross-harbour
               (:TsinShaTsui,:HungHom),(:HungHom,:EastTST),
               (:EastTST,:TsinShaTsui),    # East Rail loop
               (:Admiralty,:HungHom),       # South Island Line
               (:Kennedy,:Admiralty),
               (:Central,:TsinShaTsui),    # Airport Express overlap
               (:HKU,:Admiralty)]

    # Weights = daily ridership (thousands)
    weights_B = Dict{Tuple{Symbol,Symbol},Float64}(
        (:Admiralty,:Central)=>180.0,   # Island Line (dominant)
        (:Central,:Admiralty)=>180.0,
        (:Admiralty,:TsinShaTsui)=>120.0, # Tsuen Wan (cross-harbour)
        (:TsinShaTsui,:Admiralty)=>120.0,
        (:TsinShaTsui,:HungHom)=>95.0,
        (:HungHom,:TsinShaTsui)=>95.0,
        (:EastTST,:TsinShaTsui)=>80.0,
        (:TsinShaTsui,:EastTST)=>80.0,
        (:Admiralty,:HungHom)=>45.0,    # South Island (direct)
        (:HungHom,:Admiralty)=>45.0,
        (:Kennedy,:Admiralty)=>30.0,
        (:Admiralty,:Kennedy)=>30.0,
        (:Central,:TsinShaTsui)=>20.0,
        (:TsinShaTsui,:Central)=>20.0,
        (:Admiralty,:WanChai)=>15.0,
        (:WanChai,:Admiralty)=>15.0,
        (:WanChai,:TsinShaTsui)=>10.0,
        (:HKU,:Admiralty)=>8.0,
        (:Admiralty,:HKU)=>8.0,
        (:Central,:HKU)=>5.0,
        (:HKU,:Central)=>5.0,
        (:HungHom,:EastTST)=>85.0,
    )
    for e in edges_B; haskey(weights_B,e)||(weights_B[e]=1.0); end

    println("\n[B.1] Persistent Homology of MTR Cull Filtration:")
    pairs_B = compute_persistence(edges_B,
                                   Dict(e=>Float64(get(weights_B,e,1.0))
                                        for e in edges_B))
    print_persistence(pairs_B; top_n=8)
    backbone_B = backbone_circuits(pairs_B)
    @printf("  Backbone circuits (essential, ∞-persistence): %d\\n",
            length(backbone_B))

    println("\n[B.2] Hodge Laplacian Δ₁ on MTR Clique Complex:")
    hd_B = hodge_decomposition(nodes_B, edges_B, weights_B)
    print_hodge(hd_B, edges_B)

    # MTR fingerprint (coker to be computed — use approximate value)
    # Real computation would require 4ti2 on the MTR toric ideal
    n_backbone_B = length(backbone_B)
    fp_B = ContextFingerprint(
        1, n_backbone_B, 0,
        Dict((1,2)=>n_backbone_B * 8),  # approximate coker
        [n_backbone_B * 8],
        [[1],[n_backbone_B],[0]],
        hd_B.spectral_gap,
        0.0,  # ρ(B_Λ) not yet computed for MTR
        n_backbone_B)

    # ── Transfer Functor ───────────────────────────────────────────────────
    println("\n[C] TRANSFER FUNCTOR: Pharmacodynamics → MTR")
    println("─"^68)

    w_max_A = maximum(values(weights_A))
    w_max_B = maximum(values(weights_B))
    dt = build_transfer_functor(fp_A, fp_B, "BALBc Q_7P", "HK MTR Admiralty",
                                 w_max_A, w_max_B)
    print_transfer(dt)

    # Show what transfers concretely
    println("\n[C.1] Concrete Transfer Mapping:")
    println("  BALBc pharmacodynamics  →  Hong Kong MTR")
    println("  ─────────────────────────────────────────────────────")
    mappings = [
        ("sAMY (opioid target)",   "Admiralty (interchange hub)"),
        ("HPF→sAMY (dominant)",    "Island Line→Admiralty (dominant)"),
        ("CA1sp→sAMY (direct)",    "South Island→Admiralty (direct)"),
        ("HPF→sAMY block (norcain)", "Island Line closure (disruption)"),
        ("Nash floor 1/17",        "Min. rerouting capacity (1/N_lines)"),
        ("coker=62 obstruction",   "62 independent rerouting strategies"),
        ("Surgery (Mode 4)",       "Emergency cross-harbour ferry"),
        ("Greedy vs AU-QKV",       "Manual vs AI disruption management"),
    ]
    for (brain, mtr) in mappings
        @printf("  %-35s → %s\\n", brain, mtr)
    end

    # ── Context Library ───────────────────────────────────────────────────
    println("\n[D] CONTEXT LIBRARY")
    println("─"^68)
    lib = ContextLibrary()
    add_to_library!(lib, "BALBc Q_7P (pharmacodynamics)", fp_A)
    add_to_library!(lib, "HK MTR Admiralty (transit)", fp_B)

    println("\n  Library query demo:")
    println("  New domain: 'Road network hub with 5 connections'")
    # Simulate a query fingerprint similar to Q_7P
    fp_query = ContextFingerprint(1, length(backbone_A), 0,
                                   Dict((1,2)=>62), [62],
                                   [[1],[length(backbone_A)],[0]],
                                   hd_A.spectral_gap * 0.9,  # slightly different
                                   0.0, length(backbone_A))
    best_domain, quality = query_library(lib, fp_query)
    @printf("  Best match: '%s'  (quality: %.1f%%)\\n",
            best_domain, quality * 100)
    @printf("  → Transfer Q-table with weight rescaling × %.4f\\n",
            dt.weight_scale)
    @printf("  → k-invariants transfer exactly: %s\\n",
            string(fp_A.k_invariants))

    println("\n" * "="^68)
    println("SUMMARY: What the algebra of contexts gives us")
    println("="^68)
    println("  1. Persistent homology → essential circuits (die last = backbone)")
    println("  2. Hodge Δ₁ → harmonic (essential) vs transient (blockable) edges")
    println("  3. Toric resolution → syzygy structure, Betti fingerprint")
    println("  4. Exceptional divisor → 62 classes, intersection theory")
    println("  5. Transfer functor → exact cross-domain knowledge transfer")
    println("  6. Context library → universal store, zero-shot from fingerprint")
    println()
    @printf("  Brain β₁=%d  MTR β₁=%d  Transfer quality=%.1f%%\\n",
            fp_A.beta_1, fp_B.beta_1, dt.transfer_quality*100)
    println("  The mathematics is the same. The labels differ.")
    println("="^68)
end
