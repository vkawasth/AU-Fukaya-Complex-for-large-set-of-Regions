using LinearAlgebra, SparseArrays, Statistics
using WriteVTK
using WriteVTK.VTKCellTypes: VTK_LINE, VTK_VERTEX
using DataFrames
using JSON3
using CSV
using StaticArrays   # zero-alloc 4x4 monodromy (run_iharaSingV2 integration)


#=======================================================================================


The winding number w may differ from the previous w=±6 since the phase trajectory 
is computed from prime_path weights, which will now influence ihara_radius differently. 
The per-loop check will still confirm as long as Phi_KS = λ^w — which it will by 
construction. The key question is whether |w| > 0 (net winding exists).
What to watch for in the Klein pillars
P1/P2/P3 should now show genuine variation in ihara_radius across snapshots — the 
regression β should move away from 0 toward 1 if K and ρ are actually correlated 
in the dynamics. If β is still near 0 after the rerun, it means the run has 
already reached equilibrium (K ≈ 0, ρ stable) and the pillars are measuring a 
settled system rather than an approach — which is also a valid physics result, 
just interpreted differently.

=======================================================================================#
# Arpack: sparse SVD keeps d0/d1/d2 sparse through HH2 computation.
# Install: ] add Arpack
using Arpack: svds

# Reusable accumulator buffers (Fix 3)
# Allocated once at module load, emptied/reused in compute_m4/m5/m6 hot loops
# to avoid millions of short-lived Dict allocations.
# NOT thread-safe -- single-threaded use only (matches existing code).
const _mk_buf   = Dict{Symbol, Float64}()
const _mk_total = Dict{Symbol, Float64}()

# Sparse structure constants (Fix 4)
# Cup/bracket tensors stored as sparse Dicts instead of dense zeros(n,n,n).
const SparseTensor3 = Dict{NTuple{3,Int}, Float64}

const AlgEl = Symbol                     # an algebra element (idempotent or arrow)
const LinComb = Dict{AlgEl, Float64}     # linear combination of algebra elements
const CochainMap = Dict{Tuple, LinComb}  # a cochain: input tuple -> output lincomb

# ============================================================================
# FILTRATION INFRASTRUCTURE
# Prevents combinatorial blow-up in curved A-inf by exponentially suppressing
# long/high-curvature paths. Handles non-zero m0 correctly: m0 values enter
# as a curvature energy penalty rather than being forced to zero.
#
#   E(gamma) = sum_i edge_weights[(path_i, path_{i+1})] + sum_i |m0[path_i]|
#   ||gamma||_lambda = exp(-lambda * E(gamma))
#
# Pass a FilteredAInfAlgebra to compute_A_inf or gerstenhaber_compute_A_inf
# via the keyword argument filt= to activate filtration.
# ============================================================================

struct FilteredAInfAlgebra
    lambda::Float64                       # exponential decay rate
    max_path_len::Int                     # hard path length cutoff
    energy_cutoff::Float64                # prune paths below this weight
    m0_curvature::Dict{Symbol, Float64}  # per-element curvature (curved A-inf)
end

function FilteredAInfAlgebra(;
        lambda::Float64 = 1.0,
        max_path_len::Int = 20,
        energy_cutoff::Float64 = 1e-8,
        m0_curvature::Dict{Symbol,Float64} = Dict{Symbol,Float64}())
    return FilteredAInfAlgebra(lambda, max_path_len, energy_cutoff, m0_curvature)
end

function path_energy(path, edge_weights::Dict{Tuple{Symbol,Symbol},Float64},
                     filt::FilteredAInfAlgebra)
    n = length(path)
    e = 0.0
    for i in 1:(n-1)
        e += get(edge_weights, (path[i], path[i+1]), 1.0)
    end
    for i in 1:n
        e += abs(get(filt.m0_curvature, path[i], 0.0))
    end
    return e
end

function filtration_weight(path, edge_weights::Dict{Tuple{Symbol,Symbol},Float64},
                           filt::FilteredAInfAlgebra)
    return exp(-filt.lambda * path_energy(path, edge_weights, filt))
end

function build_Ck_filtered(basis, is_composable, k::Int,
                            edge_weights::Dict{Tuple{Symbol,Symbol},Float64},
                            filt::FilteredAInfAlgebra;
                            max_size::Int = 50000)
    if k == 1
        return [(x,) for x in basis]
    end
    prev = build_Ck_filtered(basis, is_composable, k-1,
                              edge_weights, filt; max_size=max_size)
    candidates = Vector{Tuple}()
    weights    = Vector{Float64}()
    for tup in prev
        last_sym = tup[end]
        for x in basis
            if is_composable(last_sym, x)
                cand = (tup..., x)
                w = filtration_weight(cand, edge_weights, filt)
                if w >= filt.energy_cutoff
                    push!(candidates, cand)
                    push!(weights, w)
                end
            end
        end
    end
    order = sortperm(weights; rev=true)
    if length(order) > max_size
        println("Filtration truncating C$k: $(length(order)) -> $max_size paths")
        order = order[1:max_size]
    end
    return candidates[order]
end

# ============================================================================
# 1. Global constants and data loading (static)
# ============================================================================
const NODES_FILE = "./node_regions_clean.csv"
const EDGES_FILE = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"
const FULL_BRAIN_VTU = "./nodes_edges_filtered.vtp"
# ── Graph algebra configuration ─────────────────────────────────────────────
# Loaded dynamically from graph_algebra.json based on graph_type argument.
# To switch graphs: pass graph_type as ARGS[4] (e.g. "Q_7P", "Q_7L", "Q_8")
# Do NOT edit nodes/relations here — edit graph_algebra.json instead.

const _GRAPH_ALGEBRA_FILE = joinpath(@__DIR__, "graph_algebra.json")

function load_graph_algebra(graph_type::String)
    isfile(_GRAPH_ALGEBRA_FILE) || error(
        "graph_algebra.json not found at $(_GRAPH_ALGEBRA_FILE)\n" *
        "Place graph_algebra.json in the same folder as this script.")
    all_graphs = JSON3.read(read(_GRAPH_ALGEBRA_FILE, String))
    haskey(all_graphs, graph_type) || error(
        "Graph '$graph_type' not found in graph_algebra.json. " *
        "Available: $(join(keys(all_graphs), ", "))")
    g = all_graphs[graph_type]
    nodes_sym  = Symbol.(String.(g["nodes"]))
    ridx       = Dict(String(k) => Int(v) for (k,v) in pairs(g["region_to_idx"]))
    relations  = join(String.(g["relations"]), "\n")
    desc       = String(g["description"])
    println("[graph_algebra] Loaded $graph_type: $(length(nodes_sym)) nodes, " *
            "$(length(g["relations"])) relations  ($desc)")
    return nodes_sym, ridx, relations
end

# ── Parse graph_type from ARGS early (needed for const nodes) ────────────────
const _KNOWN_GRAPHS = ("Q_6", "Q_7P", "Q_7L", "Q_8")
const _graph_type_global = begin
    local gt = "Q_7P"  # default
    for a in ARGS
        if a ∈ _KNOWN_GRAPHS
            gt = a
            break
        end
    end
    gt
end

# Load algebra for this graph type
const _nodes_loaded, _ridx_loaded, _relations_loaded = load_graph_algebra(_graph_type_global)

# Expose as module-level constants expected by the rest of the script
const nodes        = _nodes_loaded
const region_to_idx = _ridx_loaded
const relations_str = _relations_loaded

# ===========================================================================================================================================================
# Function                      Arity                                Role                                      Use for prime paths?
# ===========================================================================================================================================================
# shifted_cup_product (patch)   corrected (stored arity = degree+1)  Proper Hochschild cup product on cohomology  ✅ Yes – gives ring structure constants,
#                                                                    (HH¹ ⊗ HH¹ → HH²)                          needed to identify prime ideals.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# total_cup_product (λ3, λ4)    original arity (p+q etc.)            A∞‑corrected cup product via m₃, m₄         ✅ For higher obstructions along prime
#                                                                    (homotopy lift)                            paths; experimental.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# curved_cup_product (m0)       original arity                       First‑order curvature correction (using m0) ❌ Only if you have a non‑zero m0
#                                                                                                                (curved A∞). You don’t (m0 is zero).
# ===========================================================================================================================================================

# Helper: Extract weights from the CSV structure
# We use 'volume' as the default weight, but you can change to 'avgCrossSection'
function build_geometric_weight_map(nodes, edges_df)
    # Map node index (0‑based) to node symbol
    node_symbols = nodes  # loaded from graph_algebra.json per graph
    node_to_sym = Dict(i-1 => node_symbols[i] for i in 1:length(node_symbols))
    
    geo_weight = Dict{Tuple{Symbol,Symbol}, Float64}()
    for r in eachrow(edges_df)
        u_sym = node_to_sym[r.node1id]
        v_sym = node_to_sym[r.node2id]
        # Choose a physical quantity: volume, avgCrossSection, curveness, etc.
        w = Float64(r.volume)   # or r.avgCrossSection, r.curveness
        geo_weight[(u_sym, v_sym)] = w
        # If edges are directed, add reverse direction; otherwise ignore.
    end
    return geo_weight
end

function get_edge_weight_map(edges_df::DataFrame)
    edge_map = Dict{Tuple{Int, Int}, Float64}()
    for r in eachrow(edges_df)
        # We use node IDs as keys for the geometric weight
        edge_map[(r.node1id, r.node2id)] = Float64(r.volume)
        # If the graph is undirected, include the reverse
        edge_map[(r.node2id, r.node1id)] = Float64(r.volume)
    end
    return edge_map
end

# Inductive generation with Sorting and Truncation
function generate_weighted_paths(existing_paths::Vector, edge_weights::Dict, c2_edges::Vector, n::Int; max_paths=10000)
    new_candidates = []
    
    for path in existing_paths
        # path is a tuple of Symbols (arrows)
        last_arrow = path[end]
        # In your system, arrow names often encode nodes: f_node1_node2
        # We need to extract the target node ID from the symbol
        last_target = tgt(last_arrow) 
        
        for edge_tuple in c2_edges
            # edge_tuple is (:f_src_tgt, weight_from_relations)
            arrow_sym = edge_tuple[1]
            if src(arrow_sym) == last_target
                # Calculate the NEW path weight
                # Geometric Weight (from CSV) * Algebraic Weight (from relations)
                # Note: You may need a map from Symbol to Node ID
                push!(new_candidates, (path..., arrow_sym))
            end
        end
    end

    if isempty(new_candidates) return [] end

    # SORTING LOGIC: Magnitude of the path
    # We calculate total path weight as the product of edge volumes
    weighted = map(new_candidates) do p
        w = 1.0
        for i in 1:length(p)
            # You can define path_weight(p) here based on CSV 'volume'
            w *= get_path_score(p, edge_weights) 
        end
        return (w, p)
    end

    # Sort descending and keep top 10,000
    sort!(weighted, by=x->abs(x[1]), rev=true)
    return [x[2] for x in weighted[1:min(end, max_paths)]]
end
function generate_paths_of_length(existing_paths::Vector, edges::Vector, target_length::Int)
    # If target_length is 2, the edges ARE the paths
    if target_length == 2
        return edges
    end

    new_paths = Tuple[]
    
    # Inductive step: try to append an edge to each existing path
    # existing_paths should be of length (target_length - 1)
    for path in existing_paths
        # path is a tuple like (f1, f2, ..., fn-1)
        last_step = path[end]
        last_target = tgt(last_step)
        
        for edge in edges
            # edge is a 2-tuple (u, v)
            edge_start = edge[1]
            edge_source = src(edge_start)
            
            if last_target == edge_source
                # Composition is valid: glue them together
                push!(new_paths, (path..., edge[2]))
            end
        end
    end
    
    return unique(new_paths)
end
# Helper to get the magnitude of a path
function get_path_weight(path::Tuple, edge_weights::Dict{Tuple{Symbol, Symbol}, Float64})
    # weight = |w1 * w2 * ... * wn|
    w = 1.0
    for i in 1:(length(path)-1)
        edge = (path[i], path[i+1])
        w *= get(edge_weights, edge, 0.0)
    end
    return abs(w)
end
function get_path_score(path::Tuple, region_weights::Dict{Tuple{Symbol, Symbol}, Float64})
    score = 1.0
    for arrow in path
        # Arrow: :f_CA1sp_HPF -> parts: ["f", "CA1sp", "HPF"]
        parts = split(string(arrow), '_')
        if length(parts) >= 3
            pair = (Symbol(parts[2]), Symbol(parts[3]))
            # Multiply path score by the volume of this segment
            score *= get(region_weights, pair, 1.0)
        end
    end
    return score
end
function get_region_weight_map(edges_df::DataFrame, nodes_df::DataFrame)
    # 1. Identify the correct column name for regions
    # It might be :regions, :region, or something else in your CSV
    all_cols = names(nodes_df)
    region_col = ""
    for c in ["regions", "region", "label", "area"]
        if c in all_cols
            region_col = c
            break
        end
    end

    if region_col == ""
        error("Could not find a region-related column in nodes_df. Available columns: $all_cols")
    end

    # 2. Map Node ID -> Region Symbol
    id_to_region = Dict{Int, Symbol}()
    for r in eachrow(nodes_df)
        # Access the column dynamically using the detected name
        raw_reg = string(r[Symbol(region_col)])
        
        # Clean the string ['region'] or similar formatting
        clean_reg = replace(replace(replace(replace(raw_reg,
                    "'" => ""), "[" => ""), "]" => ""), " " => "")
        first_reg = split(clean_reg, ',')[1]
        
        # Ensure we don't have an empty string
        if isempty(first_reg)
            id_to_region[r.id] = :unknown
        else
            id_to_region[r.id] = Symbol(first_reg)
        end
    end
    
    # 3. Map Region Pair -> Max Volume
    weight_map = Dict{Tuple{Symbol, Symbol}, Float64}()
    for r in eachrow(edges_df)
        u_reg = get(id_to_region, r.node1id, :unknown)
        v_reg = get(id_to_region, r.node2id, :unknown)
        
        # Using volume as the gluing 'strength' for Reverse Hironaka
        vol = Float64(r.volume)
        
        weight_map[(u_reg, v_reg)] = max(get(weight_map, (u_reg, v_reg), 0.0), vol)
        weight_map[(v_reg, u_reg)] = max(get(weight_map, (v_reg, u_reg), 0.0), vol)
    end
    return weight_map
end
# Max 10,000 paths
struct AlgebraBasis
    C::Dict{Int, Vector}
    
    function AlgebraBasis(c2_edges::Vector, edges_df::DataFrame, nodes_df::DataFrame, max_arity::Int)
        # Prepare the mapping
        region_weights = get_region_weight_map(edges_df, nodes_df)
        
        C = Dict{Int, Vector}()
        # Initialize C2 as 1-tuples for consistency
        C[2] = [(e[1],) for e in c2_edges] 
        
        for n in 3:max_arity
            # generate_weighted_paths now uses region_weights to sort
            C[n] = generate_weighted_paths(C[n-1], region_weights, c2_edges, n; max_paths=10000)
            
            if isempty(C[n]) break end
            println("Arity $n: $(length(C[n])) paths (Sorted by volume)")
        end
        return new(C)
    end
end

# Define the function as a method of the struct -- limit computation to arity 12
function get_composable_tuples(basis::AlgebraBasis, arity::Int)
    if !haskey(basis.C, arity)
        #println("Basis for arity $arity not precomputed. - we limit to arity 12")
        return Tuple[]
    end
    return basis.C[arity]
end
###############################################################
# FULL SHIFTED-HOCHSCHILD 
#
# PURPOSE
#   Fix arity mismatch globally using stored_arity = degree + 1
#   Build:
#       HH¹ ⊗ HH¹ -> HH² cup constants
#       HH¹ Lie bracket constants
#
# ASSUMES YOUR EXISTING FILE HAS:
#   LinComb                 = Dict{Symbol,Float64}
#   CochainMap              = Dict{Tuple,LinComb}
#   add_dict!(dest,src,s)
#   get_composable_tuples(n)
#   composable(a,b)
#   m2_as_dict(a,b)
#
#   deriv_basis :: Vector{CochainMap}   # HH¹ basis (stored arity=2)
#   HH2_basis   :: Vector{CochainMap}   # HH² basis (stored arity=3)
#
###############################################################

###############################################################
# SECTION 1. SHIFTED CONVENTION HELPERS
###############################################################

# stored arity r  <-> Hochschild degree n = r - 1
cochain_degree(r::Int) = r - 1

# cup output stored arity
cup_arity(p::Int, q::Int) = p + q - 1

# bracket output stored arity
bracket_arity(p::Int, q::Int) = p + q - 1

###############################################################
# SECTION 2. Utilities
###############################################################

function tuple_ok(args::Tuple)
    if length(args) <= 1
        return true
    end
    for i in 1:length(args)-1
        if !composable(args[i], args[i+1])
            return false
        end
    end
    return true
end

function merge_lincomb!(dest::LinComb, src::LinComb, scale::Float64=1.0)
    for (k,v) in src
        dest[k] = get(dest,k,0.0) + scale*v
    end
end

###############################################################
# SECTION 3. CORRECT SHIFTED CUP PRODUCT
#
# Inputs:
#   f stored arity p
#   g stored arity q
#
# Output stored arity = p+q-1
###############################################################

function shifted_cup_product(
    f_map::CochainMap,
    g_map::CochainMap,
    p::Int,
    q::Int,
    ctx::AlgebraBasis
)
    out_r = cup_arity(p,q)
    result = CochainMap()

    tuples = get_composable_tuples(ctx, out_r)
    for args in tuples

        tup = Tuple(args)
        tuple_ok(tup) || continue

        # overlap convention:
        # left uses first p entries
        # right uses entries p:end
        left  = Tuple(tup[1:p])
        right = Tuple(tup[p:end])

        fv = get(f_map,left,LinComb())
        gv = get(g_map,right,LinComb())

        isempty(fv) && continue
        isempty(gv) && continue

        out = LinComb()

        for (a,ca) in fv
            for (b,cb) in gv
                composable(a,b) || continue
                prod = m2_as_dict(a,b)
                merge_lincomb!(out, prod, ca*cb)
            end
        end

        !isempty(out) && (result[tup] = out)
    end

    return result
end

###############################################################
# SECTION 4. BRACE INSERTION (shifted)
###############################################################

function brace_insert(f_map, g_map, p, q, ctx)
    target_arity = p + q - 1
    if target_arity > 12
        return Dict{Tuple, Dict{Symbol, Float64}}()
    end

    result = Dict{Tuple, Dict{Symbol, Float64}}()
    
    # Get paths of the correct length for the output of the brace
    paths = get_composable_tuples(ctx, target_arity)
    
    for path in paths
        # path length is n = p + q - 1
        n = length(path)
        combined_output = Dict{Symbol, Float64}()
        
        # Gerstenhaber brace sum: sum_{i=1}^p (-1)^{(i-1)(q-1)} f(a1...g(ai...ai+q-1)...an)
        for i in 1:p
            # --- THE FIX: Ensure the slice [i : i+q-1] is valid ---
            if i + q - 1 > n
                continue 
            end
            
            # Extract the sub-path for g
            sub_path_g = path[i:i+q-1]
            out_g = get(g_map, sub_path_g, nothing)
            
            if out_g !== nothing
                for (res_g, coeff_g) in out_g
                    # Construct the new path for f: (a1...ai-1, res_g, ai+q...an)
                    # Note: This requires res_g to be a composable symbol
                    new_path_f = (path[1:i-1]..., res_g, path[i+q:end]...)
                    
                    out_f = get(f_map, new_path_f, nothing)
                    if out_f !== nothing
                        # Apply Koszul sign and accumulate
                        sign = ((-1)^((i-1)*(q-1)))
                        for (final_sym, final_coeff) in out_f
                            val = sign * coeff_g * final_coeff
                            combined_output[final_sym] = get(combined_output, final_sym, 0.0) + val
                        end
                    end
                end
            end
        end
        if !isempty(combined_output)
            result[path] = combined_output
        end
    end
    return result
end

###############################################################
# SECTION 5. GERSTENHABER BRACKET
###############################################################

function shifted_bracket(
    f_map::CochainMap,
    g_map::CochainMap,
    p::Int,
    q::Int,
    ctx::AlgebraBasis
)
    A = brace_insert(f_map,g_map,p,q, ctx)
    B = brace_insert(g_map,f_map,q,p, ctx)

    sign = (-1)^(cochain_degree(p)*cochain_degree(q))

    result = CochainMap()

    keys_all = union(keys(A), keys(B))

    for k in keys_all
        out = LinComb()

        if haskey(A,k)
            merge_lincomb!(out, A[k], 1.0)
        end
        if haskey(B,k)
            merge_lincomb!(out, B[k], -sign)
        end

        !isempty(out) && (result[k] = out)
    end

    return result
end

###############################################################
# SECTION 6. VECTORIZATION OF COCHAINS
#
# Need a common ambient basis of stored-arity-3 cochains for HH²
###############################################################

function ambient_basis_keys(cochains::Vector{CochainMap})
    S = Set{Tuple}()
    for c in cochains
        for k in keys(c)
            push!(S,k)
        end
    end
    return collect(S)
end

function ambient_output_syms(cochains::Vector{CochainMap})
    S = Set{Symbol}()
    for c in cochains
        for (_,img) in c
            for s in keys(img)
                push!(S,s)
            end
        end
    end
    return collect(S)
end

function cochain_to_vector(
    c::CochainMap,
    keys_basis::Vector{Tuple},
    syms_basis::Vector{Symbol}
)
    nk = length(keys_basis)
    ns = length(syms_basis)

    v = zeros(Float64, nk*ns)

    pos = Dict{Tuple,Int}()
    for (i,k) in enumerate(keys_basis)
        pos[k]=i
    end

    spos = Dict{Symbol,Int}()
    for (j,s) in enumerate(syms_basis)
        spos[s]=j
    end

    for (k,img) in c
        haskey(pos,k) || continue
        i = pos[k]
        for (s,val) in img
            haskey(spos,s) || continue
            j = spos[s]
            idx = (i-1)*ns + j
            v[idx] += val
        end
    end

    return v
end

###############################################################
# SECTION 7. CUP CONSTANTS HH¹ x HH¹ -> HH²
###############################################################

function compute_cup_constants(
    deriv_basis::Vector{CochainMap},
    HH2_basis::Vector{CochainMap},
    ctx::AlgebraBasis
)
    # Fix 4: store as sparse Dict instead of dense zeros(n1,n1,n2).
    # For typical path algebras ~90% of entries are zero.
    # Access: get(cup_constants, (i,j,k), 0.0)
    allC = vcat(deriv_basis, HH2_basis)
    K = ambient_basis_keys(allC)
    S = ambient_output_syms(allC)
    H = hcat([cochain_to_vector(h,K,S) for h in HH2_basis]...)

    n1 = length(deriv_basis)
    n2 = length(HH2_basis)

    cup_constants = SparseTensor3()   # Dict{NTuple{3,Int}, Float64}

    for i in 1:n1, j in 1:n1
        cp     = shifted_cup_product(deriv_basis[i], deriv_basis[j], 2, 2, ctx)
        v      = cochain_to_vector(cp, K, S)
        coeffs = ldiv(H, v)
        for k in 1:n2
            abs(coeffs[k]) > 1e-12 && (cup_constants[(i,j,k)] = coeffs[k])
        end
    end

    return cup_constants
end

###############################################################
# SECTION 8. BRACKET CONSTANTS HH¹ LIE ALGEBRA
###############################################################

function compute_bracket_constants(
    deriv_basis::Vector{CochainMap},
    ctx::AlgebraBasis
)
    # Fix 4b: sparse bracket tensor
    K = ambient_basis_keys(deriv_basis)
    S = ambient_output_syms(deriv_basis)
    H = hcat([cochain_to_vector(h,K,S) for h in deriv_basis]...)

    n = length(deriv_basis)
    C = SparseTensor3()   # Dict{NTuple{3,Int}, Float64}

    for i in 1:n, j in 1:n
        br     = shifted_bracket(deriv_basis[i], deriv_basis[j], 2, 2, ctx)
        v      = cochain_to_vector(br, K, S)
        coeffs = ldiv(H, v)
        for k in 1:n
            abs(coeffs[k]) > 1e-12 && (C[(i,j,k)] = coeffs[k])
        end
    end

    return C
end

###############################################################
# SECTION 9. MAIN ENTRY
###############################################################
#
# cup_constants     = compute_cup_constants(deriv_basis, HH2_basis, ctx)
# bracket_constants = compute_bracket_constants(deriv_basis)
#
# json["cup_constants"] = cup_constants
# json["bracket_constants"] = bracket_constants
#
###############################################################
# Build region_score caches
using JLD2
# If cache needs clearing due to new fields

const CACHE_FILE = "full_connectome_cache.jld2"

# if isfile(CACHE_FILE)
#    rm(CACHE_FILE)
# end

# ============================================================================
# Derivation extractor (replaces HH¹ for curved algebras)
# ============================================================================

function basis_index(basis::Vector{Symbol})
    idx = Dict{Symbol,Int}()
    for (i,b) in enumerate(basis)
        idx[b] = i
    end
    return idx
end

var_id(j, i, n) = (i-1)*n + j

function derivation_constraint_matrix(basis::Vector{Symbol}, idempotents::Vector{Symbol}, mult_dict)
    n = length(basis)
    idx = basis_index(basis)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    rhs = Float64[]
    row = 0

    # Idempotent constraints: X(e) = 0
    for e in idempotents
        i = idx[e]
        for j in 1:n
            row += 1
            push!(rows, row); push!(cols, var_id(j, i, n)); push!(vals, 1.0)
            push!(rhs, 0.0)
        end
    end

    # Leibniz constraints: X(ab) = X(a)b + aX(b)
    for a in basis, b in basis
        prod_ab = mult_dict(a, b)   # returns Dict{Symbol,Float64}
        isempty(prod_ab) && continue
        for k in 1:n
            row += 1
            # Left side: X(ab) → coefficient of basis[k] in X(ab)
            for (t_sym, cab) in prod_ab
                t = idx[t_sym]
                push!(rows, row); push!(cols, var_id(k, t, n)); push!(vals, cab)
            end
            # Right side part 1: X(a) b
            ia = idx[a]
            for j in 1:n
                Xa_j = basis[j]
                prod1 = mult_dict(Xa_j, b)
                ck = get(prod1, basis[k], 0.0)
                if ck != 0.0
                    push!(rows, row); push!(cols, var_id(j, ia, n)); push!(vals, -ck)
                end
            end
            # Right side part 2: a X(b)
            ib = idx[b]
            for j in 1:n
                Xb_j = basis[j]
                prod2 = mult_dict(a, Xb_j)
                ck = get(prod2, basis[k], 0.0)
                if ck != 0.0
                    push!(rows, row); push!(cols, var_id(j, ib, n)); push!(vals, -ck)
                end
            end
            push!(rhs, 0.0)
        end
    end
    M = sparse(rows, cols, vals, row, n*n)
    return M, rhs
end

function numeric_nullspace(M; atol=1e-10)
    # Robust nullspace: handles NaN/Inf from large m0_curvature,
    # LAPACKException from ill-conditioned matrices, scalar V from svds.
    m, n = size(M)
    k = min(m, n, 40)
    if k < 1
        return zeros(n, 0)
    end

    # Guard: replace NaN/Inf with 0 before any linear algebra
    Mc_dense = Matrix(M)
    if any(!isfinite, Mc_dense)
        Mc_dense[.!isfinite.(Mc_dense)] .= 0.0
    end

    # Try sparse SVD first (Arpack)
    try
        _, s, V = svds(sparse(Mc_dense); nsv=k)
        r = sum(s .> atol)
        if r == n
            return zeros(n, 0)
        end
        # svds may return scalar V when matrix degenerates
        if isa(V, Number)
            return r >= 1 ? zeros(n, 0) : reshape([V], n, 1)
        end
        if ndims(V) == 1
            V = reshape(V, length(V), 1)
        end
        return V[:, r+1:end]
    catch
    end

    # Fall back to dense SVD with clamped matrix
    try
        clamp_val = 1e15
        Mc_dense = clamp.(Mc_dense, -clamp_val, clamp_val)
        F = svd(Mc_dense)
        r = sum(F.S .> atol)
        if r >= size(Mc_dense, 2)
            return zeros(size(Mc_dense, 2), 0)
        end
        V = F.V
        if ndims(V) == 1
            V = reshape(V, length(V), 1)
        end
        return V[:, r+1:end]
    catch e
        @warn "numeric_nullspace: SVD failed ($e), returning empty nullspace"
        return zeros(n, 0)
    end
end

function derivation_basis(basis::Vector{Symbol}, idempotents::Vector{Symbol}, mult_dict; atol=1e-10)
    M, _ = derivation_constraint_matrix(basis, idempotents, mult_dict)
    N = numeric_nullspace(M; atol=atol)
    return N, M
end

function decode_derivation(v, basis::Vector{Symbol})
    n = length(basis)
    X = Dict{Symbol,Dict{Symbol,Float64}}()
    for i in 1:n
        src = basis[i]
        img = Dict{Symbol,Float64}()
        for j in 1:n
            c = v[var_id(j,i,n)]
            abs(c) < 1e-10 && continue
            img[basis[j]] = c
        end
        X[src] = img
    end
    return X
end

function derivation_bracket(X, Y, basis)
    n = length(basis)
    Z = Dict{Symbol,Dict{Symbol,Float64}}()
    # Helper to apply a derivation to an element (represented as a Dict of coefficients)
    function apply_derivation(der, elem::Dict{Symbol,Float64})
        out = Dict{Symbol,Float64}()
        for (a, ca) in elem
            for (b, cb) in get(der, a, Dict{Symbol,Float64}())
                out[b] = get(out, b, 0.0) + ca * cb
            end
        end
        return out
    end
    for a in basis
        ea = Dict(a => 1.0)
        XY = apply_derivation(X, apply_derivation(Y, ea))
        YX = apply_derivation(Y, apply_derivation(X, ea))
        img = Dict{Symbol,Float64}()
        for (k, v) in XY
            img[k] = get(img, k, 0.0) + v
        end
        for (k, v) in YX
            img[k] = get(img, k, 0.0) - v
        end
        Z[a] = img
    end
    return Z
end
# Build node_regions dictionary
function parse_region_string(s)
    s = strip(s)
    if startswith(s, '[') && endswith(s, ']')
        s = s[2:end-1]
    else
        return String[]
    end
    return split(s, r"['\", ]+"; keepempty=false)
end
function preprocess_and_cache_graph_VTK()
    @info "Preprocessing full graph (first run only)..."
    nodes_df = CSV.read(NODES_FILE, DataFrame)
    edges_df = CSV.read(EDGES_FILE, DataFrame)

    # Build adjacency and coordinates
    adj = Dict{Int, Vector{Int}}()
    points = Dict{Int, Vector{Float64}}()   # node id -> [x, y, z]
    for r in eachrow(edges_df)
        u, v = r.node1id, r.node2id
        push!(get!(adj, u, []), v)
        push!(get!(adj, v, []), u)
    end
    for r in eachrow(nodes_df)
        points[r.id] = [r.pos_x, r.pos_y, r.pos_z]
    end

    # Map node ID → primary region (string)
    node_regions = Dict{Int, String}()
    region_nodes = Dict{String, Vector{Int}}()
    for r in eachrow(nodes_df)
        regions = parse_region_string(r.regions)
        if !isempty(regions)
            reg = regions[1]
            node_regions[r.id] = reg
            push!(get!(region_nodes, reg, []), r.id)
        end
    end

    jldsave(CACHE_FILE; adj, points, node_regions, region_nodes)
    @info "Cache saved to $CACHE_FILE"
end

function preprocess_and_cache_graph()
    @info "Preprocessing full graph (first run only)..."
    nodes_df = CSV.read(NODES_FILE, DataFrame)
    edges_df = CSV.read(EDGES_FILE, DataFrame)

    # Build adjacency list (directed? we'll use undirected for BFS)
    adj = Dict{Int, Vector{Int}}()
    for r in eachrow(edges_df)
        u = r.node1id
        v = r.node2id
        push!(get!(adj, u, []), v)
        push!(get!(adj, v, []), u)
    end

    # Map node ID -> region name (first region in the list)
    node_regions = Dict{Int, String}()
    for r in eachrow(nodes_df)
        # parse regions (as before)
        regions = parse_region_string(r.regions)
        if !isempty(regions)
            node_regions[r.id] = regions[1]   # primary region
        end
    end

    # Build region -> list of node IDs
    region_nodes = Dict{String, Vector{Int}}()
    for (node, reg) in node_regions
        push!(get!(region_nodes, reg, []), node)
    end

    jldsave(CACHE_FILE; adj, node_regions, region_nodes)
    @info "Cache saved to $CACHE_FILE"
end

# Load the cache (fast)
function load_cached_graph_VTK()
    if !isfile(CACHE_FILE)
        preprocess_and_cache_graph_VTK()
    end
    
    return load(CACHE_FILE, "adj", "points", "node_regions", "region_nodes")
end

# Load the cache (fast)
function load_cached_graph()
    if !isfile(CACHE_FILE)
        preprocess_and_cache_graph()
    end
    return load(CACHE_FILE, "adj", "node_regions", "region_nodes")
end

# ============================================================================
# 2. Core algebraic functions (independent of edge weights)
# ============================================================================
function is_composable_tuple(tup::Tuple)
    is_composable(x,y) = tgt(x) == src(y)
    for i in 1:length(tup)-1
        if !is_composable(tup[i], tup[i+1])
            return false
        end
    end
    return true
end

function brace_composition(f::Dict{Tuple, LinComb}, g::Dict{Tuple, LinComb}, p::Int64, q::Int64, args::Tuple)
    # n is the length of the input path (args)
    n = length(args)
    result = LinComb()
    
    # Gerstenhaber brace: f { g } = sum_{...} (-1)^k f(a1, ..., g(ai, ..., ai+q-1), ..., an)
    # The number of possible insertion points is p
    for i in 1:p
        # --- THE FIX: SAFETY CHECK ---
        # Ensure the sub-slice for g does not exceed the path length n
        if i + q - 1 > n
            continue
        end
        
        sub_args_g = args[i:i+q-1]
        out_g = get(g, sub_args_g, nothing)
        
        if out_g !== nothing
            for (res_g, coeff_g) in out_g
                # Construct the modified path: 
                # (args[1:i-1]..., res_g, args[i+q:end]...)
                # This path must have length p
                new_path_f = (args[1:i-1]..., res_g, args[i+q:n]...)
                
                out_f = get(f, new_path_f, nothing)
                if out_f !== nothing
                    sign = (-1)^((i-1)*(q-1))
                    for (final_sym, final_coeff) in out_f
                        val = sign * coeff_g * final_coeff
                        result[final_sym] = get(result, final_sym, 0.0) + val
                    end
                end
            end
        end
    end
    return result
end
# Curved A infinity algebra requires projection
#     d0 and d1 are built for the curved Hochschild complex. 
#     The projection P = I - im_d0 * pinv(im_d0) may not 
#     yield a true cohomology basis because the curvature 
#     breaks the differential property. In a curved A∞‑algebra,
#     the usual cohomology is not well‑defined. Therefore, 
#     directly computing HH¹ via ker(d1)/im(d0) may produce 
#     a space that is too large (or ill‑conditioned).

# For my application (gerstenhaber bracket and cup product), I 
# only need the derivation subalgebra. Derivation 1‑cocycles 
# have the additional property that they vanish on idempotents 
# and satisfy the derivation rule. Extracting them is more 
# involved.
# Supports Getzler–Jones brace formalism
function gerstenhaber_bracket(f::CochainMap, g::CochainMap, p::Int, q::Int, ctx::AlgebraBasis)
    bracket_res = Dict{Tuple, LinComb}()
    for input in get_composable_tuples(ctx, p + q - 1)
        if !is_composable_tuple(input)
            continue
        end
        term1 = brace_composition(f, g, p, q, input)
        term2 = brace_composition(g, f, q, p, input)
        sign = (-1)^((p-1)*(q-1))
        combined = LinComb()
        for (k, v) in term1
            combined[k] = get(combined, k, 0.0) + v
        end
        for (k, v) in term2
            combined[k] = get(combined, k, 0.0) - sign * v
        end
        if !isempty(combined)
            bracket_res[input] = combined
        end
    end
    return bracket_res
end

# ----------------------------------------------------------------------
# Cup product and curvature (obstruction)
# ----------------------------------------------------------------------
"""
cup_product(f, g, p, q, paths)
(f ⌣ g)(a1...ap+q) = (-1)^(pq) f(a1...ap) * g(ap+1...ap+q)
"""
function cup_product(f_map::CochainMap, g_map::CochainMap, p::Int, q::Int, ctx::AlgebraBasis)
    result = Dict{Tuple, LinComb}()
    for args in get_composable_tuples(ctx, p + q)
        if !is_composable_tuple(args)
            continue
        end
        left = args[1:p]
        right = args[p+1:end]
        f_val = get(f_map, left, LinComb())
        g_val = get(g_map, right, LinComb())
        combined = LinComb()
        for (a, cf) in f_val
            for (b, cg) in g_val
                if is_composable(a, b)   # note: use your existing is_composable
                    prod = m2_as_dict(a, b)   # returns LinComb (Dict{Symbol,Float64})
                    sign = (-1)^(p * q)
                    for (c, cprod) in prod
                        combined[c] = get(combined, c, 0.0) + sign * cf * cg * cprod
                    end
                end
            end
        end
        if !isempty(combined)
            result[args] = combined
        end
    end
    return result
end

function add_lincomb!(dest::Dict{Symbol,Float64}, src::Dict{Symbol,Float64}, scale::Float64=1.0)
    for (k,v) in src
        dest[k] = get(dest, k, 0.0) + scale * v
    end
end

function merge_cochain!(dest::CochainMap, src::CochainMap, scale::Float64=1.0)
    for (k, v) in src
        if !haskey(dest, k)
            dest[k] = LinComb()
        end
        add_lincomb!(dest[k], v, scale)
    end
end

# Use your existing is_composable_tuple
tuple_ok(args::Tuple) = is_composable_tuple(args)

# We use this to compute cup product of a prime path 
# (interpreted as a 6‑cochain) with itself or with another 
# prime path, I first need to convert the path (a 6‑tuple) 
# into a cochain (a map that is 1 on that tuple and 0 
# otherwise).
function path_to_cochain(path::Tuple, weight=1.0)
    # Returns a cochain (CochainMap) that sends `path` to the idempotent e_{tgt(first)} times weight
    cochain = CochainMap()
    if length(path) == 6
        out = LinComb(Symbol("e_$(tgt(path[1]))") => weight)
        cochain[path] = out
    else
        @warn "Only 6‑tuples are supported for m6 paths"
    end
    return cochain
end

function cup2_product(f_map::CochainMap, g_map::CochainMap, p::Int, q::Int, ctx::AlgebraBasis)
    result = CochainMap()
    for args in get_composable_tuples(ctx, p + q)
        !tuple_ok(args) && continue
        left = Tuple(args[1:p])
        right = Tuple(args[p+1:end])
        f_val = get(f_map, left, LinComb())
        g_val = get(g_map, right, LinComb())
        isempty(f_val) && continue
        isempty(g_val) && continue
        out = LinComb()
        for (a, ca) in f_val
            for (b, cb) in g_val
                if is_composable(a, b)
                    prod = mult_dict(a, b)   # your multiplication returning LinComb
                    coeff = (-1)^(p * q) * ca * cb
                    add_lincomb!(out, prod, coeff)
                end
            end
        end
        !isempty(out) && (result[args] = out)
    end
    return result
end

function cup3_product(f_map::CochainMap, g_map::CochainMap, p::Int, q::Int, ctx::AlgebraBasis)
    result = CochainMap()
    for args in get_composable_tuples(ctx, p + q + 1)
        !tuple_ok(args) && continue
        left = Tuple(args[1:p])
        mid = args[p+1]
        right = Tuple(args[p+2:end])
        f_val = get(f_map, left, LinComb())
        g_val = get(g_map, right, LinComb())
        isempty(f_val) && continue
        isempty(g_val) && continue
        out = LinComb()
        for (a, ca) in f_val
            for (b, cb) in g_val
                val = m3_as_dict(a, mid, b)   # returns LinComb
                add_lincomb!(out, val, ca * cb)
            end
        end
        !isempty(out) && (result[args] = out)
    end
    return result
end

function cup4_product(f_map::CochainMap, g_map::CochainMap, p::Int, q::Int, ctx::AlgebraBasis)
    result = CochainMap()
    for args in get_composable_tuples(ctx, p + q + 2)
        !tuple_ok(args) && continue
        left = Tuple(args[1:p])
        x = args[p+1]
        y = args[p+2]
        right = Tuple(args[p+3:end])
        f_val = get(f_map, left, LinComb())
        g_val = get(g_map, right, LinComb())
        isempty(f_val) && continue
        isempty(g_val) && continue
        out = LinComb()
        for (a, ca) in f_val
            for (b, cb) in g_val
                val = m4_as_dict(a, x, y, b)   # returns LinComb
                add_lincomb!(out, val, ca * cb)
            end
        end
        !isempty(out) && (result[args] = out)
    end
    return result
end

function total_cup_product(f_map::CochainMap, g_map::CochainMap, p::Int, q::Int, ctx::AlgebraBasis;
    λ3::Float64=1.0, λ4::Float64=1.0)
    total = CochainMap()
    c2 = cup2_product(f_map, g_map, p, q, ctx)
    merge_cochain!(total, c2, 1.0)
    c3 = cup3_product(f_map, g_map, p, q, ctx)
    merge_cochain!(total, c3, λ3)
    c4 = cup4_product(f_map, g_map, p, q, ctx)
    merge_cochain!(total, c4, λ4)
    return total
end

""" 
Curved Cup product will be cup product + correction term for 
curvature -- i.e. account for m0 curvature
(f⌣curved​g)=(f⌣g)+[m0​,f⌣g]
"""
function curved_cup_product(f_map::CochainMap, g_map::CochainMap, p::Int, q::Int, m0_map::CochainMap, ctx::AlgebraBasis)
    # Flat part (no extra tuple argument)
    base = cup_product(f_map, g_map, p, q, ctx)
    
    # Correction: [m0, base]  (m0 is degree 0, base is degree p+q)
    corr = gerstenhaber_bracket(m0_map, base, 0, p+q, ctx)
    
    # Combine base and correction - cochian map
    result = CochainMap() # defined as Dict{Tuple, LinComb}()
    for (k, v) in base
        result[k] = copy(v)
    end
    for (k, v) in corr
        if !haskey(result, k)
            result[k] = LinComb()
        end
        add_lincomb!(result[k], v, 1.0)
    end
    return result
end
# ------------------------------------------------------------
# Cup product mass (total absolute coefficient sum)
# ------------------------------------------------------------
function cup_mass(cp::CochainMap)
    s = 0.0
    for (_, img) in cp
        for (_, v) in img
            s += abs(v)
        end
    end
    return s
end
# Curvature of a 2‑cochain (e.g., φ ∈ HH²)
function curvature(phi_map::CochainMap)
    # [φ, φ] is a 2‑cochain (since p=q=2, input arity = 3? Wait: 
    # gerstenhaber_bracket with p=q=2 gives a 2‑cochain? Actually 
    # bracket of two 2‑cochains is a 2‑cochain (arity 2). But the
    # standard curvature is [φ,φ] which is a 3‑cochain? No – for a 
    # Maurer–Cartan element, the curvature is dφ + ½[φ,φ], which is a 
    # 2‑cochain. In our definition, a 2‑cochain takes two arguments.
    # So we need the bracket with p=q=2 and output arity 3? Let's 
    # follow the provided code: they call gerstenhaber_bracket with 
    # p=2,q=2, which yields a 2‑cochain (input length 3). That's fine.
    # We'll keep as is.
    return gerstenhaber_bracket(phi_map, phi_map, 2, 2, ctx)
end

function basis_vector_to_cochain(v::Vector{Float64}, C2::Vector)
    cochain = CochainMap()
    for (i, (x, y)) in enumerate(C2)
        coeff = v[i]
        if abs(coeff) > 1e-12
            # The basis cochain sends (x,y) -> e_{tgt(x)} * coeff
            out = LinComb(Symbol("e_$(tgt(x))") => coeff)
            cochain[(x, y)] = out
        end
    end
    return cochain
end



function parse_relations(rel_str)
    raw = Dict{Tuple{Symbol,Symbol},Float64}()
    for line in split(rel_str, '\n')
        line = strip(line)
        isempty(line) && continue
        !occursin(" - ", line) && continue
        left, right = split(line, " - ")
        left = strip(left)
        right = strip(right)
        parts = split(left, '*')
        length(parts) == 2 || continue
        sym1 = Symbol(parts[1])
        sym2 = Symbol(parts[2])
        coeff_str, _ = split(right, '*')
        coeff = parse(Float64, coeff_str)
        raw[(sym1, sym2)] = coeff
    end
    return raw
end

function build_basis(nodes, raw_coeffs)
    all_arrows = Set{Symbol}()
    for (k, _) in raw_coeffs
        push!(all_arrows, k[1], k[2])
        sx = String(k[1]); sy = String(k[2])
        if startswith(sx, "f_") && startswith(sy, "f_")
            px = split(sx, "_"); py = split(sy, "_")
            if px[3] == py[2]
                target = Symbol("f_$(px[2])_$(py[3])")
                push!(all_arrows, target)
            end
        end
    end
    all_arrows = filter(s -> !startswith(String(s), "e_"), all_arrows)
    basis = [Symbol("e_$n") for n in nodes]
    append!(basis, collect(all_arrows))
    basis = unique(basis)
    return basis
end

function src(x::Symbol)
    s = String(x)
    if startswith(s, "f_")
        return Symbol(split(s, "_")[2])
    else
        return Symbol(split(s, "_")[2])
    end
end

function tgt(x::Symbol)
    s = String(x)
    if startswith(s, "f_")
        return Symbol(split(s, "_")[3])
    else
        return Symbol(split(s, "_")[2])
    end
end

function make_m2(raw_coeffs, basis)
    function m2(x::Symbol, y::Symbol)
        if startswith(String(x), "e_")
            if startswith(String(y), "f_") && src(y) == src(x)
                return (1.0, y)
            end
            if x == y
                return (1.0, x)
            end
            return (0.0, nothing)
        end
        if startswith(String(y), "e_")
            if startswith(String(x), "f_") && tgt(x) == src(y)
                return (1.0, x)
            end
            return (0.0, nothing)
        end
        if tgt(x) != src(y)
            return (0.0, nothing)
        end
        if haskey(raw_coeffs, (x,y))
            coeff = raw_coeffs[(x,y)]
            if src(x) == tgt(y)
                target = Symbol("e_$(src(x))")
            else
                target = Symbol("f_$(src(x))_$(tgt(y))")
            end
            return (coeff, target)
        end
        return (0.0, nothing)
    end
    return m2
end

function compute_composable_chains(basis, is_composable)
    C2 = [(a,b) for a in basis for b in basis if is_composable(a,b)]
    C3 = [(a,b,c) for (a,b) in C2 for c in basis if is_composable(b,c)]
    return C2, C3
end

function build_d1(C2, basis, m2)
    rows, cols, vals = Int[], Int[], Float64[]
    for (j, a) in enumerate(basis)
        for (i, (x,y)) in enumerate(C2)
            # x * φ(y)
            if y == a
                v, t = m2(x, a)
                if t !== nothing
                    push!(rows, i); push!(cols, j); push!(vals, v)
                end
            end
            # - φ(x*y)
            v_xy, t_xy = m2(x, y)
            if t_xy == a
                push!(rows, i); push!(cols, j); push!(vals, -v_xy)
            end
            # φ(x) * y
            if x == a
                v, t = m2(a, y)
                if t !== nothing
                    push!(rows, i); push!(cols, j); push!(vals, v)
                end
            end
        end
    end
    return sparse(rows, cols, vals, length(C2), length(basis))
end

function build_d0(C2, basis, m2)
    rows, cols, vals = Int[], Int[], Float64[]
    for (j, a) in enumerate(basis)
        for (i, (x,y)) in enumerate(C2)
            if y == a
                v, t = m2(x, a)
                if t !== nothing
                    push!(rows, i); push!(cols, j); push!(vals, v)
                end
            end
            v_xy, t_xy = m2(x, y)
            if t_xy == a
                push!(rows, i); push!(cols, j); push!(vals, -v_xy)
            end
            if x == a
                v, t = m2(a, y)
                if t !== nothing
                    push!(rows, i); push!(cols, j); push!(vals, v)
                end
            end
        end
    end
    return sparse(rows, cols, vals, length(C2), length(basis))
end

function compute_m3(basis, m2, C2, C3)
    # Precompute multiplication table for efficiency
    mult_table = Dict{Symbol, Vector{Tuple{Symbol,Float64}}}()
    for a in basis
        mult_table[a] = []
        for b in basis
            c, t = m2(a,b)
            if t !== nothing && abs(c) > 1e-12
                push!(mult_table[a], (b, c))
            end
        end
    end

    m3 = Dict{Tuple{Symbol,Symbol,Symbol},Tuple{Tuple{Float64,Union{Nothing,Symbol}},Tuple{Float64,Union{Nothing,Symbol}}}}()
    for a in basis, b in basis, c in basis
        v1, t1 = m2(a,b)
        left = (t1 === nothing) ? (0.0, nothing) : m2(t1, c)
        v2, t2 = m2(b,c)
        right = (t2 === nothing) ? (0.0, nothing) : m2(a, t2)
        if left != right
            m3[(a,b,c)] = (left, right)
        end
    end
    return m3, mult_table
end

function build_d2_curved(C2, C3, mult_table, m2, C3_index)
    nC2 = length(C2)
    nC3 = length(C3)
    rows, cols, vals = Int[], Int[], Float64[]
    for (j, (a,b)) in enumerate(C2)
        for (c, _) in mult_table[b]
            # left: (a*b)*c
            c1, t1 = m2(a,b)
            if t1 !== nothing
                c2, t2 = m2(t1, c)
                if t2 !== nothing
                    i = C3_index[(a,b,c)]
                    push!(rows, i); push!(cols, j); push!(vals, c1*c2)
                end
            end
            # right: a*(b*c)
            c3, t3 = m2(b,c)
            if t3 !== nothing
                c4, t4 = m2(a, t3)
                if t4 !== nothing
                    i = C3_index[(a,b,c)]
                    push!(rows, i); push!(cols, j); push!(vals, -c3*c4)
                end
            end
        end
    end
    return sparse(rows, cols, vals, nC3, nC2)
end

function compute_HH2(d0, d1, d2)
    # Fix 1: keep d1/d2 sparse throughout -- use sparse SVD rank estimation
    # instead of converting to dense matrices.
    function sparse_rank(A; atol=1e-8)
        m, n = size(A)
        k = min(m, n, 40)
        k < 1 && return 0
        try
            _, s, _ = svds(A; nsv=k)
            return sum(s .> atol)
        catch
            return rank(Matrix(A))   # fallback only
        end
    end
    function sparse_nullity(A; atol=1e-8)
        m, n = size(A)
        return n - sparse_rank(A; atol=atol)
    end
    dim_ker = sparse_nullity(d2)
    rank_d1 = sparse_rank(d1)
    return dim_ker - rank_d1
end

# ============================================================================
# 3. m₄, m₅, m₆ computations (simplified versions for export)
# ============================================================================
function mul_elem(elem::Pair{Float64,Symbol}, y::Symbol, m2)
    coeff, sym = elem
    if sym === nothing
        return Dict{Symbol,Float64}()
    end
    c, t = m2(sym, y)
    if t === nothing
        return Dict{Symbol,Float64}()
    end
    return Dict(t => coeff * c)
end

function mul_elem_left(x::Symbol, elem::Pair{Float64,Symbol}, m2)
    coeff, sym = elem
    if sym === nothing
        return Dict{Symbol,Float64}()
    end
    c, t = m2(x, sym)
    if t === nothing
        return Dict{Symbol,Float64}()
    end
    return Dict(t => coeff * c)
end

function add_dict!(dict, other)
    for (k,v) in other
        dict[k] = get(dict, k, 0.0) + v
    end
end

function m4_obstruction_full(a,b,c,d, m2, m3)
    total = Dict{Symbol,Float64}()
    # term1: + m2( m3(a,b,c), d )
    c1, t1 = m2(a,b)
    if t1 !== nothing
        c2, t2 = m2(t1, c)
        if t2 !== nothing
            add_dict!(total, mul_elem(c1*c2 => t2, d, m2))
        end
    end
    c3, t3 = m2(b,c)
    if t3 !== nothing
        c4, t4 = m2(a, t3)
        if t4 !== nothing
            neg = mul_elem(c3*c4 => t4, d, m2)
            for (k,v) in neg
                total[k] = get(total, k, 0.0) - v
            end
        end
    end
    # term2: - m3( m2(a,b), c, d )
    c_ab, t_ab = m2(a,b)
    if t_ab !== nothing
        c5, t5 = m2(t_ab, c)
        if t5 !== nothing
            c6, t6 = m2(t5, d)
            if t6 !== nothing
                total[t6] = get(total, t6, 0.0) - c_ab * c5 * c6
            end
        end
        c7, t7 = m2(c, d)
        if t7 !== nothing
            c8, t8 = m2(t_ab, t7)
            if t8 !== nothing
                total[t8] = get(total, t8, 0.0) + c_ab * c7 * c8
            end
        end
    end
    # term3: + m3( a, m2(b,c), d )
    c_bc, t_bc = m2(b,c)
    if t_bc !== nothing
        c9, t9 = m2(a, t_bc)
        if t9 !== nothing
            c10, t10 = m2(t9, d)
            if t10 !== nothing
                total[t10] = get(total, t10, 0.0) + c_bc * c9 * c10
            end
        end
        c11, t11 = m2(t_bc, d)
        if t11 !== nothing
            c12, t12 = m2(a, t11)
            if t12 !== nothing
                total[t12] = get(total, t12, 0.0) - c_bc * c11 * c12
            end
        end
    end
    # term4: - m3( a, b, m2(c,d) )
    c_cd, t_cd = m2(c,d)
    if t_cd !== nothing
        c13, t13 = m2(a,b)
        if t13 !== nothing
            c14, t14 = m2(t13, t_cd)
            if t14 !== nothing
                total[t14] = get(total, t14, 0.0) - c_cd * c13 * c14
            end
        end
        c15, t15 = m2(b, t_cd)
        if t15 !== nothing
            c16, t16 = m2(a, t15)
            if t16 !== nothing
                total[t16] = get(total, t16, 0.0) + c_cd * c15 * c16
            end
        end
    end
    # term5: + m2( a, m3(b,c,d) )
    c17, t17 = m2(b,c)
    if t17 !== nothing
        c18, t18 = m2(t17, d)
        if t18 !== nothing
            add_dict!(total, mul_elem_left(a, c17*c18 => t18, m2))
        end
    end
    c19, t19 = m2(c,d)
    if t19 !== nothing
        c20, t20 = m2(b, t19)
        if t20 !== nothing
            neg = mul_elem_left(a, c19*c20 => t20, m2)
            for (k,v) in neg
                total[k] = get(total, k, 0.0) - v
            end
        end
    end
    return total
end

# Simplified: we only need m4_obs (obstruction) not the corrected m4 for export
function compute_m4_obs(C4, m2, m3)
    m4_obs = Dict{Tuple{Symbol,Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
    for (a,b,c,d) in C4
        obs = m4_obstruction_full(a,b,c,d, m2, m3)
        if !isempty(obs)
            m4_obs[(a,b,c,d)] = obs
        end
    end
    return m4_obs
end

function norm_dict(D::Dict{Symbol,Float64})
    isempty(D) && return 0.0
    return sqrt(sum(v^2 for v in values(D)))
end

function build_Ck(basis, is_composable, k; max_size=50000)
    if k == 1
        return [(x,) for x in basis]
    end
    prev = build_Ck(basis, is_composable, k-1; max_size=max_size)
    Ck = Vector{NTuple{k,Symbol}}()
    for tup in prev
        last = tup[end]
        for x in basis
            if is_composable(last, x)
                push!(Ck, (tup..., x))
                if length(Ck) >= max_size
                    println("Truncating C$k at $max_size")
                    return Ck
                end
            end
        end
    end
    return Ck
end

function compute_global_m5(C5, m3, m4, mult)
    # Fix 3: reuse _mk_total buffer to avoid per-path Dict allocation
    m5 = Dict{NTuple{5,Symbol}, Dict{Symbol,Float64}}()
    for (a,b,c,d,e) in C5
        empty!(_mk_total)
        # a·m4(b,c,d,e)
        d1 = get(m4, (b,c,d,e), _mk_buf)
        if !isempty(d1)
            tmp = mult_expand_left(mult, a, d1)
            merge_dicts!(_mk_total, tmp)
        end
        # m4(a,b,c,d)·e
        d2 = get(m4, (a,b,c,d), _mk_buf)
        if !isempty(d2)
            tmp = mult_expand_right(mult, d2, e)
            merge_dicts!(_mk_total, tmp)
        end
        # m4(a, m2(b,c), d, e)
        bc = mult(b,c)
        if !isempty(bc)
            tmp = mult_expand_middle_4(mult, m4, a, bc, d, e)
            merge_dicts!(_mk_total, tmp)
        end
        # m4(a, b, m2(c,d), e)
        cd = mult(c,d)
        if !isempty(cd)
            for (x, coeff) in cd
                d3 = get(m4, (a, b, x, e), _mk_buf)
                if !isempty(d3)
                    merge_dicts!(_mk_total, d3, coeff)
                end
            end
        end
        # m4(a, b, c, m2(d,e))
        de = mult(d,e)
        if !isempty(de)
            for (x, coeff) in de
                d4 = get(m4, (a, b, c, x), _mk_buf)
                if !isempty(d4)
                    merge_dicts!(_mk_total, d4, coeff)
                end
            end
        end
        # m3∘m3 terms
        m3term = compose_m3_m3(a,b,c,d,e, m3, mult)
        merge_dicts!(_mk_total, m3term)
        if !isempty(_mk_total)
            m5[(a,b,c,d,e)] = Dict(k => -v for (k,v) in _mk_total)
        end
    end
    println("Number of non-zero global m5 entries: ", length(m5))
    return m5
end

function mult_expand_left(mult, a, B)
    out = Dict{Symbol,Float64}()
    for (k,v) in B
        tmp = mult(a, k)
        for (kk, vv) in tmp
            out[kk] = get(out, kk, 0.0) + v * vv
        end
    end
    return out
end

function mult_expand_right(mult, A, b)
    out = Dict{Symbol,Float64}()
    for (k,v) in A
        tmp = mult(k, b)
        for (kk, vv) in tmp
            out[kk] = get(out, kk, 0.0) + v * vv
        end
    end
    return out
end

function mult_expand_middle_4(mult, m4, a, bc, d, e)
    total = Dict{Symbol,Float64}()
    for (x, coeff) in bc
        if haskey(m4, (a, x, d, e))
            merge_dicts!(total, m4[(a, x, d, e)], coeff)
        end
    end
    return total
end

function compose_m3_m3(a,b,c,d,e, m3, mult)
    total = Dict{Symbol,Float64}()
    # + m3(m3(a,b,c), d, e)
    if haskey(m3, (a,b,c))
        for (x, coeff) in m3[(a,b,c)]
            if haskey(m3, (x, d, e))
                merge_dicts!(total, m3[(x, d, e)], coeff)
            end
        end
    end
    # - m3(a, m3(b,c,d), e)
    if haskey(m3, (b,c,d))
        for (x, coeff) in m3[(b,c,d)]
            if haskey(m3, (a, x, e))
                merge_dicts!(total, m3[(a, x, e)], -coeff)
            end
        end
    end
    # + m3(a, b, m3(c,d,e))
    if haskey(m3, (c,d,e))
        for (x, coeff) in m3[(c,d,e)]
            if haskey(m3, (a, b, x))
                merge_dicts!(total, m3[(a, b, x)], coeff)
            end
        end
    end
    return total
end

function merge_dicts!(target, source, scale=1.0)
    for (k,v) in source
        target[k] = get(target, k, 0.0) + scale * v
    end
end

# ============================================================
# CORRECTED m6 COMPUTATION (A∞ IDENTITY AT LEVEL 6)
# ============================================================
# ----------------------------------------------------------------------
# m6 obstruction (lower terms of the A∞ identity at level 6)
# Returns a dictionary mapping output basis element -> coefficient
# for the sum of all terms that do NOT involve m6 itself.
#
# Arguments:
#   a,b,c,d,e,f : Symbol – the six input arguments
#   m2          : function (Symbol, Symbol) -> Dict{Symbol,Float64}
#   m3          : Dict{NTuple{3,Symbol}, Dict{Symbol,Float64}}
#   m4          : Dict{NTuple{4,Symbol}, Dict{Symbol,Float64}}
#   m5          : Dict{NTuple{5,Symbol}, Dict{Symbol,Float64}}
# ----------------------------------------------------------------------
function m6_obstruction_full(a,b,c,d,e,f, m2, m3, m4, m5)
    total = Dict{Symbol,Float64}()

    # ------------------------------------------------------------------
    # 1. j = 5, i = 2   ->   m2( m5(...), ... )   (2 terms)
    # ------------------------------------------------------------------
    # term: + m2( m5(a,b,c,d,e), f )
    if haskey(m5, (a,b,c,d,e))
        for (x, coeff) in m5[(a,b,c,d,e)]
            tmp = m2(x, f)
            add_dict!(total, tmp, coeff)
        end
    end
    # term: - m2( a, m5(b,c,d,e,f) )
    if haskey(m5, (b,c,d,e,f))
        for (x, coeff) in m5[(b,c,d,e,f)]
            tmp = m2(a, x)
            add_dict!(total, tmp, -coeff)
        end
    end

    # ------------------------------------------------------------------
    # 2. j = 4, i = 3   ->   m3( ..., m4(...), ... )   (3 terms)
    # ------------------------------------------------------------------
    # k=0: + m3( m4(a,b,c,d), e, f )
    if haskey(m4, (a,b,c,d))
        for (x, coeff) in m4[(a,b,c,d)]
            if haskey(m3, (x, e, f))
                add_dict!(total, m3[(x, e, f)], coeff)
            end
        end
    end
    # k=1: - m3( a, m4(b,c,d,e), f )
    if haskey(m4, (b,c,d,e))
        for (x, coeff) in m4[(b,c,d,e)]
            if haskey(m3, (a, x, f))
                add_dict!(total, m3[(a, x, f)], -coeff)
            end
        end
    end
    # k=2: + m3( a, b, m4(c,d,e,f) )
    if haskey(m4, (c,d,e,f))
        for (x, coeff) in m4[(c,d,e,f)]
            if haskey(m3, (a, b, x))
                add_dict!(total, m3[(a, b, x)], coeff)
            end
        end
    end

    # ------------------------------------------------------------------
    # 3. j = 3, i = 4   ->   m4( ..., m3(...), ... )   (4 terms)
    # ------------------------------------------------------------------
    # k=0: + m4( m3(a,b,c), d, e, f )
    if haskey(m3, (a,b,c))
        for (x, coeff) in m3[(a,b,c)]
            if haskey(m4, (x, d, e, f))
                add_dict!(total, m4[(x, d, e, f)], coeff)
            end
        end
    end
    # k=1: - m4( a, m3(b,c,d), e, f )
    if haskey(m3, (b,c,d))
        for (x, coeff) in m3[(b,c,d)]
            if haskey(m4, (a, x, e, f))
                add_dict!(total, m4[(a, x, e, f)], -coeff)
            end
        end
    end
    # k=2: + m4( a, b, m3(c,d,e), f )
    if haskey(m3, (c,d,e))
        for (x, coeff) in m3[(c,d,e)]
            if haskey(m4, (a, b, x, f))
                add_dict!(total, m4[(a, b, x, f)], coeff)
            end
        end
    end
    # k=3: - m4( a, b, c, m3(d,e,f) )
    if haskey(m3, (d,e,f))
        for (x, coeff) in m3[(d,e,f)]
            if haskey(m4, (a, b, c, x))
                add_dict!(total, m4[(a, b, c, x)], -coeff)
            end
        end
    end

    # ------------------------------------------------------------------
    # 4. j = 2, i = 5   ->   m5( ..., m2(...), ... )   (5 terms)
    # ------------------------------------------------------------------
    # k=0: + m5( m2(a,b), c, d, e, f )
    ab = m2(a,b)
    for (x, coeff) in ab
        if haskey(m5, (x, c, d, e, f))
            add_dict!(total, m5[(x, c, d, e, f)], coeff)
        end
    end
    # k=1: - m5( a, m2(b,c), d, e, f )
    bc = m2(b,c)
    for (x, coeff) in bc
        if haskey(m5, (a, x, d, e, f))
            add_dict!(total, m5[(a, x, d, e, f)], -coeff)
        end
    end
    # k=2: + m5( a, b, m2(c,d), e, f )
    cd = m2(c,d)
    for (x, coeff) in cd
        if haskey(m5, (a, b, x, e, f))
            add_dict!(total, m5[(a, b, x, e, f)], coeff)
        end
    end
    # k=3: - m5( a, b, c, m2(d,e), f )
    de = m2(d,e)
    for (x, coeff) in de
        if haskey(m5, (a, b, c, x, f))
            add_dict!(total, m5[(a, b, c, x, f)], -coeff)
        end
    end
    # k=4: + m5( a, b, c, d, m2(e,f) )
    ef = m2(e,f)
    for (x, coeff) in ef
        if haskey(m5, (a, b, c, d, x))
            add_dict!(total, m5[(a, b, c, d, x)], coeff)
        end
    end

    return total
end


function compute_m6_selective(C6, mult, m3, m4, m5; tol=1e-6)
    m6 = Dict{NTuple{6,Symbol}, Dict{Symbol,Float64}}()
    max_norm = 0.0
    for (a,b,c,d,e,f) in C6
        #total = m6_obstruction(a,b,c,d,e,f, mult, m3, m4, m5)
        total = m6_obstruction_full(a,b,c,d,e,f, mult, m3, m4, m5)
        nrm = norm_dict(total)
        if nrm > tol
            m6_cancel = Dict{Symbol,Float64}()
            for (k, v) in total
                m6_cancel[k] = -v
            end
            m6[(a,b,c,d,e,f)] = m6_cancel
            max_norm = max(max_norm, nrm)
        end
    end
    println("Nonzero m6 entries: ", length(m6))
    println("Max m6 obstruction: ", max_norm)
    return m6
end



# For prime paths we need m5, m6 – we will not implement full recursion here,
# but provide placeholders. In a full implementation you would compute them
# as in the original script. For export we assume they are already computed.
function extract_prime_paths_primitive(m6; top_n=100)
    scored = []
    for (path, outdict) in m6
        weight = sum(abs(v) for v in values(outdict))
        push!(scored, (path, weight))
    end
    sort!(scored, by=x->-x[2])
    return scored[1:min(top_n, end)]
end

# Helper: add contents of dict B into dict A with an optional scale factor
function add_dict!(A::Dict{Symbol,Float64}, B::Dict{Symbol,Float64}, scale::Float64=1.0)
    for (k, v) in B
        A[k] = get(A, k, 0.0) + scale * v
    end
    return A
end

function m6_obstruction(a,b,c,d,e,f, mult, m3, m4, m5)
    total = Dict{Symbol,Float64}()

    # --- (1) m2 ∘ m4 terms ---
    if haskey(m4, (a,b,c,d))
        tmp = mult_expand_left(mult, a, m4[(b,c,d,e)])
        add_dict!(total, tmp, +1.0)
    end

    if haskey(m4, (b,c,d,e))
        tmp = mult_expand_left(mult, a, m4[(b,c,d,e)])
        add_dict!(total, tmp, -1.0)
    end

    # --- (2) m3 ∘ m3 terms ---
    if haskey(m3, (a,b,c)) && haskey(m3, (d,e,f))
        left = m3[(a,b,c)]
        tmp = mult_expand_right(mult, left, d)
        tmp2 = mult_expand_right(mult, tmp, e)
        tmp3 = mult_expand_right(mult, tmp2, f)
        add_dict!(total, tmp3, +1.0)
    end

    if haskey(m3, (c,d,e)) && haskey(m3, (a,b,c))
        right = m3[(c,d,e)]
        tmp = mult_expand_left(mult, a, right)
        tmp2 = mult_expand_right(mult, tmp, f)
        add_dict!(total, tmp2, -1.0)
    end

    # --- (3) m4 ∘ m2 terms ---
    if haskey(m4, (b,c,d,e))
        tmp = mult_expand_right(mult, m4[(b,c,d,e)], f)
        add_dict!(total, tmp, +1.0)
    end

    if haskey(m4, (a,b,c,d))
        tmp = mult_expand_right(mult, m4[(a,b,c,d)], e)
        tmp2 = mult_expand_right(mult, tmp, f)
        add_dict!(total, tmp2, -1.0)
    end

    # --- (4) m5 differential term (optional if available) ---
    if haskey(m5, (a,b,c,d,e))
        tmp = mult_expand_right(mult, m5[(a,b,c,d,e)], f)
        add_dict!(total, tmp, +1.0)
    end

    if haskey(m5, (b,c,d,e,f))
        tmp = mult_expand_left(mult, a, m5[(b,c,d,e,f)])
        add_dict!(total, tmp, -1.0)
    end

    return total
end

# convert m2 (returns tuple) to a dict return for consistency
function compute_global_m6(C6, m2_dict, m3, m4, m5; tol=1e-6)
    # Fix 3b: avoid Dict allocation for zero-obs paths (common in stable zone)
    m6 = Dict{NTuple{6,Symbol}, Dict{Symbol,Float64}}()
    for tup in C6
        a,b,c,d,e,f = tup
        obs = m6_obstruction_full(a,b,c,d,e,f, m2_dict, m3, m4, m5)
        isempty(obs) && continue          # fast path: no alloc needed
        norm_dict(obs) > tol || continue  # check tolerance
        m6[tup] = Dict(k => -v for (k,v) in obs)
    end
    println("Number of non‑zero global m6 entries: ", length(m6))
    return m6
end

# ============================================================================
# 4. High‑level A∞ computation (returns data structures)
# ============================================================================
# Convert a 2‑cochain vector (length |C2|) to CochainMap
function cochain2_to_map(vec::Vector{Float64}, C2::Vector)
    cochain = CochainMap()
    for (i, (x, y)) in enumerate(C2)
        coeff = vec[i]
        if abs(coeff) > 1e-12
            # 2‑cochain sends (x,y) -> e_{tgt(x)} * coeff? Or maybe the identity.
            # For cup product, the explicit output basis is needed. Here we assume
            # the basis cochain returns the idempotent e_{tgt(x)}.
            out = LinComb(Symbol("e_$(tgt(x))") => coeff)
            cochain[(x, y)] = out
        end
    end
    return cochain
end

function cochain3_to_map(vec::Vector{Float64}, C3::Vector)
    cochain = CochainMap()
    # disruptor kill dimensions 
    # Calculate the safe limit
    v_len = length(vec)
    c_len = length(C3)
    limit = min(v_len, c_len)

    for i in 1:limit
        path = C3[i]
        coeff = vec[i]
        
        if abs(coeff) > 1e-12
            # Destructure the path (x, y, z)
            x, y, z = path
            # Use your tgt() logic to find the endpoint
            out = LinComb(Symbol("e_$(tgt(x))") => coeff)
            cochain[path] = out
        end
    end
    return cochain
end

function project_cochain_onto_basis(cochain::CochainMap, basis_matrix::Matrix{Float64}, tuple_space::Vector)
    # basis_matrix: each column is a basis cochain represented as vector over tuple_space
    # cochain: Dict{Tuple, LinComb}
    # Convert cochain to a vector over tuple_space
    vec = zeros(length(tuple_space))
    for (i, tup) in enumerate(tuple_space)
        lincomb = get(cochain, tup, LinComb())
        # sum of coefficients (since output basis is idempotent e_{tgt(first)} with coefficient 1)
        # For simplicity, we just take the total coefficient (assuming all outputs are idempotents)
        # Actually we need to map linear combination to scalar: we treat the cochain as giving a scalar (the coefficient of the idempotent).
        # Here we assume the cochain outputs only the idempotent e_{tgt(first)} with some coefficient.
        total = sum(values(lincomb))
        vec[i] = total
    end
    # Project onto basis_matrix using least squares
    coeffs = basis_matrix \ vec   # solves basis_matrix * coeffs = vec
    return coeffs
end

# =====================================================================================
# CURVED A∞ SUPPORT / ANNIHILATOR / HIGHER-IDEAL FRAMEWORK
# =====================================================================================
#
# PURPOSE
# -------
# This codebase models the graph/path system not as a strictly associative algebra,
# but as a curved A∞-algebra with explicit higher multiplications:
#
#     m0, m2, m3, m4, m5, m6
#
# where:
#
#   m2 = ordinary path composition / pairwise multiplication
#   m3 = first associativity correction (ternary interaction)
#   m4 = higher obstruction / coherence correction
#   m5 = deeper multi-step interaction tensor
#   m6 = highest currently implemented obstruction / hidden structure detector
#   m0 = curvature term (background forcing / non-flatness)
#
# Thus the system cannot be faithfully analyzed using only classical associative
# ideal theory. Important structure may live entirely in m3..m6.
#
#
# MOTIVATION
# ----------
# Standard graph analysis sees only connectivity.
# Standard path algebra sees only m2 composition.
# Our objective is to detect higher-order organization, hidden corridors,
# coherent subsystems, and obstruction loci using the full curved A∞ structure.
#
# This framework defines computational analogues of:
#
#   • ideals
#   • prime ideals
#   • annihilators
#   • support varieties
#
# directly from the explicit sparse tensors m2..m6 already computed in Julia.
#
#
# ---------------------------------------------------------------------
# 1. HIGHER IDEAL (Curved A∞ analogue of a classical ideal)
# ---------------------------------------------------------------------
#
# A subset I ⊆ basis is called a higher ideal if it is closed under all higher
# operations:
#
#   if any input of m_n lies in I, then outputs of m_n remain in I
#
# for n = 2,3,4,5,6.
#
# In code this is computed by:
#
#   higher_closure(seed, basis, m3_element, m4_obs, m5, m6)
#
# starting from seed paths and repeatedly adding outputs of interactions touching
# the current set until closure is reached.
#
# Interpretation:
#   A self-contained higher-order functional subsystem.
#
#
# ---------------------------------------------------------------------
# 2. PRIME HIGHER IDEAL / PRIMITIVE SECTOR
# ---------------------------------------------------------------------
#
# Classical prime ideals are not the correct notion in a non-associative curved
# A∞ setting. Instead we use prime_paths (already extracted from m6) as seeds for
# primitive sectors.
#
# Each prime path is:
#
#   • high obstruction weight
#   • not reducible into smaller strong subpaths
#
# We then close it under higher interactions:
#
#   prime_higher_ideals(...)
#
# yielding irreducible higher-order sectors.
#
# Interpretation:
#   Minimal coherent dynamic corridor carrying hidden structure.
#
#
# ---------------------------------------------------------------------
# 3. HOMOTOPY ANNIHILATOR
# ---------------------------------------------------------------------
#
# Classical annihilator:
#
#   Ann(α) = {a : aα = 0}
#
# In curved A∞ form we instead detect basis elements that do not participate in
# any higher interaction.
#
# Implemented by:
#
#   annihilator_infty(...)
#
# using support_score(x, ...) across m3..m6.
#
# If score ≈ 0, x lies in the curved annihilator.
#
# Interpretation:
#   Regions/paths dynamically invisible to higher-order effects.
#
#
# ---------------------------------------------------------------------
# 4. SUPPORT LOCUS / SUPPORT VARIETY
# ---------------------------------------------------------------------
#
# Elements with nonzero higher interaction score define:
#
#   support_infty(...)
#
# These are basis paths actively involved in m3..m6 structure.
#
# Interpretation:
#   Geometric locus where hidden dynamics actually live.
#
# This can be exported to graph / voxel geometry for visualization.
#
#
# ---------------------------------------------------------------------
# 5. RELATION TO EXISTING PRIME PATHS
# ---------------------------------------------------------------------
#
# Existing:
#
#   prime_paths = extract_prime_paths(m6, top_n=...)
#
# are cohomology-guided primitive obstruction generators.
#
# New upgrade:
#
#   prime_paths  --> higher_closure(...) --> higher_prime_ideals
#
# so paths become full curved A∞ sectors rather than isolated motifs.
#
#
# ---------------------------------------------------------------------
# 6. WHY THIS IS BETTER THAN CLASSICAL IDEAL THEORY
# ---------------------------------------------------------------------
#
# Classical ideals only see m2.
#
# Our data suggest essential structure appears in:
#
#   m3  context-sensitive interactions
#   m4  coherence failures
#   m5  deeper multi-step coupling
#   m6  obstruction concentration / hidden backbone
#
# Therefore strict associative algebra alone is incomplete.
#
#
# ---------------------------------------------------------------------
# 7. SCIENTIFIC INTERPRETATION
# ---------------------------------------------------------------------
#
# ann_infty:
#   inactive or insulated paths
#
# supp_infty:
#   active hidden skeleton
#
# higher_prime_ideals:
#   irreducible functional modules
#
# m6 spikes:
#   transition / obstruction loci
#
# persistent higher ideals over time:
#   stable memory / regime structures
#
#
# ---------------------------------------------------------------------
# 8. COMPUTATIONAL PHILOSOPHY
# ---------------------------------------------------------------------
#
# We preserve mapping back to the original graph at all times.
# No Morita reduction or quotienting that destroys geometry.
#
# Every algebraic object is represented by actual basis paths / nodes / regions,
# so results remain interpretable in physical graph space.
#
#
# ---------------------------------------------------------------------
# 9. PRACTICAL PIPELINE
# ---------------------------------------------------------------------
#
# compute_A∞(...)
#     -> m3_element, m4_obs, m5, m6
#     -> prime_paths
#     -> ann_infty
#     -> supp_infty
#     -> higher_prime_ideals
#
# Then visualize on graph / ParaView / time dynamics.
#
#
# ---------------------------------------------------------------------
# 10. STATUS
# ---------------------------------------------------------------------
#
# This is a computational research definition of curved A∞ support geometry.
# It is principled, practical, and tailored to explicit sparse higher operations.
# It extends classical algebraic ideas into a setting where strict associativity
# is too restrictive.
#
# =====================================================================================
function collect_ops(m3_element, m4_obs, m5, m6)
    return Dict(
        3 => m3_element,
        4 => m4_obs,
        5 => m5,
        6 => m6
    )
end
# tiny scores may render this useless. 1e-8
function support_score(x, m3_element, m4_obs, m5, m6)
    score = 0.0

    for op in (m3_element, m4_obs, m5, m6)
        for (inp, out) in op
            if x in inp
                score += norm_dict(out)
            end
        end
    end

    return score
end

function annihilator_infty(basis, m3_element, m4_obs, m5, m6; tol=1e-8)
    ann = Symbol[]

    for x in basis
        s = support_score(x, m3_element, m4_obs, m5, m6)
        if s < tol
            push!(ann, x)
        end
    end

    return ann
end

function support_infty(basis, m3_element, m4_obs, m5, m6; tol=1e-8)
    supp = Symbol[]

    for x in basis
        s = support_score(x, m3_element, m4_obs, m5, m6)
        if s >= tol
            push!(supp, x)
        end
    end

    return supp
end

function higher_closure(seed, basis, m3_element, m4_obs, m5, m6)
    I = Set(seed)
    changed = true

    ops = (m3_element, m4_obs, m5, m6)

    while changed
        changed = false

        for op in ops
            for (inp, out) in op
                if any(x in I for x in inp)
                    for y in keys(out)
                        if !(y in I)
                            push!(I, y)
                            changed = true
                        end
                    end
                end
            end
        end
    end

    return collect(I)
end

function prime_higher_ideals_sortK(prime_paths, basis, m3_element, m4_obs, m5, m6)
    ideals = []
    for (pth, wt) in prime_paths
        seed = collect(pth)
        I = higher_closure(seed, basis, m3_element, m4_obs, m5, m6)
        # Compute total support score for this ideal
        total_supp = sum(support_score(x, m3_element, m4_obs, m5, m6) for x in I)
        push!(ideals, (pth, wt, I, total_supp))
    end
    # Sort ideals by total_supp descending (optional)
    sort!(ideals, by=x->x[4], rev=true)
    return ideals
end

function prime_higher_ideals(prime_paths, basis, m3_element, m4_obs, m5, m6)
    ideals = []

    for (pth, wt) in prime_paths
        seed = collect(pth)
        I = higher_closure(seed, basis, m3_element, m4_obs, m5, m6)
        push!(ideals, (pth, wt, I))
    end

    return ideals
end
"""
    In a quiver path algebra, a path is indecomposable 
    as an element if it cannot be written as a product of 
    two non‑zero paths. That would mean its length is 1 
    (an arrow). But here, my m6 paths are 6‑tuples, not 
    single arrows. Still, the idea of “prime” as indecomposable 
    in the cohomology ring (via cup product) requires a 
    different test. This heuristic – requiring both halves 
    to also have high obstruction – ensures the path’s 
    obstruction does not come from a simple composition of 
    two other highly obstructed paths. Hence, it is a 
    “primitive” obstruction generator.
"""
function extract_prime_paths(m6; top_n=100, weight_threshold_ratio=0.3)
    # Compute total weight (sum of absolute coefficients) for each path
    scored = [(path, sum(abs(v) for v in values(outdict))) for (path, outdict) in m6]
    sort!(scored, by=x->-x[2])
    if isempty(scored)
        return []
    end
    max_weight = scored[1][2]
    threshold = max_weight * weight_threshold_ratio

    # Build a dictionary for fast weight lookup
    weight_dict = Dict(path => w for (path, w) in scored)

    prime_candidates = []
    for (path, w) in scored
        if w < threshold
            break
        end
        # Check if path is decomposable into two subpaths that both have weight > threshold
        decomposable = false
        for i in 1:5  # split after i-th element (1 to 5)
            left = Tuple(path[1:i])
            right = Tuple(path[i+1:end])
            left_w = get(weight_dict, left, 0.0)
            right_w = get(weight_dict, right, 0.0)
            if left_w > threshold && right_w > threshold
                decomposable = true
                break
            end
        end
        if !decomposable
            push!(prime_candidates, (path, w))
            if length(prime_candidates) >= top_n
                break
            end
        end
    end
    return prime_candidates
end

function prime_to_cochain(path, weight=1.0)
    cochain = CochainMap()
    out = LinComb(Symbol("e_$(tgt(path[1]))") => weight)
    cochain[path] = out
    return cochain
end
function prime_path_interaction_score(path_i, path_j, ctx::AlgebraBasis)
    ci = prime_to_cochain(path_i)
    cj = prime_to_cochain(path_j)
    cup_m = cup_mass(total_cup_product(ci, cj, 6, 6, ctx))
    brack_m = cup_mass(gerstenhaber_bracket(ci, cj, 6, 6, ctx))
    return cup_m + brack_m
end
# ============================================================================
# PERVERSE SHEAVES t- Structure
# ============================================================================
# ============================================================================
# PERVERSIVE t‑STRUCTURE AND PERVERSITY FUNCTION (tailored for A∞)
# ============================================================================

"""
    shift_cochain(cochain::CochainMap, shift::Int)

Shift the arity of all keys in the cochain by `shift` (i.e., add or remove entries).
For simplicity, we shift the degree (stored arity = degree+1). A positive shift
increases the stored arity by `shift`. This is a placeholder; a real implementation
would need to adjust the tuple lengths consistently.
"""
function shift_cochain(cochain::CochainMap, shift::Int)
    if shift == 0
        return cochain
    end
    result = CochainMap()
    for (k, v) in cochain
        # For a cochain, the domain tuple arity is fixed (e.g., (x,y) for arity 2).
        # Shifting by an integer changes the degree, hence the stored arity by `shift`.
        # We simply keep the same keys – in a proper derived category setting,
        # shifting would move the complex index, not the input arity.
        # Here we define a dummy operation: we return the same cochain.
        # For truncation, we will work with complexes (not implemented yet).
        result[k] = v
    end
    @warn "shift_cochain is a stub – does not change actual homological degree."
    return result
end

"""
    truncate_above(cochain::CochainMap, degree::Int)

Truncate the complex (cochain) at a given cohomological degree. In our simple
model, we assume the cochain represents a single degree (or we ignore higher
degrees). This stub returns the input unchanged.
"""
function truncate_above(cochain::CochainMap, degree::Int)
    # In a full implementation, we would decompose the complex and keep only
    # components with cohomological index ≤ degree.
    @warn "truncate_above is a stub – returns original cochain."
    return cochain
end

function truncate_below(cochain::CochainMap, degree::Int)
    @warn "truncate_below is a stub – returns original cochain."
    return cochain
end

"""
    stratum_of_symbol(sym::Symbol, prime_ideals)

Return the index (1‑based) of the prime higher ideal that contains `sym`,
or 0 if none.
"""
function stratum_of_symbol(sym::Symbol, prime_ideals)
    for (idx, (_, _, closure, _)) in enumerate(prime_ideals)
        if sym in closure
            return idx
        end
    end
    return 0
end

"""
    perversity(stratum_idx, prime_ideals; max_perv=2)

Map the total_support of a prime higher ideal to an integer perversity
in the range [0, max_perv].
"""
function perversity(stratum_idx, prime_ideals; max_perv=2)
    if stratum_idx == 0
        return 0
    end
    _, _, _, total_support = prime_ideals[stratum_idx]
    # Find global maximum total_support among ideals
    max_supp = maximum(ideals[4] for ideals in prime_ideals)
    if max_supp == 0
        return 0
    end
    # Map linearly from [0, max_supp] to [0, max_perv]
    p = round(Int, (total_support / max_supp) * max_perv)
    return clamp(p, 0, max_perv)
end

"""
    restrict_to_stratum(cochain::CochainMap, stratum::Set{Symbol})

Return a new cochain where only tuples whose every element belongs to `stratum`
are kept.
"""
function restrict_to_stratum(cochain::CochainMap, stratum::Set{Symbol})
    filtered = CochainMap()
    for (tup, img) in cochain
        if all(s -> s in stratum, tup)
            filtered[tup] = img
        end
    end
    return filtered
end

"""
    perverse_truncation_above(module::CochainMap, degree::Int, prime_ideals; kwargs...)

Apply perverse truncation above `degree`. The module is decomposed by strata,
shifted by perversity, truncated, then shifted back.
"""
function perverse_truncation_above(cochain::CochainMap, degree::Int, prime_ideals; max_perv=2)
    result = CochainMap()
    for (idx, (_, _, closure, _)) in enumerate(prime_ideals)
        p = perversity(idx, prime_ideals; max_perv=max_perv)
        restricted = restrict_to_stratum(cochain, closure)
        shifted_down = shift_cochain(restricted, -p)
        truncated = truncate_above(shifted_down, degree)
        shifted_up = shift_cochain(truncated, p)
        merge_cochain!(result, shifted_up)
    end
    return result
end

"""
    perverse_truncation_below(module::CochainMap, degree::Int, prime_ideals; max_perv=2)

Perverse truncation below `degree`.
"""
function perverse_truncation_below(cochain::CochainMap, degree::Int, prime_ideals; max_perv=2)
    result = CochainMap()
    for (idx, (_, _, closure, _)) in enumerate(prime_ideals)
        p = perversity(idx, prime_ideals; max_perv=max_perv)
        restricted = restrict_to_stratum(cochain, closure)
        shifted_down = shift_cochain(restricted, -p)
        truncated = truncate_below(shifted_down, degree)
        shifted_up = shift_cochain(truncated, p)
        merge_cochain!(result, shifted_up)
    end
    return result
end

"""
    is_perverse(module::CochainMap, prime_ideals, degree_range)

Check whether the module satisfies the perverse t‑structure conditions:
cohomology above perversity vanishes and below perversity vanishes.
This is a stub; a full implementation would need actual cohomology computation.
"""
function is_perverse(cochain::CochainMap, prime_ideals, degree_range=( -10, 10 ))
    # Placeholder: always returns true
    return true
end

# ============================================================================
# MAIN COMPUTE FUNCTION
# ============================================================================
function gerstenhaber_compute_A∞(raw_coeffs, nodes;
        filt::Union{FilteredAInfAlgebra,Nothing}=nothing,
        edge_weights::Union{Dict{Tuple{Symbol,Symbol},Float64},Nothing}=nothing)
    # 1. Basis and multiplication
    basis = build_basis(nodes, raw_coeffs)
    m2 = make_m2(raw_coeffs, basis)
    is_composable(x,y) = tgt(x) == src(y)

    # 2. Composable chains -- use filtration if provided
    C2, C3 = compute_composable_chains(basis, is_composable)
    C3_index = Dict(c => i for (i,c) in enumerate(C3))
    if filt !== nothing
        ew = edge_weights !== nothing ? edge_weights : Dict{Tuple{Symbol,Symbol},Float64}(
            (a,b) => abs(get(raw_coeffs, (a,b), 1.0)) for (a,b) in keys(raw_coeffs))
        println("gerstenhaber_compute_A∞: filtration active (lambda=$(filt.lambda))")
        C4 = build_Ck_filtered(basis, is_composable, 4, ew, filt; max_size=20000)
        C5 = build_Ck_filtered(basis, is_composable, 5, ew, filt; max_size=50000)
        C6 = build_Ck_filtered(basis, is_composable, 6, ew, filt; max_size=100000)
    else
        C4 = build_Ck(basis, is_composable, 4; max_size=20000)
        C5 = build_Ck(basis, is_composable, 5; max_size=50000)
        C6 = build_Ck(basis, is_composable, 6; max_size=100000)
    end
    println("|C2| = $(length(C2)), |C3| = $(length(C3)), |C4| = $(length(C4)), |C5| = $(length(C5)), |C6| = $(length(C6))")

    # 1. Capture your base edges from your filtered graph
    c2_edges = C2
    # get path weights -- TODO : will update to cached version 
    nodes_df = CSV.read(NODES_FILE, DataFrame; delim=';')
    edges_df = CSV.read(EDGES_FILE, DataFrame; delim=';')

    ctx = AlgebraBasis(c2_edges, edges_df, nodes_df, 12) # C3..C12 paths generated.

    # 3. m3 (associator) and multiplication table
    m3_raw, mult_table = compute_m3(basis, m2, C2, C3)
    m3_element = Dict{Tuple{Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
    for (triple, (left, right)) in m3_raw
        diff = Dict{Symbol,Float64}()
        if left[2] !== nothing
            diff[left[2]] = get(diff, left[2], 0.0) + left[1]
        end
        if right[2] !== nothing
            diff[right[2]] = get(diff, right[2], 0.0) - right[1]
        end
        for (k, v) in diff
            if abs(v) < 1e-12
                delete!(diff, k)
            end
        end
        if !isempty(diff)
            m3_element[triple] = diff
        end
    end

    # 4. Differentials d0, d1, d2 and HH² dimension (still useful)
    d0 = build_d0(C2, basis, m2)
    d1 = build_d1(C2, basis, m2)
    d2 = build_d2_curved(C2, C3, mult_table, m2, C3_index)
    HH2_dim = compute_HH2(d0, d1, d2)

    # 5. Derivation basis (replaces HH¹)
    idempotents = [Symbol("e_$n") for n in nodes]
    # Build multiplication dictionary for derivations
    mult_dict_raw = Dict{Tuple{Symbol,Symbol}, Dict{Symbol,Float64}}()
    for a in basis, b in basis
        c, t = m2(a,b)
        if t !== nothing && abs(c) > 1e-12
            mult_dict_raw[(a,b)] = Dict(t => c)
        end
    end
    function mult_dict(a,b)
        return get(mult_dict_raw, (a,b), Dict{Symbol,Float64}())
    end

    N, M = derivation_basis(basis, idempotents, mult_dict)
    der_dim = size(N,2)
    println("Derivation dimension = $der_dim")

    # Build derivation basis info - Gerstenhaber Monodromy deformation
    deriv_basis_info = []
    n = length(basis)   # n = 39
    for j in 1:size(N, 2)
        v = N[:, j]
        region_weights = Dict{String, Float64}()
        for (idx, coeff) in enumerate(v)
            if abs(coeff) > 1e-8
                # Convert flat index to (src, tgt) pair
                src_idx = ((idx - 1) % n) + 1
                tgt_idx = div(idx - 1, n) + 1
                src_sym = basis[src_idx]
                tgt_sym = basis[tgt_idx]
                # Choose which region(s) to credit; here we credit the target region
                reg = symbol_to_region(tgt_sym)
                if reg != "UNK"
                    region_weights[reg] = get(region_weights, reg, 0.0) + abs(coeff)
                end
                # Optionally also credit the source region:
                # reg_src = symbol_to_region(src_sym)
                # if reg_src != "UNK"
                #     region_weights[reg_src] = get(region_weights, reg_src, 0.0) + abs(coeff)
                # end
            end
        end
        push!(deriv_basis_info, Dict("vector" => v, "regions" => region_weights))
    end

    # Convert each derivation column to a cochain map (for bracket computation)
    deriv_cochains = []
    for j in 1:der_dim
        v = N[:, j]
        X_raw = decode_derivation(v, basis) # Dict{Symbol, Dict{Symbol,Float64}}
        
        # Convert to a formal CochainMap (arity 1)
        # This wraps the Symbol key in a 1-tuple: (:arrow,)
        X_cochain = CochainMap()
        for (arrow_sym, lin_comb) in X_raw
            X_cochain[(arrow_sym,)] = lin_comb
        end
        
        push!(deriv_cochains, X_cochain)
    end

    # ---------- Gerstenhaber bracket on derivations ----------
    gerstenhaber_constants = []
    if der_dim > 0
        # Build matrix of derivation actions as vectors (to project bracket)
        # Represent each derivation as a vector of length |basis|^2 (flattened matrix)
        basis_vecs = zeros(length(basis)^2, der_dim)
        for j in 1:der_dim
            v = N[:, j]
            basis_vecs[:, j] = v
        end
        # Use QR for least squares
        Q, R = qr(basis_vecs)
        for i in 1:der_dim
            for j in 1:der_dim
                bracket_dict = derivation_bracket(deriv_cochains[i], deriv_cochains[j], basis)
                # Convert bracket_dict to a vector (flattened)
                vec = zeros(length(basis)^2)
                idx_dict = Dict(basis[k] => k for k in 1:length(basis))
                for (src, img) in bracket_dict
                    src_idx = idx_dict[src]
                    for (tgt, coeff) in img
                        tgt_idx = idx_dict[tgt]
                        vec[(src_idx-1)*length(basis) + tgt_idx] = coeff
                    end
                end
                # Project onto basis_vecs
                # following is incorrect as we need to 
                # project with correct dimension.
                # coeffs = R \ (Q' * vec)
                # After computing bracket_cochain, convert it to a vector `vec` of length n^2 (1521)
                # (you must construct `vec` correctly – each entry corresponds to a pair (source, target) in the basis)
                # Then:
                v_col = reshape(vec, :, 1)   # column vector
                basis_matrix = N              # N is your derivation basis matrix of size (n^2 × der_dim)
                coeffs = basis_matrix \ v_col   # solves N * coeffs ≈ vec

                for k in 1:der_dim
                    if abs(coeffs[k]) > 1e-8
                        push!(gerstenhaber_constants, (i-1, j-1, k-1, coeffs[k]))
                    end
                end
            end
        end
    else
        @warn "No derivations found, Gerstenhaber bracket empty"
    end

    # ---------- Cup product HH¹ ⊗ HH¹ → HH² (optional, skip for now) ----------
    cup_constants = []   # placeholder

    # 6. m4 obstruction
    m4_obs = compute_m4_obs(C4, m2, m3_raw)

    function m3_as_dict(a, b, c)
        return get(m3_element, (a, b, c), LinComb())
    end
    
    function m4_as_dict(a, b, c, d)
        return get(m4_obs, (a, b, c, d), LinComb())
    end
    # 5. Build HH² basis (stored arity = 3)
    # ... (you already have d0, d1, d2; compute ker_d2 and im_d1)
    ker_d2 = nullspace(Matrix(d2))
    im_d1 = Matrix(d1) * Matrix(I, size(d1,2), size(d1,2))
    P2 = I - im_d1 * pinv(im_d1)
    HH2_basis_mat = P2 * ker_d2
    keep2 = [norm(HH2_basis_mat[:,j]) > 1e-8 for j in 1:size(HH2_basis_mat,2)]
    HH2_basis_mat = HH2_basis_mat[:, keep2]
    # Convert each column to a CochainMap (arity 3)
    HH2_cochains = [cochain3_to_map(HH2_basis_mat[:, j], C3) for j in 1:size(HH2_basis_mat,2)]
    # Shifted Cup product to honor arity for projection.
    # Explicitly cast the first argument to match the candidate signature
    phi_typed = Vector{Dict{Tuple, Dict{Symbol, Float64}}}(deriv_cochains)
    cup_constants_tensor = compute_cup_constants(phi_typed, HH2_cochains, ctx)
    bracket_constants_tensor = compute_bracket_constants(phi_typed, ctx)
    # 7. Multiplication function returning Dict (for m5,m6)
    # We could use static cached mult_dict_raw, but for 
    # dynamics that is evolving that will be incorrect.
    mult_dict_changed_m2(x,y) = begin
        c, t = m2(x,y)
        return t === nothing ? Dict{Symbol,Float64}() : Dict(t => c)
    end

    # 8. m5 and m6
    m5 = compute_global_m5(C5, m3_element, m4_obs, mult_dict_changed_m2)
    mult_dict_fast(a,b) = begin
        c, t = m2(a,b)
        return t === nothing ? Dict{Symbol,Float64}() : Dict(t => c)
    end
    m6 = compute_global_m6(C6, mult_dict_fast, m3_element, m4_obs, m5)

    if der_dim > 0
        for i in 1:der_dim
            for j in 1:der_dim
                cp = total_cup_product(deriv_cochains[i], deriv_cochains[j], 2, 2, ctx; λ3=0.5, λ4=0.25)
                mass = cup_mass(cp)   # add cup_mass definition (same as your existing)
                # store mass or structure constants as needed
            end
        end
    end

    # 9. Prime paths
    prime_paths = extract_prime_paths(m6, top_n=100)
    # Annihilator and Support of A infinity
    ann_infty = annihilator_infty(basis, m3_element, m4_obs, m5, m6)

    all_active_symbols = Set{Symbol}()
    for m in (m3_element, m4_obs, m5, m6)
        for (inp, out_dict) in m
            union!(all_active_symbols, inp)
            union!(all_active_symbols, keys(out_dict))
        end
    end
    supp = collect(all_active_symbols)
    supp_infty = support_infty(basis, m3_element, m4_obs, m5, m6)

    # 1. Final Support Variety
    support_final = collect(Set(vcat(supp, supp_infty)))

    # Compute annihilator as complement of support (if you want)
    basis_syms = basis
    ann = [x for x in basis_syms if !(x in all_active_symbols)]

    # 2. Final Annihilator (The complement of all active logic)
    annihilator_final = collect(Set(vcat(ann, ann_infty)))   

    prime_ideals_4 =
        prime_higher_ideals_sortK(prime_paths, basis, m3_element, m4_obs, m5, m6)
    
    # If no prime ideals found (i.e., prime_paths empty or closure adds nothing), 
    # create a degenerate ideal per prime path (still 4‑tuple, total_support = 0)
    if isempty(prime_ideals_4) && !isempty(prime_paths)
        prime_ideals_4 = [(pth, wt, collect(pth), 0.0) for (pth, wt) in prime_paths]
    end

    # No need for separate prime_ideals_3 or export_ideals.
    prime_ideals_final = prime_ideals_4

    # Compute perversity.
    max_perv = 3   # e.g., 0–3 perversity levels
    perv_values = []
    for (idx, (path, wt, closure, total_supp)) in enumerate(prime_ideals_final)
        p = perversity(idx, prime_ideals_final; max_perv=max_perv)
        println("Ideal $idx: total_support=$total_supp → perversity=$p")
        push!(perv_values, (idx, p, closure))
    end
    
    # ---------- Prime path interaction via cup and bracket ----------
    prime_path_interactions = []
    if length(prime_paths) >= 2
        # Limit to top 20 prime paths for performance
        n_pp = min(20, length(prime_paths))
        for i in 1:n_pp
            for j in i+1:n_pp
                path_i = prime_paths[i][1]
                path_j = prime_paths[j][1]
                ci = prime_to_cochain(path_i)
                cj = prime_to_cochain(path_j)
                # Total cup product (with A∞ corrections)
                cp = total_cup_product(ci, cj, 6, 6, ctx; λ3=0.5, λ4=0.25)
                cup_mass_val = cup_mass(cp)
                # Gerstenhaber bracket
                br = gerstenhaber_bracket(ci, cj, 6, 6, ctx)
                bracket_mass_val = cup_mass(br)
                # Interaction score = sum of masses
                score = cup_mass_val + bracket_mass_val
                push!(prime_path_interactions, Dict(
                    "i" => i-1,
                    "j" => j-1,
                    "cup_mass" => cup_mass_val,
                    "bracket_mass" => bracket_mass_val,
                    "score" => score
                ))
            end
        end
        # Sort by score descending
        sort!(prime_path_interactions, by=x->x["score"], rev=true)
        # Keep top 100 or all
        if length(prime_path_interactions) > 100
            prime_path_interactions = prime_path_interactions[1:100]
        end
    end

    # Monodromy Deformation for Gerstenhaber Monodromy per region.
    println("\n=== Derivation basis info (first 3 derivations) ===")
    for (j, info) in enumerate(deriv_basis_info[1:min(3, end)])
        println("Derivation $j:")
        println("  vector (first 10 entries): ", info["vector"][1:min(10, end)])
        println("  region weights: ", info["regions"])
    end

    println("all_active_symbols size: ", length(all_active_symbols))
    println("support_final size: ", length(support_final))
    println("annihilator_final size: ", length(annihilator_final))
    return (m3_element, m4_obs, m5, m6, HH2_dim, 
            prime_paths,
            gerstenhaber_constants, 
            cup_constants, prime_path_interactions,
            annihilator_final,
            support_final,
            prime_ideals_final,
            deriv_basis_info)
end


# My edits to add gerstenhaber.
function compute_A∞(raw_coeffs, nodes;
        filt::Union{FilteredAInfAlgebra,Nothing}=nothing,
        edge_weights::Union{Dict{Tuple{Symbol,Symbol},Float64},Nothing}=nothing)
    # 1. Basis and multiplication
    basis = build_basis(nodes, raw_coeffs)
    m2 = make_m2(raw_coeffs, basis)
    is_composable(x,y) = tgt(x) == src(y)

    # 2. Composable chains -- use filtration if provided (prevents blow-up)
    C2, C3 = compute_composable_chains(basis, is_composable)
    C3_index = Dict(c => i for (i,c) in enumerate(C3))
    if filt !== nothing
        ew = edge_weights !== nothing ? edge_weights : Dict{Tuple{Symbol,Symbol},Float64}(
            (a,b) => abs(get(raw_coeffs, (a,b), 1.0)) for (a,b) in keys(raw_coeffs))
        println("compute_A∞: using filtration (lambda=$(filt.lambda), cutoff=$(filt.energy_cutoff))")
        C4 = build_Ck_filtered(basis, is_composable, 4, ew, filt; max_size=20000)
        C5 = build_Ck_filtered(basis, is_composable, 5, ew, filt; max_size=50000)
        C6 = build_Ck_filtered(basis, is_composable, 6, ew, filt; max_size=100000)
    else
        C4 = build_Ck(basis, is_composable, 4; max_size=20000)
        C5 = build_Ck(basis, is_composable, 5; max_size=50000)
        C6 = build_Ck(basis, is_composable, 6; max_size=100000)
    end
    println("|C2| = $(length(C2)), |C3| = $(length(C3)), |C4| = $(length(C4)), |C5| = $(length(C5)), |C6| = $(length(C6))")

    # 3. m3 (associator) and multiplication table
    m3_raw, mult_table = compute_m3(basis, m2, C2, C3)
    # convert m3 to element form (left - right)
    m3_element = Dict{Tuple{Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
    for (triple, (left, right)) in m3_raw
        diff = Dict{Symbol,Float64}()
        if left[2] !== nothing
            diff[left[2]] = get(diff, left[2], 0.0) + left[1]
        end
        if right[2] !== nothing
            diff[right[2]] = get(diff, right[2], 0.0) - right[1]
        end
        for (k, v) in diff
            if abs(v) < 1e-12
                delete!(diff, k)
            end
        end
        if !isempty(diff)
            m3_element[triple] = diff
        end
    end

    # 4. Differential d0, d1, d2 and HH²
    d0 = build_d0(C2, basis, m2)
    d1 = build_d1(C2, basis, m2)
    d2 = build_d2_curved(C2, C3, mult_table, m2, C3_index)
    HH2_dim = compute_HH2(d0, d1, d2)
    """
    ker_d1 = nullspace(Matrix(d1))
    dimA = length(basis)
    im_d0 = Matrix(d0) * I(dimA)
    # projection onto orthogonal complement of im_d0
    # C0=A to C1=Hom⁡(A⊗A,A)
    P = I - im_d0 * pinv(im_d0)
    K_mod = P * ker_d1
    keep = [norm(K_mod[:,j]) > 1e-8 for j in 1:size(K_mod,2)]
    HH1_basis = K_mod[:, keep]   # each column is a HH¹ class
    ker_d2 = nullspace(Matrix(d2))   # you already have this as `ker_d2`
    im_d1 = Matrix(d1) * Matrix(I, size(d1,2), size(d1,2))   # or use d1 * I
    P2 = I - im_d1 * pinv(im_d1)
    HH2_basis = P2 * ker_d2
    keep2 = [norm(HH2_basis[:,j]) > 1e-8 for j in 1:size(HH2_basis,2)]
    HH2_basis = HH2_basis[:, keep2]
    # After HH1_basis is computed (as a matrix |C2| × r)
    HH1_cochains = [basis_vector_to_cochain(HH1_basis[:, j], C2) for j in 1:size(HH1_basis,2)]

    # Compute structure constants
    n = length(HH1_cochains)
    struct_const = []
    for i in 1:n
        for j in 1:n
            bracket = gerstenhaber_bracket(HH1_cochains[i], HH1_cochains[j], 2, 2)
            # bracket is a Dict{Tuple, LinComb}. To get coefficients in the HH1 basis,
            # you would need to project each output cochain (on C2) back to the basis.
            # This is a linear algebra problem – we can skip here for brevity.
        end
    end

    # Compute cup product on HH² (example)
    if HH2_dim > 0
        HH2_cochains = [cochain2_to_map(HH2_basis[:, j], C2) for j in 1:HH2_dim]
        cup_2_2 = cup_product(HH2_cochains[1], HH2_cochains[2], 2, 2)  # example
    end
    """

    # 5. m4 obstruction
    m4_obs = compute_m4_obs(C4, m2, m3_raw)   # uses raw m3 with left/right

    # 6. Multiplication function returning Dict (for m5,m6)
    mult_dict(x,y) = begin
        c, t = m2(x,y)
        return t === nothing ? Dict{Symbol,Float64}() : Dict(t => c)
    end

    # 7. m5 and m6
    m5 = compute_global_m5(C5, m3_element, m4_obs, mult_dict)
    #m6 = compute_global_m6(C6, m2, m3_element, m4_obs, m5)
    mult_dict_fast(a,b) = begin
        c, t = m2(a,b)
        return t === nothing ? Dict{Symbol,Float64}() : Dict(t => c)
    end
    m6 = compute_global_m6(C6, mult_dict_fast, m3_element, m4_obs, m5)
    # 8. Prime paths
    prime_paths = extract_prime_paths(m6, top_n=100)

    return m3_element, m4_obs, m5, m6, HH2_dim, prime_paths
end

# ============================================================================
# 5. Dynamic edge weight replacement
# ============================================================================

function update_raw_coeffs_with_weights(raw_coeffs, edge_weight_map, nodes)
    # node_to_idx mapping
    node_to_idx = Dict(n => i-1 for (i, n) in enumerate(nodes))
    new_raw_coeffs = Dict{Tuple{Symbol,Symbol},Float64}()
    for ((x, y), old_coeff) in raw_coeffs
        # x and y are symbols like :f_CA1sp_HPF, :f_HPF_BLA
        # Extract source and target regions
        x_str = String(x)
        y_str = String(y)
        if !startswith(x_str, "f_") || !startswith(y_str, "f_")
            # Keep idempotent relations unchanged
            new_raw_coeffs[(x,y)] = old_coeff
            continue
        end
        x_parts = split(x_str, '_')
        y_parts = split(y_str, '_')
        x_src = Symbol(x_parts[2])
        x_tgt = Symbol(x_parts[3])
        y_src = Symbol(y_parts[2])
        y_tgt = Symbol(y_parts[3])
        if x_tgt != y_src
            continue
        end
        # Get edge weights
        w_xy = get(edge_weight_map, (node_to_idx[x_src], node_to_idx[x_tgt]), 1.0)
        w_yz = get(edge_weight_map, (node_to_idx[y_src], node_to_idx[y_tgt]), 1.0)
        new_coeff = w_xy * w_yz
        new_raw_coeffs[(x, y)] = new_coeff
    end
    return new_raw_coeffs
end

# ============================================================================
# 6. JSON export helpers
# ============================================================================
function tuple_to_key(tup)
    parts = [string(x) for x in tup]
    return "(" * join(parts, ", ") * ")"
end



function export_ainf_to_json(
    m3, m4, m5, m6, HH2_dim, prime_paths,
    gerstenhaber, cup, prime_path_interactions,
    filename;
    ann=[],
    supp=[],
    prime_ideals=[],
    deriv_basis_info=[],  # list of dicts with "vector" and "regions"
    graph_type="Q_6"      # which connectome graph: Q_6 | Q_7L | Q_7P | Q_8
)
    m3_json = Dict(tuple_to_key(k) => Dict(string(tgt) => coeff for (tgt, coeff) in v) for (k, v) in m3)
    m4_json = Dict(tuple_to_key(k) => Dict(string(tgt) => coeff for (tgt, coeff) in v) for (k, v) in m4)
    m5_json = Dict(tuple_to_key(k) => Dict(string(tgt) => coeff for (tgt, coeff) in v) for (k, v) in m5)
    m6_json = Dict(tuple_to_key(k) => Dict(string(tgt) => coeff for (tgt, coeff) in v) for (k, v) in m6)

    prime_paths_json = [Dict("path" => [string(s) for s in path], "weight" => weight) for (path, weight) in prime_paths]

    # Gerstenhaber structure constants
    gerstenhaber_json = [Dict("i" => i, "j" => j, "k" => k, "coeff" => coeff) for (i,j,k,coeff) in gerstenhaber]

    # Cup product constants
    cup_product_json = [Dict("i" => i, "j" => j, "k" => k, "coeff" => coeff) for (i,j,k,coeff) in cup]

    # Prime path interactions
    interactions_json = [Dict("i" => d["i"], "j" => d["j"],
                              "cup_mass" => d["cup_mass"],
                              "bracket_mass" => d["bracket_mass"],
                              "score" => d["score"]) for d in prime_path_interactions]

    # Convert annihilator and support to list of strings
    ann_json = [string(x) for x in ann]
    supp_json = [string(x) for x in supp]

    # Convert prime higher ideals (supports both 3‑tuple and 4‑tuple)
    prime_ideals_json = []
    for (idx, item) in enumerate(prime_ideals)
        # Compute perversity for this ideal using its index
        perv_val = perversity(idx, prime_ideals; max_perv=2)
        if length(item) == 4
            path, weight, closure, total_supp = item
            push!(prime_ideals_json, Dict(
                "path" => [string(s) for s in path],
                "weight" => weight,
                "closure" => [string(s) for s in closure],
                "total_support" => total_supp,
                "perversity" => perv_val
            ))
        else
            path, weight, closure = item
            # For 3‑tuple (older format), we still include perversity (computed via weight? Or just 0)
            push!(prime_ideals_json, Dict(
                "path" => [string(s) for s in path],
                "weight" => weight,
                "closure" => [string(s) for s in closure],
                "total_support" => 0.0,
                "perversity" => perv_val
            ))
        end
    end

    # Convert derivation basis info (already a list of serializable dicts)
    deriv_basis_json = deriv_basis_info

    # ---------------------------------------------------------------
    # Ihara spectral radius — computed from prime path weights.
    #
    # IMPORTANT: do NOT column-normalise T. A column-stochastic matrix
    # always has spectral radius = 1.0 (Perron-Frobenius), destroying
    # all per-snapshot variation. The raw weighted T gives the actual
    # A∞-deformed transfer amplitude, which varies with dynamics.
    #
    # Three variants are exported:
    #   ihara_radius         — raw ρ(T), log-weighted prime paths
    #   ihara_radius_norm    — ρ(T)/√(n_regions), normalised for graph size
    #   ihara_radius_h1      — H1-cycle-restricted ρ (cylinder/trinion)
    # ---------------------------------------------------------------
    ihara_radius, ihara_radius_norm, ihara_radius_q = let
        # Determine region set from graph_type (support all 4 graph types)
        region_order = if graph_type ∈ ("Q_7P",)
            [:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY]
        elseif graph_type ∈ ("Q_7L",)
            [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY]
        elseif graph_type == "Q_8"
            [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :PAL, :sAMY]
        else  # Q_6 default
            [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY]
        end
        n_r = length(region_order)
        reg_idx = Dict(r => i for (i, r) in enumerate(region_order))

        T_raw = zeros(Float64, n_r, n_r)   # raw log-weighted transfer matrix

        for (path, weight) in prime_paths
            abs_w = log(1 + abs(weight))   # log-compression keeps values finite
            for k in 1:length(path)-1
                parts_a = split(string(path[k]),   "_")
                parts_b = split(string(path[k+1]), "_")
                if length(parts_a) >= 3 && length(parts_b) >= 3
                    src_r = Symbol(parts_a[2])
                    tgt_r = Symbol(parts_b[2])
                    a = get(reg_idx, src_r, 0)
                    b = get(reg_idx, tgt_r, 0)
                    if a > 0 && b > 0
                        T_raw[b, a] += abs_w
                    end
                end
            end
        end

        # Raw spectral radius — varies per snapshot, reflects A∞ dynamics
        eigs_raw = eigvals(T_raw)
        rho_raw  = maximum(abs.(eigs_raw))

        # Size-normalised: divide by √n_r so different graph types are comparable
        rho_norm = rho_raw / sqrt(n_r)

        # Mean out-degree q_eff of T_raw (for Ramanujan ratio ρ/√q in test script)
        q_eff = mean(vec(sum(T_raw, dims=2)))

        rho_raw, rho_norm, q_eff
    end

    # ── Bridge B: H1 transfer eigenvalue ─────────────────────────────────────
    h1_data = let
        # Define H1 cycles per graph type
        if graph_type ∈ ("Q_6", "Q_7L")
            h1_cycles = [[:f_BLA_LA, :f_LA_sAMY, :f_sAMY_BLA]]
        elseif graph_type ∈ ("Q_7P", "Q_8")
            h1_cycles = [[:f_BLA_LA, :f_LA_sAMY, :f_sAMY_BLA],
                         [:f_HY_sAMY, :f_sAMY_PAL, :f_PAL_HY]]
        else
            h1_cycles = [[:f_BLA_LA, :f_LA_sAMY, :f_sAMY_BLA]]
        end

        # Build weighted transfer matrix on H1 cycles from prime paths
        cycle_eigenvalues = Float64[]
        for cycle in h1_cycles
            n_c = length(cycle)
            W = zeros(Float64, n_c, n_c)
            for (path, weight) in prime_paths
                abs_w = abs(weight)
                for k in 1:length(path)-1
                    src_sym = path[k]
                    tgt_sym = path[k+1]
                    i_pos = findfirst(==(src_sym), cycle)
                    j_pos = findfirst(==(tgt_sym), cycle)
                    if i_pos !== nothing && j_pos !== nothing
                        W[i_pos, j_pos] += abs_w
                    end
                end
            end
            eigs_W = eigvals(W)
            push!(cycle_eigenvalues, maximum(abs.(eigs_W)))
        end

        # Arrow index maps (1-based)
        if graph_type ∈ ("Q_6", "Q_7L")
            arrow_indices = [[5, 11, 12]]
        elseif graph_type == "Q_7P"
            arrow_indices = [[3, 10, 13], [8, 12, 18]]
        elseif graph_type == "Q_8"
            arrow_indices = [[5, 12, 15], [10, 14, 20]]
        else
            arrow_indices = [[3, 10, 13]]  # default: BLA cycle
        end

        h1_cycle_eigenvalues = Float64[]
        h1_cycle_traces = Int[]

        for h1_idx in arrow_indices
            n_c = length(h1_idx)
            cycle_trace = n_c
            push!(h1_cycle_traces, cycle_trace)

            # Compute cycle weight from prime paths
            cycle_wt = 0.0
            for (path, weight) in prime_paths
                abs_w = abs(weight)
                # Determine if path contains this cycle
                # Arrow indices use MAGMA canonical Q_7P numbering:
                #   [3,10,13] = f_BLA_LA, f_LA_sAMY, f_sAMY_BLA  (BLA cycle)
                #   [8,12,18] = f_HY_PAL, f_PAL_sAMY, f_sAMY_HY  (PAL cycle)
                if h1_idx == [3,10,13]
                    target_arrows = ["f_BLA_LA","f_LA_sAMY","f_sAMY_BLA"]
                elseif h1_idx == [8,12,18] || h1_idx == [8,14,20]
                    target_arrows = ["f_HY_PAL","f_PAL_sAMY","f_sAMY_HY"]
                elseif h1_idx == [3,10,13]  # Q_6/Q_7L fallback (same as above)
                    target_arrows = ["f_BLA_LA","f_LA_sAMY","f_sAMY_BLA"]
                else
                    target_arrows = String[]
                end
                path_syms = [string(s) for s in path]
                # init=0 prevents crash when target_arrows is empty
                hits = sum((1 for s in target_arrows if s ∈ path_syms); init=0)
                if !isempty(target_arrows) && hits == length(target_arrows)
                    cycle_wt += abs_w
                end
            end

            λ_h1 = cycle_wt / max(n_c, 1)
            push!(h1_cycle_eigenvalues, λ_h1)
        end

        # SAFETY: If h1_cycle_traces is empty, provide defaults
        if isempty(h1_cycle_traces)
            h1_cycle_traces = [3]
            h1_cycle_eigenvalues = [0.0]
        end

        Dict(
            "h1_cycles"                 => [[string(s) for s in c] for c in h1_cycles],
            "h1_transfer_eigenvalues"   => cycle_eigenvalues,
            "h1_transfer_max"           => isempty(cycle_eigenvalues) ? 0.0 : maximum(cycle_eigenvalues),
            "h1_arrow_indices"          => arrow_indices,
            "h1_cycle_traces"           => h1_cycle_traces,
            "h1_cycle_eigenvalues"      => h1_cycle_eigenvalues,
            "h1_cycle_eigenvalue_max"   => isempty(h1_cycle_eigenvalues) ? 0.0 : maximum(h1_cycle_eigenvalues),
            "ihara_radius_from_paths"   => ihara_radius,
            # bridge_b_ratio: H1 cycle eigenvalue / full transfer radius
            # (meaningful now that ihara_radius is the raw, varying quantity)
            "bridge_b_ratio"            => isempty(h1_cycle_eigenvalues) ? 0.0 :
                                           maximum(h1_cycle_eigenvalues) / max(ihara_radius, 1e-10),
            "bridge_b_algebraic"        => isempty(h1_cycle_traces) ? 0.0 :
                                           Float64(h1_cycle_traces[1]) / max(ihara_radius, 1e-10),
            "graph_type"                => graph_type,
            "b1"                        => length(h1_cycles)
        )
    end

    data = Dict(
        "m3" => m3_json,
        "m4" => m4_json,
        "m5" => m5_json,
        "m6" => m6_json,
        "HH2_dim" => HH2_dim,
        "prime_paths" => prime_paths_json,
        "gerstenhaber" => gerstenhaber_json,
        "cup_product" => cup_product_json,
        "prime_path_interactions" => interactions_json,
        "annihilator_infty" => ann_json,
        "support_infty" => supp_json,
        "prime_higher_ideals" => prime_ideals_json,
        "derivation_basis" => deriv_basis_json,
        "ihara_radius"             => ihara_radius,
        "ihara_radius_norm"        => ihara_radius_norm,    # ρ/√n_regions (graph-size normalised)
        "ihara_radius_q"           => ihara_radius_q,       # mean out-degree of T_raw (= q_eff for test script)
        "H1_cycles"                => h1_data["h1_cycles"],
        "H1_transfer_eigenvalues"  => h1_data["h1_transfer_eigenvalues"],
        "H1_transfer_max"          => h1_data["h1_transfer_max"],
        "H1_arrow_indices"         => h1_data["h1_arrow_indices"],
        "H1_cycle_traces"          => h1_data["h1_cycle_traces"],
        "H1_cycle_eigenvalues"     => h1_data["h1_cycle_eigenvalues"],
        "H1_cycle_eigenvalue_max"  => h1_data["h1_cycle_eigenvalue_max"],
        "bridge_b_ratio"           => h1_data["bridge_b_ratio"],
        "bridge_b_algebraic"       => h1_data["bridge_b_algebraic"],
        "bridge_b_b1"              => h1_data["b1"],
        "graph_type"               => graph_type
    )
    open(filename, "w") do f
        JSON3.write(f, data)
    end
end

using Graphs, GraphPlot, Plots

# ------------------------------------------------------------
# Function to compute node/edge scores from m6 (sum of m6 path weights)
# ------------------------------------------------------------
function compute_m6_scores_cached(m6_dict, region_to_nodes)
    node_score = Dict{Int, Float64}()
    edge_score = Dict{Tuple{Int,Int}, Float64}()
    
    for (path, coeff_dict) in m6_dict
        path_weight = sum(abs(v) for v in values(coeff_dict))
        if !isempty(path)
            reg = symbol_to_region(path[1])
            # Get node list for this region (direct lookup)
            nodes_in_region = get(region_to_nodes, reg, [])
            for nid in nodes_in_region
                node_score[nid] = get(node_score, nid, 0.0) + path_weight
            end
        end
    end
    return node_score, edge_score
end
function compute_m6_scores(m6_dict, node_regions_dict, edges_df)
    node_score = Dict{Int, Float64}()
    edge_score = Dict{Tuple{Int,Int}, Float64}()
    
    for (path, coeff_dict) in m6_dict
        path_weight = sum(abs(v) for v in values(coeff_dict))
        # path is a tuple of symbols (like (:f_CA1sp_HPF, :f_HPF_BLA, ...))
        # We need to map these symbols to node IDs. For now, we only have region-level.
        # Simplified: assign weight to all nodes in the first region of the path
        if !isempty(path)
            first_sym = path[1]
            reg = symbol_to_region(first_sym)  # you have this function
            # find nodes in that region
            for (nid, reglist) in node_regions_dict
                if reg in reglist
                    node_score[nid] = get(node_score, nid, 0.0) + path_weight
                end
            end
        end
        # Similarly, we could assign edge weights for consecutive region pairs
    end
    
    # Also, add the per‑edge curveness from the original data as a baseline
    for i in 1:nrow(edges_df)
        u = edges_df.node1id[i]
        v = edges_df.node2id[i]
        edge_score[(u,v)] = get(edge_score, (u,v), 0.0) + edges_df.curveness[i]
    end
    
    return node_score, edge_score
end

function region_obstruction_from_m6(m6, region_names)
    region_score = Dict{Symbol,Float64}()
    for (path, coeff_dict) in m6
        weight = sum(abs(v) for v in values(coeff_dict))
        for sym in path
            reg = symbol_to_region(sym)   # you have this function
            region_score[Symbol(reg)] = get(region_score, Symbol(reg), 0.0) + weight
        end
    end
    return region_score
end

# ------------------------------------------------------------
# Extract a subgraph around a seed node, up to a given number of nodes/edges
# ------------------------------------------------------------
function extract_subgraph(edges_df, seed_node_id, max_nodes=100)
    # Build adjacency list
    adj = Dict{Int, Vector{Int}}()
    for i in 1:nrow(edges_df)
        u = edges_df.node1id[i]
        v = edges_df.node2id[i]
        push!(get!(adj, u, []), v)
        push!(get!(adj, v, []), u)
    end
    
    # BFS from seed
    visited = Set{Int}()
    queue = [seed_node_id]
    while length(visited) < max_nodes && !isempty(queue)
        node = popfirst!(queue)
        node in visited && continue
        push!(visited, node)
        for nb in get(adj, node, [])
            if !(nb in visited)
                push!(queue, nb)
            end
        end
    end
    
    # Collect edges within the visited set
    sub_edges = []
    for i in 1:nrow(edges_df)
        u = edges_df.node1id[i]
        v = edges_df.node2id[i]
        if u in visited && v in visited
            push!(sub_edges, (u, v))
        end
    end
    
    return collect(visited), sub_edges
end

# ------------------------------------------------------------
# Create a graph and plot it with obstruction scores
# ------------------------------------------------------------
function extract_subgraph_from_adj(adj, seed, max_nodes)
    visited = Set{Int}()
    queue = [seed]
    while length(visited) < max_nodes && !isempty(queue)
        node = popfirst!(queue)
        node in visited && continue
        push!(visited, node)
        for nb in get(adj, node, [])
            if !(nb in visited)
                push!(queue, nb)
            end
        end
    end
    # Collect edges within visited set
    sub_edges = []
    for u in visited
        for v in get(adj, u, [])
            if v in visited && u < v   # avoid duplicates
                push!(sub_edges, (u, v))
            end
        end
    end
    return collect(visited), sub_edges
end

using Compose

using Dates

function plot_blowup_subgraph_cached(seed_region, sub_nodes, sub_edges, node_score, edge_score, node_region)
    node_index = Dict(node => i for (i, node) in enumerate(sub_nodes))
    g = SimpleGraph(length(sub_nodes))
    edge_weights = Float64[]
    for (u, v) in sub_edges
        add_edge!(g, node_index[u], node_index[v])
        w = get(edge_score, (u, v), 0.0) + get(edge_score, (v, u), 0.0)
        push!(edge_weights, w)
    end

    color_dict = Dict(
        "CA1sp" => "red", "HPF" => "blue", "BLA" => "green",
        "sAMY" => "orange", "HY" => "purple", "LA" => "brown"
    )
    node_colors = [get(color_dict, get(node_region, n, "unknown"), "gray") for n in sub_nodes]
    node_sizes = max.(1.0, log10.(1 .+ [get(node_score, n, 1.0) for n in sub_nodes])) * 10
    edge_widths = max.(0.5, log10.(1 .+ edge_weights))

    x, y = spring_layout(g)
    ctx = gplot(g, x, y;
                nodefillc=node_colors,
                nodesize=node_sizes,
                edgelinewidth=edge_widths)
    
    # Unique filename using timestamp (second precision)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    filename = "blowup_$(seed_region)_$(timestamp).svg"
    draw(SVG(filename, 16cm, 16cm), ctx)
    println("Diagram saved as $filename")
end

function plot_blowup_subgraph(seed_region, nodes_df, edges_df, node_score, edge_score, max_nodes=100)
    # Find seed node using node_regions (must be defined)
    if !isdefined(Main, :node_regions)
        error("node_regions not defined. Build it first.")
    end
    region_nodes = [node for (node, regs) in node_regions if string(seed_region) in regs]
    if isempty(region_nodes)
        error("No node found for region $seed_region")
    end
    seed = first(region_nodes)
    
    # Extract subgraph
    sub_nodes, sub_edges = extract_subgraph(edges_df, seed, max_nodes)
    
    # Build graph
    node_index = Dict(node => i for (i, node) in enumerate(sub_nodes))
    g = Graph(length(sub_nodes))
    edge_weights = Float64[]
    for (u,v) in sub_edges
        add_edge!(g, node_index[u], node_index[v])
        w = get(edge_score, (u,v), 0.0) + get(edge_score, (v,u), 0.0)
        push!(edge_weights, w)
    end
    
    # Node colours
    color_dict = Dict(
        "CA1sp" => "red", "HPF" => "blue", "BLA" => "green",
        "sAMY" => "orange", "HY" => "purple", "LA" => "brown"
    )
    node_colors = []
    for node in sub_nodes
        regs = get(node_regions, node, [])
        reg = isempty(regs) ? "unknown" : regs[1]
        color = get(color_dict, reg, "gray")
        push!(node_colors, color)
    end
    
    # Node sizes
    node_sizes = [get(node_score, n, 1.0) for n in sub_nodes]
    node_sizes = max.(1.0, log10.(1 .+ node_sizes)) * 10
    
    # Edge widths
    edge_widths = max.(0.5, log10.(1 .+ edge_weights))
    
    # Layout and plot
    layout = spring_layout(g)
    p = graphplot(g, layout,
                  nodecolor=node_colors, nodesize=node_sizes,
                  edgewidth=edge_widths, edgelabel="")
    plot(p, title="Blow‑up of $seed_region (m₆ obstruction intensity)")
    savefig("blowup_$(seed_region).png")
    println("Diagram saved as blowup_$(seed_region).png")
end

# ============================================================================
# 7. Main entry point and A∞-only mode
# ============================================================================
function compute_and_export(input_weights_file::String, output_json_file::String;
        filt::Union{FilteredAInfAlgebra,Nothing}=nothing,
        graph_type::String="Q_6")

    # Remove any stray commas or whitespace
    input_weights_file = strip(input_weights_file, [',', ' '])
    output_json_file = strip(output_json_file, [',', ' '])

    # Debug: print the exact filename
    #println("Reading edge weights from: ", repr(input_weights_file))
    println("Reading edge weights from:  $input_weights_file")
    weights_dict = JSON3.read(read(input_weights_file, String))
    # Build edge_weight_map: (Int,Int) -> Float64
    edge_weight_map = Dict{Tuple{Int,Int},Float64}()
    for (key, w) in weights_dict
        # Skip any key that does not contain "->" (like "seed_region")
        if !occursin("->", String(key))
            continue
        end
        parts = split(String(key), "->")
        u = parse(Int, parts[1])
        v = parse(Int, parts[2])
        edge_weight_map[(u, v)] = w
    end

    # Parse original static relations
    raw_coeffs = parse_relations(relations_str)
    # Update coefficients with dynamic weights
    new_raw_coeffs = update_raw_coeffs_with_weights(raw_coeffs, edge_weight_map, nodes)
    # Compute A∞
    # Build Symbol edge weights for filtration (if Phase 2)
    _ew_sym_ce = Dict{Tuple{Symbol,Symbol},Float64}()
    if filt !== nothing
        _node_id_to_sym_ce = Dict(0=>:CA1sp,1=>:HPF,2=>:BLA,3=>:sAMY,4=>:HY,5=>:LA)
        for ((u,v), w) in edge_weight_map
            us = get(_node_id_to_sym_ce, u, nothing)
            vs = get(_node_id_to_sym_ce, v, nothing)
            (us !== nothing && vs !== nothing) && (_ew_sym_ce[(us,vs)] = w)
        end
    end
    m3, m4, m5, m6, HH2_dim, prime_paths = compute_A∞(new_raw_coeffs, nodes;
        filt=filt, edge_weights=isempty(_ew_sym_ce) ? nothing : _ew_sym_ce)
    # graph_type passed through for Bridge B H1 export
    _graph_type_ce = @isdefined(_filt_graph_type) ? _filt_graph_type : "Q_6"
    #m3, m4, m5, m6, HH2_dim, prime_paths, gerstenhaber, cup, prime_path_interactions, deriv_basis_info = gerstenhaber_compute_A∞(new_raw_coeffs, nodes)
    # Export
    gerstenhaber = []
    cup = []
    prime_path_interactions = []
    ann = []
    supp = []
    prime_ideals = []
    deriv_basis_info = [] # for Gerstenhaber monodromy to apply infinitesimal deformations.
    export_ainf_to_json(m3, m4, m5, m6, HH2_dim, 
        prime_paths, gerstenhaber, cup, 
        prime_path_interactions, 
        output_json_file;
        ann = ann, 
        supp = supp,
        prime_ideals=prime_ideals,
        deriv_basis_info=deriv_basis_info,
        graph_type=graph_type
    )
    println("A∞ data written to $output_json_file")
end
# Compute region obstruction from m₆ (optional, for dynamic seed selection)
function symbol_to_region(sym)
    s = String(sym)
    if startswith(s, "e_")
        return s[3:end]
    elseif startswith(s, "f_")
        return split(s, "_")[2]
    else
        return "UNK"
    end
end

# Write VTK file
function write_subgraph_vtk(sub_nodes, sub_edges, node_score, edge_score, points, node_region, region_name, timestamp)
    if isempty(sub_nodes)
        @warn "No nodes in subgraph for $region_name. VTK will be empty."
        return
    end

    # Ensure coordinates are a 3xN matrix (REQUIRED for WriteVTK)
    points_mat = Matrix{Float64}(undef, 3, length(sub_nodes))
    for (i, node) in enumerate(sub_nodes)
        points_mat[:, i] = points[node]
    end

    # Zero-based indexing for VTK cells
    node_to_idx = Dict(node => i-1 for (i, node) in enumerate(sub_nodes))
    
    cells = MeshCell[]
    valid_edge_scores = Float64[]
    
    for (u, v) in sub_edges
        if haskey(node_to_idx, u) && haskey(node_to_idx, v)
            push!(cells, MeshCell(VTK_LINE, [node_to_idx[u], node_to_idx[v]]))
            # Map obstruction magnitude to the edge
            w = get(edge_score, (u, v), 0.0) + get(edge_score, (v, u), 0.0)
            push!(valid_edge_scores, w)
        end
    end

    filename = "blowup_$(region_name)_$(timestamp)" # .vtu added by vtk_grid
    vtk_grid(filename, points_mat, cells) do vtk
        vtk["node_score"] = [get(node_score, n, 0.0) for n in sub_nodes]
        vtk["edge_score"] = valid_edge_scores
        # Use log10 scale if dealing with 10^18 magnitudes to make it visible
        vtk["log_node_score"] = log10.(abs.([get(node_score, n, 0.0) for n in sub_nodes]) .+ 1.0)
    end
    println("VTK blow‑up file saved as $filename")
end

function write_marker_vtk(x::Float64, y::Float64, z::Float64, magnitude::Float64, filename::String)
    if !endswith(filename, ".vtk")
        filename *= ".vtk"
    end
    open(filename, "w") do f
        println(f, "# vtk DataFile Version 3.0")
        println(f, "Blowup marker with obstruction magnitude")
        println(f, "ASCII")
        println(f, "DATASET POLYDATA")
        println(f, "POINTS 1 float")
        println(f, "$x $y $z")
        println(f, "VERTICES 1 2")
        println(f, "1 0")
        println(f, "POINT_DATA 1")
        println(f, "SCALARS blowup_magnitude float 1")
        println(f, "LOOKUP_TABLE default")
        println(f, "$magnitude")
    end
    println("Marker saved as $filename (magnitude = $magnitude)")
end

function write_subgraph_vtk_deepseek(sub_nodes, sub_edges, node_score, edge_score, points, node_region, region_name, timestamp)
    # Prepare points array
    coords = Float64[]
    for node in sub_nodes
        pt = points[node]
        append!(coords, pt)
    end
    points_mat = reshape(coords, 3, length(sub_nodes))

    node_index = Dict(node => i-1 for (i, node) in enumerate(sub_nodes))
    cells = MeshCell[]
    for (u, v) in sub_edges
        push!(cells, MeshCell(VTK_LINE, [node_index[u], node_index[v]]))
    end

    node_scores = Float64[get(node_score, n, 0.0) for n in sub_nodes]
    region_to_idx = Dict("CA1sp"=>0, "HPF"=>1, "BLA"=>2, "sAMY"=>3, "HY"=>4, "LA"=>5)
    region_ids = Int[get(region_to_idx, get(node_region, n, "unknown"), 0) for n in sub_nodes]

    edge_scores = Float64[]
    for (u, v) in sub_edges
        w = get(edge_score, (u, v), 0.0) + get(edge_score, (v, u), 0.0)
        push!(edge_scores, w)
    end

    filename = "blowup_$(region_name)_$(timestamp).vtu"
    vtk_grid(filename, points_mat, cells) do vtk
        vtk["node_score"] = node_scores
        vtk["region_id"] = region_ids
        vtk["edge_score"] = edge_scores
    end
    println("VTK blow‑up file saved as $filename")
end
# Generate VTU from brain where rees blew up
function extract_cube_vtu(full_vtu_file::String, cx::Float64, cy::Float64, cz::Float64, half_size::Float64, output_file::String)
    println("Extracting cube at center ($cx, $cy, $cz) with half‑size $half_size")
    bounds = [cx - half_size, cx + half_size,
              cy - half_size, cy + half_size,
              cz - half_size, cz + half_size]
    println("Bounds: ", bounds)
    python_script = """
import pyvista as pv
import numpy as np
mesh = pv.read("$full_vtu_file")
print(f"Mesh has {mesh.n_points} points and {mesh.n_cells} cells")
bounds = $bounds
print(f"Clipping box: {bounds}")
clipped = mesh.clip_box(bounds, invert=False)
print(f"Clipped mesh has {clipped.n_points} points and {clipped.n_cells} cells")
if clipped.n_points == 0:
    print("WARNING: No points inside the clipping box. Try increasing half_size.")
clipped.save("$output_file")
print("Cube saved to", "$output_file")
"""
    script_file = tempname() * ".py"
    write(script_file, python_script)
    run(`python $script_file`)
    rm(script_file)
    return output_file
end
# Check for command‑line argument
# ============================================================================
# 7. Main entry point – three modes
# ============================================================================
# ARGS parsing — two modes, both support optional filt_config.json
#
# Mode 1 (--ainf-only):
#   julia script.jl --ainf-only weights.json output.json [filt_config.json]
#   ARGS: [1]=--ainf-only  [2]=weights  [3]=output  [4]=filt_config (optional)
#
# Mode 2 (--full):
#   julia script.jl --full weights.json output.json region [filt_config.json]
#   ARGS: [1]=--full  [2]=weights  [3]=output  [4]=region  [5]=filt_config (optional)
#
# filt_config.json schema (Phase 2, from BALBc_Opiate_Norcain.py):
#   { "phase": 2, "lambda": 1.0, "energy_cutoff": 1e-8,
#     "max_path_len": 20, "m0_curvature": {"sAMY": 0.043, "HPF": 0.012} }
#
# Phase 1 (no filt_config): filt = nothing → flat A∞, m0=0, no filtration
# Phase 2 (filt_config present): filt = FilteredAInfAlgebra(...) → curved A∞
# ============================================================================

"""
    load_filt_config(path) -> Union{FilteredAInfAlgebra, Nothing}

Read a filtration config JSON and return a FilteredAInfAlgebra,
or nothing if the file doesn't exist or has phase=1.
"""
function load_filt_config(path::String)
    isfile(path) || return nothing
    data = JSON3.read(read(path, String))
    phase = Int(get(data, "phase", 1))
    phase == 1 && return nothing

    lam   = Float64(get(data, "lambda",        1.0))
    ecut  = Float64(get(data, "energy_cutoff", 1e-8))
    mlen  = Int(get(data,    "max_path_len",   20))
    raw_m0 = get(data, "m0_curvature", Dict())
    m0 = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k,v) in raw_m0)

    println("  [Filtration] Phase 2 active: lambda=$lam cutoff=$ecut max_len=$mlen")
    println("  [Filtration] m0_curvature: $m0")
    return FilteredAInfAlgebra(
        lambda        = lam,
        max_path_len  = mlen,
        energy_cutoff = ecut,
        m0_curvature  = m0
    )
end

# Optional: build Symbol edge weights from Int-keyed map (reused across modes)
function sym_edge_weights(edge_weight_map, graph_type::String="Q_6")
    node_id_to_sym = if graph_type == "Q_7P"
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:PAL, 6=>:sAMY)
    elseif graph_type == "Q_7L"
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:LSX, 6=>:sAMY)
    elseif graph_type == "Q_8"
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:LSX, 6=>:PAL, 7=>:sAMY)
    else  # Q_6 default
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:sAMY)
    end
    ew = Dict{Tuple{Symbol,Symbol},Float64}()
    for ((u,v), w) in edge_weight_map
        us = get(node_id_to_sym, u, nothing)
        vs = get(node_id_to_sym, v, nothing)
        (us !== nothing && vs !== nothing) && (ew[(us,vs)] = w)
    end
    return ew
end

if length(ARGS) >= 1 && ARGS[1] == "--ainf-only"
    # Mode 1: export A∞ JSON only
    # Usage: --ainf-only weights.json output.json [filt_config.json]
    if length(ARGS) < 3
        error("Usage: --ainf-only input_weights.json output.json [filt_config.json]")
    end
    input_weights_file = ARGS[2]
    output_json_file   = ARGS[3]
    # ARGS[4..N]: graph_type (Q_6|Q_7P|Q_7L|Q_8) and/or filt_config path
    # Python may pass them in any order: filt before graph_type or after
    _known_graphs    = ("Q_6", "Q_7P", "Q_7L", "Q_8")
    _graph_type      = _graph_type_global  # from top-level const (already parsed)
    filt_config_path = ""
    for arg in ARGS[4:end]
        if arg ∈ _known_graphs
            _graph_type = arg
        elseif !isempty(arg) && endswith(arg, ".json")
            filt_config_path = arg
        end
    end
    _filt = isempty(filt_config_path) ? nothing : load_filt_config(filt_config_path)

    if _filt === nothing
        println("Mode: --ainf-only  Phase 1 (flat A∞, m0=0)  graph=$(_graph_type)")
        compute_and_export(input_weights_file, output_json_file;
                           graph_type=_graph_type)
    else
        println("Mode: --ainf-only  Phase 2 (curved A∞ + filtration)")
        # Load weights and run filtered computation
        weights_dict = JSON3.read(read(input_weights_file, String))
        edge_weight_map = Dict{Tuple{Int,Int},Float64}()
        for (key, w) in weights_dict
            key_str = String(key)
            !occursin("->", key_str) && continue
            parts = split(key_str, "->")
            u = parse(Int, parts[1]); v = parse(Int, parts[2])
            edge_weight_map[(u,v)] = Float64(w)
        end
        raw_coeffs = parse_relations(relations_str)
        new_raw_coeffs = update_raw_coeffs_with_weights(raw_coeffs, edge_weight_map, nodes)
        _ew_sym = sym_edge_weights(edge_weight_map, _graph_type)
        m3, m4, m5, m6, HH2_dim, prime_paths = compute_A∞(
            new_raw_coeffs, nodes;
            filt=_filt, edge_weights=_ew_sym)
        export_ainf_to_json(m3, m4, m5, m6, HH2_dim, prime_paths, 
                            [], [], [],            # gerstenhaber, cup, prime_path_interactions 
                            output_json_file;
                            graph_type=_graph_type)
        println("Phase 2 --ainf-only export complete.")
    end
    exit(0)
elseif ARGS[1] == "--full"
    println("=== FULL MODE: computing A∞, exporting JSON, and plotting blow-up ===")
    if length(ARGS) < 4
        error("Usage: --full input_weights.json output.json region_name [filt_config.json]")
    end
    input_weights_file = ARGS[2]
    output_json_file   = ARGS[3]
    seed_region_str    = ARGS[4]
    # ARGS[5..N]: graph_type and/or filt_config path in any order
    _known_graphs_full = ("Q_6", "Q_7P", "Q_7L", "Q_8")
    _graph_type_full   = _graph_type_global  # from top-level const
    filt_config_path   = ""
    for arg in ARGS[5:end]
        if arg ∈ _known_graphs_full
            _graph_type_full = arg
        elseif !isempty(arg) && endswith(arg, ".json")
            filt_config_path = arg
        end
    end
    _filt = isempty(filt_config_path) ? nothing : load_filt_config(filt_config_path)
    if _filt === nothing
        println("Phase 1 (flat A∞, m0=0, no filtration)")
    else
        println("Phase 2 (curved A∞ + filtration active)")
    end
    seed_region = Symbol(seed_region_str)
    println("Seed region: $seed_region")

    # Load node/edge data (check paths)
    # Load cached graph (extremely fast)
    # For SVG
    # adj, node_regions, region_nodes = load_cached_graph()
    # For VTK showing 3 d graph of brain where REES blow up was 
    # applied
    adj, points, node_regions, region_nodes = load_cached_graph_VTK()

    # Get a seed node in the requested region
    seed_nodes = get(region_nodes, seed_region_str, [])
    if isempty(seed_nodes)
        error("Region $seed_region_str not found in cached graph")
    end
    seed = first(seed_nodes)
    println("Seed node: $seed")

    # Extract subgraph of up to 100 nodes
    sub_nodes, sub_edges = extract_subgraph_from_adj(adj, seed, 1000)
    println("Subgraph: $(length(sub_nodes)) nodes, $(length(sub_edges)) edges")
    println("Seed node $seed has neighbors: ", get(adj, seed, []))
    # Build node_score dictionary (from m6) – you already compute this
    # For plotting, we need edge_score as well (from m6)
    # Read weights and skip non‑edge keys
    weights_dict = JSON3.read(read(input_weights_file, String))
    cx = get(weights_dict, "centroid_x", 0.0)
    cy = get(weights_dict, "centroid_y", 0.0)
    cz = get(weights_dict, "centroid_z", 0.0)
    println("Centroid from JSON: ($cx, $cy, $cz)")
    edge_weight_map = Dict{Tuple{Int,Int},Float64}()
    for (key, w) in weights_dict
        key_str = String(key)
        if !occursin("->", key_str)
            println("Skipping non‑edge key: $key_str")
            continue
        end
        parts = split(key_str, "->")
        u = parse(Int, parts[1])
        v = parse(Int, parts[2])
        edge_weight_map[(u, v)] = w
    end
    println("Number of edge weights loaded: ", length(edge_weight_map))

    # Compute A∞
    raw_coeffs = parse_relations(relations_str)
    new_raw_coeffs = update_raw_coeffs_with_weights(raw_coeffs, edge_weight_map, nodes)
    #m3, m4, m5, m6, HH2_dim, prime_paths = compute_A∞(new_raw_coeffs, nodes)
    
    #export_ainf_to_json(m3, m4, m5, m6, HH2_dim, prime_paths, output_json_file)
    # Build Symbol-keyed edge weights for filtration (Int keys -> Symbol keys)
    # node_id_to_sym uses CSV index order from region_to_idx
    # Q_6:  BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,sAMY=5
    # Q_7P: BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,PAL=5,sAMY=6
    # Q_7L: BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,LSX=5,sAMY=6
    # Q_8:  BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,LSX=5,PAL=6,sAMY=7
    _node_id_to_sym = if _graph_type_full == "Q_7P"
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:PAL, 6=>:sAMY)
    elseif _graph_type_full == "Q_7L"
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:LSX, 6=>:sAMY)
    elseif _graph_type_full == "Q_8"
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:LSX, 6=>:PAL, 7=>:sAMY)
    else  # Q_6
        Dict(0=>:BLA, 1=>:CA1sp, 2=>:HPF, 3=>:HY, 4=>:LA, 5=>:sAMY)
    end
    # Build symbol edge weights using the per-graph node_id_to_sym already defined above
    _ew_sym = Dict{Tuple{Symbol,Symbol},Float64}()
    for ((u,v), w) in edge_weight_map
        us = get(_node_id_to_sym, u, nothing)
        vs = get(_node_id_to_sym, v, nothing)
        (us !== nothing && vs !== nothing) && (_ew_sym[(us,vs)] = w)
    end
    println("Symbol edge weights loaded: ", length(_ew_sym))

    # Use _filt from filt_config.json (Phase 2) or default Phase 1 flat
    # _ew_sym already computed above from edge_weight_map
    _full_filt = _filt !== nothing ? _filt :
                 FilteredAInfAlgebra(lambda=1.0, max_path_len=20, energy_cutoff=1e-8)

    m3, m4, m5, m6, HH2_dim, prime_paths, gerstenhaber, cup, prime_path_interactions, ann, supp, prime_ideal_paths, deriv_basis_info = gerstenhaber_compute_A∞(
        new_raw_coeffs, nodes;
        filt         = _full_filt,
        edge_weights = _ew_sym
    )
    # Export JSON
    export_ainf_to_json(m3, m4, m5, m6, HH2_dim, 
        prime_paths, 
        gerstenhaber, 
        cup, 
        prime_path_interactions, 
        output_json_file;
        ann = ann, 
        supp = supp,
        prime_ideals = prime_ideal_paths,
        deriv_basis_info=deriv_basis_info,
        graph_type=_graph_type_full)

    println("Gerstenhaber/Cup JSON exported to $output_json_file")

    region_to_nodes = Dict{String, Vector{Int}}()
    for (nid, reg_str) in node_regions
        push!(get!(region_to_nodes, reg_str, []), nid)
    end

    # Compute node/edge scores from m6
    node_score, edge_score = compute_m6_scores_cached(m6, region_to_nodes)
    println("Node score computed for ", length(node_score), " nodes")

    # Plot blow‑up subgraph
    SAVE_VTK=true
    SAVE_SVG=false
    if (SAVE_SVG)
        println("Plotting blow‑up subgraph for region $seed_region")
        # Now plot using the subgraph
        plot_blowup_subgraph_cached(seed_region, sub_nodes, sub_edges, node_score, edge_score, node_regions)
        println("Blow‑up diagram saved.")
    end
    if (SAVE_VTK)
        timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
        # Single marker is not well supported in non ascii format
        # in paraview
        marker_file = "blowup_marker_$(seed_region)_$(timestamp).vtk"
        if !isfinite(cx) || !isfinite(cy) || !isfinite(cz)
            println("WARNING: Invalid centroid ($cx, $cy, $cz); skipping marker.")
            return
        end
        # Determine a meaningful magnitude for the marker
        if !isempty(prime_paths)
            # Use the weight of the heaviest prime path (largest m6 obstruction)
            magnitude = maximum(w for (_, w) in prime_paths)
            println("Using prime path max weight as magnitude: $magnitude")
        else
            # Fallback: use the sum of all node scores (total obstruction)
            magnitude = sum(values(node_score))
            println("No prime paths; using node_score sum as magnitude: $magnitude")
        end

        # Clip to a reasonable range for visualisation (avoid huge numbers or zeros)
        if magnitude <= 0.0
            magnitude = 1.0
        else
            magnitude = min(magnitude, 1e6)  # prevent overflow in colour map
        end
        write_marker_vtk(cx, cy, cz, magnitude, marker_file)
        # Marker files were not getting generated.
        if isfile(marker_file)
            println("Marker file saved: $(abspath(marker_file))")
        else
            println("ERROR: Marker file not created!")
        end
        write_subgraph_vtk(sub_nodes, sub_edges, node_score, edge_score, points, node_regions, String(seed_region), timestamp)
        println("Blow‑up subgraph VTK saved.")
        timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
        cube_file = "blowup_cube_$(seed_region)_$(timestamp).vtu"
        extract_cube_vtu(FULL_BRAIN_VTU, cx, cy, cz, 200.0, cube_file)
        println("Blow‑up cube saved as $cube_file")
    end

    exit(0)
else
    # Mode 3: original full simulation (no arguments)
    println("=== Curved A∞ Hochschild HH² + m₄ Obstruction (Full Simulation) ===")

    # Load node and edge DataFrames
    nodes_df = CSV.read(NODES_FILE, DataFrame)
    edges_df = CSV.read(EDGES_FILE, DataFrame)

    node_regions = Dict{Int, Vector{String}}()
    for r in eachrow(nodes_df)
        node_regions[r.id] = parse_region_string(r.regions)
    end

    # Compute A∞ using static relations
    raw_coeffs = parse_relations(relations_str)
    m3, m4, m5, m6, HH2_dim, prime_paths = compute_A∞(raw_coeffs, nodes)

    
    region_obstruction = Dict{String,Float64}()
    for (path, outdict) in m6
        weight = sum(abs(v) for v in values(outdict))
        for sym in path
            reg = symbol_to_region(sym)
            region_obstruction[reg] = get(region_obstruction, reg, 0.0) + weight
        end
    end
    if !isempty(region_obstruction)
        seed_region = first(sort(collect(region_obstruction), by=x->-x[2]))[1]
        seed_region_sym = Symbol(seed_region)
        println("Most obstructed region (blow‑up center): $seed_region")
    else
        seed_region_sym = :sAMY
        println("No m₆ obstruction found; using default region sAMY")
    end

    node_score, edge_score = compute_m6_scores(m6, node_regions, edges_df)
    plot_blowup_subgraph(seed_region_sym, nodes_df, edges_df, node_score, edge_score, 100)

    # Continue with original VTK, time evolution, etc.
    # ... (paste your original large script's code here) ...

    println("Full simulation completed.")
end
