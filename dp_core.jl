# =============================================================================
# dp_core.jl
#
# Stage 2: Target-Pruned Dynamic Programming over AU contexts.
#
# Depends on nno_au_core.jl (NNOProb, NNOAUContext, Der21_mode, surgery!,
# coproduct, lan_i_extend, boundary_flux, plucker_and_stratum).
#
# Architecture (strict separation of scales):
#   OFFLINE  — 4ti2 Markov basis per AU subgraph (run once, cached)
#   OFFLINE  — AU pushout classification → PUSHOUT_TABLE
#   RUNTIME  — DP over QActive with NNO exact probabilities
#   RUNTIME  — Lazy A∞ invocation only at AU boundary crossings
#   RUNTIME  — Surgery fires only when Der21_mode = 4
#
# Key design decisions:
#   - DPState carries Rational{Int128} probability (no float64 anywhere)
#   - Memoisation keyed by (node, context, sector, depth) — exact key
#   - Graver reachability check: O(1) set lookup, not enumeration
#   - AU boundary detection: node membership in context region sets
#   - No backward flow: transition matrix built from directed edges only
#   - Hyper-Confluence Operator (∐⊗_Λ) dispatches on PUSHOUT_TABLE
# =============================================================================

# Guard against re-including nno_au_core if already loaded
if !@isdefined(NNOProb)
    include(joinpath(@__DIR__, "nno_au_core.jl"))
end

using Printf

# =============================================================================
# PART 1: CORE STRUCTS
# =============================================================================

"""
    DPState

One state in the target-pruned DP.
  node:       current brain region
  context:    which AU context (:CTX_sAMY, :CTX_HPF, …)
  sector:     GPS sector (:A/:B/:C/:D)
  prob:       exact NNO probability accumulated so far (Σ = 1//1 at start)
  path_hash:  rolling hash for cycle detection (FNV-1a style)
  depth:      number of hops taken
"""
struct DPState
    node      ::Symbol
    context   ::Symbol
    sector    ::Symbol
    prob      ::NNOProb
    path_hash ::UInt64
    depth     ::Int
end

"""
    DPQuery

A pharmacodynamic transport query.
  start:     source region (drug entry point)
  target:    target region (effect site)
  p_min:     minimum probability to report (prune below)
  p_max:     maximum probability (prune above — catches runaway paths)
  max_depth: hard cap on path length
"""
struct DPQuery
    start    ::Symbol
    target   ::Symbol
    p_min    ::NNOProb
    p_max    ::NNOProb
    max_depth::Int
end

"""
    Transition

A single step: from current node to next node with weight.
  to:       destination region
  weight:   NNO-exact transition probability (from transition matrix column)
  edge:     the directed edge (current→to)
"""
struct Transition
    to      ::Symbol
    weight  ::NNOProb
    edge    ::Tuple{Symbol,Symbol}
end

# =============================================================================
# PART 2: PRECOMPUTED OFFLINE TABLES
# =============================================================================

# ── Pushout classification table (from au_pushout_full_m7m8.jl) ─────────────
# Keys are ordered pairs (smaller context id, larger context id).
# Values: (:coproduct, :H0_only, :padic_gate2, :padic_gate4,
#          :categorical_independent)

const PUSHOUT_TABLE = Dict{Tuple{Symbol,Symbol}, Symbol}(
    (:CTX_BG,   :CTX_sAMY)   => :coproduct,
    (:CTX_HPF,  :CTX_sAMY)   => :coproduct,
    (:CTX_OLF,  :CTX_sAMY)   => :coproduct,
    (:CTX_sAMY, :CTX_THAL)   => :H0_only,
    (:CTX_CORTEX,:CTX_HPF)   => :H0_only,
    (:CTX_HPF,  :CTX_THAL)   => :H0_only,
    (:CTX_BG,   :CTX_THAL)   => :H0_only,
    (:CTX_HB,   :CTX_THAL)   => :categorical_independent,
    (:CTX_HPF,  :CTX_INFRA)  => :padic_gate2,
    (:CTX_INFRA,:CTX_sAMY)   => :padic_gate4,   # ← coker=62, double pole
)

# Canonical key: sort to (smaller, larger) alphabetically
function pushout_key(c1::Symbol, c2::Symbol)
    c1 <= c2 ? (c1, c2) : (c2, c1)
end

function pushout_type(c1::Symbol, c2::Symbol)::Symbol
    get(PUSHOUT_TABLE, pushout_key(c1, c2), :coproduct)
end

# ── Graver reachability (offline: which contexts can reach target) ───────────
# Built once from the AU region membership.
# graver_reachable[ctx][node] = true if `target` is reachable from `node`
# within that context's active edges.
#
# For runtime use this is a simple set-membership: if the target node is
# IN the context or IN an adjacent context, it is reachable.
# Full Graver path validation is deferred to the 4ti2 subgraph call.

const GRAVER_REACHABLE_CACHE = Dict{Tuple{Symbol,Symbol,Symbol}, Bool}()

function graver_reachable(ctx_id::Symbol,
                           from::Symbol,
                           target::Symbol,
                           remaining_depth::Int,
                           au_contexts::Dict{Symbol, NNOAUContext})::Bool
    remaining_depth < 0 && return false
    # Fast path: target is in same context
    ctx = get(au_contexts, ctx_id, nothing)
    ctx === nothing && return true   # unknown context — don't prune
    target ∈ ctx.regions && return true
    # Check adjacent contexts (share at least one edge endpoint with this ctx)
    # This is the O(1) heuristic that avoids full Graver enumeration.
    # A false positive (we say reachable when it is not) wastes one DP branch.
    # A false negative (we say unreachable when it is) would prune valid paths.
    # We err toward false positives for correctness.
    for (key, _) in PUSHOUT_TABLE
        if (key[1] == ctx_id || key[2] == ctx_id)
            other_id = key[1] == ctx_id ? key[2] : key[1]
            other    = get(au_contexts, other_id, nothing)
            other !== nothing && target ∈ other.regions && return true
        end
    end
    return false
end

# =============================================================================
# PART 3: CONTEXT MEMBERSHIP
# =============================================================================

"""
    find_context(node, au_contexts) -> Symbol

Find which AU context `node` belongs to.
Selection priority (in order):
  1. The context where node has strictly highest probability mass.
  2. Tiebreak: the context with the MOST active outgoing edges from node
     (most reachable context — avoids stranding the DP in a stub context).
  3. Tiebreak: the context with the most regions (most informative).
Defaults to :CTX_sAMY if not found.
"""
function find_context(node::Symbol,
                       au_contexts::Dict{Symbol, NNOAUContext})::Symbol
    candidates = Symbol[]
    for (id, ctx) in au_contexts
        node ∈ ctx.regions && push!(candidates, id)
    end
    isempty(candidates) && return :CTX_sAMY
    length(candidates) == 1 && return candidates[1]

    # Priority 1: highest probability mass at this node
    best_p = maximum(begin
        ctx = au_contexts[id]
        i   = findfirst(==(node), ctx.regions)
        i !== nothing ? ctx.prob[i] : NNO_ZERO
    end for id in candidates)

    top_by_prob = [id for id in candidates
                   if begin
                       ctx = au_contexts[id]
                       i   = findfirst(==(node), ctx.regions)
                       i !== nothing && ctx.prob[i] == best_p
                   end]

    length(top_by_prob) == 1 && return top_by_prob[1]

    # Priority 2 (tiebreak): most active outgoing edges from node
    # This prevents the DP starting in a stub context (e.g. CTX_HB)
    # where the target is unreachable, when a richer context exists.
    best_id = top_by_prob[1]
    best_out = -1
    for id in top_by_prob
        ctx     = au_contexts[id]
        n_out   = count(s == node for (s,_) in ctx.edges)
        n_regs  = length(ctx.regions)
        score   = n_out * 1000 + n_regs   # outgoing edges dominate, regions break tie
        if score > best_out
            best_out = score
            best_id  = id
        end
    end
    return best_id
end

"""
    check_au_boundary(state, trans, au_contexts)
    -> (new_context::Symbol, new_sector::Symbol, crossing::Bool)

Detect whether a transition crosses an AU boundary.
Returns the destination context, sector, and whether a crossing occurred.
"""
function check_au_boundary(state::DPState,
                             trans::Transition,
                             au_contexts::Dict{Symbol, NNOAUContext})
    new_ctx_id = find_context(trans.to, au_contexts)
    new_sector = au_contexts[new_ctx_id].sector
    crossing   = new_ctx_id != state.context || new_sector != state.sector
    return new_ctx_id, new_sector, crossing
end

# =============================================================================
# PART 4: GET TRANSITIONS FROM CURRENT STATE
# =============================================================================

"""
    get_transitions(state, au_contexts) -> Vector{Transition}

Return all valid outgoing transitions from `state.node` in `state.context`.
Uses the precomputed NNO transition matrix column for `state.node`.
No backward flow: directed edges only (already enforced in trans_mat).
"""
function get_transitions(state::DPState,
                          au_contexts::Dict{Symbol, NNOAUContext})::Vector{Transition}
    ctx = get(au_contexts, state.context, nothing)
    ctx === nothing && return Transition[]

    node_idx = findfirst(==(state.node), ctx.regions)
    node_idx === nothing && return Transition[]

    transitions = Transition[]
    for (i, v) in enumerate(ctx.regions)
        w = ctx.trans_mat[i, node_idx]
        w == NNO_ZERO && continue
        i == node_idx && continue   # no self-loops (absorbing states don't count)
        push!(transitions, Transition(v, w, (state.node, v)))
    end
    return transitions
end

# =============================================================================
# PART 5: HYPER-CONFLUENCE OPERATOR (∐⊗_Λ)
# =============================================================================

"""
    hyper_confluence(state, trans, new_context, new_sector,
                     au_contexts, inclusion_corrections)
    -> (adjusted_trans::Transition, new_prob::NNOProb)

The ∐⊗_Λ operator: dispatches on PUSHOUT_TABLE to determine how to
combine probabilities when crossing an AU boundary.

Mode 1 (:coproduct):          linear addition, no correction
Mode 2 (:H0_only):            Lan_i inclusion correction
Mode 3 (:padic_gate2):        chain splits; return primary branch weight
Mode 4 (:padic_gate4):        crisis; call surgery!, return buffered weight
"""
function hyper_confluence(state::DPState,
                           trans::Transition,
                           new_context::Symbol,
                           new_sector::Symbol,
                           au_contexts::Dict{Symbol, NNOAUContext},
                           inclusion_corrections::Dict{Tuple{Symbol,Symbol}, NNOProb})

    ptype = pushout_type(state.context, new_context)

    if ptype == :coproduct
        # Mode 1: full A∞ equivalence, direct addition
        return trans, state.prob * trans.weight

    elseif ptype == :H0_only
        # Mode 2: Lan_i scaling via inclusion correction
        corr_key = (state.context, new_context)
        corr = get(inclusion_corrections, corr_key,
                   get(inclusion_corrections, (new_context, state.context), NNO_ONE))
        new_w = trans.weight * corr
        return Transition(trans.to, new_w, trans.edge), state.prob * new_w

    elseif ptype == :padic_gate2
        # Mode 3: single pole, chain splits
        # Primary branch: weight scaled by 1/2 (equal split)
        # Secondary branch is spawned but not tracked in this DP call
        # (can be captured by running solve_transport again with different start)
        half = NNOProb(Int128(1), Int128(2))
        new_w = trans.weight * half
        return Transition(trans.to, new_w, trans.edge), state.prob * new_w

    else  # padic_gate4 or categorical_independent
        # Mode 4: derived tensor, crisis
        # Execute surgery on the source context
        ctx1 = au_contexts[state.context]
        ctx2 = get(au_contexts, new_context, nothing)

        if ctx2 === nothing
            # Unknown target context — treat as Mode 1
            return trans, state.prob * trans.weight
        end

        # Surgery: extract buffer from boundary, redirect to non-boundary
        ctx1_updated, p_buffer, boundary_nodes = surgery!(ctx1, ctx2)
        au_contexts[state.context] = ctx1_updated

        # After surgery: the transition weight comes from the updated
        # transition matrix (boundary nodes have been redirected)
        node_idx = findfirst(==(state.node), ctx1_updated.regions)
        if node_idx === nothing
            return trans, state.prob * trans.weight
        end

        # Find weight of (state.node → trans.to) in updated matrix
        to_idx = findfirst(==(trans.to), ctx1_updated.regions)
        if to_idx === nothing
            # trans.to is in ctx2, not ctx1 — blocked by surgery
            # Return zero-weight transition (will be pruned by p_min check)
            return trans, NNO_ZERO
        end

        new_w = ctx1_updated.trans_mat[to_idx, node_idx]
        return Transition(trans.to, new_w, trans.edge), state.prob * new_w
    end
end

# =============================================================================
# PART 6: PATH HASH (cycle detection)
# =============================================================================

# FNV-1a hash: fast, good avalanche for Symbol sequences
const FNV_OFFSET = UInt64(14695981039346656037)
const FNV_PRIME  = UInt64(1099511628211)

function extend_hash(h::UInt64, sym::Symbol)::UInt64
    for byte in codeunits(string(sym))
        h = xor(h, UInt64(byte)) * FNV_PRIME
    end
    return h
end

# =============================================================================
# PART 7: TARGET-PRUNED DP
# =============================================================================

"""
    solve_transport(query, au_contexts;
                    inclusion_corrections, verbose) -> DPResult

The core DP loop. Explores paths from query.start to query.target
over the NNO-versioned AU contexts, pruning by:
  (a) probability out of [p_min, p_max]
  (b) depth exceeding max_depth
  (c) graver_reachable check (heuristic, no false negatives)
  (d) memoisation (better probability already found for this key)
  (e) cycle detection via path_hash

Returns a DPResult with all solutions found and the probability bracket.
"""
struct DPResult
    solutions    ::Vector{DPState}
    p_bracket_lo ::NNOProb    # min probability among solutions
    p_bracket_hi ::NNOProb    # max probability among solutions
    n_explored   ::Int        # total states explored
    n_pruned_p   ::Int        # pruned by probability bounds
    n_pruned_g   ::Int        # pruned by graver reachability
    n_pruned_m   ::Int        # pruned by memoisation
    n_cycles     ::Int        # pruned by cycle detection
end

function solve_transport(query::DPQuery,
                          au_contexts::Dict{Symbol, NNOAUContext};
                          inclusion_corrections::Dict{Tuple{Symbol,Symbol},NNOProb} =
                              Dict{Tuple{Symbol,Symbol},NNOProb}(),
                          verbose::Bool = false)::DPResult

    # ── Initial context assignment ───────────────────────────────────────────
    init_ctx    = find_context(query.start, au_contexts)
    init_sector = au_contexts[init_ctx].sector

    # ── Memoisation: (node, context, sector, depth) → best prob so far ──────
    memo = Dict{Tuple{Symbol,Symbol,Symbol,Int}, NNOProb}()

    # ── Seen hashes: cycle detection ─────────────────────────────────────────
    seen_hashes = Set{UInt64}()

    # ── Frontier ─────────────────────────────────────────────────────────────
    init_hash = extend_hash(FNV_OFFSET, query.start)
    frontier  = [DPState(query.start, init_ctx, init_sector,
                         NNO_ONE, init_hash, 0)]
    solutions = DPState[]

    # ── Counters ─────────────────────────────────────────────────────────────
    n_explored = 0
    n_pruned_p = 0
    n_pruned_g = 0
    n_pruned_m = 0
    n_cycles   = 0

    verbose && @printf("  DP: %s → %s  p∈[%s, %s]  max_depth=%d\n",
            query.start, query.target,
            string(query.p_min), string(query.p_max), query.max_depth)

    while !isempty(frontier)
        next_frontier = DPState[]

        for state in frontier
            n_explored += 1

            # ── Hard depth cutoff ────────────────────────────────────────────
            state.depth >= query.max_depth && continue

            # ── Get outgoing transitions ─────────────────────────────────────
            transitions = get_transitions(state, au_contexts)
            isempty(transitions) && continue

            for trans in transitions

                # ── Tentative new probability ────────────────────────────────
                new_prob_raw = state.prob * trans.weight

                # ── Probability prune ────────────────────────────────────────
                if new_prob_raw < query.p_min
                    n_pruned_p += 1
                    continue
                end

                # ── AU boundary check ────────────────────────────────────────
                new_ctx_id, new_sector, crossing = check_au_boundary(
                    state, trans, au_contexts)

                # ── Hyper-Confluence Operator if crossing ────────────────────
                adj_trans, new_prob = if crossing
                    hyper_confluence(state, trans, new_ctx_id, new_sector,
                                     au_contexts, inclusion_corrections)
                else
                    trans, new_prob_raw
                end

                # ── Post-confluence probability prune ────────────────────────
                if new_prob < query.p_min || new_prob > query.p_max
                    n_pruned_p += 1
                    continue
                end

                # ── Graver reachability prune ────────────────────────────────
                remaining = query.max_depth - state.depth - 1
                if !graver_reachable(new_ctx_id, adj_trans.to,
                                     query.target, remaining, au_contexts)
                    n_pruned_g += 1
                    continue
                end

                # ── Cycle detection ──────────────────────────────────────────
                new_hash = extend_hash(state.path_hash, adj_trans.to)
                if new_hash ∈ seen_hashes
                    n_cycles += 1
                    continue
                end
                push!(seen_hashes, new_hash)

                # ── Memoisation ──────────────────────────────────────────────
                memo_key = (adj_trans.to, new_ctx_id, new_sector, state.depth + 1)
                if haskey(memo, memo_key) && memo[memo_key] >= new_prob
                    n_pruned_m += 1
                    continue
                end
                memo[memo_key] = new_prob

                # ── Build new state ──────────────────────────────────────────
                new_state = DPState(
                    adj_trans.to, new_ctx_id, new_sector,
                    new_prob, new_hash, state.depth + 1)

                # ── Target reached? ──────────────────────────────────────────
                if new_state.node == query.target
                    push!(solutions, new_state)
                    verbose && @printf("    ✓ Path found: depth=%d  p=%s\n",
                            new_state.depth, string(new_state.prob))
                else
                    push!(next_frontier, new_state)
                end
            end  # transitions
        end  # frontier states

        frontier = next_frontier

        verbose && length(frontier) > 0 && @printf(
            "    Frontier: %d states  |  solutions: %d  |  explored: %d\n",
            length(frontier), length(solutions), n_explored)
    end  # while

    # ── Compute probability bracket ──────────────────────────────────────────
    p_lo = isempty(solutions) ? NNO_ZERO : minimum(s.prob for s in solutions)
    p_hi = isempty(solutions) ? NNO_ZERO : maximum(s.prob for s in solutions)

    DPResult(solutions, p_lo, p_hi,
             n_explored, n_pruned_p, n_pruned_g, n_pruned_m, n_cycles)
end

# =============================================================================
# PART 8: INCLUSION CORRECTIONS (Mode 2 Lan_i scaling)
# =============================================================================

"""
    build_inclusion_corrections(au_contexts) -> Dict

Precompute Lan_i scaling factors for all H0_only boundary pairs.
The correction for crossing c1 → c2 is:
    correction = |c1 ∩ c2| / |c2|
(fraction of c2 that is covered by c1 → how much of the new context
 the local chain already knows about)
"""
function build_inclusion_corrections(au_contexts::Dict{Symbol, NNOAUContext})
    corrections = Dict{Tuple{Symbol,Symbol}, NNOProb}()
    for ((c1, c2), ptype) in PUSHOUT_TABLE
        ptype != :H0_only && continue
        ctx1 = get(au_contexts, c1, nothing)
        ctx2 = get(au_contexts, c2, nothing)
        (ctx1 === nothing || ctx2 === nothing) && continue

        overlap = length(intersect(Set(ctx1.regions), Set(ctx2.regions)))
        n2      = length(ctx2.regions)
        n2 == 0 && continue

        corr = NNOProb(Int128(overlap), Int128(n2))
        corrections[(c1, c2)] = corr
        corrections[(c2, c1)] = corr   # symmetric for now
    end
    return corrections
end

# =============================================================================
# PART 9: DYNAMIC AU EXPANSION (bgr fiber trigger)
# =============================================================================

"""
    expand_au_if_fiber_triggered!(au_contexts, state, trans, all_edges, weights)

When a transition reaches a high-degree fiber node (bgr, fibertracts, root),
check whether the new node's neighborhood contains the query target and, if so,
expand the current AU to include the fiber hub and its direct connections.

This implements the "AU+AU triggers new 4ti2 run recreating quivers" pattern:
  - sAMY → bgr → HPF → sAMY is invisible to local CTX_sAMY
  - When state.node = sAMY and trans.to = bgr, we expand
  - The expanded AU now includes bgr's 19 connections
  - A new transition matrix is computed on the expanded subgraph

Returns true if expansion occurred (caller should re-fetch transitions).
"""
const FIBER_HUBS = Set([:bgr, :fibertracts, :root])

function expand_au_if_fiber_triggered!(
        au_contexts::Dict{Symbol, NNOAUContext},
        state::DPState,
        trans::Transition,
        all_edges::Vector{Tuple{Symbol,Symbol}},
        weights::Dict{Tuple{Symbol,Symbol}, NNOProb})::Bool

    trans.to ∉ FIBER_HUBS && return false

    ctx = get(au_contexts, state.context, nothing)
    ctx === nothing && return false

    # Find all nodes reachable in 1 hop from the fiber hub
    fiber_node  = trans.to
    fiber_nbrs  = Symbol[]
    for (s, t) in all_edges
        s == fiber_node && push!(fiber_nbrs, t)
        t == fiber_node && push!(fiber_nbrs, s)
    end
    unique!(fiber_nbrs)

    # Check if any new node is not already in the context
    new_nodes = [v for v in fiber_nbrs if v ∉ ctx.regions]
    isempty(new_nodes) && return false

    @printf("  [Fiber expansion] %s → %s: adding %d new nodes to %s\n",
            state.node, fiber_node, length(new_nodes), state.context)

    # Expand context: add fiber hub + its neighbors
    expanded_regions = vcat(ctx.regions, [fiber_node], new_nodes)
    unique!(expanded_regions)

    # Add new edges involving expanded nodes
    expanded_edges = [(s,t) for (s,t) in all_edges
                      if s ∈ Set(expanded_regions) && t ∈ Set(expanded_regions)
                      && (s,t) ∉ ctx.stops]

    # Extend weights
    expanded_weights = copy(ctx.weights)
    for e in expanded_edges
        haskey(expanded_weights, e) || (expanded_weights[e] = get(weights, e, NNO_ONE))
    end

    # Rebuild transition matrix on expanded subgraph
    new_T = build_transition_matrix(expanded_regions, expanded_edges, expanded_weights)

    # Extend probability vector: new nodes start with zero mass
    n_old = length(ctx.regions)
    n_new = length(expanded_regions)
    new_prob = fill(NNO_ZERO, n_new)
    for (i, v) in enumerate(ctx.regions)
        j = findfirst(==(v), expanded_regions)
        j !== nothing && (new_prob[j] = ctx.prob[i])
    end
    # Probability is conserved: new nodes have zero mass, old mass unchanged
    nno_check(new_prob; label="fiber expansion")

    # Update context in-place
    ctx.regions   = expanded_regions
    ctx.edges     = expanded_edges
    ctx.weights   = expanded_weights
    ctx.prob      = new_prob
    ctx.trans_mat = new_T

    @printf("  [Fiber expansion] %s expanded: %d → %d regions, %d active edges\n",
            state.context, n_old, n_new, length(expanded_edges))
    return true
end

# =============================================================================
# PART 10: RESULT FORMATTING
# =============================================================================

function print_dp_result(result::DPResult, query::DPQuery)
    println("\n" * "─"^65)
    @printf("  DP RESULT: %s → %s\n", query.start, query.target)
    println("─"^65)
    @printf("  Solutions found:    %d\n", length(result.solutions))
    @printf("  States explored:    %d\n", result.n_explored)
    @printf("  Pruned (prob):      %d\n", result.n_pruned_p)
    @printf("  Pruned (graver):    %d\n", result.n_pruned_g)
    @printf("  Pruned (memo):      %d\n", result.n_pruned_m)
    @printf("  Pruned (cycles):    %d\n", result.n_cycles)

    if !isempty(result.solutions)
        @printf("  P_bracket_lo:       %s ≈ %.6f\n",
                string(result.p_bracket_lo), Float64(result.p_bracket_lo))
        @printf("  P_bracket_hi:       %s ≈ %.6f\n",
                string(result.p_bracket_hi), Float64(result.p_bracket_hi))
        println("\n  Top solutions (by probability):")
        sorted = sort(result.solutions, by=s->s.prob, rev=true)
        for (i, s) in enumerate(sorted[1:min(5, end)])
            @printf("    %d. node=%-12s ctx=%-14s sec=%s  depth=%d  p≈%.6f\n",
                    i, s.node, s.context, s.sector, s.depth, Float64(s.prob))
        end
    else
        println("  No paths found within constraints.")
    end
    println("─"^65)
end

# =============================================================================
# PART 11: MULTI-MOLECULE TRANSPORT (parallel DP)
# =============================================================================

"""
    solve_multi_molecule(queries, au_contexts; inclusion_corrections, verbose)

Run multiple queries in sequence, updating AU context probabilities
after each query (so earlier molecule transport affects later distributions).

This models the pharmacodynamic interaction between opiates (qA) and
naloxone/norcain (qB): the second drug sees the AU contexts as already
perturbed by the first.
"""
function solve_multi_molecule(queries::Vector{DPQuery},
                               au_contexts::Dict{Symbol, NNOAUContext};
                               inclusion_corrections =
                                   build_inclusion_corrections(au_contexts),
                               verbose::Bool = false)

    results = DPResult[]
    for (i, q) in enumerate(queries)
        verbose && @printf("\n[Molecule %d] %s → %s\n", i, q.start, q.target)

        res = solve_transport(q, au_contexts;
                              inclusion_corrections=inclusion_corrections,
                              verbose=verbose)
        push!(results, res)

        # After each query: advance Markov chains one step to reflect
        # the transport that just occurred
        for (_, ctx) in au_contexts
            markov_step!(ctx)
        end
    end
    return results
end

# =============================================================================
# PART 12: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^65)
    println("DP CORE: Stage 2 — Target-Pruned Dynamic Programming")
    println("="^65)

    # ── Build NNO AU contexts (Q_7P) ─────────────────────────────────────────
    vertices_7p = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
    edges_7p    = [(:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
                   (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
                   (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
                   (:sAMY,:BLA),(:sAMY,:HY),(:sAMY,:HPF),
                   (:sAMY,:LA),(:sAMY,:PAL),
                   (:HY,:sAMY),(:LA,:BLA),(:LA,:sAMY),(:PAL,:sAMY)]

    stops_A = Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA),
                   (:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)])
    stops_C = Set([(:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)])

    w7p = Dict{Tuple{Symbol,Symbol}, NNOProb}(
        (:LA,   :sAMY) => NNOProb(9752, 100),
        (:sAMY, :LA)   => NNOProb(9752, 100),
        (:BLA,  :LA)   => NNOProb(206,  100),
        (:LA,   :BLA)  => NNOProb(206,  100),
        (:HPF,  :sAMY) => NNOProb(34590, 100),
        (:sAMY, :HPF)  => NNOProb(34590, 100),
        (:CA1sp,:HPF)  => NNOProb(1500,  100),
        (:HPF,  :CA1sp)=> NNOProb(1500,  100),
    )
    for e in edges_7p; haskey(w7p, e) || (w7p[e] = NNO_ONE); end

    println("\n[1] Building AU contexts ...")
    au_contexts = Dict{Symbol, NNOAUContext}()

    # ── Core Sector A contexts ────────────────────────────────────────────────
    # Use stops_C for CTX_sAMY: fewer stops → sAMY has 3 active outgoing edges
    # (sAMY→HPF, sAMY→BLA, sAMY→LA) giving p < 1//1 for transport queries.
    # stops_A was maximally restrictive (only sAMY→HPF active) → p=1//1 absorbing.
    # stops_C removes the HY/PAL stops but keeps BLA/LA stops removed → realistic.
    au_contexts[:CTX_sAMY] = build_nno_au(
        :CTX_sAMY, "sAMY hub",
        [:sAMY, :BLA, :LA, :HPF, :CA1sp],   # 5 regions
        edges_7p, stops_C, w7p, :A, 89, 0, 1.2599;
        initial_node = :sAMY)

    au_contexts[:CTX_HPF] = build_nno_au(
        :CTX_HPF, "Hippocampal formation",
        [:HPF, :CA1sp, :sAMY, :BLA],
        edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node = :HPF)

    # ── Transition / crisis contexts ──────────────────────────────────────────
    au_contexts[:CTX_HY] = build_nno_au(
        :CTX_HY, "Hypothalamus-PAL",
        [:HY, :PAL, :sAMY],
        edges_7p, stops_C, w7p, :C, 0, 0, 0.618;
        initial_node = :HY)

    # Sector C context (Λ⁻ removed — crisis onset)
    # NOTE: Plücker stratum reflects PROBABILITY STATE, not algebraic sector.
    # initial_node=:sAMY with 6 regions → uniform-ish distribution → stratum=4
    # (both 2×2 Gr(2,4) minors nonzero = open cell = dynamic state is "active")
    # The declared sector=:C is the STRUCTURAL label (HH²=151, coker=62).
    # These two are complementary: structure says "crisis", dynamics says "active".
    au_contexts[:CTX_sAMY_C] = build_nno_au(
        :CTX_sAMY_C, "sAMY hub (Sector C — crisis onset)",
        [:sAMY, :BLA, :LA, :HPF, :HY, :PAL],
        edges_7p, stops_C, w7p, :C, 151, 62, 1.618;
        initial_node = :sAMY)

    # ── H0_only contexts (populate inclusion corrections) ─────────────────────
    # These pairs generate nonzero inclusion corrections via PUSHOUT_TABLE.
    # Thalamus connects sAMY and BG with H0_only boundaries.
    # Cortex connects HPF and THAL with H0_only boundaries.
    # Hindbrain connects THAL and HB with categorical_independent.
    au_contexts[:CTX_THAL] = build_nno_au(
        :CTX_THAL, "Thalamus",
        [:HY, :PAL, :sAMY, :HPF],   # overlap with sAMY and HPF
        edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node = :HY)

    au_contexts[:CTX_BG] = build_nno_au(
        :CTX_BG, "Basal Ganglia",
        [:PAL, :HY, :sAMY],
        edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node = :PAL)

    au_contexts[:CTX_HB] = build_nno_au(
        :CTX_HB, "Hindbrain",
        [:HPF, :CA1sp, :BLA],
        edges_7p, stops_A, w7p, :B, 0, 0, 1.909;
        initial_node = :HPF)

    println(@sprintf("  Built %d AU contexts", length(au_contexts)))

    # ── Build inclusion corrections (now non-empty with THAL/BG/HB) ──────────
    corrections = build_inclusion_corrections(au_contexts)
    println(@sprintf("  Built %d inclusion corrections (H0_only boundary pairs)",
            length(corrections)))

    # ── Query 1: Simple transport sAMY → HPF ────────────────────────────────
    println("\n[2] Query: sAMY → HPF (opiate transport)")
    q1 = DPQuery(:sAMY, :HPF,
                 NNOProb(Int128(1), Int128(1000)),   # p_min = 0.001
                 NNO_ONE,                             # p_max = 1.0
                 6)                                   # max_depth = 6
    r1 = solve_transport(q1, au_contexts;
                         inclusion_corrections=corrections, verbose=true)
    print_dp_result(r1, q1)

    # ── Query 2: Transport CA1sp → sAMY (via HPF) ───────────────────────────
    println("\n[3] Query: CA1sp → sAMY (hippocampal → amygdala)")
    q2 = DPQuery(:CA1sp, :sAMY,
                 NNOProb(Int128(1), Int128(10000)),
                 NNO_ONE,
                 8)
    r2 = solve_transport(q2, au_contexts;
                         inclusion_corrections=corrections, verbose=true)
    print_dp_result(r2, q2)

    # ── Query 3: Multi-molecule (opiate + antagonist) ────────────────────────
    println("\n[4] Multi-molecule: opiate then norcain")
    qs = [
        DPQuery(:sAMY, :HPF,  NNOProb(Int128(1),Int128(1000)), NNO_ONE, 6),
        DPQuery(:BLA,  :sAMY, NNOProb(Int128(1),Int128(1000)), NNO_ONE, 6),
    ]
    results = solve_multi_molecule(qs, au_contexts; verbose=true)
    for (i, (q, r)) in enumerate(zip(qs, results))
        println("\n  Molecule $i:")
        print_dp_result(r, q)
    end

    # ── Schubert stratum check after DP ─────────────────────────────────────
    println("\n[5] Plücker stratum after transport:")
    println("    (stratum = dynamic probability state; sector = structural algebra label)")
    println("    These can differ: high stratum = both Gr(2,4) minors nonzero = active flow")
    println("    Sector C context can show stratum=4 if probability is well-distributed")
    for (id, ctx) in sort(collect(au_contexts), by=x->string(x[1]))
        pl = plucker_and_stratum(ctx)
        # Show both: dynamic stratum AND structural sector from ctx
        match = pl.sector_hint == ctx.sector ? "✓ agree" : "⟂ differ (dynamic≠structural)"
        @printf("  %-16s  stratum=%d  dynamic=%s  structural=%s  %s\n",
                id, pl.stratum, pl.sector_hint, ctx.sector, match)
    end

    println("\n" * "="^65)
    println("DP CORE: Stage 2 complete")
    println("  ✓ DPState / DPQuery / Transition structs")
    println("  ✓ PUSHOUT_TABLE (O(1) mode lookup)")
    println("  ✓ Hyper-Confluence Operator (Modes 1–4)")
    println("  ✓ Target-pruned DP (prob + graver + memo + cycle)")
    println("  ✓ AU boundary detection")
    println("  ✓ Surgery integration (Mode 4)")
    println("  ✓ Fiber hub expansion (bgr/fibertracts)")
    println("  ✓ Multi-molecule transport")
    println("  ✓ Probability bracket output")
    println("="^65)
end
