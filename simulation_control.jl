# =============================================================================
# simulation_control.jl
#
# Three runtime capabilities:
#   1. MCMC look-ahead: simulate k steps forward before committing
#   2. SimulationResult: structured output for end users
#   3. SimulationCheckpoint: save/restore/branch at any point
#
# Depends on: nno_au_core.jl, dp_core.jl, qkv_truncation.jl
# =============================================================================

if !@isdefined(NNOProb)
    include(joinpath(@__DIR__, "tool_paths.jl"))
    include(joinpath(@__DIR__, "nno_au_core.jl"))
end
if !@isdefined(DPState)
    include(joinpath(@__DIR__, "dp_core.jl"))
end
if !@isdefined(SheetProbeResult)
    include(joinpath(@__DIR__, "qkv_truncation.jl"))
end

using Printf, Dates

# =============================================================================
# PART 1: SIMULATION CHECKPOINT
# =============================================================================

"""
    SimulationCheckpoint

Complete, serialisable snapshot of simulation state at one instant.
Cheap to create (shallow copy of NNO rational vectors).
Supports branching: multiple independent continuations from one point.
"""
struct SimulationCheckpoint
    id          ::String                              # unique checkpoint ID
    step        ::Int                                 # DP step number
    t           ::Float64                             # simulation time
    au_contexts ::Dict{Symbol, NNOAUContext}          # probability state
    h_current   ::NNOProb                             # toric height
    pushout_snapshot ::Dict{Tuple{Symbol,Symbol}, Symbol}  # restriction maps
    alert_registry   ::Dict{Tuple{Symbol,Symbol}, NamedTuple}  # past crises
    dp_memo     ::Dict{Tuple{Symbol,Symbol,Symbol,Int}, NNOProb}
    gr35_plucker::Vector{Float64}                     # Gr(3,5) Plücker coords
    description ::String                              # human-readable label
end

"""
    save_checkpoint(au_contexts, h, step, t; description) -> SimulationCheckpoint

Snapshot the current simulation state. Deep-copies NNO probability
vectors so the checkpoint is independent of future evolution.
"""
function save_checkpoint(au_contexts ::Dict{Symbol, NNOAUContext},
                          h           ::NNOProb,
                          step        ::Int,
                          t           ::Float64;
                          description ::String = "",
                          dp_memo     ::Dict = Dict{Tuple{Symbol,Symbol,Symbol,Int},NNOProb}(),
                          alert_registry::Dict = Dict{Tuple{Symbol,Symbol},NamedTuple}(),
                          gr35_plucker::Vector{Float64} = Float64[])

    # Deep-copy the NNO probability vectors (cheap — just rational arrays)
    ctx_copy = Dict{Symbol, NNOAUContext}()
    for (id, ctx) in au_contexts
        ctx_new = NNOAUContext(
            ctx.id, ctx.label,
            copy(ctx.regions), copy(ctx.edges),
            copy(ctx.stops), copy(ctx.weights),
            copy(ctx.prob),       # the critical NNO state
            copy(ctx.trans_mat),
            ctx.sector, ctx.hh2, ctx.coker, ctx.rho, ctx.step)
        ctx_copy[id] = ctx_new
    end

    id_str = string(Dates.now())[12:23] * "_step$(step)"

    SimulationCheckpoint(
        id_str, step, t,
        ctx_copy, h,
        copy(PUSHOUT_TABLE),
        copy(alert_registry),
        copy(dp_memo),
        copy(gr35_plucker),
        description)
end

"""
    restore_checkpoint(cp) -> (au_contexts, h, step, t)

Restore simulation from a checkpoint.
Returns deep copies so the checkpoint remains usable for other branches.
"""
function restore_checkpoint(cp::SimulationCheckpoint)
    ctx_restored = Dict{Symbol, NNOAUContext}()
    for (id, ctx) in cp.au_contexts
        ctx_new = NNOAUContext(
            ctx.id, ctx.label,
            copy(ctx.regions), copy(ctx.edges),
            copy(ctx.stops), copy(ctx.weights),
            copy(ctx.prob),
            copy(ctx.trans_mat),
            ctx.sector, ctx.hh2, ctx.coker, ctx.rho, ctx.step)
        ctx_restored[id] = ctx_new
    end
    return ctx_restored, cp.h_current, cp.step, cp.t
end

"""
    branch_checkpoint(cp, new_stops, affected_ctx_ids; description)
    -> SimulationCheckpoint

Create a new branch from cp with modified stop architecture.
Changing stops = changing drug state = changing restriction maps.

new_stops: Dict{Symbol, Set{Tuple{Symbol,Symbol}}}
  maps context id → new stop set for that context
  
affected_ctx_ids: which contexts are affected by the drug change
"""
function branch_checkpoint(cp          ::SimulationCheckpoint,
                             new_stops  ::Dict{Symbol, Set{Tuple{Symbol,Symbol}}},
                             affected_ids::Vector{Symbol};
                             description::String = "branch")::SimulationCheckpoint

    ctx_branched = Dict{Symbol, NNOAUContext}()

    for (id, ctx) in cp.au_contexts
        if id ∈ affected_ids && haskey(new_stops, id)
            # Apply new stop set → rebuild transition matrix
            new_stop_set = new_stops[id]
            new_edges = [(s,t) for (s,t) in ctx.edges
                         if (s,t) ∉ new_stop_set]
            new_T = build_transition_matrix(ctx.regions, new_edges, ctx.weights)

            # Redistribute probability from newly-stopped nodes
            new_prob = copy(ctx.prob)
            n = length(ctx.regions)
            isolated_mass = NNO_ZERO
            for i in 1:n
                # Check if node i is now isolated (all outgoing edges stopped)
                has_out = any(new_T[j,i] != NNO_ZERO for j in 1:n if j != i)
                if !has_out
                    isolated_mass += new_prob[i]
                    new_prob[i] = NNO_ZERO
                end
            end
            # Redistribute to non-isolated nodes
            active_idx = [i for i in 1:n if new_prob[i] > NNO_ZERO ||
                          any(new_T[j,i] != NNO_ZERO for j in 1:n if j!=i)]
            if !isempty(active_idx) && isolated_mass > NNO_ZERO
                share = isolated_mass // NNOProb(Int128(length(active_idx)), 1)
                for i in active_idx; new_prob[i] += share; end
            end
            nno_check(new_prob; label="branch $id")

            ctx_new = NNOAUContext(id, ctx.label * " [branched]",
                copy(ctx.regions), new_edges, new_stop_set,
                copy(ctx.weights), new_prob, new_T,
                ctx.sector, ctx.hh2, ctx.coker, ctx.rho, ctx.step)
            ctx_branched[id] = ctx_new
        else
            # Unchanged context — deep copy
            ctx_branched[id] = NNOAUContext(
                ctx.id, ctx.label,
                copy(ctx.regions), copy(ctx.edges), copy(ctx.stops),
                copy(ctx.weights), copy(ctx.prob), copy(ctx.trans_mat),
                ctx.sector, ctx.hh2, ctx.coker, ctx.rho, ctx.step)
        end
    end

    id_str = cp.id * "_branch_" * string(Dates.now())[12:23]
    SimulationCheckpoint(
        id_str, cp.step, cp.t,
        ctx_branched, cp.h_current,
        copy(cp.pushout_snapshot),
        copy(cp.alert_registry),
        Dict{Tuple{Symbol,Symbol,Symbol,Int},NNOProb}(),  # clear memo for new maps
        copy(cp.gr35_plucker),
        description)
end

# =============================================================================
# PART 2: MCMC LOOK-AHEAD
# =============================================================================

"""
    LookAheadResult

Result of a k-step look-ahead simulation from a candidate path.
"""
struct LookAheadResult
    path_first_step ::Symbol          # first hop of this candidate
    k_steps         ::Int
    final_prob      ::Dict{Symbol, Float64}   # probability at each node after k steps
    risk_trajectory ::Vector{Float64}          # risk gradient at each step
    tee_sign_flips  ::Vector{Int}              # steps where T_eee changed sign
    crisis_step     ::Union{Int,Nothing}       # first step where risk > 0.8
    bracket_lo      ::Float64
    bracket_hi      ::Float64
    recommended     ::Bool                     # should we take this path?
end

"""
    mcmc_lookahead(ctx, candidate_next_nodes, target, k_steps, basis, weights, h)
    -> Vector{LookAheadResult}

For each candidate next node from the current position,
simulate k_steps forward and return probability trajectories.

Does NOT modify ctx (uses deep copies).
The Fisher/Amari risk trajectory is computed at each step.
"""
function mcmc_lookahead(ctx         ::NNOAUContext,
                         candidates  ::Vector{Symbol},
                         target      ::Symbol,
                         k_steps     ::Int,
                         basis       ::Vector{Vector{Int}},
                         weights     ::Dict{Tuple{Symbol,Symbol}, NNOProb},
                         h           ::Float64;
                         target_ctx  ::Union{NNOAUContext, Nothing} = nothing)

    results = LookAheadResult[]

    for candidate in candidates
        # Deep copy — simulation does not affect real state
        ctx_sim = NNOAUContext(
            ctx.id, ctx.label,
            copy(ctx.regions), copy(ctx.edges), copy(ctx.stops),
            copy(ctx.weights), copy(ctx.prob), copy(ctx.trans_mat),
            ctx.sector, ctx.hh2, ctx.coker, ctx.rho, ctx.step)

        # Set initial distribution to δ_candidate (we are AT this node)
        candidate_idx = findfirst(==(candidate), ctx_sim.regions)
        if candidate_idx === nothing
            continue  # candidate not in this context
        end
        n = length(ctx_sim.regions)
        ctx_sim.prob = fill(NNO_ZERO, n)
        ctx_sim.prob[candidate_idx] = NNO_ONE

        risk_traj = Float64[]
        tee_flips  = Int[]
        crisis_step = nothing
        prev_tee_sign = 0

        for step in 1:k_steps
            markov_step!(ctx_sim)

            # Compute risk at this simulated step
            if target_ctx !== nothing
                flux = Float64(boundary_flux(ctx_sim, target_ctx))
            else
                # Proxy: probability at target node
                tgt_idx = findfirst(==(target), ctx_sim.regions)
                flux = tgt_idx !== nothing ? Float64(ctx_sim.prob[tgt_idx]) : 0.0
            end

            # Sheet B proxy: active circuits above h
            dim_h1, _ = probe_sheet_b(ctx_sim, basis, weights, h, 62)

            # Sheet C: max circuit length
            max_len, has_m7 = probe_sheet_c(ctx_sim, basis, weights, h)

            risk = 0.3 * flux + 0.5 * dim_h1 + 0.2 * (has_m7 ? 1.0 : 0.0)
            push!(risk_traj, risk)

            # Amari-Chentsov sign proxy: sign of (risk_new - risk_old)
            if length(risk_traj) >= 2
                tee_sign = sign(risk_traj[end] - risk_traj[end-1])
                if tee_sign != prev_tee_sign && prev_tee_sign != 0
                    push!(tee_flips, step)
                end
                prev_tee_sign = tee_sign
            end

            if risk > 0.8 && crisis_step === nothing
                crisis_step = step
            end
        end

        # Final probability distribution
        final_prob = Dict(ctx_sim.regions[i] => Float64(ctx_sim.prob[i])
                          for i in 1:n)

        # Probability bracket for this path
        target_p = get(final_prob, target, 0.0)
        bracket_lo = minimum(values(final_prob))
        bracket_hi = maximum(values(final_prob))

        # Recommend this path if: target probability is high AND no early crisis
        recommended = target_p > 0.1 && crisis_step === nothing

        push!(results, LookAheadResult(
            candidate, k_steps, final_prob, risk_traj,
            tee_flips, crisis_step,
            bracket_lo, bracket_hi, recommended))
    end

    # Sort by target probability descending
    target_idx_sort = [get(r.final_prob, target, 0.0) for r in results]
    perm = sortperm(target_idx_sort, rev=true)
    return results[perm]
end

# =============================================================================
# PART 3: SIMULATION RESULT (END USER OUTPUT)
# =============================================================================

"""
    SimulationResult

Structured output for the end user: probability bracket, top pathways,
risk trajectory, GPS sector, and look-ahead summary.
"""
struct SimulationResult
    query           ::DPQuery
    dp_result       ::DPResult
    lookahead       ::Vector{LookAheadResult}
    current_sector  ::Symbol
    current_stratum ::Int
    risk_at_query   ::Float64
    bracket_lo      ::NNOProb
    bracket_hi      ::NNOProb
    bracket_width   ::Float64    # hi - lo in Float64
    checkpoint      ::SimulationCheckpoint
    timestamp       ::String
end

"""
    run_simulation(query, au_contexts, h; k_lookahead, verbose)
    -> SimulationResult

Complete simulation run: DP + look-ahead + sector detection + checkpoint.
"""
function run_simulation(query       ::DPQuery,
                         au_contexts ::Dict{Symbol, NNOAUContext},
                         h           ::NNOProb;
                         basis       ::Vector{Vector{Int}} = Vector{Vector{Int}}(),
                         weights     ::Dict{Tuple{Symbol,Symbol},NNOProb} =
                             Dict{Tuple{Symbol,Symbol},NNOProb}(),
                         k_lookahead ::Int = 8,
                         verbose     ::Bool = true)::SimulationResult

    # ── Step 1: Run DP ────────────────────────────────────────────────────
    dp_result = solve_transport(query, au_contexts; verbose=verbose)

    # ── Step 2: Look-ahead from starting node ────────────────────────────
    start_ctx_id = find_context(query.start, au_contexts)
    start_ctx    = au_contexts[start_ctx_id]
    target_ctx   = get(au_contexts, find_context(query.target, au_contexts), nothing)

    # Candidate next nodes from query.start
    candidates = Symbol[]
    start_idx  = findfirst(==(query.start), start_ctx.regions)
    if start_idx !== nothing
        for i in 1:length(start_ctx.regions)
            start_ctx.trans_mat[i, start_idx] != NNO_ZERO &&
            start_ctx.regions[i] != query.start &&
                push!(candidates, start_ctx.regions[i])
        end
    end

    lookahead = if !isempty(candidates)
        mcmc_lookahead(start_ctx, candidates[1:min(3,end)],
                       query.target, k_lookahead, basis, weights,
                       Float64(h); target_ctx=target_ctx)
    else
        LookAheadResult[]
    end

    # ── Step 3: GPS sector from current probability state ─────────────────
    pl = plucker_and_stratum(start_ctx)
    current_sector  = pl.sector_hint
    current_stratum = pl.stratum

    # ── Step 4: Current risk from sheet probes ────────────────────────────
    risk_now = if target_ctx !== nothing && !isempty(basis)
        probe = probe_all_sheets(start_ctx, target_ctx, basis, weights,
                                  Float64(h))
        probe.risk_gradient
    else
        Float64(boundary_flux(start_ctx, isnothing(target_ctx) ?
                               start_ctx : target_ctx))
    end

    # ── Step 5: Save checkpoint ───────────────────────────────────────────
    cp = save_checkpoint(au_contexts, h, 0, 0.0;
                         description="Query: $(query.start)→$(query.target)")

    SimulationResult(
        query, dp_result, lookahead,
        current_sector, current_stratum, risk_now,
        dp_result.p_bracket_lo, dp_result.p_bracket_hi,
        isempty(dp_result.solutions) ? 0.0 :
            Float64(dp_result.p_bracket_hi) - Float64(dp_result.p_bracket_lo),
        cp, string(Dates.now())[1:19])
end

"""
    print_simulation_result(result)

End-user display of a simulation result.
"""
function print_simulation_result(r::SimulationResult)
    println("\n" * "┌" * "─"^58 * "┐")
    println("│ TRANSPORT PREDICTION" * " "^37 * "│")
    println("│" * "─"^58 * "│")
    @printf("│  Source: %-12s  Target: %-12s           │\n",
            r.query.start, r.query.target)
    println("│" * "─"^58 * "│")
    @printf("│  Probability bracket:  [%.6f, %.6f]          │\n",
            Float64(r.bracket_lo), Float64(r.bracket_hi))
    @printf("│  Bracket width:         %.6f                    │\n",
            r.bracket_width)
    println("│" * "─"^58 * "│")
    println("│  Top pathways:                                           │")
    sorted = sort(r.dp_result.solutions, by=s->s.prob, rev=true)
    for (i, s) in enumerate(sorted[1:min(5,end)])
        warn = s.prob > NNOProb(8,10) ? " ⚠" : ""
        @printf("│   %d. %-10s depth=%-2d p≈%.6f%s%s│\n",
                i, s.node, s.depth, Float64(s.prob), warn,
                " "^max(0, 18-length(warn)))
    end
    println("│" * "─"^58 * "│")
    @printf("│  Crisis risk: %.4f   GPS sector: %s   Stratum: %d      │\n",
            r.risk_at_query, r.current_sector, r.current_stratum)
    println("│" * "─"^58 * "│")

    if !isempty(r.lookahead)
        println("│  Look-ahead ($(r.lookahead[1].k_steps) steps):                                    │")
        for la in r.lookahead[1:min(3,end)]
            target_p = get(la.final_prob, r.query.target, 0.0)
            crisis_str = la.crisis_step !== nothing ?
                "⚠ crisis@step$(la.crisis_step)" : "stable"
            rec = la.recommended ? "✓" : "✗"
            @printf("│   %s %-8s  p_target=%.4f  %s  %s      │\n",
                    rec, la.path_first_step, target_p,
                    crisis_str,
                    isempty(la.tee_sign_flips) ? "" :
                    "flip@$(la.tee_sign_flips[1])")
        end
        # Risk trajectory for best path
        if !isempty(r.lookahead[1].risk_trajectory)
            best = r.lookahead[1]
            print("│  Risk traj: ")
            for (i, rk) in enumerate(best.risk_trajectory)
                bar = rk > 0.8 ? "█" : rk > 0.5 ? "▓" : rk > 0.3 ? "▒" : "░"
                print(bar)
            end
            println(" " * " "^max(0, 44-length(best.risk_trajectory)) * "│")
        end
    end

    println("│" * "─"^58 * "│")
    @printf("│  Checkpoint: %-20s                   │\n", r.checkpoint.id)
    @printf("│  Timestamp:  %-20s                   │\n", r.timestamp)
    println("└" * "─"^58 * "┘")
end

# =============================================================================
# PART 4: COUNTERFACTUAL COMPARISON
# =============================================================================

"""
    compare_branches(cp, branch_specs, query; k_steps, verbose)
    -> Vector{SimulationResult}

Run multiple branches from a checkpoint and compare outcomes.

branch_specs: Vector of (description, new_stops_dict) pairs
  Each entry creates one counterfactual: "what if we change Λ this way?"

Example:
  compare_branches(cp, [
    ("Drug A only",    Dict()),
    ("Add Drug B",     Dict(:CTX_sAMY => Set([(:BLA,:sAMY)]))),
    ("Add Drug C",     Dict(:CTX_HPF  => Set([(:HPF,:sAMY)]))),
  ], query)
"""
function compare_branches(cp           ::SimulationCheckpoint,
                            branch_specs ::Vector,   # Vector of (String, Dict) pairs
                            query        ::DPQuery;
                            basis        ::Vector{Vector{Int}} = Vector{Vector{Int}}(),
                            weights      ::Dict{Tuple{Symbol,Symbol},NNOProb} =
                                Dict{Tuple{Symbol,Symbol},NNOProb}(),
                            k_steps      ::Int = 8,
                            verbose      ::Bool = false)

    results = SimulationResult[]

    for (desc, new_stops) in branch_specs
        # Restore and branch
        branched_cp = if isempty(new_stops)
            cp  # baseline: no change
        else
            affected = collect(keys(new_stops))
            branch_checkpoint(cp,
                Dict(k => Set(v) for (k,v) in new_stops),
                affected; description=desc)
        end

        ctx_branch, h_branch, _, _ = restore_checkpoint(branched_cp)

        r = run_simulation(query, ctx_branch, h_branch;
                            basis=basis, weights=weights,
                            k_lookahead=k_steps, verbose=verbose)
        push!(results, r)
        verbose && @printf("  Branch '%s': bracket [%.4f, %.4f]\n",
                desc, Float64(r.bracket_lo), Float64(r.bracket_hi))
    end

    return results
end

"""
    print_branch_comparison(results, branch_names)

Compare outcomes across counterfactual branches.
"""
function print_branch_comparison(results::Vector{SimulationResult},
                                  names::Vector{String})
    println("\n" * "═"^65)
    println("COUNTERFACTUAL COMPARISON")
    println("═"^65)
    @printf("  %-20s  %-12s  %-12s  %-8s  %s\n",
            "Branch", "P_bracket_lo", "P_bracket_hi", "Width", "Risk")
    println("  " * "─"^62)
    for (r, name) in zip(results, names)
        @printf("  %-20s  %-12.6f  %-12.6f  %-8.4f  %.4f\n",
                name,
                Float64(r.bracket_lo), Float64(r.bracket_hi),
                r.bracket_width, r.risk_at_query)
    end
    println("═"^65)
    # Best branch
    best_idx = argmax([Float64(r.bracket_hi) for r in results])
    println("  Best outcome: '$(names[best_idx])'")
    println("  (highest P_max = $(round(Float64(results[best_idx].bracket_hi), digits=6)))")
    println("═"^65)
end


# =============================================================================
# PART 6: MARKOV CHAIN DISPLAY
# =============================================================================

"""
    print_markov_chain(ctx; top_n=5, label="")

Show the current Markov chain probability distribution for an AU context.
Displays the top_n nodes by probability mass with a bar chart.
"""
function print_markov_chain(ctx::NNOAUContext; top_n::Int=5, label::String="")
    n = length(ctx.regions)
    probs_f64 = [Float64(ctx.prob[i]) for i in 1:n]
    order = sortperm(probs_f64, rev=true)

    title = isempty(label) ? string(ctx.id) : label
    @printf("  %-18s [step %d, sector %s]\n", title, ctx.step, ctx.sector)

    shown = 0
    for idx in order
        p = probs_f64[idx]
        p < 1e-6 && break
        shown >= top_n && break
        bar_len = round(Int, p * 20)
        bar = "█"^bar_len * "░"^(20 - bar_len)
        @printf("    %-10s %s %.4f\n", ctx.regions[idx], bar, p)
        shown += 1
    end
    shown == 0 && println("    (all mass below 1e-6)")
end

"""
    print_markov_evolution(au_contexts, n_steps; target_ctx=nothing)

Advance all AU contexts n_steps and print the chain state after each step.
Shows how probability mass flows through the network over time.
"""
function print_markov_evolution(au_contexts::Dict{Symbol,NNOAUContext},
                                 n_steps::Int;
                                 show_contexts::Vector{Symbol} = Symbol[],
                                 label::String = "")
    ctx_ids = isempty(show_contexts) ? collect(keys(au_contexts)) : show_contexts

    println("  " * "─"^55)
    isempty(label) || println("  $label")
    println("  " * "─"^55)

    for step in 1:n_steps
        # Advance all contexts
        for (_, ctx) in au_contexts
            markov_step!(ctx)
        end

        println("  Step $step:")
        for id in sort(ctx_ids, by=string)
            ctx = au_contexts[id]
            n = length(ctx.regions)
            # Find top-2 nodes
            pf = [Float64(ctx.prob[i]) for i in 1:n]
            top2 = partialsortperm(pf, 1:min(2,n), rev=true)
            parts = [@sprintf("%s=%.3f", ctx.regions[i], pf[i]) for i in top2
                     if pf[i] > 1e-4]
            @printf("    %-14s  %s\n", id, join(parts, "  "))
        end
    end
    println("  " * "─"^55)
end

# =============================================================================
# PART 5: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^65)
    println("SIMULATION CONTROL: Look-ahead + Checkpoints + Branches")
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
        (:LA,:sAMY)    => NNOProb(9752,100),
        (:sAMY,:LA)    => NNOProb(9752,100),
        (:BLA,:LA)     => NNOProb(206,100),
        (:HPF,:sAMY)   => NNOProb(34590,100),
        (:sAMY,:HPF)   => NNOProb(34590,100),
        (:CA1sp,:HPF)  => NNOProb(1500,100),
    )
    for e in edges_7p; haskey(w7p,e) || (w7p[e]=NNO_ONE); end

    # Build demo basis
    au_contexts = Dict{Symbol,NNOAUContext}()
    au_contexts[:CTX_sAMY] = build_nno_au(:CTX_sAMY, "sAMY",
        [:sAMY,:BLA,:LA,:HPF,:CA1sp], edges_7p, stops_A, w7p, :A, 89,0,1.26;
        initial_node=:sAMY)
    au_contexts[:CTX_HPF] = build_nno_au(:CTX_HPF, "HPF",
        [:HPF,:CA1sp,:sAMY,:BLA], edges_7p, stops_A, w7p, :A, 89,0,1.26;
        initial_node=:HPF)

    n_e = length(au_contexts[:CTX_sAMY].edges)
    basis_demo = [[i==j ? 1 : (i==mod1(j+1,n_e) ? -1 : 0) for i in 1:n_e]
                  for j in 1:min(n_e,15)]
    n_e >= 7 && push!(basis_demo,
        [k<=7 ? (isodd(k) ? 1 : -1) : 0 for k in 1:n_e])

    h_val = NNOProb(Int128(10), Int128(1))
    query = DPQuery(:CA1sp, :sAMY,
                    NNOProb(Int128(1),Int128(10000)), NNO_ONE, 8)

    println("\n[1] Full simulation with look-ahead:")
    result = run_simulation(query, au_contexts, h_val;
                             basis=basis_demo, weights=w7p,
                             k_lookahead=8, verbose=false)
    print_simulation_result(result)

    println("\n[2] Markov chain evolution (5 steps):")
    # Show the chains BEFORE and AFTER stepping
    println("  Initial state (δ_CA1sp):")
    for (id, ctx) in sort(collect(au_contexts), by=x->string(x[1]))
        any(p > NNOProb(1,Int128(1000)) for p in ctx.prob) || continue
        print_markov_chain(ctx; top_n=3)
    end

    println("\n  Evolving 5 steps:")
    # Deep-copy so the checkpoint remains clean
    au_evolving = Dict(id => NNOAUContext(
        ctx.id, ctx.label, copy(ctx.regions), copy(ctx.edges),
        copy(ctx.stops), copy(ctx.weights), copy(ctx.prob),
        copy(ctx.trans_mat), ctx.sector, ctx.hh2, ctx.coker,
        ctx.rho, ctx.step) for (id,ctx) in au_contexts)
    print_markov_evolution(au_evolving, 5;
        show_contexts=[:CTX_sAMY, :CTX_HPF],
        label="Opiate transport: CA1sp → sAMY → HPF")

    println("\n  Final state after 5 steps:")
    for (id, ctx) in sort(collect(au_evolving), by=x->string(x[1]))
        id ∈ [:CTX_sAMY, :CTX_HPF] || continue
        print_markov_chain(ctx; top_n=4)
    end

    println("\n[3] Save checkpoint and branch:")
    cp = result.checkpoint
    @printf("  Checkpoint saved: %s\n", cp.id)
    @printf("  AU contexts: %d\n", length(cp.au_contexts))
    @printf("  h = %s\n", string(cp.h_current))

    println("\n[4] Counterfactual: Drug A alone vs Drug A+B vs Drug A+C")
    # branch_specs: (description, Dict mapping ctx_id → new edges to stop)
    branch_specs = Tuple{String,Dict}[
        ("Drug A only",
         Dict{Symbol,Vector{Tuple{Symbol,Symbol}}}()),
        ("Add Drug B (block BLA→HPF)",
         Dict{Symbol,Vector{Tuple{Symbol,Symbol}}}(:CTX_sAMY => [(:BLA,:HPF)])),
        ("Add Drug C (block HPF→sAMY)",
         Dict{Symbol,Vector{Tuple{Symbol,Symbol}}}(:CTX_HPF  => [(:HPF,:sAMY)])),
    ]
    branch_results = compare_branches(cp, branch_specs, query;
                                       basis=basis_demo, weights=w7p,
                                       k_steps=8, verbose=true)
    print_branch_comparison(branch_results,
                             [b[1] for b in branch_specs])

    println("\n[5] Look-ahead trajectories from CA1sp:")
    ctx_start = au_contexts[:CTX_sAMY]
    candidates = [:HPF, :BLA, :sAMY]  # possible first hops
    la_results = mcmc_lookahead(ctx_start, candidates, :sAMY,
                                  8, basis_demo, w7p, 10.0;
                                  target_ctx=au_contexts[:CTX_HPF])
    for la in la_results
        target_p = get(la.final_prob, :sAMY, 0.0)
        @printf("  First hop: %-8s  p(sAMY)=%.4f  risk_final=%.3f  %s\n",
                la.path_first_step, target_p,
                isempty(la.risk_trajectory) ? 0.0 : last(la.risk_trajectory),
                la.recommended ? "✓ RECOMMENDED" : "")
        print("    Risk: ")
        for rk in la.risk_trajectory
            print(rk > 0.7 ? "█" : rk > 0.5 ? "▓" : rk > 0.3 ? "▒" : "░")
        end
        println()
    end

    println("\n" * "="^65)
    println("Simulation control complete.")
    println("  save_checkpoint()       → snapshot NNO state")
    println("  restore_checkpoint()    → load any past state")
    println("  branch_checkpoint()     → counterfactual (new Λ)")
    println("  mcmc_lookahead()        → 8-step MCMC planning")
    println("  run_simulation()        → complete query + output")
    println("  compare_branches()      → drug A vs A+B vs A+C")
    println("="^65)
end
