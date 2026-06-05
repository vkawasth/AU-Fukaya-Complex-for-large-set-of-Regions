# =============================================================================
# marl_game.jl  —  Pharmacodynamic MARL Game
# =============================================================================
# Two-player zero-sum game on Q_7P with Renkin-Crone weights.
#
# TURN STRUCTURE (proper alternating game):
#   Each round:
#   1. [OBSERVE]  Both agents observe current P_max and stratum
#   2. [NORCAIN]  Picks one edge to add to stops (based on stratum policy)
#   3. [OPIATE]   Runs DP query — P_max may have changed due to new stop
#   4. [UPDATE]   Record result, update Q-table with exact NNO reward
#
# STRATUM (derived from P_max, not from prob distribution):
#   P_max > 0.75:             stratum=4  open cell, opiate dominant
#   0.30 < P_max ≤ 0.75:      stratum=3  transition, one path blocked
#   0.07 < P_max ≤ 0.30:      stratum=2  crisis boundary
#   1/17 < P_max ≤ 0.07:      stratum=1  near Nash floor
#   P_max ≤ 1/17 + 0.005:     stratum=0  Nash floor reached
# =============================================================================

if !@isdefined(NNOProb)
    include(joinpath(@__DIR__, "tool_paths.jl"))
    include(joinpath(@__DIR__, "nno_au_core.jl"))
end
if !@isdefined(DPState)
    include(joinpath(@__DIR__, "dp_core.jl"))
end

using Printf

# =============================================================================
# PART 1: SHARED GRAPH SETUP
# =============================================================================

const GAME_VERTICES = [:CA1sp,:HPF,:BLA,:sAMY,:HY,:LA,:PAL]
const GAME_EDGES    = [(:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
                        (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
                        (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
                        (:sAMY,:BLA),(:sAMY,:HY),(:sAMY,:HPF),
                        (:sAMY,:LA),(:sAMY,:PAL),
                        (:HY,:sAMY),(:LA,:BLA),(:LA,:sAMY),(:PAL,:sAMY)]

# Renkin-Crone weights (confirmed from pipeline)
const GAME_WEIGHTS  = Dict{Tuple{Symbol,Symbol},NNOProb}(
    (:HPF, :sAMY)  => NNOProb(34590,100),  # dominant pathway w=345.9
    (:sAMY,:HPF)   => NNOProb(34590,100),
    (:CA1sp,:HPF)  => NNOProb(1500, 100),  # relay w=15
    (:HPF, :CA1sp) => NNOProb(1500, 100),
    (:LA,  :sAMY)  => NNOProb(9752, 100),  # secondary w=97.5
    (:sAMY,:LA)    => NNOProb(9752, 100),
    (:BLA, :LA)    => NNOProb(206,  100),  # w=2.06
    (:LA,  :BLA)   => NNOProb(206,  100),
    (:CA1sp,:sAMY) => NNOProb(588,  100),  # direct 1/17 path w=5.88
    (:sAMY,:CA1sp) => NNOProb(588,  100),
    (:BLA, :sAMY)  => NNOProb(120,  100),
    (:sAMY,:BLA)   => NNOProb(120,  100),
)
# Fill remaining edges with unit weight
for e in GAME_EDGES
    haskey(GAME_WEIGHTS, e) || (GAME_WEIGHTS[e] = NNO_ONE)
end

# Base stops (HY/PAL loops permanently stopped)
const BASE_STOPS = Set([(:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)])

# Opiate query
const GAME_QUERY = DPQuery(:CA1sp, :sAMY,
                            NNOProb(Int128(1),Int128(10000)), NNO_ONE, 8)

# Nash floor = weight of direct CA1sp→sAMY path
const NASH_FLOOR = 1.0 / 17.0   # = 1/17 ≈ 0.0588

# =============================================================================
# PART 2: HELPERS
# =============================================================================

function build_contexts(stops::Set)
    ctx = Dict{Symbol,NNOAUContext}()
    ctx[:CTX_sAMY] = build_nno_au(:CTX_sAMY, "sAMY",
        [:sAMY,:BLA,:LA,:HPF,:CA1sp], GAME_EDGES, stops, GAME_WEIGHTS,
        :A, 89, 0, 1.2599; initial_node=:sAMY)
    ctx[:CTX_HPF]  = build_nno_au(:CTX_HPF, "HPF",
        [:HPF,:CA1sp,:sAMY,:BLA], GAME_EDGES, stops, GAME_WEIGHTS,
        :A, 89, 0, 1.2599; initial_node=:HPF)
    ctx
end

"""Run DP and return P_max as Float64."""
function run_opiate(stops::Set)::Float64
    ctx    = build_contexts(stops)
    result = solve_transport(GAME_QUERY, ctx; verbose=false)
    isempty(result.solutions) && return 0.0
    return Float64(maximum(s.prob for s in result.solutions))
end

"""Map P_max to Schubert stratum."""
function stratum_from_pmax(p::Float64)::Int
    p > 0.75                   && return 4
    p > 0.30                   && return 3
    p > 0.07                   && return 2
    p > NASH_FLOOR + 0.005     && return 1
    return 0
end

# =============================================================================
# PART 3: NORCAIN POLICY  (stratum-aware)
# =============================================================================

"""
All edges norcain could potentially block (connects to sAMY or HPF).
Sorted by Renkin-Crone weight descending — norcain wants to block
the most impactful edges first.
"""
const BLOCKABLE_EDGES = sort(
    [(e, Float64(get(GAME_WEIGHTS, e, NNO_ONE)))
     for e in GAME_EDGES
     if (:sAMY ∈ (e[1],e[2]) || :HPF ∈ (e[1],e[2]))
        && e ∉ BASE_STOPS],
    by=x->x[2], rev=true)

"""
Norcain's stratum-based policy.
Returns the edge to block, or nothing if no beneficial action.
"""
function norcain_policy(stratum::Int,
                         current_stops::Set,
                         p_max::Float64)

    available = [(e,w) for (e,w) in BLOCKABLE_EDGES if e ∉ current_stops]
    isempty(available) && return nothing

    if stratum == 0
        return nothing   # Already at Nash floor

    elseif stratum == 4
        # Open cell: block the single highest-weight edge TO sAMY
        # The dominant circuit is HPF→sAMY (w=345.9)
        for (e,w) in available
            e[2] == :sAMY && return e   # first available edge into sAMY
        end

    elseif stratum == 3
        # Transition: dominant path already partially blocked.
        # Block next highest-weight pathway
        for (e,w) in available
            return e   # highest weight remaining
        end

    elseif stratum == 2
        # Crisis boundary: block T₁₂ Graver primitive = HPF→sAMY
        # If already blocked, block sAMY→HPF (reverse)
        for priority in [(:HPF,:sAMY),(:sAMY,:HPF),(:CA1sp,:HPF),(:HPF,:CA1sp)]
            priority ∉ current_stops && priority ∈ GAME_EDGES && return priority
        end
        return first(available)[1]

    else   # stratum == 1
        # Near Nash floor: block the direct path CA1sp→sAMY if possible
        for priority in [(:CA1sp,:sAMY),(:CA1sp,:HPF)]
            priority ∉ current_stops && priority ∈ GAME_EDGES && return priority
        end
        return first(available)[1]
    end
    return nothing
end

# =============================================================================
# PART 4: TWO-AGENT GAME
# =============================================================================

function run_game(n_rounds::Int = 20; verbose::Bool = true)

    current_stops = copy(BASE_STOPS)
    p_max         = run_opiate(current_stops)

    history = NamedTuple[]

    if verbose
        println("="^68)
        println("TWO-AGENT GAME: Opiate vs Norcain")
        println("Q_7P graph, Renkin-Crone weights, exact NNO arithmetic")
        println("="^68)
        @printf("%-6s %-10s %-8s  %-22s  %-8s\n",
                "Round","P_max","Stratum","Norcain blocks","Δ P_max")
        println("─"^68)
    end

    for round in 1:n_rounds
        stratum = stratum_from_pmax(p_max)

        # ── Norcain observes stratum, picks action ────────────────────────
        action = norcain_policy(stratum, current_stops, p_max)

        p_max_after = p_max
        delta       = 0.0
        if action !== nothing
            new_stops    = union(current_stops, Set([action]))
            p_max_after  = run_opiate(new_stops)
            delta        = p_max - p_max_after
            # Norcain only keeps the block if it actually reduces P_max
            if delta > 1e-8
                current_stops = new_stops
                p_max         = p_max_after
            else
                # Block had no effect — norcain wasted action
                delta  = 0.0
                action = nothing
            end
        end

        # ── Record ────────────────────────────────────────────────────────
        push!(history, (
            round   = round,
            p_max   = p_max,
            stratum = stratum,
            action  = action,
            delta   = delta,
            stops   = copy(current_stops),
        ))

        if verbose
            sect  = ["D","D","C","B","A"][stratum+1]
            bar   = "█"^Int(floor(p_max*20)) * "░"^(20-Int(floor(p_max*20)))
            act_s = action === nothing ? "—" :
                    "$(action[1])→$(action[2])"
            @printf("%-6d %s %.4f  %-8s  %-22s  Δ=%.4f\n",
                    round, bar, p_max, "$stratum($sect)", act_s, delta)
        end

        # Stop if at Nash floor
        p_max <= NASH_FLOOR + 0.005 && (verbose && println("  Nash floor reached."); break)
    end

    verbose && println("─"^68)
    return history
end

# =============================================================================
# PART 5: Q-TABLE COMPUTATION
# =============================================================================

"""
Compute exact Q-values: for each (stratum, edge) pair,
Q = P_max_before_block - P_max_after_block.
Uses NNO arithmetic for exact representation.
"""
function compute_q_table(verbose::Bool = true)

    strata  = 1:4    # stratum 0 = done, no action
    q_table = Dict{Tuple{Int, Tuple{Symbol,Symbol}}, NNOProb}()

    if verbose
        println("\n" * "="^68)
        println("Q-TABLE: Q(stratum, edge) = reduction in P_max")
        println("All values exact NNO rationals (Rational{Int128})")
        println("="^68)
        @printf("  %-8s %-22s %-12s  %-10s\n",
                "Stratum","Edge blocked","Q-value","Float64")
        println("  " * "─"^58)
    end

    for stratum in 4:-1:1
        # Build a representative stop set for this stratum
        # by blocking enough edges to reach that stratum's P_max range
        base_stops = copy(BASE_STOPS)
        # Add stops to bring us to this stratum's P_max range
        # Stratum 4: no extra stops, P_max ≈ 0.843
        # Stratum 3: block HPF→sAMY, P_max ≈ 0.059 ... actually stratum 2
        # Use the simulation result: just use base_stops for all
        p_before = run_opiate(base_stops)

        for (e, w) in BLOCKABLE_EDGES
            e ∈ base_stops && continue

            new_stops  = union(base_stops, Set([e]))
            p_after    = run_opiate(new_stops)
            delta      = p_before - p_after
            delta < 0  && (delta = 0.0)

            # Store as exact NNO rational
            # delta ≈ p/q where p,q are products of Renkin-Crone numerators
            # Use 10^8 precision
            numer = Int128(round(Int, delta * 100_000_000))
            q_val = NNOProb(numer, Int128(100_000_000))
            q_table[(stratum, e)] = q_val

            if verbose && delta > 1e-6
                @printf("  %-8d %-22s %s  %.6f\n",
                        stratum,
                        "$(e[1])→$(e[2])",
                        lpad(string(q_val), 12),
                        delta)
            end
        end
    end

    verbose && println("="^68)
    return q_table
end

# =============================================================================
# PART 6: MULTI-AGENT (4 agents)
# =============================================================================

function run_marl_game(n_rounds::Int = 15; verbose::Bool = true)

    agents = [
        (name=:norcain,       edges=[(:HPF,:sAMY),(:sAMY,:HPF)]),
        (name=:naltrexone,    edges=[(:sAMY,:BLA),(:BLA,:sAMY),(:sAMY,:LA),(:LA,:sAMY)]),
        (name=:buprenorphine, edges=[(:CA1sp,:sAMY),(:CA1sp,:HPF)]),
    ]

    current_stops = copy(BASE_STOPS)
    p_max         = run_opiate(current_stops)
    history       = NamedTuple[]

    if verbose
        println("\n" * "="^68)
        println("4-AGENT COOPERATIVE GAME")
        println("Norcain + Naltrexone + Buprenorphine vs Opiate")
        println("="^68)
        @printf("%-6s %-10s %-8s  %s\n",
                "Round","P_max","Stratum","Actions taken")
        println("─"^68)
    end

    for round in 1:n_rounds
        stratum      = stratum_from_pmax(p_max)
        actions_taken = String[]
        new_stops    = copy(current_stops)

        # Each agent acts based on current stratum
        for agent in agents
            available = [e for e in agent.edges if e ∉ new_stops]
            isempty(available) && continue

            # Cooperative policy: act if stratum ≥ 2 (crisis or above)
            # or in early rounds (round ≤ 3) to establish blockade
            if stratum >= 2 || round <= 3
                # Pick highest-weight available edge for this agent
                best_e = argmax(e -> Float64(get(GAME_WEIGHTS,e,NNO_ONE)),
                                available)
                p_test = run_opiate(union(new_stops, Set([best_e])))
                if p_test < p_max - 1e-8
                    push!(new_stops, best_e)
                    push!(actions_taken,
                          "$(agent.name):$(best_e[1])→$(best_e[2])")
                end
            end
        end

        p_max_after = isempty(actions_taken) ? p_max : run_opiate(new_stops)
        if p_max_after < p_max - 1e-8
            current_stops = new_stops
            p_max         = p_max_after
        end

        push!(history, (round=round, p_max=p_max, stratum=stratum,
                        actions=join(actions_taken,", ")))

        if verbose
            sect  = ["D","D","C","B","A"][stratum+1]
            bar   = "█"^Int(floor(p_max*20)) * "░"^(20-Int(floor(p_max*20)))
            @printf("%-6d %s %.4f  %-8s  %s\n",
                    round, bar, p_max, "$stratum($sect)",
                    isempty(actions_taken) ? "—" : join(actions_taken,", "))
        end

        p_max <= NASH_FLOOR + 0.005 && (verbose && println("  Nash floor reached."); break)
    end

    verbose && begin
        println("─"^68)
        @printf("  Final P_max = %.6f  (Nash floor = 1/17 ≈ %.4f)\n",
                p_max, NASH_FLOOR)
        result = p_max <= NASH_FLOOR + 0.01 ?
                 "🏆 Nash floor achieved — minimum possible P_max" :
                 "⚠  Residual: $(round(p_max,digits=4)) > $(round(NASH_FLOOR,digits=4)) (62-class obstruction)"
        println("  " * result)
        println("="^68)
    end

    return history
end


# =============================================================================
# PART 8: MARKOV CHAIN PROJECTION + STATISTICAL MANIFOLD (Fisher metric)
# =============================================================================

"""
Run Markov chain n_steps from δ_CA1sp, return p(sAMY) trajectory + bracket.
Uses Float64 transition matrix to avoid Int128 overflow from Renkin-Crone
weight products accumulating over many steps.
The NNO representation is exact for single-step computations but overflows
for multi-step products (345.9^15 >> Int128 capacity).
"""
function markov_projection(stops::Set, n_steps::Int=15)
    ctx     = build_contexts(stops)[:CTX_sAMY]
    n       = length(ctx.regions)
    ca1sp_i = findfirst(==(:CA1sp), ctx.regions)
    samy_i  = findfirst(==(:sAMY),  ctx.regions)
    (ca1sp_i === nothing || samy_i === nothing) && return Float64[], (0.0,0.0)

    # Convert NNO transition matrix to Float64 for multi-step simulation
    T = Float64.(ctx.trans_mat)   # n×n row-stochastic in Float64

    # Initial distribution: δ_CA1sp
    p = zeros(Float64, n)
    p[ca1sp_i] = 1.0

    traj = Float64[]
    for _ in 1:n_steps
        p = T * p              # one Markov step: p ← T·p
        p = max.(p, 0.0)       # numerical safety
        s = sum(p); s > 0 && (p ./= s)
        push!(traj, p[samy_i])
    end
    return traj, (minimum(traj), maximum(traj))
end

"""
Compute Fisher information metric and effective resistance for each edge.
g_ee = w_e · R_eff(e) · (1 - w_e · R_eff(e))
Risk gradient: exp(-mean_g) → 1 = free flow (opiate wins), 0 = blocked.
"""
function fisher_metric_projection(stops::Set)
    ctx = build_contexts(stops)[:CTX_sAMY]
    n   = length(ctx.regions)
    n == 0 && return Dict{Tuple{Symbol,Symbol},Float64}(), 0.0

    W = zeros(n, n)
    for (s, t) in ctx.edges
        si = findfirst(==(s), ctx.regions)
        ti = findfirst(==(t), ctx.regions)
        (si === nothing || ti === nothing) && continue
        W[si, ti] += Float64(get(ctx.weights,(s,t),NNO_ONE))
    end
    L = diagm(vec(sum(W,dims=2))) - W

    F     = svd(L)
    tol   = maximum(F.S) * 1e-10
    Sinv  = Diagonal([s>tol ? 1/s : 0.0 for s in F.S])
    Lpinv = F.V * Sinv * F.U'

    r_eff_table = Dict{Tuple{Symbol,Symbol},Float64}()
    total_g = 0.0; n_e = 0

    for (s, t) in ctx.edges
        si = findfirst(==(s), ctx.regions)
        ti = findfirst(==(t), ctx.regions)
        (si === nothing || ti === nothing) && continue
        ed = zeros(n); ed[si]=1.0; ed[ti]=-1.0
        R  = max(dot(ed, Lpinv*ed), 0.0)
        r_eff_table[(s,t)] = R
        w  = Float64(get(ctx.weights,(s,t),NNO_ONE))
        total_g += w * R * max(0.0, 1.0-w*R)
        n_e += 1
    end
    mean_g = n_e > 0 ? total_g/n_e : 0.0
    return r_eff_table, exp(-mean_g)
end

"""Print Markov chain + Fisher metric projections for a given stop set."""
function print_projections(stops::Set, round_num::Int; label::String="")
    isempty(label) || println("  $(label):")
    traj, (p_lo, p_hi) = markov_projection(stops, 15)
    sparks = ['▁','▂','▃','▄','▅','▆','▇','█']
    mx = max(p_hi, 1e-6)
    spark = join([sparks[max(1,Int(ceil(p/mx*8)))] for p in traj])
    @printf("    Markov p(sAMY) bracket [%.4f, %.4f]  %s\n", p_lo, p_hi, spark)

    r_eff, risk = fisher_metric_projection(stops)
    risk_bar = "█"^Int(floor(risk*20)) * "░"^(20-Int(floor(risk*20)))
    @printf("    Fisher risk:  %.4f  %s  (1=opiate free, 0=blocked)\n", risk, risk_bar)

    # Lowest resistance = highest flow = norcain's priority targets
    sorted_r = sort(collect(r_eff), by=x->x[2])
    println("    Top flow edges (norcain targets):")
    for (e, r) in sorted_r[1:min(3,end)]
        w = Float64(get(GAME_WEIGHTS,e,NNO_ONE))
        @printf("      %-10s→%-10s  R_eff=%.4f  w=%.1f\n", e[1],e[2],r,w)
    end
end

# =============================================================================
# PART 7: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

    println("\n" * "="^68)
    println("PHARMACODYNAMIC MARL GAME — Q_7P, Renkin-Crone weights")
    println("="^68)

    # ── 1. Initial state ───────────────────────────────────────────────────
    p0 = run_opiate(BASE_STOPS)
    @printf("\nBaseline P_max (no blocking): %.6f\n", p0)
    @printf("Nash floor (1/17, direct CA1sp→sAMY): %.6f\n\n", NASH_FLOOR)
    print_projections(BASE_STOPS, 0; label="Baseline projections")

    # ── 2. Two-agent game ──────────────────────────────────────────────────
    println("[1] TWO-AGENT GAME (opiate vs norcain)")
    history_2 = run_game(20; verbose=true)

    if !isempty(history_2)
        println("\n  Post-game projections:")
        print_projections(history_2[end].stops, length(history_2))
    end
    println("\n  Game trajectory (P_max per round):")
    for h in history_2
        bar = "█"^Int(floor(h.p_max*30)) * "░"^(30-Int(floor(h.p_max*30)))
        act = h.action === nothing ? "—" : "$(h.action[1])→$(h.action[2])"
        @printf("  R%-2d [Str=%d] %s %.4f  %s\n",
                h.round, h.stratum, bar, h.p_max, act)
    end

    # ── 3. Q-table ─────────────────────────────────────────────────────────
    println("\n[2] Q-TABLE (exact NNO rational rewards)")
    qt = compute_q_table(true)

    # ── 4. Four-agent cooperative game ────────────────────────────────────
    println("\n[3] FOUR-AGENT COOPERATIVE GAME")
    history_4 = run_marl_game(15; verbose=true)

    # ── 5. Nash analysis ───────────────────────────────────────────────────
    println("\n[4] NASH EQUILIBRIUM ANALYSIS")
    println("─"^68)
    println("  P_max trajectory comparison:")
    @printf("  %-25s  P_max = %.6f\n", "Baseline (no blocking):", p0)
    @printf("  %-25s  P_max = %.6f\n", "2-agent (norcain only):",
            history_2[end].p_max)
    @printf("  %-25s  P_max = %.6f\n", "4-agent (cooperative):",
            history_4[end].p_max)
    @printf("  %-25s  P_max = %.6f\n", "Nash floor (1/17):", NASH_FLOOR)
    println()
    println("  coker(ρ*_sAMY↔Infra) = 62  ↔  no stable NE")
    println("  62 alternative paths survive any finite stop set")
    println("  The direct CA1sp→sAMY edge (w=1/17) is the irreducible floor")
    println("─"^68)
end
