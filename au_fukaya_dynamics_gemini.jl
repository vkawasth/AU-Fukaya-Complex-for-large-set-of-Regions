using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Plots, DelimitedFiles

const RealType = Float64

# Pharmacological constants
const HALF_LIFE_A = 6.0
const HALF_LIFE_B = 3.0
const EC50_A = 0.2
const EC50_B = 0.2
const HILL_A = 2.0
const HILL_B = 2.0
const ALPHA = 3.0
const BETA = 1.0
const THRESHOLD = 0.2

const DOSE_TIMES = [2.5, 6.0, 9.5]
const DOSE_AMOUNT = 4.0

const MOLECULE_RADIUS = Dict(:A => 0.5e-9, :B => 0.8e-9)
const PORE_RADIUS = 1.0e-9
const DIFFUSIVITY = Dict(:A => 5e-10, :B => 8e-10)
const MEMBRANE_THICKNESS = 1e-6
const BASE_FLOW = Dict(:A => 4.0, :B => 6.0)

const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY]
const REGION_INDEX = Dict(r => i for (i, r) in enumerate(REGIONS))
const REGION_NAMES = ["BLA", "CA1sp", "HPF", "HY", "LA", "sAMY"]

const CORE_EDGES = [
    (:CA1sp, :HPF), (:HPF, :CA1sp),
    (:BLA, :LA), (:LA, :BLA),
    (:BLA, :sAMY), (:sAMY, :BLA),
    (:CA1sp, :sAMY), (:sAMY, :CA1sp),
    (:HPF, :BLA), (:HPF, :sAMY),
    (:LA, :sAMY), (:sAMY, :LA),
    (:HY, :sAMY), (:sAMY, :HY),
    (:BLA, :BLA), (:CA1sp, :CA1sp), (:HPF, :HPF),
    (:HY, :HY), (:LA, :LA), (:sAMY, :sAMY)
]

const BASE_EDGE_WEIGHTS = Dict(
    (:CA1sp, :HPF) => 15.0, (:HPF, :CA1sp) => 5.0,
    (:BLA, :LA) => 8.0, (:LA, :BLA) => 6.0,
    (:BLA, :sAMY) => 12.0, (:sAMY, :BLA) => 10.0,
    (:CA1sp, :sAMY) => 9.0, (:sAMY, :CA1sp) => 7.0,
    (:HPF, :BLA) => 4.0, (:HPF, :sAMY) => 6.0,
    (:LA, :sAMY) => 5.0, (:sAMY, :LA) => 5.0,
    (:HY, :sAMY) => 3.0, (:sAMY, :HY) => 8.0,
    (:BLA, :BLA) => 2.0, (:CA1sp, :CA1sp) => 2.0,
    (:HPF, :HPF) => 2.0, (:HY, :HY) => 2.0,
    (:LA, :LA) => 2.0, (:sAMY, :sAMY) => 2.0
)

# Compile static factors outside the loop to save massive cycles
function renkin_crone_factor(radius_ratio::RealType)::RealType
    if radius_ratio >= 1.0 return 0.0 end
    return (1 - radius_ratio)^2 * (1 - 2.104*radius_ratio + 2.09*radius_ratio^3 - 0.95*radius_ratio^5)
end

const RENKIN_FACTOR_A = renkin_crone_factor(MOLECULE_RADIUS[:A] / PORE_RADIUS)
const RENKIN_FACTOR_B = renkin_crone_factor(MOLECULE_RADIUS[:B] / PORE_RADIUS)

function flow_rate(edge::Tuple{Symbol,Symbol}, t::RealType, molecule::Symbol, edge_weights::Dict)::RealType
    heartbeat = 1.0 + 0.3 * sin(2 * π * t)
    weight = get(edge_weights, edge, 1.0)
    return BASE_FLOW[molecule] * weight * heartbeat
end

function transition_rate(edge::Tuple{Symbol,Symbol}, t::RealType, molecule::Symbol, edge_weights::Dict)::RealType
    flow = flow_rate(edge, t, molecule, edge_weights)
    factor = (molecule == :A) ? RENKIN_FACTOR_A : RENKIN_FACTOR_B
    permeability = (DIFFUSIVITY[molecule] / MEMBRANE_THICKNESS) * factor
    return flow * permeability * 300.0
end

mutable struct SimulationState
    t::RealType
    step::Int
    dt::RealType
    qA::Vector{RealType}
    qA_trap::Vector{RealType}
    qB::Vector{RealType}
    qB_trap::Vector{RealType}
    C::Vector{RealType}
    C_history::Vector{Vector{RealType}}
    qB_history::Vector{Vector{RealType}}
    edge_weights::Dict{Tuple{Symbol,Symbol}, RealType}
    Lax_L::Matrix{RealType}
    HH2::RealType
    HH2_history::Vector{RealType}
    crisis_countdown::Int
    norcain_injected::Bool
    crisis_events::Vector{Float64}
    dose_applied::Dict{Float64, Bool}
    loopy_nodes::Set{Symbol}
end

function create_initial_state(dt::RealType)::SimulationState
    n_regions = length(REGIONS)
    qA, qA_trap, qB, qB_trap = zeros(n_regions), zeros(n_regions), zeros(n_regions), zeros(n_regions)
    C = ones(n_regions)
    qA[REGION_INDEX[:CA1sp]] = 3.5
    
    C_history = [copy(C) for _ in 1:3]
    qB_history = [copy(qB) for _ in 1:3]
    
    # Initialize Lax Operator matrix dynamically dimensioned to match the actual number of nodes
    Lax_L = Matrix{RealType}(I, n_regions, n_regions)
    
    dose_applied = Dict(t_val => false for t_val in DOSE_TIMES)
    
    return SimulationState(
        0.0, 0, dt, qA, qA_trap, qB, qB_trap, C, C_history, qB_history,
        copy(BASE_EDGE_WEIGHTS), Lax_L, 0.0, RealType[], 0, false, Float64[],
        dose_applied, Set([:CA1sp, :BLA])
    )
end

function compute_renkin_crone_rates(state::SimulationState, t::RealType, region_idx::Int)
    region = REGIONS[region_idx]
    loopy = region in state.loopy_nodes
    
    qA_free_cur = state.qA[region_idx]
    qA_trap_cur = loopy ? state.qA_trap[region_idx] : 0.0
    qB_free_cur = state.qB[region_idx]
    qB_trap_cur = loopy ? state.qB_trap[region_idx] : 0.0
    C_cur = state.C[region_idx]
    
    lambda_A = log(2) / HALF_LIFE_A
    lambda_B = log(2) / HALF_LIFE_B
    inflow_A, inflow_B = 0.0, 0.0
    
    for (src, tgt) in CORE_EDGES
        if tgt == region
            src_idx = REGION_INDEX[src]
            inflow_A += transition_rate((src, tgt), t, :A, state.edge_weights) * state.qA[src_idx] * state.dt
            inflow_B += transition_rate((src, tgt), t, :B, state.edge_weights) * state.qB[src_idx] * state.dt
        end
    end
    
    outflow_A, outflow_B = 0.0, 0.0
    for (src, tgt) in CORE_EDGES
        if src == region
            outflow_A += transition_rate((src, tgt), t, :A, state.edge_weights) * qA_free_cur * state.dt
            outflow_B += transition_rate((src, tgt), t, :B, state.edge_weights) * qB_free_cur * state.dt
        end
    end
    
    dqA_free = inflow_A - outflow_A - qA_free_cur * lambda_A * state.dt
    dqB_free = inflow_B - outflow_B - qB_free_cur * lambda_B * state.dt
    dqA_trap, dqB_trap = 0.0, 0.0
    
    if loopy
        alpha_in, alpha_out = 0.15, 0.015
        dqA_free += (-alpha_in * qA_free_cur + alpha_out * qA_trap_cur) * state.dt
        dqB_free += (-alpha_in * qB_free_cur + alpha_out * qB_trap_cur) * state.dt
        dqA_trap = (alpha_in * qA_free_cur - alpha_out * qA_trap_cur) * state.dt
        dqB_trap = (alpha_in * qB_free_cur - alpha_out * qB_trap_cur) * state.dt
    end
    
    activation = ALPHA * (1 - C_cur) * (qB_free_cur^HILL_B) / (EC50_B^HILL_B + qB_free_cur^HILL_B)
    dampening = BETA * C_cur * (qA_free_cur^HILL_A) / (EC50_A^HILL_A + qA_free_cur^HILL_A)
    oscillation = 0.1 * sin(2 * π * 0.6 * t) * (1 - C_cur)
    dC = (activation - dampening + oscillation) * state.dt
    
    return dqA_free, dqA_trap, dqB_free, dqB_trap, dC
end

function apply_dose!(state::SimulationState, t::RealType)
    for dose_time in DOSE_TIMES
        if abs(t - dose_time) < state.dt && !state.dose_applied[dose_time]
            state.dose_applied[dose_time] = true
            ca1sp_idx = REGION_INDEX[:CA1sp]
            state.qB[ca1sp_idx] += DOSE_AMOUNT
            state.qB[ca1sp_idx] = clamp(state.qB[ca1sp_idx], 0.0, 6.0)
            @printf("  💉 DOSE at t=%.2f: +%.1f norcain at CA1sp\n", t, DOSE_AMOUNT)
            state.norcain_injected = true
        end
    end
end

function update_molecular_dynamics!(state::SimulationState)
    t = state.t
    new_qA, new_qA_trap = copy(state.qA), copy(state.qA_trap)
    new_qB, new_qB_trap = copy(state.qB), copy(state.qB_trap)
    new_C = copy(state.C)
    
    for (idx, region) in enumerate(REGIONS)
        dqA_f, dqA_t, dqB_f, dqB_t, dC = compute_renkin_crone_rates(state, t, idx)
        
        new_qA[idx] += dqA_f
        new_qB[idx] += dqB_f
        new_C[idx] += dC
        if region in state.loopy_nodes
            new_qA_trap[idx] += dqA_t
            new_qB_trap[idx] += dqB_t
        end
        
        new_qA[idx] = clamp(new_qA[idx], 0.0, 5.0)
        new_qA_trap[idx] = clamp(new_qA_trap[idx], 0.0, 5.0)
        new_qB[idx] = clamp(new_qB[idx], 0.0, 6.0)
        new_qB_trap[idx] = clamp(new_qB_trap[idx], 0.0, 6.0)
        new_C[idx] = clamp(new_C[idx], 0.0, 1.0)
    end
    
    state.qA, state.qA_trap = new_qA, new_qA_trap
    state.qB, state.qB_trap = new_qB, new_qB_trap
    state.C = new_C
    apply_dose!(state, t)
end

function update_history!(state::SimulationState)
    push!(state.C_history, copy(state.C))
    push!(state.qB_history, copy(state.qB))
    if length(state.C_history) > 100
        popfirst!(state.C_history)
        popfirst!(state.qB_history)
    end
end

function compute_HH2!(state::SimulationState)
    if length(state.C_history) < 3
        state.HH2 = 0.0
        return
    end
    
    C_cur, C_prev, C_prev2 = state.C_history[end], state.C_history[end-1], state.C_history[end-2]
    dt_val = state.dt
    
    HH2 = norm((C_cur .- 2*C_prev .+ C_prev2) ./ (dt_val^2))
    
    comm = 0.0
    for i in 1:min(3, length(C_cur)), j in i+1:min(3, length(C_cur))
        comm += abs(C_cur[i] * state.qB_history[end][j] - C_cur[j] * state.qB_history[end][i])
    end
    
    state.HH2 = HH2 + 0.3 * comm
    push!(state.HH2_history, state.HH2)
    if length(state.HH2_history) > 300 popfirst!(state.HH2_history) end
end

function detect_crisis(state::SimulationState)::Tuple{Bool,RealType}
    if length(state.HH2_history) < 50 return false, 0.0 end
    baseline_data = state.HH2_history[max(1, end-100):end]
    
    med = median(baseline_data)
    current = state.HH2
    
    if med > 0 && current > med * 2.0
        return true, current / med
    elseif current > 1.0
        return true, current
    end
    return false, 0.0
end

function rees_blowup!(state::SimulationState)
    @printf("  🔥 REES BLOW-UP at t=%.2f: Injecting norcain at ALL nodes\n", state.t)
    state.qB .+= 1.0
    state.qB .= clamp.(state.qB, 0.0, 6.0)
    push!(state.crisis_events, state.t)
    state.norcain_injected = true
    state.crisis_countdown = 20
end

function prolate_feedback!(state::SimulationState, prolate_ratio::RealType)
    if state.step > 100 && prolate_ratio > 0.8 && !state.norcain_injected
        @printf("  🔒 Prolate lock achieved at t=%.2f: Extra norcain injection\n", state.t)
        state.qB[REGION_INDEX[:CA1sp]] = clamp(state.qB[REGION_INDEX[:CA1sp]] + 1.0, 0.0, 6.0)
        state.norcain_injected = true
    end
end

function build_lax_matrix(state::SimulationState)::Matrix{RealType}
    n = length(REGIONS)
    L = zeros(RealType, n, n)
    for i in 1:n
        L[i, i] = state.C[i] * (1.0 - state.qA[i])
        if i < n
            L[i, i+1] = atan(state.qB[i] - state.qA[i], state.qA[i] + state.qB[i] + 1e-5)
            L[i+1, i] = L[i, i+1]
        end
    end
    return L
end

function toda_flow_step!(state::SimulationState, dt::RealType)
    L = state.Lax_L
    n = size(L, 1)
    B = zeros(n, n)
    for i in 1:n-1
        B[i, i+1] = L[i, i+1]
        B[i+1, i] = -L[i, i+1]
    end
    
    dL = B * L - L * B - 0.1 * (L - diagm(diag(L))) + 0.5 * sin(2 * π * 8.0 * state.t) * I
    state.Lax_L += dL * dt
    state.Lax_L = (state.Lax_L + state.Lax_L') / 2
end

function compute_prolate_ratio(state::SimulationState)::RealType
    if length(state.C_history) < 10 return 0.0 end
    C_mat = hcat(state.C_history[end-9:end]...)
    evals = eigvals(cov(C_mat'))
    return sum(evals) > 0 ? maximum(evals) / sum(evals) : 0.0
end

function compute_plucker_coordinates(state::SimulationState)::Vector{RealType}
    p = [mean(state.C), mean(state.qA), mean(state.qB), 1.0, 0.0, 0.0]
    return p ./ (norm(p) + 1e-8)
end

function compute_ghost_signal(state::SimulationState)::RealType
    return mean(state.C) < 0.3 ? 0.6 : sqrt(mean(state.qA)^2 + mean(state.qB)^2)
end

function run_simulation(t_span::Tuple{RealType,RealType}, dt::RealType, save_interval::Int = 25)
    state = create_initial_state(dt)
    n_steps = Int(floor((t_span[2] - t_span[1]) / dt)) + 1
    n_save = ceil(Int, n_steps / save_interval)
    
    times, C_mean, C_std = zeros(n_save), zeros(n_save), zeros(n_save)
    qA_mean, qB_mean, HH2_values = zeros(n_save), zeros(n_save), zeros(n_save)
    ghost_signal, plucker_norm = zeros(n_save), zeros(n_save)
    C_regions = zeros(length(REGIONS), n_save)
    
    save_idx = 1
    times[1] = state.t
    C_mean[1], qA_mean[1], qB_mean[1] = mean(state.C), mean(state.qA), mean(state.qB)
    C_regions[:, 1] = state.C
    
    for step in 1:n_steps-1
        state.t = t_span[1] + (step - 1) * dt
        state.step = step
        
        update_molecular_dynamics!(state)
        update_history!(state)
        compute_HH2!(state)
        
        state.Lax_L = build_lax_matrix(state)
        toda_flow_step!(state, dt)
        
        prolate_ratio = compute_prolate_ratio(state)
        prolate_feedback!(state, prolate_ratio)
        
        is_crisis, ratio = detect_crisis(state)
        if is_crisis && state.crisis_countdown == 0
            @printf("  ⚠️ CRISIS at t=%.2f: HH² spike ratio = %.2f\n", state.t, ratio)
            rees_blowup!(state)
        elseif state.crisis_countdown > 0
            state.crisis_countdown -= 1
        end
        
        if step % save_interval == 0 || step == n_steps-1
            save_idx += 1
            times[save_idx] = state.t
            C_mean[save_idx], C_std[save_idx] = mean(state.C), std(state.C)
            C_regions[:, save_idx] = state.C
            qA_mean[save_idx], qB_mean[save_idx] = mean(state.qA), mean(state.qB)
            HH2_values[save_idx] = state.HH2
            ghost_signal[save_idx] = compute_ghost_signal(state)
            plucker_norm[save_idx] = norm(compute_plucker_coordinates(state))
        end
    end
    return (times, C_mean, C_std, C_regions, qA_mean, qB_mean, HH2_values, ghost_signal, plucker_norm, state)
end

function create_dashboard(times, C_mean, C_std, C_regions, qA_mean, qB_mean, HH2_values, ghost_signal, dose_times, crisis_times, threshold, region_names)
    p1 = plot(title="1. Consciousness Dynamics", xlabel="Time (s)", ylabel="C")
    plot!(p1, times, C_mean, lw=2, color=:blue, label="Mean C")
    plot!(p1, times, C_mean .+ C_std, lw=1, color=:blue, alpha=0.3, fillrange=C_mean .- C_std, fillalpha=0.2, label="±1σ")
    hline!(p1, [threshold], lw=2, ls=:dash, color=:red, label="Threshold")
    
    for (i, dt) in enumerate(dose_times)
        vline!(p1, [dt], lw=1, ls=:dot, color=:green, alpha=0.7, label=(i == 1 ? "Dose" : ""))
    end
    for (i, ct) in enumerate(crisis_times)
        vline!(p1, [ct], lw=2, ls=:dash, color=:orange, alpha=0.8, label=(i == 1 ? "Crisis" : ""))
    end
    
    p2 = plot(title="2. Norcain Concentration", xlabel="Time (s)", ylabel="qB")
    plot!(p2, times, qB_mean, lw=2, color=:green, label="Mean qB")
    for (i, dt) in enumerate(dose_times) vline!(p2, [dt], lw=1, ls=:dot, color=:green, alpha=0.7, label=(i == 1 ? "Dose" : "")) end
    
    p3 = plot(title="3. HH² Obstruction", xlabel="Time (s)", ylabel="HH²")
    plot!(p3, times, HH2_values, lw=2, color=:red, label="HH²")
    for (i, ct) in enumerate(crisis_times) vline!(p3, [ct], lw=2, ls=:dash, color=:orange, label=(i == 1 ? "Crisis" : "")) end
    
    p4 = plot(title="4. Ghost Signal", xlabel="Time (s)", ylabel="Value")
    plot!(p4, times, ghost_signal, lw=2, color=:gold, label="Ghost")
    
    p5 = plot(title="5. Regional Consciousness", xlabel="Time (s)", ylabel="C")
    colors_pal = palette(:tab10, length(region_names))
    for i in 1:length(region_names)
        plot!(p5, times, C_regions[i, :], lw=1.5, color=colors_pal[i], label=region_names[i])
    end
    
    p6 = plot(title="6. Phase Space", xlabel="C", ylabel="dC/dt")
    dC = diff(C_mean) ./ (diff(times) .+ 1e-5)
    scatter!(p6, C_mean[2:end], dC, markersize=2, color=:blue, alpha=0.5, label="Trajectory")
    
    return plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1400, 900))
end

function main()
    t_span = (0.0, 25.0)
    dt = 0.02
    times, C_mean, C_std, C_regions, qA_mean, qB_mean, HH2_values, ghost_signal, plucker_norm, state = run_simulation(t_span, dt, 25)
    
    dashboard = create_dashboard(times, C_mean, C_std, C_regions, qA_mean, qB_mean, HH2_values, ghost_signal, DOSE_TIMES, state.crisis_events, THRESHOLD, REGION_NAMES)
    savefig(dashboard, "simulation_dashboard.png")
    println("✓ Saved: simulation_dashboard.png")
end

main()
