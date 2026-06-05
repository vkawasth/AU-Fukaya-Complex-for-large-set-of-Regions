# =============================================================================
# AU_FUKAYA_INTEGRATED_WITH_PLOTS.jl
# =============================================================================
# Fully integrated pipeline with:
#   1. Full Renkin-Crone hindered transport (matching Python)
#   2. Norcain injection schedule (matching Python's dose_times)
#   3. Crisis detection via HH² spikes (matching Python's obstruction metric)
#   4. Rees blow-up (norcain injection on crisis)
#   5. Toda lattice flow with isospectral evolution
#   6. Complete visualization with Plots.jl
#
# Run: julia AU_FUKAYA_INTEGRATED_WITH_PLOTS.jl
# =============================================================================

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Plots, DelimitedFiles

# =============================================================================
# PART 0: TYPE DEFINITIONS & CONSTANTS
# =============================================================================

const RealType = Float64

# Pharmacological constants (matching Python)
const HALF_LIFE_A = 6.0      # Opiate half-life (seconds)
const HALF_LIFE_B = 3.0      # Norcain half-life (seconds)
const EC50_A = 0.2           # Half-maximal concentration for opiate
const EC50_B = 0.2           # Half-maximal concentration for norcain
const HILL_A = 2.0           # Hill coefficient for opiate
const HILL_B = 2.0           # Hill coefficient for norcain
const ALPHA = 3.0            # Activation rate
const BETA = 1.0             # Dampening rate
const THRESHOLD = 0.2        # Consciousness threshold for crisis detection

# Dose schedule (matching Python)
const DOSE_TIMES = [2.5, 6.0, 9.5]  # seconds
const DOSE_AMOUNT = 4.0              # norcain units per dose

# Physical constants for Renkin-Crone (matching Python)
const MOLECULE_RADIUS = Dict(:A => 0.5e-9, :B => 0.8e-9)  # meters
const PORE_RADIUS = 1.0e-9                                 # meters
const DIFFUSIVITY = Dict(:A => 5e-10, :B => 8e-10)        # m²/s
const MEMBRANE_THICKNESS = 1e-6                           # meters
const BASE_FLOW = Dict(:A => 4.0, :B => 6.0)              # flow rate scaling

# Graph definition (6 regions matching Python)
const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY]
const REGION_INDEX = Dict(r => i for (i, r) in enumerate(REGIONS))
const REGION_NAMES = ["BLA", "CA1sp", "HPF", "HY", "LA", "sAMY"]

# Core edges with self-loops
const CORE_EDGES = [
    # CA1sp ↔ HPF
    (:CA1sp, :HPF), (:HPF, :CA1sp),
    # BLA ↔ LA  
    (:BLA, :LA), (:LA, :BLA),
    # BLA ↔ sAMY
    (:BLA, :sAMY), (:sAMY, :BLA),
    # CA1sp ↔ sAMY
    (:CA1sp, :sAMY), (:sAMY, :CA1sp),
    # HPF → BLA, HPF → sAMY
    (:HPF, :BLA), (:HPF, :sAMY),
    # LA → sAMY, sAMY → LA
    (:LA, :sAMY), (:sAMY, :LA),
    # HY ↔ sAMY
    (:HY, :sAMY), (:sAMY, :HY),
    # Self-loops
    (:BLA, :BLA), (:CA1sp, :CA1sp), (:HPF, :HPF),
    (:HY, :HY), (:LA, :LA), (:sAMY, :sAMY)
]

# Base edge weights (matching Python's edge_weights)
const BASE_EDGE_WEIGHTS = Dict(
    # CA1sp ↔ HPF
    (:CA1sp, :HPF) => 15.0, (:HPF, :CA1sp) => 5.0,
    # BLA ↔ LA
    (:BLA, :LA) => 8.0, (:LA, :BLA) => 6.0,
    # BLA ↔ sAMY
    (:BLA, :sAMY) => 12.0, (:sAMY, :BLA) => 10.0,
    # CA1sp ↔ sAMY
    (:CA1sp, :sAMY) => 9.0, (:sAMY, :CA1sp) => 7.0,
    # HPF → BLA, HPF → sAMY
    (:HPF, :BLA) => 4.0, (:HPF, :sAMY) => 6.0,
    # LA → sAMY, sAMY → LA
    (:LA, :sAMY) => 5.0, (:sAMY, :LA) => 5.0,
    # HY ↔ sAMY
    (:HY, :sAMY) => 3.0, (:sAMY, :HY) => 8.0,
    # Self-loops (all 2.0)
    (:BLA, :BLA) => 2.0, (:CA1sp, :CA1sp) => 2.0,
    (:HPF, :HPF) => 2.0, (:HY, :HY) => 2.0,
    (:LA, :LA) => 2.0, (:sAMY, :sAMY) => 2.0
)

# =============================================================================
# PART 1: RENKIN-CRONE HINDERED TRANSPORT
# =============================================================================

function renkin_crone_factor(radius_ratio::RealType)::RealType
    if radius_ratio >= 1.0
        return 0.0
    end
    return (1 - radius_ratio)^2 * (1 - 2.104*radius_ratio + 
           2.09*radius_ratio^3 - 0.95*radius_ratio^5)
end

function flow_rate(edge::Tuple{Symbol,Symbol}, t::RealType, molecule::Symbol, 
                   edge_weights::Dict)::RealType
    heartbeat = 1.0 + 0.3 * sin(2 * π * t)
    weight = get(edge_weights, edge, 1.0)
    base = BASE_FLOW[molecule]
    return base * weight * heartbeat
end

function transition_rate(edge::Tuple{Symbol,Symbol}, t::RealType, molecule::Symbol,
                         edge_weights::Dict)::RealType
    flow = flow_rate(edge, t, molecule, edge_weights)
    
    radius = MOLECULE_RADIUS[molecule]
    radius_ratio = radius / PORE_RADIUS
    renkin_factor = renkin_crone_factor(radius_ratio)
    
    diffusivity = DIFFUSIVITY[molecule]
    permeability = (diffusivity / MEMBRANE_THICKNESS) * renkin_factor
    
    return flow * permeability * 300.0
end

# =============================================================================
# PART 2: DYNAMICAL SYSTEM STATE
# =============================================================================

mutable struct SimulationState
    # Time tracking
    t::RealType
    step::Int
    dt::RealType
    
    # Concentrations [region]
    qA::Vector{RealType}
    qA_trap::Vector{RealType}
    qB::Vector{RealType}
    qB_trap::Vector{RealType}
    C::Vector{RealType}
    
    # History buffers
    C_history::Vector{Vector{RealType}}
    qB_history::Vector{Vector{RealType}}
    
    # Dynamic edge weights
    edge_weights::Dict{Tuple{Symbol,Symbol}, RealType}
    
    # Toda Lax matrix
    Lax_L::Matrix{RealType}
    
    # Crisis tracking
    HH2::RealType
    HH2_history::Vector{RealType}
    crisis_countdown::Int
    norcain_injected::Bool
    crisis_events::Vector{Float64}
    
    # Dose tracking
    dose_applied::Dict{Float64, Bool}
    
    # Region properties
    loopy_nodes::Set{Symbol}
end

function create_initial_state(dt::RealType)::SimulationState
    n_regions = length(REGIONS)
    
    qA = zeros(n_regions)
    qA_trap = zeros(n_regions)
    qB = zeros(n_regions)
    qB_trap = zeros(n_regions)
    C = ones(n_regions)
    
    # Opiate at CA1sp (index 2)
    ca1sp_idx = REGION_INDEX[:CA1sp]
    qA[ca1sp_idx] = 3.5
    
    C_history = [copy(C), copy(C), copy(C)]
    qB_history = [copy(qB), copy(qB), copy(qB)]
    
    edge_weights = copy(BASE_EDGE_WEIGHTS)
    Lax_L = Matrix{RealType}(I, 3, 3)
    
    HH2_history = RealType[]
    crisis_events = Float64[]
    
    dose_applied = Dict(dt => false for dt in DOSE_TIMES)
    loopy_nodes = Set([:CA1sp, :BLA])
    
    return SimulationState(
        0.0, 0, dt,
        qA, qA_trap, qB, qB_trap, C,
        C_history, qB_history,
        edge_weights,
        Lax_L,
        0.0, HH2_history, 0, false, crisis_events,
        dose_applied,
        loopy_nodes
    )
end

# =============================================================================
# PART 3: MOLECULAR DYNAMICS
# =============================================================================

function compute_renkin_crone_rates(state::SimulationState, t::RealType, 
                                    region_idx::Int, step_idx::Int)
    region = REGIONS[region_idx]
    loopy = region in state.loopy_nodes
    
    if loopy
        qA_free_cur = state.qA[region_idx]
        qA_trap_cur = state.qA_trap[region_idx]
        qB_free_cur = state.qB[region_idx]
        qB_trap_cur = state.qB_trap[region_idx]
    else
        qA_free_cur = state.qA[region_idx]
        qA_trap_cur = 0.0
        qB_free_cur = state.qB[region_idx]
        qB_trap_cur = 0.0
    end
    C_cur = state.C[region_idx]
    
    lambda_A = log(2) / HALF_LIFE_A
    lambda_B = log(2) / HALF_LIFE_B
    
    inflow_A = 0.0
    inflow_B = 0.0
    
    for (src, tgt) in CORE_EDGES
        if tgt == region
            src_idx = REGION_INDEX[src]
            
            src_loopy = src in state.loopy_nodes
            if src_loopy
                qA_src = state.qA[src_idx]
                qB_src = state.qB[src_idx]
            else
                qA_src = state.qA[src_idx]
                qB_src = state.qB[src_idx]
            end
            
            rate_A = transition_rate((src, tgt), t, :A, state.edge_weights)
            rate_B = transition_rate((src, tgt), t, :B, state.edge_weights)
            
            inflow_A += rate_A * qA_src * state.dt
            inflow_B += rate_B * qB_src * state.dt
        end
    end
    
    outflow_A = 0.0
    outflow_B = 0.0
    
    for (src, tgt) in CORE_EDGES
        if src == region
            rate_A = transition_rate((src, tgt), t, :A, state.edge_weights)
            rate_B = transition_rate((src, tgt), t, :B, state.edge_weights)
            
            outflow_A += rate_A * qA_free_cur * state.dt
            outflow_B += rate_B * qB_free_cur * state.dt
        end
    end
    
    dqA_free = inflow_A - outflow_A - qA_free_cur * lambda_A * state.dt
    dqB_free = inflow_B - outflow_B - qB_free_cur * lambda_B * state.dt
    
    dqA_trap = 0.0
    dqB_trap = 0.0
    
    if loopy
        alpha_in_A = 0.15
        alpha_out_A = 0.015
        alpha_in_B = 0.15
        alpha_out_B = 0.015
        
        dqA_free -= alpha_in_A * qA_free_cur * state.dt
        dqA_free += alpha_out_A * qA_trap_cur * state.dt
        dqB_free -= alpha_in_B * qB_free_cur * state.dt
        dqB_free += alpha_out_B * qB_trap_cur * state.dt
        
        dqA_trap = alpha_in_A * qA_free_cur * state.dt - alpha_out_A * qA_trap_cur * state.dt
        dqB_trap = alpha_in_B * qB_free_cur * state.dt - alpha_out_B * qB_trap_cur * state.dt
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

function update_molecular_dynamics!(state::SimulationState, step_idx::Int)
    t = state.t
    
    new_qA = copy(state.qA)
    new_qA_trap = copy(state.qA_trap)
    new_qB = copy(state.qB)
    new_qB_trap = copy(state.qB_trap)
    new_C = copy(state.C)
    
    for (region_idx, region) in enumerate(REGIONS)
        dqA_free, dqA_trap, dqB_free, dqB_trap, dC = 
            compute_renkin_crone_rates(state, t, region_idx, step_idx)
        
        loopy = region in state.loopy_nodes
        
        if loopy
            new_qA[region_idx] = state.qA[region_idx] + dqA_free
            new_qA_trap[region_idx] = state.qA_trap[region_idx] + dqA_trap
            new_qB[region_idx] = state.qB[region_idx] + dqB_free
            new_qB_trap[region_idx] = state.qB_trap[region_idx] + dqB_trap
        else
            new_qA[region_idx] = state.qA[region_idx] + dqA_free
            new_qB[region_idx] = state.qB[region_idx] + dqB_free
        end
        new_C[region_idx] = state.C[region_idx] + dC
        
        if loopy
            new_qA[region_idx] = clamp(new_qA[region_idx], 0.0, 5.0)
            new_qA_trap[region_idx] = clamp(new_qA_trap[region_idx], 0.0, 5.0)
            new_qB[region_idx] = clamp(new_qB[region_idx], 0.0, 6.0)
            new_qB_trap[region_idx] = clamp(new_qB_trap[region_idx], 0.0, 6.0)
        else
            new_qA[region_idx] = clamp(new_qA[region_idx], 0.0, 5.0)
            new_qB[region_idx] = clamp(new_qB[region_idx], 0.0, 6.0)
        end
        new_C[region_idx] = clamp(new_C[region_idx], 0.0, 1.0)
    end
    
    state.qA = new_qA
    state.qA_trap = new_qA_trap
    state.qB = new_qB
    state.qB_trap = new_qB_trap
    state.C = new_C
    
    apply_dose!(state, t)
end

# =============================================================================
# PART 4: HH² COMPUTATION (CRISIS DETECTION)
# =============================================================================

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
    
    C_cur = state.C_history[end]
    C_prev = state.C_history[end-1]
    C_prev2 = state.C_history[end-2]
    qB_cur = state.qB_history[end]
    
    dt_val = state.dt
    
    dC_dt = (C_cur .- C_prev) ./ dt_val
    HH1 = norm(dC_dt)
    
    d2C_dt2 = (C_cur .- 2*C_prev .+ C_prev2) ./ (dt_val^2)
    HH2 = norm(d2C_dt2)
    
    comm = 0.0
    for i in 1:min(3, length(C_cur))
        for j in i+1:min(3, length(C_cur))
            comm += abs(C_cur[i] * qB_cur[j] - C_cur[j] * qB_cur[i])
        end
    end
    HH2 += 0.3 * comm
    
    state.HH2 = HH2
    
    push!(state.HH2_history, HH2)
    if length(state.HH2_history) > 300
        popfirst!(state.HH2_history)
    end
end

function detect_crisis(state::SimulationState)::Tuple{Bool,RealType}
    if length(state.HH2_history) < 50
        return false, 0.0
    end
    
    window = min(100, length(state.HH2_history))
    start_idx = length(state.HH2_history) - window + 1
    baseline_data = state.HH2_history[start_idx:end]
    
    if isempty(baseline_data)
        return false, 0.0
    end
    
    baseline_median = median(baseline_data)
    baseline_std = std(baseline_data)
    current = state.HH2
    
    crisis_condition = false
    spike_ratio = 0.0
    
    if baseline_median > 0
        if current > baseline_median * 2.0
            crisis_condition = true
            spike_ratio = current / baseline_median
        elseif baseline_std > 0 && current > baseline_median + 3*baseline_std
            crisis_condition = true
            spike_ratio = current / baseline_median
        end
    elseif current > 1.0
        crisis_condition = true
        spike_ratio = current
    end
    
    return crisis_condition, spike_ratio
end

# =============================================================================
# PART 5: REES BLOW-UP & PROLATE FEEDBACK
# =============================================================================

function rees_blowup!(state::SimulationState)
    @printf("  🔥 REES BLOW-UP at t=%.2f: Injecting norcain at ALL nodes\n", state.t)
    
    for i in 1:length(REGIONS)
        state.qB[i] += 1.0
        state.qB[i] = clamp(state.qB[i], 0.0, 6.0)
    end
    
    push!(state.crisis_events, state.t)
    state.norcain_injected = true
    state.crisis_countdown = 20
end

function prolate_feedback!(state::SimulationState, prolate_ratio::RealType)
    if state.step < 100
        return
    end
    
    if prolate_ratio > 0.8 && !state.norcain_injected
        @printf("  🔒 Prolate lock achieved at t=%.2f: Extra norcain injection\n", state.t)
        ca1sp_idx = REGION_INDEX[:CA1sp]
        state.qB[ca1sp_idx] += 1.0
        state.qB[ca1sp_idx] = clamp(state.qB[ca1sp_idx], 0.0, 6.0)
        state.norcain_injected = true
    end
end

# =============================================================================
# PART 6: TODA FLOW AND LAX MATRIX
# =============================================================================

function build_lax_matrix(state::SimulationState)::Matrix{RealType}
    mean_C = mean(state.C)
    mean_qA = mean(state.qA)
    mean_qB = mean(state.qB)
    
    a = mean_C * (1 - mean_qA)
    b = (1 - mean_C) * (1 - mean_qB)
    w1 = 1.0 + mean_qA
    w2 = 2.0 + mean_qB
    phi = atan(mean_qB - mean_qA, mean_qA + mean_qB)
    kappa = mean_qA * mean_qB / ((mean_qA + mean_qB)^2 + 1e-8)
    
    L = [
        a     phi    kappa
        phi   b      (w1 + w2)/2
        kappa (w1 + w2)/2 mean_C
    ]
    
    return (L + L') / 2
end

function toda_flow_step!(state::SimulationState, dt::RealType)
    L = state.Lax_L
    n = size(L, 1)
    
    B = zeros(n, n)
    for i in 1:n-1
        B[i, i+1] = L[i, i+1]
        B[i+1, i] = -L[i, i+1]
    end
    
    dL = B * L - L * B
    damping = -0.1 * (L - diagm(diag(L)))
    dL += damping
    
    drive = 0.5 * sin(2 * π * 8.0 * state.t) * I
    dL += drive
    
    state.Lax_L += dL * dt
    state.Lax_L = (state.Lax_L + state.Lax_L') / 2
end

function compute_prolate_ratio(state::SimulationState)::RealType
    if length(state.C_history) < 10
        return 0.0
    end
    
    n_steps = min(10, length(state.C_history))
    C_mat = hcat(state.C_history[end-n_steps+1:end]...)
    
    if size(C_mat, 2) < 2
        return 0.0
    end
    
    cov_mat = cov(C_mat')
    evals = eigvals(cov_mat)
    
    if sum(evals) > 0
        return maximum(evals) / sum(evals)
    end
    return 0.0
end

# =============================================================================
# PART 7: PLÜCKER COORDINATES & GHOST SIGNAL
# =============================================================================

function compute_plucker_coordinates(state::SimulationState)::Vector{RealType}
    mean_C = mean(state.C)
    mean_qA = mean(state.qA)
    mean_qB = mean(state.qB)
    
    a = mean_C * (1 - mean_qA)
    b = (1 - mean_C) * (1 - mean_qB)
    w1 = 1.0 + mean_qA
    w2 = 2.0 + mean_qB
    phi = atan(mean_qB - mean_qA, mean_qA + mean_qB)
    kappa = mean_qA * mean_qB / ((mean_qA + mean_qB)^2 + 1e-8)
    
    denom = w1 * w2
    if abs(denom) < 1e-12
        return zeros(6)
    end
    
    q12 = a * b * denom
    q13 = a * phi * denom
    q14 = a * kappa * denom
    q23 = b * phi * denom
    q24 = b * kappa * denom
    q34 = phi * kappa * denom
    
    plucker = [q12, q13, q14, q23, q24, q34]
    norm_val = norm(plucker)
    
    if norm_val > 0
        plucker ./= norm_val
    end
    
    return plucker
end

function compute_ghost_signal(state::SimulationState)::RealType
    mean_qA = mean(state.qA)
    mean_qB = mean(state.qB)
    mean_C = mean(state.C)
    
    coordination = sqrt(mean_qA^2 + mean_qB^2)
    
    if mean_C < 0.3
        return 0.6
    end
    
    return coordination
end

# =============================================================================
# PART 8: MAIN SIMULATION LOOP
# =============================================================================

function run_simulation(t_span::Tuple{RealType,RealType}, dt::RealType, 
                        save_interval::Int = 25)
    println("="^70)
    println("AU-FUKAYA SIMULATION WITH RENKIN-CRONE & NORCAIN INJECTION")
    println("="^70)
    println("Time span: $(t_span[1]) to $(t_span[2]) seconds")
    println("Time step: dt = $dt")
    println("Save interval: every $save_interval steps")
    println()
    
    state = create_initial_state(dt)
    
    n_steps = Int(floor((t_span[2] - t_span[1]) / dt)) + 1
    n_save = ceil(Int, n_steps / save_interval)
    
    times = zeros(n_save)
    C_mean = zeros(n_save)
    C_std = zeros(n_save)
    qA_mean = zeros(n_save)
    qB_mean = zeros(n_save)
    HH2_values = zeros(n_save)
    ghost_signal = zeros(n_save)
    plucker_norm = zeros(n_save)
    
    C_regions = zeros(length(REGIONS), n_save)
    qA_regions = zeros(length(REGIONS), n_save)
    qB_regions = zeros(length(REGIONS), n_save)
    
    save_idx = 1
    times[save_idx] = state.t
    C_mean[save_idx] = mean(state.C)
    C_regions[:, save_idx] = state.C
    qA_mean[save_idx] = mean(state.qA)
    qA_regions[:, save_idx] = state.qA
    qB_mean[save_idx] = mean(state.qB)
    qB_regions[:, save_idx] = state.qB
    
    progress_step = max(1, div(n_steps, 50))
    
    for step in 1:n_steps-1
        state.t = t_span[1] + (step - 1) * dt
        state.step = step
        
        update_molecular_dynamics!(state, step)
        update_history!(state)
        compute_HH2!(state)
        
        state.Lax_L = build_lax_matrix(state)
        toda_flow_step!(state, dt)
        
        prolate_ratio = compute_prolate_ratio(state)
        prolate_feedback!(state, prolate_ratio)
        
        is_crisis, spike_ratio = detect_crisis(state)
        if is_crisis && state.crisis_countdown == 0
            @printf("  ⚠️ CRISIS at t=%.2f: HH² spike ratio = %.2f\n", state.t, spike_ratio)
            rees_blowup!(state)
        elseif state.crisis_countdown > 0
            state.crisis_countdown -= 1
        end
        
        plucker = compute_plucker_coordinates(state)
        ghost = compute_ghost_signal(state)
        
        if step % save_interval == 0 || step == n_steps-1
            save_idx += 1
            times[save_idx] = state.t
            C_mean[save_idx] = mean(state.C)
            C_std[save_idx] = std(state.C)
            C_regions[:, save_idx] = state.C
            qA_mean[save_idx] = mean(state.qA)
            qA_regions[:, save_idx] = state.qA
            qB_mean[save_idx] = mean(state.qB)
            qB_regions[:, save_idx] = state.qB
            HH2_values[save_idx] = state.HH2
            ghost_signal[save_idx] = ghost
            plucker_norm[save_idx] = norm(plucker)
        end
        
        if step % progress_step == 0
            percent = 100 * step / n_steps
            @printf("  Progress: %.1f%% | t=%.2f | C=%.3f | qB=%.3f | HH²=%.4f\n", 
                    percent, state.t, C_mean[save_idx], qB_mean[save_idx], state.HH2)
        end
    end
    
    println()
    println("="^70)
    println("SIMULATION COMPLETE")
    println("="^70)
    @printf("  Final mean consciousness: %.4f\n", C_mean[end])
    @printf("  Final mean norcain: %.4f\n", qB_mean[end])
    @printf("  Final HH²: %.4f\n", HH2_values[end])
    @printf("  Crisis events: %d\n", length(state.crisis_events))
    if !isempty(state.crisis_events)
        @printf("  Crisis times: %s\n", join(round.(state.crisis_events, digits=2), ", "))
    end
    
    return (times, C_mean, C_std, C_regions, qA_mean, qA_regions, 
            qB_mean, qB_regions, HH2_values, ghost_signal, plucker_norm, state)
end

# =============================================================================
# PART 9: VISUALIZATION
# =============================================================================

function create_dashboard(times, C_mean, C_std, C_regions, qA_mean, qB_mean, 
                          HH2_values, ghost_signal, dose_times, crisis_times, 
                          threshold, region_names)
    
    # Panel 1: Consciousness Dynamics
    p1 = plot(title="1. Consciousness Dynamics", xlabel="Time (s)", ylabel="Consciousness (C)")
    plot!(p1, times, C_mean, label="Mean C", lw=2, color=:blue)
    plot!(p1, times, C_mean .+ C_std, label="±1σ", lw=1, color=:blue, alpha=0.3, 
          fillrange=C_mean .- C_std, fillalpha=0.2)
    hline!(p1, [threshold], label="Threshold", lw=2, ls=:dash, color=:red)
    
    for dt in dose_times
        vline!(p1, [dt], label="Dose", lw=1, ls=:dot, color=:green, alpha=0.7)
    end
    
    for ct in crisis_times
        vline!(p1, [ct], label="Crisis", lw=2, ls=:dash, color=:orange, alpha=0.8)
    end
    
    # Panel 2: Norcain Concentration
    p2 = plot(title="2. Norcain Concentration", xlabel="Time (s)", ylabel="Norcain (qB)")
    plot!(p2, times, qB_mean, label="Mean qB", lw=2, color=:green)
    for dt in dose_times
        vline!(p2, [dt], label="Dose", lw=1, ls=:dot, color=:green, alpha=0.7)
    end
    
    # Panel 3: HH² Obstruction
    p3 = plot(title="3. HH² Obstruction (Crisis Detection)", xlabel="Time (s)", ylabel="HH²")
    plot!(p3, times, HH2_values, label="HH²", lw=2, color=:red)
    
    for ct in crisis_times
        vline!(p3, [ct], label="Crisis", lw=2, ls=:dash, color=:orange)
    end
    hline!(p3, [median(HH2_values[HH2_values .> 0]) * 2], label="2× Baseline", lw=1, ls=:dot, color=:gray)
    
    # Panel 4: Ghost Signal
    p4 = plot(title="4. Ghost Signal (Siegel Lock)", xlabel="Time (s)", ylabel="Ghost Signal")
    plot!(p4, times, ghost_signal, label="Ghost Signal", lw=2, color=:gold)
    hline!(p4, [0.6], label="Anchor", lw=1, ls=:dash, color=:gray)
    
    # Panel 5: Regional Consciousness
    p5 = plot(title="5. Regional Consciousness", xlabel="Time (s)", ylabel="Consciousness")
    colors_pal = palette(:tab10, length(region_names))
    for (i, region) in enumerate(region_names)
        plot!(p5, times, C_regions[i, :], label=region, lw=1.5, color=colors_pal[i])
    end
    hline!(p5, [threshold], label="Threshold", lw=2, ls=:dash, color=:red)
    
    # Panel 6: Phase Space
    p6 = plot(title="6. Phase Space", xlabel="Consciousness (C)", ylabel="dC/dt")
    dC = diff(C_mean) ./ diff(times)
    C_phase = C_mean[2:end]
    scatter(p6, C_phase, dC, label="Trajectory", markersize=2, color=:blue, alpha=0.5)
    scatter!([C_mean[1]], [0], label="Start", markersize=8, color=:green, marker=:star)
    scatter!([C_mean[end]], [0], label="End", markersize=8, color=:red, marker=:star)
    
    dashboard = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1400, 900),
                     titlefontsize=12, legendfontsize=8)
    
    return dashboard
end

function create_crisis_analysis_plot(times, HH2_values, crisis_times, dose_times)
    p = plot(title="Crisis Detection Analysis", xlabel="Time (s)", ylabel="HH² Obstruction")
    
    plot!(p, times, HH2_values, lw=2, color=:blue, label="HH²")
    
    window = 51
    baseline = [median(HH2_values[max(1,i-window÷2):min(end,i+window÷2)]) for i in 1:length(HH2_values)]
    plot!(p, times, baseline, lw=1.5, ls=:dash, color=:gray, label="Baseline (median)")
    
    threshold_line = 2 * baseline
    plot!(p, times, threshold_line, lw=1, ls=:dot, color=:orange, label="2× Baseline")
    
    for ct in crisis_times
        vline!(p, [ct], lw=3, color=:red, alpha=0.7, label=ct == first(crisis_times) ? "Crisis" : "")
    end
    
    for dt in dose_times
        vline!(p, [dt], lw=1, ls=:dash, color=:green, alpha=0.5, label=dt == first(dose_times) ? "Dose" : "")
    end
    
    above_threshold = HH2_values .> threshold_line
    for i in 1:length(times)-1
        if above_threshold[i] || above_threshold[i+1]
            plot!(p, [times[i], times[i+1]], [threshold_line[i], threshold_line[i+1]], 
                  fillrange=[threshold_line[i], threshold_line[i+1]], 
                  fillalpha=0.3, color=:red, label="")
        end
    end
    
    return p
end

function save_all_plots(times, C_mean, C_std, C_regions, qA_mean, qB_mean, 
                        HH2_values, ghost_signal, crisis_times, state)
    
    println("\n📊 Generating visualizations...")
    
    dashboard = create_dashboard(times, C_mean, C_std, C_regions, qA_mean, qB_mean,
                                  HH2_values, ghost_signal, DOSE_TIMES, crisis_times,
                                  THRESHOLD, REGION_NAMES)
    savefig(dashboard, "simulation_dashboard.png")
    println("  ✓ Saved: simulation_dashboard.png")
    
    crisis_plot = create_crisis_analysis_plot(times, HH2_values, crisis_times, DOSE_TIMES)
    savefig(crisis_plot, "crisis_analysis.png")
    println("  ✓ Saved: crisis_analysis.png")
    
    p_individual = plot(title="Individual Region Consciousness", xlabel="Time (s)", ylabel="Consciousness")
    colors_pal = palette(:tab10, length(REGION_NAMES))
    for i in 1:length(REGION_NAMES)
        plot!(p_individual, times, C_regions[i, :], label=REGION_NAMES[i], lw=1.5, color=colors_pal[i])
    end
    hline!(p_individual, [THRESHOLD], label="Threshold", lw=2, ls=:dash, color=:red)
    savefig(p_individual, "regional_consciousness.png")
    println("  ✓ Saved: regional_consciousness.png")
    
    p_summary = plot(title="Simulation Summary", layout=(2, 1), size=(1000, 600))
    plot!(p_summary[1], times, C_mean, label="Mean Consciousness", lw=2, color=:blue)
    hline!(p_summary[1], [THRESHOLD], label="Threshold", lw=1, ls=:dash, color=:red)
    for ct in crisis_times
        vline!(p_summary[1], [ct], lw=2, color=:orange, alpha=0.7)
    end
    ylabel!(p_summary[1], "Consciousness")
    
    plot!(p_summary[2], times, HH2_values, label="HH²", lw=2, color=:red)
    for ct in crisis_times
        vline!(p_summary[2], [ct], lw=2, color=:orange, alpha=0.7)
    end
    ylabel!(p_summary[2], "HH² Obstruction")
    xlabel!(p_summary[2], "Time (s)")
    
    savefig(p_summary, "simulation_summary.png")
    println("  ✓ Saved: simulation_summary.png")
end

# =============================================================================
# PART 10: MAIN
# =============================================================================

function main()
    println("\n" * "="^70)
    println("AU-FUKAYA INTEGRATED SIMULATION WITH PLOTTING")
    println("="^70)
    
    t_span = (0.0, 25.0)
    dt = 0.02
    save_interval = 25
    
    println("\n🚀 Starting simulation...\n")
    @time times, C_mean, C_std, C_regions, qA_mean, qA_regions, 
          qB_mean, qB_regions, HH2_values, ghost_signal, plucker_norm, state = 
          run_simulation(t_span, dt, save_interval)
    
    # Save data to CSV
    data = hcat(times, C_mean, C_std, qA_mean, qB_mean, HH2_values, ghost_signal)
    writedlm("simulation_output.csv", data, ',')
    println("\n  ✓ Saved: simulation_output.csv")
    
    # Generate plots
    save_all_plots(times, C_mean, C_std, C_regions, qA_mean, qB_mean, 
                  HH2_values, ghost_signal, state.crisis_events, state)
    
    println("\n" * "="^70)
    println("SIMULATION COMPLETE")
    println("="^70)
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
