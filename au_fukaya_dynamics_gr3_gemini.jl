# ============================================================================
# GR(3,5) TO GR(2,4) VIA HYPERPLANE SECTIONS - FULLY INTEGRATED & FIXED
# ============================================================================

using LinearAlgebra, SparseArrays, Printf, Random, Statistics
using Plots, DelimitedFiles

const RealType = Float64

# ============================================================================
# PART 0: PHYSICAL CONSTANTS
# ============================================================================

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

# Pre-compute Renkin-Crone factors
function renkin_crone_factor(radius_ratio::RealType)::RealType
    if radius_ratio >= 1.0
        return 0.0
    end
    return (1 - radius_ratio)^2 * (1 - 2.104*radius_ratio + 
           2.09*radius_ratio^3 - 0.95*radius_ratio^5)
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

# ============================================================================
# PART 1: PHYSICAL SIMULATION STATE (6-node brain model)
# ============================================================================

mutable struct PhysicalState
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

function create_physical_state(dt::RealType)::PhysicalState
    n_regions = length(REGIONS)
    qA = zeros(n_regions)
    qA_trap = zeros(n_regions)
    qB = zeros(n_regions)
    qB_trap = zeros(n_regions)
    C = ones(n_regions)
    qA[REGION_INDEX[:CA1sp]] = 3.5
    
    C_history = [copy(C) for _ in 1:3]
    qB_history = [copy(qB) for _ in 1:3]
    
    Lax_L = Matrix{RealType}(I, n_regions, n_regions)
    dose_applied = Dict(t_val => false for t_val in DOSE_TIMES)
    
    return PhysicalState(
        0.0, 0, dt, qA, qA_trap, qB, qB_trap, C, C_history, qB_history,
        copy(BASE_EDGE_WEIGHTS), Lax_L, 0.0, RealType[], 0, false, Float64[],
        dose_applied, Set([:CA1sp, :BLA])
    )
end

function compute_renkin_crone_rates(state::PhysicalState, t::RealType, region_idx::Int)
    region = REGIONS[region_idx]
    loopy = region in state.loopy_nodes
    
    qA_free_cur = state.qA[region_idx]
    qA_trap_cur = loopy ? state.qA_trap[region_idx] : 0.0
    qB_free_cur = state.qB[region_idx]
    qB_trap_cur = loopy ? state.qB_trap[region_idx] : 0.0
    C_cur = state.C[region_idx]
    
    lambda_A = log(2) / HALF_LIFE_A
    lambda_B = log(2) / HALF_LIFE_B
    inflow_A = 0.0
    inflow_B = 0.0
    
    for (src, tgt) in CORE_EDGES
        if tgt == region
            src_idx = REGION_INDEX[src]
            inflow_A += transition_rate((src, tgt), t, :A, state.edge_weights) * state.qA[src_idx] * state.dt
            inflow_B += transition_rate((src, tgt), t, :B, state.edge_weights) * state.qB[src_idx] * state.dt
        end
    end
    
    outflow_A = 0.0
    outflow_B = 0.0
    for (src, tgt) in CORE_EDGES
        if src == region
            outflow_A += transition_rate((src, tgt), t, :A, state.edge_weights) * qA_free_cur * state.dt
            outflow_B += transition_rate((src, tgt), t, :B, state.edge_weights) * qB_free_cur * state.dt
        end
    end
    
    dqA_free = inflow_A - outflow_A - qA_free_cur * lambda_A * state.dt
    dqB_free = inflow_B - outflow_B - qB_free_cur * lambda_B * state.dt
    dqA_trap = 0.0
    dqB_trap = 0.0
    
    if loopy
        alpha_in = 0.15
        alpha_out = 0.015
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

function apply_dose!(state::PhysicalState, t::RealType)
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

function update_physical_dynamics!(state::PhysicalState)
    t = state.t
    new_qA = copy(state.qA)
    new_qA_trap = copy(state.qA_trap)
    new_qB = copy(state.qB)
    new_qB_trap = copy(state.qB_trap)
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
    
    state.qA = new_qA
    state.qA_trap = new_qA_trap
    state.qB = new_qB
    state.qB_trap = new_qB_trap
    state.C = new_C
    apply_dose!(state, t)
end

function update_history!(state::PhysicalState)
    push!(state.C_history, copy(state.C))
    push!(state.qB_history, copy(state.qB))
    if length(state.C_history) > 100
        popfirst!(state.C_history)
        popfirst!(state.qB_history)
    end
end

function compute_HH2!(state::PhysicalState)
    if length(state.C_history) < 3
        state.HH2 = 0.0
        return
    end
    
    C_cur = state.C_history[end]
    C_prev = state.C_history[end-1]
    C_prev2 = state.C_history[end-2]
    dt_val = state.dt
    
    HH2 = norm((C_cur .- 2*C_prev .+ C_prev2) ./ (dt_val^2))
    
    comm = 0.0
    for i in 1:min(3, length(C_cur))
        for j in i+1:min(3, length(C_cur))
            comm += abs(C_cur[i] * state.qB_history[end][j] - C_cur[j] * state.qB_history[end][i])
        end
    end
    
    state.HH2 = HH2 + 0.3 * comm
    push!(state.HH2_history, state.HH2)
    if length(state.HH2_history) > 300
        popfirst!(state.HH2_history)
    end
end

function rees_blowup!(state::PhysicalState)
    @printf("  🔥 REES BLOW-UP at t=%.2f: Injecting norcain at ALL nodes\n", state.t)
    state.qB .+= 1.0
    state.qB .= clamp.(state.qB, 0.0, 6.0)
    push!(state.crisis_events, state.t)
    state.norcain_injected = true
    state.crisis_countdown = 20
end

# ============================================================================
# PART 2: GRASSMANNIAN GR(3,5) REPRESENTATION
# ============================================================================

struct GrassmannianGr35
    plucker::Vector{RealType}
    matrix::Matrix{RealType}
    
    function GrassmannianGr35(matrix::Matrix{RealType})
        @assert size(matrix) == (3, 5) "Need 3×5 matrix, got $(size(matrix))"
        plucker = compute_plucker_gr35(matrix)
        norm_val = norm(plucker)
        if norm_val > 0
            plucker = plucker ./ norm_val
        end
        return new(plucker, matrix)
    end
end

function compute_plucker_gr35(M::Matrix{RealType})::Vector{RealType}
    @assert size(M) == (3, 5) "Need 3×5 matrix, got $(size(M))"
    plucker = zeros(10)
    idx = 1
    for j in 1:5
        for k in j+1:5
            for l in k+1:5
                plucker[idx] = det(M[:, [j, k, l]])
                idx += 1
            end
        end
    end
    return plucker
end

function orthonormalize_rows(M::Matrix{RealType})::Matrix{RealType}
    Q, R = qr(M')
    Q_reduced = Q[:, 1:3]  # Dimensions: 5 × 3
    return Matrix(Q_reduced')  # Dimensions: 3 × 5
end

# ============================================================================
# PART 3: HYPERPLANE SECTIONS
# ============================================================================

struct HyperplaneC5
    normal::Vector{RealType}
    
    function HyperplaneC5(normal::Vector{RealType})
        @assert length(normal) == 5
        norm_val = norm(normal)
        if norm_val > 0
            normal = normal ./ norm_val
        end
        return new(normal)
    end
end

const STANDARD_HYPERPLANES = Dict(
    :A => HyperplaneC5([1.0, 0.0, 0.0, 0.0, 0.0]),
    :B => HyperplaneC5([0.0, 1.0, 0.0, 0.0, 0.0]),
    :C => HyperplaneC5([0.0, 0.0, 1.0, 0.0, 0.0]),
    :D => HyperplaneC5([0.0, 0.0, 0.0, 1.0, 0.0]),
    :generic => HyperplaneC5([1.0, 1.0, 1.0, 1.0, 1.0] ./ sqrt(5)),
    :crisis => HyperplaneC5([1.0, 2.0, 3.0, 4.0, 5.0] ./ sqrt(55)),
)

function nullspace_basis(A::Matrix{RealType})::Matrix{RealType}
    F = svd(A)
    rank_val = count(F.S .> 1e-10)
    if rank_val >= size(A, 2)
        return zeros(size(A, 2), 0)
    end
    return F.V[:, rank_val+1:end]
end

function project_to_gr24(gr35::GrassmannianGr35, H::HyperplaneC5)::Matrix{RealType}
    M = gr35.matrix
    n = H.normal
    
    Mn = M * n
    if norm(Mn) < 1e-10
        return zeros(2, 4)
    end
    
    u = Mn / norm(Mn)
    X = zeros(3, 2)
    e1 = [1.0, 0.0, 0.0]
    if abs(dot(e1, u)) > 0.9999
        e1 = [0.0, 1.0, 0.0]
    end
    v1 = e1 - dot(e1, u) * u
    v1 = v1 / norm(v1)
    X[:, 1] = v1
    X[:, 2] = cross(u, v1)
    
    H_basis = nullspace_basis(reshape(n, 1, 5))
    result = zeros(2, 4)
    
    if size(H_basis, 2) == 4
        for j in 1:2
            v = M' * X[:, j]
            result[j, :] = H_basis \ v
        end
    end
    return result
end

# ============================================================================
# PART 4: CRISIS DETECTION IN GR(3,5)
# ============================================================================

function detect_crisis_gr35(gr35::GrassmannianGr35, H::HyperplaneC5)::Tuple{Bool, Int, RealType}
    M = gr35.matrix
    n = H.normal
    
    U, S, Vt = svd(M)
    rank_val = count(S .> 1e-10)
    
    if rank_val < 3
        return true, 62, 0.0
    end
    
    # Vt is 5×5. Its first 3 rows perfectly form the 3×5 row space basis.
    row_basis = Vt[1:3, :]
    
    # Corrected Dimension Mapping:
    # row_basis * n yields (3×5) * (5×1) = (3×1) Vector
    # row_basis' * (row_basis * n) yields (5×3) * (3×1) = (5×1) Vector
    n_proj = row_basis' * (row_basis * n)
    n_orth = n - n_proj
    tangency_angle = norm(n_orth) / (norm(n) + 1e-10)
    
    if tangency_angle < 0.01
        return true, 62, tangency_angle
    end
    
    gr24 = project_to_gr24(gr35, H)
    if size(gr24, 1) == 2 && size(gr24, 2) == 4
        if LinearAlgebra.rank(gr24) < 2
            return true, 62, tangency_angle
        end
    end
    
    return false, 0, tangency_angle
end

# ============================================================================
# PART 5: DYNAMICAL SYSTEM ON GR(3,5)
# ============================================================================

mutable struct GrassmannianSystem
    gr35::GrassmannianGr35
    hyperplane::HyperplaneC5
    physical::PhysicalState
    gr35_history::Vector{GrassmannianGr35}
    crisis_events::Vector{Float64}
    coker_history::Vector{Int}
    Lax_L::Matrix{RealType}
    crisis_countdown::Int
end

function create_grassmannian_system(dt::RealType, sector::Symbol=:generic)::GrassmannianSystem
    phys = create_physical_state(dt)
    
    mean_C = mean(phys.C)
    mean_qA = mean(phys.qA)
    mean_qB = mean(phys.qB)
    
    M = zeros(RealType, 3, 5)
    M[1, 1] = mean_C; M[1, 2] = mean_qA; M[1, 3] = mean_qB; M[1, 4] = 1.0; M[1, 5] = 0.0
    M[2, 1] = 1.0 - mean_C; M[2, 2] = 1.0 - mean_qA; M[2, 3] = 1.0 - mean_qB; M[2, 4] = 0.0; M[2, 5] = 1.0
    M[3, 1] = mean_C * mean_qA; M[3, 2] = mean_C * mean_qB; M[3, 3] = mean_qA * mean_qB; M[3, 4] = 1.0; M[3, 5] = 1.0
    
    M = orthonormalize_rows(M)
    gr35 = GrassmannianGr35(M)
    H = get(STANDARD_HYPERPLANES, sector, STANDARD_HYPERPLANES[:generic])
    Lax_L = Matrix{RealType}(I, 5, 5)
    
    return GrassmannianSystem(gr35, H, phys, GrassmannianGr35[], Float64[], Int[], Lax_L, 0)
end

function update_grassmannian_from_physical!(sys::GrassmannianSystem)
    s = sys.physical
    mean_C = mean(s.C)
    mean_qA = mean(s.qA)
    mean_qB = mean(s.qB)
    
    M = zeros(RealType, 3, 5)
    M[1, 1] = mean_C; M[1, 2] = mean_qA; M[1, 3] = mean_qB; M[1, 4] = 1.0; M[1, 5] = 0.0
    M[2, 1] = 1.0 - mean_C; M[2, 2] = 1.0 - mean_qA; M[2, 3] = 1.0 - mean_qB; M[2, 4] = 0.0; M[2, 5] = 1.0
    M[3, 1] = mean_C * mean_qA; M[3, 2] = mean_C * mean_qB; M[3, 3] = mean_qA * mean_qB; M[3, 4] = 1.0; M[3, 5] = 1.0
    
    B = sys.Lax_L[1:3, 1:3]
    dM = B * M
    M = M + dM * s.dt
    M = orthonormalize_rows(M)
    
    sys.gr35 = GrassmannianGr35(M)
    push!(sys.gr35_history, sys.gr35)
end

function toda_flow_gr35!(sys::GrassmannianSystem, dt::RealType)
    L = sys.Lax_L
    n = size(L, 1)
    B = zeros(n, n)
    for i in 1:n-1
        B[i, i+1] = L[i, i+1]
        B[i+1, i] = -L[i, i+1]
    end
    
    dL = B * L - L * B
    damping = -0.1 * (L - diagm(diag(L)))
    drive = 0.5 * sin(2 * π * 8.0 * sys.physical.t) * I
    
    sys.Lax_L = sys.Lax_L + (dL + damping + drive) * dt
    sys.Lax_L = (sys.Lax_L + sys.Lax_L') / 2
end

# ============================================================================
# PART 6: MAIN SIMULATION
# ============================================================================

function run_gr35_simulation(t_span::Tuple{RealType,RealType}, dt::RealType, 
                             sector::Symbol=:generic, save_interval::Int=25)
    println("="^70)
    println("GR(3,5) → GR(2,4) SIMULATION | Sector: $sector")
    println("="^70)
    
    sys = create_grassmannian_system(dt, sector)
    phys = sys.physical
    
    n_steps = Int(floor((t_span[2] - t_span[1]) / dt)) + 1
    
    # Pre-allocating with strict safety boundaries
    times = Float64[]
    C_mean = Float64[]
    qB_mean = Float64[]
    HH2_values = Float64[]
    coker_dim = Int[]
    tangency_angle = Float64[]
    
    push!(times, phys.t)
    push!(C_mean, mean(phys.C))
    push!(qB_mean, mean(phys.qB))
    push!(HH2_values, phys.HH2)
    push!(coker_dim, 0)
    push!(tangency_angle, 1.0)
    
    progress_step = max(1, div(n_steps, 5))
    
    for step in 1:n_steps-1
        phys.t = t_span[1] + (step - 1) * dt
        phys.step = step
        
        update_physical_dynamics!(phys)
        update_history!(phys)
        compute_HH2!(phys)
        
        update_grassmannian_from_physical!(sys)
        toda_flow_gr35!(sys, dt)
        
        is_crisis, coker, tangency = detect_crisis_gr35(sys.gr35, sys.hyperplane)
        
        if is_crisis && sys.crisis_countdown == 0
            push!(sys.crisis_events, phys.t)
            @printf("  ⚠️ CRISIS at t=%.2f: coker=%d, tangency=%.4f\n", phys.t, coker, tangency)
            
            if coker == 62
                sectors_list = [:A, :B, :C, :D]
                new_sector = sectors_list[rand(1:4)]
                sys.hyperplane = STANDARD_HYPERPLANES[new_sector]
                @printf("     → Structural change: Switching to section %s\n", new_sector)
            end
            
            rees_blowup!(phys)
            sys.crisis_countdown = 25
        end
        
        if step % save_interval == 0 || step == n_steps-1
            push!(times, phys.t)
            push!(C_mean, mean(phys.C))
            push!(qB_mean, mean(phys.qB))
            push!(HH2_values, phys.HH2)
            push!(coker_dim, coker)
            push!(tangency_angle, tangency)
        end
        
        if step % progress_step == 0
            percent = 100 * step / n_steps
            @printf("  Progress: %.1f%% | t=%.2f | Mean C=%.3f\n", percent, phys.t, C_mean[end])
        end
        
        if sys.crisis_countdown > 0
            sys.crisis_countdown -= 1
        end
    end
    
    return (times, C_mean, qB_mean, HH2_values, coker_dim, tangency_angle, sys)
end

# ============================================================================
# PART 7: RUNTIME EXECUTIVE
# ============================================================================

function main()
    sectors = [:A, :B, :C, :D, :generic]
    
    for sector in sectors
        println("\n🔬 Starting Execution Domain: $sector")
        println("-"^50)
        
        times, C_mean, qB_mean, HH2_values, coker_dim, tangency_angle, sys = 
            run_gr35_simulation((0.0, 25.0), 0.02, sector, 25)
        
        p = plot(times, C_mean, lw=2, label="Consciousness", title="Sector $sector", xlabel="Time (s)", ylabel="Value")
        norm_HH2 = HH2_values ./ (maximum(HH2_values) + 1e-10)
        plot!(p, times, norm_HH2, lw=2, label="HH² (normalized)")
        
        for ct in sys.crisis_events
            vline!(p, [ct], lw=1, ls=:dash, color=:red, alpha=0.6, label="")
        end
        
        savefig(p, "gr35_sector_$(sector).png")
        
        data = hcat(times, C_mean, qB_mean, HH2_values, coker_dim, tangency_angle)
        writedlm("gr35_data_$(sector).csv", data, ',')
        
        println("  ✓ Output generated successfully.")
    end
    println("\n" * "="^70)
    println("ALL MODULES COMPLETED SUCCESSFULLY")
    println("="^70)
end

main()
