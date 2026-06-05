# ============================================================================
# GR(3,5) → GR(2,4) VIA HYPERPLANE SECTIONS - FIXED
# ============================================================================
# Key fixes from original:
#   1. orthonormalize_rows: Matrix(F.Q) to materialise QRPackedQ before slicing
#   2. detect_crisis_gr35: Matrix(Vt[...]) to materialise Adjoint before multiply
#   3. nullspace_basis: use F.V (not Vt) for right singular vectors
# ============================================================================

using LinearAlgebra, SparseArrays, Printf, Random, Statistics

const RealType = Float64

# ============================================================================
# PART 0: PHYSICAL CONSTANTS
# ============================================================================

const HALF_LIFE_A = 6.0
const HALF_LIFE_B = 3.0
const EC50_A      = 0.2
const EC50_B      = 0.2
const HILL_A      = 2.0
const HILL_B      = 2.0
const ALPHA       = 3.0
const BETA        = 1.0
const THRESHOLD   = 0.2

const DOSE_TIMES  = [2.5, 6.0, 9.5]
const DOSE_AMOUNT = 4.0

const MOLECULE_RADIUS   = Dict(:A => 0.5e-9, :B => 0.8e-9)
const PORE_RADIUS       = 1.0e-9
const DIFFUSIVITY       = Dict(:A => 5e-10,  :B => 8e-10)
const MEMBRANE_THICKNESS = 1e-6
const BASE_FLOW         = Dict(:A => 4.0,    :B => 6.0)

const REGIONS      = [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY]
const REGION_INDEX = Dict(r => i for (i, r) in enumerate(REGIONS))
const REGION_NAMES = ["BLA", "CA1sp", "HPF", "HY", "LA", "sAMY"]

const CORE_EDGES = [
    (:CA1sp,:HPF), (:HPF,:CA1sp), (:BLA,:LA),   (:LA,:BLA),
    (:BLA,:sAMY),  (:sAMY,:BLA),  (:CA1sp,:sAMY),(:sAMY,:CA1sp),
    (:HPF,:BLA),   (:HPF,:sAMY),  (:LA,:sAMY),  (:sAMY,:LA),
    (:HY,:sAMY),   (:sAMY,:HY),
    (:BLA,:BLA),   (:CA1sp,:CA1sp),(:HPF,:HPF),  (:HY,:HY),
    (:LA,:LA),     (:sAMY,:sAMY)
]

const BASE_EDGE_WEIGHTS = Dict(
    (:CA1sp,:HPF) => 15.0, (:HPF,:CA1sp) => 5.0,
    (:BLA,:LA)    =>  8.0, (:LA,:BLA)    => 6.0,
    (:BLA,:sAMY)  => 12.0, (:sAMY,:BLA)  =>10.0,
    (:CA1sp,:sAMY)=>  9.0, (:sAMY,:CA1sp)=> 7.0,
    (:HPF,:BLA)   =>  4.0, (:HPF,:sAMY)  => 6.0,
    (:LA,:sAMY)   =>  5.0, (:sAMY,:LA)   => 5.0,
    (:HY,:sAMY)   =>  3.0, (:sAMY,:HY)   => 8.0,
    (:BLA,:BLA)   =>  2.0, (:CA1sp,:CA1sp)=>2.0,
    (:HPF,:HPF)   =>  2.0, (:HY,:HY)     => 2.0,
    (:LA,:LA)     =>  2.0, (:sAMY,:sAMY) => 2.0,
)

function renkin_crone_factor(radius_ratio::RealType)::RealType
    radius_ratio >= 1.0 && return 0.0
    return (1 - radius_ratio)^2 *
           (1 - 2.104*radius_ratio + 2.09*radius_ratio^3 - 0.95*radius_ratio^5)
end

const RENKIN_FACTOR_A = renkin_crone_factor(MOLECULE_RADIUS[:A] / PORE_RADIUS)
const RENKIN_FACTOR_B = renkin_crone_factor(MOLECULE_RADIUS[:B] / PORE_RADIUS)

function flow_rate(edge, t, molecule, edge_weights)
    return BASE_FLOW[molecule] * get(edge_weights, edge, 1.0) *
           (1.0 + 0.3 * sin(2π * t))
end

function transition_rate(edge, t, molecule, edge_weights)
    flow   = flow_rate(edge, t, molecule, edge_weights)
    factor = (molecule == :A) ? RENKIN_FACTOR_A : RENKIN_FACTOR_B
    return flow * (DIFFUSIVITY[molecule] / MEMBRANE_THICKNESS) * factor * 300.0
end

# ============================================================================
# PART 1: PHYSICAL SIMULATION STATE
# ============================================================================

mutable struct PhysicalState
    t                ::RealType
    step             ::Int
    dt               ::RealType
    qA               ::Vector{RealType}
    qA_trap          ::Vector{RealType}
    qB               ::Vector{RealType}
    qB_trap          ::Vector{RealType}
    C                ::Vector{RealType}
    C_history        ::Vector{Vector{RealType}}
    qB_history       ::Vector{Vector{RealType}}
    edge_weights     ::Dict{Tuple{Symbol,Symbol}, RealType}
    Lax_L            ::Matrix{RealType}
    HH2              ::RealType
    HH2_history      ::Vector{RealType}
    crisis_countdown ::Int
    norcain_injected ::Bool
    crisis_events    ::Vector{Float64}
    dose_applied     ::Dict{Float64, Bool}
    loopy_nodes      ::Set{Symbol}
end

function create_physical_state(dt::RealType)::PhysicalState
    n = length(REGIONS)
    qA = zeros(n); qA[REGION_INDEX[:CA1sp]] = 3.5
    PhysicalState(0.0, 0, dt,
        qA, zeros(n), zeros(n), zeros(n), ones(n),
        [ones(n) for _ in 1:3], [zeros(n) for _ in 1:3],
        copy(BASE_EDGE_WEIGHTS),
        Matrix{RealType}(I, n, n),
        0.0, RealType[], 0, false, Float64[],
        Dict(t => false for t in DOSE_TIMES),
        Set([:CA1sp, :BLA]))
end

function compute_renkin_crone_rates(state::PhysicalState, t::RealType, ri::Int)
    region  = REGIONS[ri]
    loopy   = region in state.loopy_nodes
    qA_f    = state.qA[ri];  qA_t = loopy ? state.qA_trap[ri] : 0.0
    qB_f    = state.qB[ri];  qB_t = loopy ? state.qB_trap[ri] : 0.0
    C_cur   = state.C[ri]
    λA      = log(2) / HALF_LIFE_A
    λB      = log(2) / HALF_LIFE_B

    in_A = out_A = in_B = out_B = 0.0
    for (src, tgt) in CORE_EDGES
        if tgt == region
            si = REGION_INDEX[src]
            in_A += transition_rate((src,tgt), t, :A, state.edge_weights) * state.qA[si] * state.dt
            in_B += transition_rate((src,tgt), t, :B, state.edge_weights) * state.qB[si] * state.dt
        end
        if src == region
            out_A += transition_rate((src,tgt), t, :A, state.edge_weights) * qA_f * state.dt
            out_B += transition_rate((src,tgt), t, :B, state.edge_weights) * qB_f * state.dt
        end
    end

    dqA_f = in_A - out_A - qA_f * λA * state.dt
    dqB_f = in_B - out_B - qB_f * λB * state.dt
    dqA_t = dqB_t = 0.0
    if loopy
        αi, αo = 0.15, 0.015
        dqA_f += (-αi*qA_f + αo*qA_t) * state.dt
        dqB_f += (-αi*qB_f + αo*qB_t) * state.dt
        dqA_t  = (αi*qA_f - αo*qA_t)  * state.dt
        dqB_t  = (αi*qB_f - αo*qB_t)  * state.dt
    end

    act  = ALPHA * (1-C_cur) * (qB_f^HILL_B) / (EC50_B^HILL_B + qB_f^HILL_B)
    damp = BETA  *   C_cur   * (qA_f^HILL_A) / (EC50_A^HILL_A + qA_f^HILL_A)
    osc  = 0.1 * sin(2π * 0.6 * t) * (1 - C_cur)
    dC   = (act - damp + osc) * state.dt

    return dqA_f, dqA_t, dqB_f, dqB_t, dC
end

function apply_dose!(state::PhysicalState, t::RealType)
    for dt_val in DOSE_TIMES
        if abs(t - dt_val) < state.dt && !state.dose_applied[dt_val]
            state.dose_applied[dt_val] = true
            i = REGION_INDEX[:CA1sp]
            state.qB[i] = clamp(state.qB[i] + DOSE_AMOUNT, 0.0, 6.0)
            @printf("  💉 DOSE at t=%.2f: +%.1f norcain at CA1sp\n", t, DOSE_AMOUNT)
            state.norcain_injected = true
        end
    end
end

function update_physical_dynamics!(state::PhysicalState)
    t    = state.t
    n    = length(REGIONS)
    nqA  = copy(state.qA);  nqAt = copy(state.qA_trap)
    nqB  = copy(state.qB);  nqBt = copy(state.qB_trap)
    nC   = copy(state.C)
    for (idx, region) in enumerate(REGIONS)
        da_f, da_t, db_f, db_t, dC = compute_renkin_crone_rates(state, t, idx)
        nqA[idx]  = clamp(state.qA[idx]      + da_f, 0.0, 5.0)
        nqB[idx]  = clamp(state.qB[idx]      + db_f, 0.0, 6.0)
        nC[idx]   = clamp(state.C[idx]        + dC,   0.0, 1.0)
        if region in state.loopy_nodes
            nqAt[idx] = clamp(state.qA_trap[idx] + da_t, 0.0, 5.0)
            nqBt[idx] = clamp(state.qB_trap[idx] + db_t, 0.0, 6.0)
        end
    end
    state.qA = nqA;  state.qA_trap = nqAt
    state.qB = nqB;  state.qB_trap = nqBt
    state.C  = nC
    apply_dose!(state, t)
end

function update_history!(state::PhysicalState)
    push!(state.C_history,  copy(state.C))
    push!(state.qB_history, copy(state.qB))
    length(state.C_history)  > 100 && popfirst!(state.C_history)
    length(state.qB_history) > 100 && popfirst!(state.qB_history)
end

function compute_HH2!(state::PhysicalState)
    length(state.C_history) < 3 && (state.HH2 = 0.0; return)
    C0, C1, C2 = state.C_history[end], state.C_history[end-1], state.C_history[end-2]
    hh = norm((C0 .- 2C1 .+ C2) ./ (state.dt^2))
    comm = 0.0
    for i in 1:min(3,length(C0)), j in i+1:min(3,length(C0))
        comm += abs(C0[i]*state.qB_history[end][j] - C0[j]*state.qB_history[end][i])
    end
    state.HH2 = hh + 0.3*comm
    push!(state.HH2_history, state.HH2)
    length(state.HH2_history) > 300 && popfirst!(state.HH2_history)
end

function detect_crisis_physical(state::PhysicalState)::Tuple{Bool,RealType}
    length(state.HH2_history) < 50 && return false, 0.0
    base = state.HH2_history[max(1,end-100):end]
    med  = median(base)
    cur  = state.HH2
    (med > 0 && cur > med*2.0) && return true, cur/med
    cur > 1.0 && return true, cur
    return false, 0.0
end

function rees_blowup!(state::PhysicalState)
    @printf("  🔥 REES BLOW-UP at t=%.2f: Injecting norcain at ALL nodes\n", state.t)
    state.qB .= clamp.(state.qB .+ 1.0, 0.0, 6.0)
    push!(state.crisis_events, state.t)
    state.norcain_injected = true
    state.crisis_countdown = 20
end

function toda_flow_step!(state::PhysicalState, dt::RealType)
    L = state.Lax_L
    n = size(L, 1)
    B = zeros(n, n)
    for i in 1:n-1
        B[i,i+1] =  L[i,i+1]
        B[i+1,i] = -L[i,i+1]
    end
    dL = B*L - L*B - 0.1*(L - diagm(diag(L))) +
         0.5*sin(2π*8.0*state.t)*Matrix{RealType}(I,n,n)
    state.Lax_L += dL * dt
    state.Lax_L  = (state.Lax_L + state.Lax_L') / 2
end

# ============================================================================
# PART 2: GRASSMANNIAN GR(3,5)
# ============================================================================

struct GrassmannianGr35
    plucker ::Vector{RealType}   # length 10
    matrix  ::Matrix{RealType}   # 3×5

    function GrassmannianGr35(M::Matrix{RealType})
        @assert size(M) == (3,5) "Need 3×5 matrix, got $(size(M))"
        p     = compute_plucker_gr35(M)
        nrm   = norm(p)
        nrm > 0 && (p = p ./ nrm)
        new(p, M)
    end
end

function compute_plucker_gr35(M::Matrix{RealType})::Vector{RealType}
    p   = zeros(10)
    idx = 1
    for j in 1:5, k in j+1:5, l in k+1:5
        p[idx] = det(M[:, [j,k,l]])
        idx   += 1
    end
    return p
end

# ── FIX 1: orthonormalize_rows ─────────────────────────────────────────────
# Julia's qr() returns a QRPackedQ lazy object; slicing it gives wrong shapes.
# Force materialisation with Matrix(F.Q) before indexing.
function orthonormalize_rows(M::Matrix{RealType})::Matrix{RealType}
    # M is 3×5; we want 3 orthonormal rows spanning the same row space.
    # QR of M^T (5×3): Q is 5×k, take first 3 columns → 5×3, transpose → 3×5
    F      = qr(M')                   # factorise 5×3 matrix
    Q_full = Matrix(F.Q)              # materialise: avoids QRPackedQ slice bugs
    # Q_full is 5×5 (full) or 5×3 (thin depending on Julia version)
    # Either way take first 3 columns and transpose
    ncols  = min(3, size(Q_full, 2))
    result = Matrix(Q_full[:, 1:ncols]')   # (ncols×5) → ensure concrete Matrix
    # If we only got fewer than 3 cols, pad with SVD fallback
    if ncols < 3
        _, _, Vt = svd(M)
        result = Matrix(Vt[1:3, :])   # materialise Adjoint → concrete 3×5
    end
    @assert size(result) == (3,5) "orthonormalize_rows returned $(size(result)), expected (3,5)"
    return result
end

# ============================================================================
# PART 3: HYPERPLANE SECTIONS
# ============================================================================

struct HyperplaneC5
    normal::Vector{RealType}
    function HyperplaneC5(n::Vector{RealType})
        @assert length(n) == 5
        nrm = norm(n)
        new(nrm > 0 ? n ./ nrm : n)
    end
end

const STANDARD_HYPERPLANES = Dict(
    :A       => HyperplaneC5([1.0,0,0,0,0]),
    :B       => HyperplaneC5([0,1.0,0,0,0]),
    :C       => HyperplaneC5([0,0,1.0,0,0]),
    :D       => HyperplaneC5([0,0,0,1.0,0]),
    :generic => HyperplaneC5(fill(1/sqrt(5), 5)),
    :crisis  => HyperplaneC5([1,2,3,4,5.0] ./ sqrt(55)),
)

# ============================================================================
# ANALYTIC PLÜCKER COORDINATES (matching Python's compute_plucker_trajectory)
# ============================================================================
# These use the pharmacodynamic state directly — no SVD, no 3-plane geometry.
# Formula from BALBc_Opiate_Norcain.py:
#   p12 = C_mean / (1 + qA_mean/EC50_A)         ← consciousness / opiate saturation
#   p13 = qB_mean / (1 + qB_mean/EC50_B)        ← norcain saturation
#   p14 = (1 - C_mean) * exp(-λ_A * t)           ← unconscious × opiate decay
#   p23 = C_mean * exp(-λ_B * t)                 ← conscious × norcain decay
#   p24 = qA_mean * exp(-λ_A * t)               ← opiate × decay
#   p34 = qB_mean * exp(-λ_B * t)               ← norcain × decay
# The jumps at dose times come from qB_mean spiking → p13, p24, p34 jump.

function compute_analytic_plucker(state::PhysicalState)::Vector{RealType}
    C_m  = mean(state.C)
    qA_m = mean(state.qA)
    qB_m = mean(state.qB)
    t    = state.t
    λA   = log(2) / HALF_LIFE_A
    λB   = log(2) / HALF_LIFE_B

    p12 = C_m  / (1.0 + qA_m / EC50_A + 1e-10)
    p13 = qB_m / (1.0 + qB_m / EC50_B + 1e-10)
    p14 = (1.0 - C_m) * exp(-λA * t)
    p23 = C_m  * exp(-λB * t)
    p24 = qA_m * exp(-λA * t)
    p34 = qB_m * exp(-λB * t)

    p   = [p12, p13, p14, p23, p24, p34]
    nrm = norm(p)
    nrm > 0 && (p ./= nrm)
    return p
end

function nullspace_basis(A::Matrix{RealType})::Matrix{RealType}
    F    = svd(A)
    rnk  = count(F.S .> 1e-10)
    rnk >= size(A,2) && return zeros(size(A,2), 0)
    # F.V is 5×5 (or n×n); null vectors = columns beyond rank
    V_full = Matrix(F.V)   # materialise
    return V_full[:, rnk+1:end]
end

function project_to_gr24(gr35::GrassmannianGr35, H::HyperplaneC5)::Matrix{RealType}
    # Correct approach: find the intersection V ∩ H as a 2-plane.
    # V is the 3-plane spanned by rows of M (3×5).
    # H = {x ∈ R^5 : n·x = 0}.
    #
    # A vector in V has the form M^T α for α ∈ R^3.
    # For this to lie in H: n · (M^T α) = 0  ↔  (Mn) · α = 0.
    # So α must be in null(Mn^T) = the 2D null space of the 3-vector Mn.
    #
    # Basis of V∩H: {M^T α₁, M^T α₂} where {α₁,α₂} = null(Mn^T).
    # Express in H-coordinates: H has an orthonormal basis of 4 vectors
    # orthogonal to n. Project M^T αᵢ onto H using those 4 basis vectors.

    M  = gr35.matrix   # 3×5
    n  = H.normal      # length-5

    # Step 1: Mn = M*n is a 3-vector; find its null space (2D)
    Mn = M * n         # (3,)
    if norm(Mn) < 1e-12
        # n is perpendicular to all rows of M → V ⊂ H → projection undefined
        return zeros(2, 4)
    end

    # Two orthogonal vectors in null(Mn):
    # Any vector perpendicular to Mn in R^3
    Mn_hat = Mn / norm(Mn)
    e1 = abs(Mn_hat[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    a1 = e1 - dot(e1, Mn_hat) * Mn_hat;  a1 ./= norm(a1)
    a2 = cross(Mn_hat, a1);               a2 ./= norm(a2)
    # a1, a2 span null(Mn^T) in R^3

    # Step 2: corresponding vectors in V∩H ⊂ R^5
    v1 = M' * a1   # (5,) — lies in V and in H (since n·v1 = n·M'a1 = (Mn)·a1 = 0)
    v2 = M' * a2

    # Step 3: build orthonormal basis of H (4 vectors in R^5, all ⊥ n)
    # Build directly via Gram-Schmidt on the 5 standard basis vectors,
    # projecting out the n direction. This avoids SVD size issues.
    n_hat   = n / norm(n)
    H_vecs  = Vector{Float64}[]
    for i in 1:5
        e = zeros(5); e[i] = 1.0
        e .-= dot(e, n_hat) * n_hat   # project out n component
        norm(e) < 1e-10 && continue   # skip if e was parallel to n
        e ./= norm(e)
        # Orthogonalise against already-found vectors
        for u in H_vecs
            e .-= dot(e, u) * u
        end
        norm(e) < 1e-10 && continue
        e ./= norm(e)
        push!(H_vecs, e)
        length(H_vecs) == 4 && break
    end
    length(H_vecs) < 4 && return zeros(2, 4)
    H_basis = hcat(H_vecs...)   # 5×4, each column ⊥ n

    # Step 4: express v1, v2 in H-coordinates (4D)
    c1 = H_basis' * v1   # (4,)
    c2 = H_basis' * v2   # (4,)

    # The 2-plane in Gr(2,4) is represented by the 2×4 matrix [c1; c2]
    result = Matrix{RealType}([c1'; c2'])
    return result
end

# ============================================================================
# PART 4: CRISIS DETECTION — FIXED
# ============================================================================

function detect_crisis_gr35(gr35::GrassmannianGr35,
                              H::HyperplaneC5)::Tuple{Bool,Int,RealType}
    # Crisis = the projection V → V∩H drops below 2D.
    # This happens when dim(V∩H) < 2, i.e., rank of the projected 2×4 matrix < 2.
    # Equivalently: Mn = M*n has rank 0 (V⊂H) or dim null(Mn) < 2 (V nearly ⊂H⊥).
    #
    # We use the norm of Mn as the tangency signal:
    #   norm(Mn) ≈ 0 → V nearly inside H → crisis (projection degenerates)
    #   norm(Mn) large → V transverse to H → healthy projection

    M  = gr35.matrix   # 3×5
    n  = H.normal      # length-5

    Mn       = M * n          # (3,)
    tangency = norm(Mn)       # large = transverse, small = tangent/crisis

    # Crisis when Mn is nearly zero (V tangent to H)
    # Threshold: if ||Mn|| < 0.05 the intersection is nearly degenerate
    if tangency < 0.05
        return true, 62, tangency
    end

    # Secondary check: compute projection and verify rank
    gr24 = project_to_gr24(gr35, H)
    if size(gr24) == (2,4)
        S = svd(gr24).S
        rnk = count(S .> 1e-8 * maximum(S))
        rnk < 2 && return true, 62, tangency
    end

    return false, 0, tangency
end

# ============================================================================
# PART 5: DYNAMICAL SYSTEM ON GR(3,5)
# ============================================================================

mutable struct GrassmannianSystem
    gr35              ::GrassmannianGr35
    hyperplane        ::HyperplaneC5
    physical          ::PhysicalState
    gr35_history      ::Vector{GrassmannianGr35}
    plucker_history   ::Vector{Vector{RealType}}  # pharmacodynamic embedding (Panel 9)
    geo_gr24_history  ::Vector{Vector{RealType}}  # geometric Gr(2,4) projection (singularity tracking)
    crisis_events     ::Vector{Float64}
    coker_history     ::Vector{Int}
    Lax_L             ::Matrix{RealType}
    crisis_countdown  ::Int
end

function create_grassmannian_system(dt::RealType, sector::Symbol=:generic)
    phys = create_physical_state(dt)
    mC   = mean(phys.C); mqA = mean(phys.qA); mqB = mean(phys.qB)

    M = Matrix{RealType}([
        mC         mqA        mqB        1.0  0.0;
        1.0-mC     1.0-mqA    1.0-mqB    0.0  1.0;
        mC*mqA     mC*mqB     mqA*mqB    1.0  1.0
    ])
    M = orthonormalize_rows(M)

    GrassmannianSystem(
        GrassmannianGr35(M),
        get(STANDARD_HYPERPLANES, sector, STANDARD_HYPERPLANES[:generic]),
        phys,
        GrassmannianGr35[], Vector{RealType}[], Vector{RealType}[], Float64[], Int[],
        Matrix{RealType}(I, 5, 5), 0)
end

function update_grassmannian_from_physical!(sys::GrassmannianSystem)
    s    = sys.physical
    t_n  = clamp(s.t / 25.0, 0.0, 1.0)   # normalised time ∈ [0,1]
    qB_mean = mean(s.qB)

    # Use 3 pharmacologically distinct nodes as the 3 rows of M.
    # These are the nodes most relevant to the opioid-norcain dynamics:
    #   sAMY (index 6): primary opiate target
    #   CA1sp (index 2): norcain injection site  
    #   HPF (index 3): hippocampal relay
    # Each row: [C_node, qA_node, qB_node, t_norm, qB_mean]
    # Dose events → qB spike at CA1sp → immediate jump in row 2 → Plücker jumps
    i_sAMY  = REGION_INDEX[:sAMY]
    i_CA1sp = REGION_INDEX[:CA1sp]
    i_HPF   = REGION_INDEX[:HPF]

    row1 = [s.C[i_sAMY],  s.qA[i_sAMY],  s.qB[i_sAMY],  t_n,     qB_mean]
    row2 = [s.C[i_CA1sp], s.qA[i_CA1sp], s.qB[i_CA1sp], 1.0-t_n, s.qA[i_sAMY]]
    row3 = [s.C[i_HPF],   s.qA[i_HPF],   s.qB[i_HPF],   t_n*0.5, s.qB[i_sAMY]]

    M = Matrix{RealType}([row1'; row2'; row3'])

    # Apply Toda flow perturbation
    B  = sys.Lax_L[1:3, 1:3]
    M  = M + B * M * s.dt
    M  = orthonormalize_rows(M)
    sys.gr35 = GrassmannianGr35(M)
    push!(sys.gr35_history, sys.gr35)

    # PATH A: pharmacodynamic embedding (Panel 9 style, matches Python)
    push!(sys.plucker_history, compute_analytic_plucker(sys.physical))

    # PATH B: geometric Gr(2,4) projection (singularity / exceptional divisor tracking)
    # This is the rigorous algebraic geometry path.
    # The projection V ∩ H gives the true Gr(2,4) Plücker coords.
    # When this projection degenerates (rank < 2), a singularity is detected:
    #   - minor_bot → 0: Schubert wall crossing (stratum 2)
    #   - coker = 62:    exceptional divisor (62-class obstruction)
    #   - surgery fires: Picard-Lefschetz redistribution
    geo24 = project_to_gr24(sys.gr35, sys.hyperplane)
    if size(geo24) == (2,4)
        push!(sys.geo_gr24_history, compute_plucker_gr24(geo24))
    else
        # Degenerate projection → singularity, record zero vector as sentinel
        push!(sys.geo_gr24_history, zeros(6))
    end
end

function toda_flow_gr35!(sys::GrassmannianSystem, dt::RealType)
    L  = sys.Lax_L
    n  = size(L, 1)
    B  = zeros(n, n)
    for i in 1:n-1
        B[i,i+1] =  L[i,i+1]
        B[i+1,i] = -L[i,i+1]
    end
    dL = B*L - L*B - 0.1*(L - diagm(diag(L))) +
         0.5*sin(2π*8.0*sys.physical.t)*Matrix{RealType}(I,n,n)
    sys.Lax_L += dL * dt
    sys.Lax_L  = (sys.Lax_L + sys.Lax_L') / 2
end

# ============================================================================
# PART 6: MAIN SIMULATION LOOP
# ============================================================================

function run_gr35_simulation(t_span::Tuple{RealType,RealType}, dt::RealType,
                              sector::Symbol=:generic, save_interval::Int=25)
    println("="^70)
    println("GR(3,5) → GR(2,4) SIMULATION  |  sector=$sector")
    println("="^70)

    sys  = create_grassmannian_system(dt, sector)
    phys = sys.physical

    n_steps = Int(floor((t_span[2] - t_span[1]) / dt)) + 1
    n_save  = ceil(Int, n_steps / save_interval) + 1

    times           = zeros(n_save)
    C_mean          = zeros(n_save)
    qB_mean         = zeros(n_save)
    HH2_vals        = zeros(n_save)
    coker_dim       = zeros(Int, n_save)
    tangency_angle  = zeros(n_save)

    save_idx   = 1
    times[1]   = phys.t
    C_mean[1]  = mean(phys.C)
    qB_mean[1] = mean(phys.qB)

    prog_step = max(1, div(n_steps, 20))

    for step in 1:n_steps-1
        phys.t    = t_span[1] + (step-1) * dt
        phys.step = step

        update_physical_dynamics!(phys)
        update_history!(phys)
        compute_HH2!(phys)
        update_grassmannian_from_physical!(sys)
        toda_flow_gr35!(sys, dt)

        is_crisis, coker, tangency = detect_crisis_gr35(sys.gr35, sys.hyperplane)
        push!(sys.coker_history, coker)

        if is_crisis
            push!(sys.crisis_events, phys.t)
            @printf("  ⚠  CRISIS at t=%.2f: coker=%d, tangency=%.4f\n",
                    phys.t, coker, tangency)
            if coker == 62
                sectors_list = [:A, :B, :C, :D]
                new_sector   = sectors_list[rand(1:4)]
                sys.hyperplane = STANDARD_HYPERPLANES[new_sector]
                @printf("     → Switching to sector %s\n", new_sector)
            end
            rees_blowup!(phys)
            sys.crisis_countdown = 20
        end
        sys.crisis_countdown > 0 && (sys.crisis_countdown -= 1)

        if step % save_interval == 0 || step == n_steps-1
            save_idx  = min(save_idx + 1, n_save)
            times[save_idx]          = phys.t
            C_mean[save_idx]         = mean(phys.C)
            qB_mean[save_idx]        = mean(phys.qB)
            HH2_vals[save_idx]       = phys.HH2
            coker_dim[save_idx]      = coker
            tangency_angle[save_idx] = tangency
        end

        step % prog_step == 0 && @printf(
            "  %.0f%%  t=%.2f  C=%.3f  HH²=%.3f  coker=%d\n",
            100*step/n_steps, phys.t,
            mean(phys.C), phys.HH2, coker)
    end

    valid = 1:save_idx
    println()
    println("="^70)
    @printf("  Crisis events : %d\n", length(sys.crisis_events))
    @printf("  Final C       : %.4f\n", C_mean[save_idx])
    @printf("  Final HH²     : %.4f\n", HH2_vals[save_idx])
    println("="^70)

    return (times[valid], C_mean[valid], qB_mean[valid],
            HH2_vals[valid], coker_dim[valid], tangency_angle[valid], sys)
end


# ============================================================================
# PART 8: PLÜCKER TRAJECTORY PLOTS
# ============================================================================


# ============================================================================
# PART 8: PLÜCKER TRAJECTORY PLOTS
# ============================================================================
# Four panels per sector:
#   1. Phase portrait: p12 vs p34 coloured by time
#   2. Klein quadric residual: |p12·p34 - p13·p24 + p14·p23| (should ≈ 0)
#   3. Gr(3,5) PCA projection: 10-dim Plücker → 2D via SVD (no external packages)
#   4. Time series: p12(t), p34(t), HH²(t)
# ============================================================================

using Plots

"""Compute Gr(2,4) Plücker coordinates from a 2×4 matrix (6 values)."""
function compute_plucker_gr24(M::Matrix{RealType})::Vector{RealType}
    @assert size(M) == (2,4) "Need 2×4 matrix, got $(size(M))"
    p   = zeros(6)
    idx = 1
    for j in 1:4, k in j+1:4
        p[idx] = det(M[:, [j,k]])
        idx   += 1
    end
    nrm = norm(p); nrm > 0 && (p ./= nrm)
    return p
end

"""Extract Gr(3,5) and projected Gr(2,4) Plücker trajectories from a simulation."""
function extract_plucker_trajectory(sys::GrassmannianSystem, n_points::Int)
    n        = min(n_points, length(sys.gr35_history))
    step     = max(1, length(sys.gr35_history) ÷ n)
    indices  = 1:step:length(sys.gr35_history)

    gr35_traj = [sys.gr35_history[i].plucker for i in indices]

    gr24_traj = Vector{Float64}[]
    for i in indices
        gr24 = project_to_gr24(sys.gr35_history[i], sys.hyperplane)
        push!(gr24_traj, size(gr24) == (2,4) ?
              compute_plucker_gr24(gr24) : zeros(6))
    end

    return gr35_traj, gr24_traj, indices
end

"""
PCA via plain SVD — no external packages needed.
Returns projection of each row of P (n×d) onto first k principal components.
"""
function pca_svd(P::Matrix{Float64}; k::Int=3)::Matrix{Float64}
    μ     = mean(P, dims=1)
    Pc    = P .- μ
    _, S, V = svd(Pc)
    # V is d×d; first k columns are the principal directions
    Vk    = Matrix(V[:, 1:min(k, size(V,2))])   # d×k
    return Pc * Vk                                # n×k
end

"""Phase portrait: p₁₂ vs p₃₄ (the two 2×2 corner minors)."""
function plot_phase_portrait(gr24_traj, times, crisis_events, sector)
    isempty(gr24_traj) && return plot(title="No data")
    x = [p[1] for p in gr24_traj]   # p₁₂ = minor_top
    y = [p[2] for p in gr24_traj]   # p₁₃
    z = [p[6] for p in gr24_traj]   # p₃₄ = minor_bot (→0 = Schubert wall)
    n = length(x)
    pal = cgrad(:viridis)

    fig = plot(xlabel="p₁₂", ylabel="p₁₃", zlabel="p₃₄",
               title="Gr(2,4) Phase Portrait 3D  [Sector $sector]",
               legend=:topright, camera=(30, 20), size=(700,600))

    for i in 1:n-1
        t_nrm = (i-1) / max(n-1, 1)
        plot!(fig, x[i:i+1], y[i:i+1], z[i:i+1];
              lw=1.5, color=pal[t_nrm], alpha=0.8, label="")
    end
    scatter!(fig, [x[1]],   [y[1]],   [z[1]];
             ms=10, color=:green, marker=:circle, label="Start")
    scatter!(fig, [x[end]], [y[end]], [z[end]];
             ms=10, color=:red,   marker=:circle, label="End")
    for ct in crisis_events
        idx = findfirst(t -> t >= ct, times)
        idx === nothing && continue
        scatter!(fig, [x[idx]], [y[idx]], [z[idx]]; ms=12, color=:orangered,
                 marker=:diamond,
                 label= ct == first(crisis_events) ? "Crisis" : "")
    end
    return fig
end


"""Klein quadric residual over time: |p₁₂p₃₄ - p₁₃p₂₄ + p₁₄p₂₃|."""
function plot_klein_quadric(gr24_traj, times, crisis_events, sector)
    err = [abs(p[1]*p[6] - p[2]*p[5] + p[3]*p[4]) for p in gr24_traj]
    # Replace zeros with minimum nonzero for log scale
    min_nonzero = maximum([1e-15; filter(x->x>0, err)])
    err_plot = max.(err, 1e-15)

    fig = plot(times, err_plot, lw=2, color=:purple, yscale=:log10,
               title="Klein Quadric Residual  [Sector $sector]",
               xlabel="Time (s)", ylabel="|p₁₂p₃₄ - p₁₃p₂₄ + p₁₄p₂₃|",
               legend=:topright)
    hline!(fig, [1e-6], lw=1, ls=:dot, color=:green, label="tol=1e-6")
    for ct in crisis_events
        vline!(fig, [ct], lw=1, ls=:dash, color=:red, alpha=0.6,
               label= ct == first(crisis_events) ? "Crisis" : "")
    end
    return fig
end

"""Gr(3,5) Plücker trajectory projected to first 2 PCs (plain SVD)."""
function plot_gr35_pca(gr35_traj, times, sector)
    isempty(gr35_traj) && return plot(title="No data")
    P     = Matrix(hcat(gr35_traj...)')   # n×10
    P_pca = pca_svd(P; k=3)              # n×3 for true 3D
    size(P_pca, 1) < 2 && return plot(title="Too few points")
    n   = size(P_pca, 1)
    pal = cgrad(:plasma)

    fig = plot(xlabel="PC₁", ylabel="PC₂", zlabel="PC₃",
               title="Gr(3,5) Plücker PCA 3D  [Sector $sector]",
               legend=:topright, camera=(35, 20), size=(700,600))

    for i in 1:n-1
        t_nrm = (i-1) / max(n-1, 1)
        plot!(fig, P_pca[i:i+1,1], P_pca[i:i+1,2], P_pca[i:i+1,3];
              lw=1.5, color=pal[t_nrm], alpha=0.8, label="")
    end
    scatter!(fig, [P_pca[1,1]],   [P_pca[1,2]],   [P_pca[1,3]];
             ms=10, color=:green, marker=:circle, label="Start")
    scatter!(fig, [P_pca[end,1]], [P_pca[end,2]], [P_pca[end,3]];
             ms=10, color=:red,   marker=:circle, label="End")
    return fig
end


"""Time series of all 6 Gr(2,4) Plücker coordinates."""
function plot_plucker_timeseries(gr24_traj, times, HH2_vals, crisis_events, sector)
    labels = ["p₁₂","p₁₃","p₁₄","p₂₃","p₂₄","p₃₄"]
    colors = [:blue,:orange,:green,:red,:purple,:brown]

    fig = plot(title="Gr(2,4) Plücker Coordinates  [Sector $sector]",
               xlabel="Time (s)", ylabel="Plücker coord", legend=:topright)
    for (k, (lbl, col)) in enumerate(zip(labels, colors))
        vals = [p[k] for p in gr24_traj]
        plot!(fig, times, vals, lw=1.5, color=col, label=lbl, alpha=0.85)
    end
    for ct in crisis_events
        vline!(fig, [ct], lw=1, ls=:dash, color=:red, alpha=0.5,
               label= ct == first(crisis_events) ? "Crisis" : "")
    end
    # Overlay HH² on secondary-axis-style (scaled to fit)
    if !isempty(HH2_vals) && maximum(HH2_vals) > 0
        hh_scaled = HH2_vals ./ maximum(HH2_vals)
        plot!(fig, times, hh_scaled, lw=2, color=:black, ls=:dash,
              alpha=0.6, label="HH² (norm)")
    end
    return fig
end

"""Generate all 4 plots for one sector and save as PNG files."""
function make_sector_plots(times::Vector{RealType}, HH2_vals::Vector{RealType}, sys::GrassmannianSystem, sector::Symbol)
    # ── Subsample both trajectory histories ─────────────────────────────────
    ph_analytic = sys.plucker_history      # pharmacodynamic embedding (Panel 9)
    ph_geometric = sys.geo_gr24_history    # geometric Gr(2,4) projection

    isempty(ph_analytic) && (println("  No plucker data for sector $sector"); return)

    n_full = length(ph_analytic)
    n_pts  = min(length(times), n_full)
    stride = max(1, div(n_full, 500))
    idx_s  = 1:stride:n_full

    t_sub    = Float64[times[min(i, n_pts)] for i in idx_s]
    n_sub    = length(t_sub)

    # PATH A — pharmacodynamic (Panel 9, time series, panel9 scatter)
    pharma_sub = [ph_analytic[i]  for i in idx_s]

    # PATH B — geometric Gr(2,4) (phase portrait p12/p34, Klein quadric)
    # Zero vectors mark genuine singularities (degenerate projection)
    geo_sub = isempty(ph_geometric) ? fill(zeros(6), n_sub) :
              [ph_geometric[min(i, length(ph_geometric))] for i in idx_s]

    # Gr(3,5) 10-component geometric plucker for PCA
    gr35_sub = [sys.gr35_history[i].plucker for i in idx_s
                if i <= length(sys.gr35_history)]

    # For backward-compat aliases used by existing plot functions:
    gr24_sub = geo_sub     # geometric path → phase portrait, Klein quadric

    crisis   = sys.crisis_events

    # Mark singularity events: steps where geometric projection degenerated
    # (zero vector sentinel from update_grassmannian_from_physical!)
    singularity_times = Float64[]
    for (k, i) in enumerate(idx_s)
        i > length(ph_geometric) && break
        if norm(ph_geometric[i]) < 1e-10 && (isempty(singularity_times) ||
                t_sub[k] - singularity_times[end] > 0.5)
            push!(singularity_times, t_sub[k])
        end
    end
    isempty(singularity_times) || @printf(
        "  Singularities (degenerate projection): %s\n",
        join(round.(singularity_times, digits=2), ", "))

    # Plot 1: phase portrait — geometric Gr(2,4) path
    # Minor_bot → 0 (p34 → 0) marks Schubert wall crossings.
    # Zero-vectors (singularities) shown as orange diamonds.
    fig1 = plot_phase_portrait(geo_sub, t_sub, crisis, sector)
    # Overlay singularity markers on phase portrait
    for st in singularity_times
        idx_sing = findfirst(t -> t >= st, t_sub)
        idx_sing === nothing && continue
        scatter!(fig1, [geo_sub[idx_sing][1]], [geo_sub[idx_sing][6]];
                 ms=14, color=:orange, marker=:hexagon,
                 label= st == first(singularity_times) ? "Singularity (Exc. Div.)" : "")
    end
    savefig(fig1, "gr35_phase_$(sector).png")
    println("  ✓ gr35_phase_$(sector).png")

    # Plot 2: Klein quadric — geometric Gr(2,4) path only
    # True Gr(2,4) coords satisfy p12p34 - p13p24 + p14p23 = 0 exactly.
    # Deviations from zero signal numerical degradation or near-singularity.
    # Large spikes = exceptional divisor (Picard-Lefschetz territory).
    fig2 = plot_klein_quadric(geo_sub, t_sub, crisis, sector)
    savefig(fig2, "gr35_klein_$(sector).png")
    println("  ✓ gr35_klein_$(sector).png")

    # Plot 3: PCA
    fig3 = plot_gr35_pca(gr35_sub, t_sub, sector)
    savefig(fig3, "gr35_pca_$(sector).png")
    println("  ✓ gr35_pca_$(sector).png")

    # Plot 4: time series — pharmacodynamic path (shows drug dose dynamics)
    HH2_sub = begin
        n_h = length(HH2_vals)
        [HH2_vals[min(i, n_h)] for i in idx_s]
    end
    fig4 = plot_plucker_timeseries(pharma_sub, t_sub, HH2_sub, crisis, sector)
    savefig(fig4, "gr35_timeseries_$(sector).png")
    println("  ✓ gr35_timeseries_$(sector).png")

    # Plot 5: Panel-9 style TRUE 3D — matches Python's ax9 (projection='3d')
    # Axes: p12 (x), p13 (y), p14 (z), coloured by time (viridis segments)
    # Norcain dose jumps visible as sharp hops in 3D Plücker space.
    if length(pharma_sub) > 2
        x3 = [p[1] for p in pharma_sub]   # p12 = C/(1+qA/EC50)
        y3 = [p[2] for p in pharma_sub]   # p13 = qB/(1+qB/EC50)
        z3 = [p[3] for p in pharma_sub]   # p14 = (1-C)*exp(-λA*t)

        # Split trajectory into segments between dose events for colour coding
        dose_times_sorted = sort(DOSE_TIMES)
        seg_colors = [:dodgerblue, :darkorange, :green4, :purple]

        # ── TRUE 3D PLOT — p12/p13/p14, coloured by time (viridis segments) ───
        # Replicates Python's ax9 = fig.add_subplot(..., projection='3d')
        n_pts_3d = length(x3)
        palette  = cgrad(:viridis)

        fig5 = plot(xlabel="p₁₂", ylabel="p₁₃", zlabel="p₁₄",
                    title="Plücker 3D Trajectory  [Sector $sector]",
                    legend=:topright, camera=(30, 25),
                    size=(700, 600))

        # Draw each segment coloured by normalised time (Python style loop)
        for i in 1:n_pts_3d-1
            t_nrm = (i - 1) / max(n_pts_3d - 1, 1)
            col   = palette[t_nrm]
            plot!(fig5, x3[i:i+1], y3[i:i+1], z3[i:i+1];
                  lw=1.5, color=col, alpha=0.8, label="")
        end

        # Start (green) and end (red)
        scatter!(fig5, [x3[1]],   [y3[1]],   [z3[1]];
                 ms=10, color=:green, marker=:circle, label="Start (t=0)")
        scatter!(fig5, [x3[end]], [y3[end]], [z3[end]];
                 ms=10, color=:red,   marker=:circle, label="End (t=25)")

        # Dose jump markers — red triangles in 3D
        dose_times_sorted = sort(DOSE_TIMES)
        for (di, dt_val) in enumerate(dose_times_sorted)
            idx = findfirst(t -> t >= dt_val, t_sub)
            idx === nothing || idx < 2 && continue
            scatter!(fig5, [x3[idx]], [y3[idx]], [z3[idx]];
                     ms=14, color=:red, marker=:utriangle,
                     label= di == 1 ? "Norcain dose" : "")
            # Arrow showing jump direction in 3D
            dx, dy, dz = x3[idx]-x3[idx-1], y3[idx]-y3[idx-1], z3[idx]-z3[idx-1]
            if abs(dx)+abs(dy)+abs(dz) > 1e-6
                plot!(fig5, [x3[idx-1],x3[idx]], [y3[idx-1],y3[idx]],
                            [z3[idx-1],z3[idx]];
                      lw=3, color=:red, alpha=0.9, label="")
            end
        end

        # Crisis events in 3D
        for ct in crisis
            idx = findfirst(t -> t >= ct, t_sub)
            idx === nothing && continue
            scatter!(fig5, [x3[idx]], [y3[idx]], [z3[idx]];
                     ms=12, color=:darkred, marker=:diamond, label="")
        end

        # Geometric singularities (exceptional divisors) in 3D
        for st in singularity_times
            idx_s2 = findfirst(t -> t >= st, t_sub)
            idx_s2 === nothing && continue
            scatter!(fig5, [x3[idx_s2]], [y3[idx_s2]], [z3[idx_s2]];
                     ms=14, color=:orange, marker=:hexagon,
                     label= st == first(singularity_times) ?
                            "Singularity (Exc. Div.)" : "")
        end

        savefig(fig5, "gr35_panel9_$(sector).png")
        println("  ✓ gr35_panel9_$(sector).png  ← 3D Panel 9: Plücker trajectory")
    end
end

function main()
    println("\n" * "="^70)
    println("GR(3,5) → GR(2,4) WITH HYPERPLANE SECTIONS + PLÜCKER PLOTS")
    println("="^70)
    println("Output: gr35_data_<sector>.csv + 4 PNG plots per sector")
    println()

    for sector in [:generic, :A, :C]   # :generic and :A show norcain jumps best
        println("\n🔬 Sector: $sector")
        println("-"^50)

        times, C_mean, qB_mean, HH2_vals, coker_dim, tangency, sys =
            run_gr35_simulation((0.0, 25.0), 0.02, sector, 25)

        @printf("  Crisis times : %s\n",
                isempty(sys.crisis_events) ? "none" :
                join(round.(sys.crisis_events, digits=2), ", "))
        @printf("  Max coker    : %d\n", maximum(coker_dim))
        @printf("  Final C      : %.4f\n", C_mean[end])

        # CSV
        open("gr35_data_$(sector).csv", "w") do f
            println(f, "time,C_mean,qB_mean,HH2,coker,tangency")
            for i in eachindex(times)
                @printf(f, "%.4f,%.6f,%.6f,%.6f,%d,%.6f\n",
                        times[i], C_mean[i], qB_mean[i],
                        HH2_vals[i], coker_dim[i], tangency[i])
            end
        end
        println("  ✓ gr35_data_$(sector).csv")

        # 4 Plücker plots
        make_sector_plots(times, HH2_vals, sys, sector)
    end

    println()
    println("="^70)
    println("COMPLETE — 5 CSV files + 20 PNG plots saved to current directory")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
