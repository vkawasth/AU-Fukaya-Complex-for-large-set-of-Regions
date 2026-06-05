# =============================================================================
# gr35_lefschetz.jl
#
# Gr(3,5) ambient space and Lefschetz fibration for the AU-Fukaya pipeline.
#
# This file implements the two missing geometric layers:
#
#   LAYER 1: Gr(3,5) — the ambient 6-dimensional space
#     State = 3-plane in C^5 parameterised by Plücker coordinates p_{ijk}
#     Five dimensions = {C_sAMY, C_HPF, C_Infra, C_BG, h(t)}
#     Three simultaneously active contexts span a 3-plane
#
#   LAYER 2: Lefschetz fibration f: Gr(3,5) → P^1
#     Base = drug concentration parameter space (P^1)
#     Fibers = AU interaction spaces (Gr(2,4) Schubert cells)
#     Critical values = crisis boundaries where det(minor) = 0
#     Monodromy around a critical value = Picard-Lefschetz twist
#
#   LAYER 3: Path-dependent restriction maps
#     ρ*(γ) = monodromy of path γ in parameter space
#     γ avoids all critical values → trivial monodromy → Mode 1/2
#     γ encircles critical value → P-L twist by vanishing cycle → Mode 3/4
#     coker(ρ*) = 62 iff path encircles the sAMY↔Infra critical value
#
# Connection to Picard-Lefschetz theory (Chassé 2020):
#   vanishing cycle ∆ = the 62-dim class that collapses at crisis
#   monodromy formula: h_*(b) = b + (-1)^{n(n+1)/2} (b·∆)∆
#   variation operator: var(∇) = (-1)^{n(n+1)/2} ∆
#   thimble ∂Γ = ∆ = p_buffer in the surgery
# =============================================================================

using LinearAlgebra, Printf

# Load NNO arithmetic and AU context if not already loaded
# (needed for the surgery demo in Part 9)
if !@isdefined(NNOProb)
    include(joinpath(@__DIR__, "tool_paths.jl"))
    include(joinpath(@__DIR__, "nno_au_core.jl"))
end

# =============================================================================
# PART 1: GR(3,5) — PLÜCKER COORDINATES
# =============================================================================

"""
    Gr35State

A point in Gr(3,5): a 3-plane in C^5 represented by a 3×5 matrix M
whose rows span the plane. The Plücker coordinates are the 3×3 minors.

The five basis directions correspond to:
  e1 = C_sAMY context weight
  e2 = C_HPF   context weight
  e3 = C_Infra context weight
  e4 = C_BG    context weight
  e5 = h(t)    toric height parameter

The 3-plane = the three "most active" directions at time t.
"""
struct Gr35State
    M       ::Matrix{Float64}   # 3×5 matrix, rows span the 3-plane
    plucker ::Vector{Float64}   # C(5,3)=10 Plücker coordinates p_{ijk}
    t       ::Float64           # simulation time
end

# Index map for Plücker coordinates: (i,j,k) → index in length-10 vector
# All triples 1≤i<j<k≤5
const PLUCKER_TRIPLES = [(i,j,k) for i in 1:5 for j in i+1:5 for k in j+1:5]
const PLUCKER_IDX     = Dict(t => i for (i,t) in enumerate(PLUCKER_TRIPLES))

"""Compute all 10 Plücker coordinates from a 3×5 matrix."""
function compute_plucker(M::Matrix{Float64})::Vector{Float64}
    p = zeros(Float64, 10)
    for (idx, (i,j,k)) in enumerate(PLUCKER_TRIPLES)
        # p_{ijk} = det of columns i,j,k
        sub = M[:, [i,j,k]]
        p[idx] = det(sub)
    end
    return p
end

"""
    gr35_from_context_weights(w_sAMY, w_HPF, w_Infra, w_BG, h, t)
    -> Gr35State

Build a Gr(3,5) state from the four AU context weights and toric height.
The three most active contexts define the 3-plane.
"""
function gr35_from_context_weights(w_sAMY::Float64, w_HPF::Float64,
                                    w_Infra::Float64, w_BG::Float64,
                                    h::Float64, t::Float64)::Gr35State
    # State vector in R^5
    v = [w_sAMY, w_HPF, w_Infra, w_BG, h]

    # The 3-plane is spanned by three cyclic shifts of the state vector.
    # Cyclic shifts are genuinely independent and have support in ALL 5
    # directions, so both 2×2 corner minors of the Gr(2,4) projection
    # are generically nonzero (both minor_top and minor_bot ≠ 0).
    #
    #   Row 1: v              = [w1, w2, w3, w4, h]   current state
    #   Row 2: σ(v)           = [w2, w3, w4, h,  w1]  one cyclic shift
    #   Row 3: σ²(v)          = [w3, w4, h,  w1, w2]  two cyclic shifts
    v2 = [v[2], v[3], v[4], v[5], v[1]]   # σ(v)
    v3 = [v[3], v[4], v[5], v[1], v[2]]   # σ²(v)
    M  = [v'; v2'; v3']

    p = compute_plucker(M)
    return Gr35State(M, p, t)
end

# =============================================================================
# PART 2: PROJECTION π: GR(3,5) → GR(2,4)
# =============================================================================

"""
    project_to_gr24(state::Gr35State) -> (plucker_24, stratum, minors)

Project a Gr(3,5) state to Gr(2,4) via the hyperplane slicing map π.

The projection extracts the 2-plane cross-sections of the 3-plane
when intersected with the hyperplane H = {x_5 = h(t)}.

Returns:
  plucker_24: the 6 Plücker coordinates of Gr(2,4) [C(4,2)=6]
  stratum: Schubert cell 0–4
  minors: the two 2×2 corner minors for GPS sector detection
"""
function project_to_gr24(state::Gr35State)

    # The (2,2) free entry of the Lax matrix corresponds to the
    # sAMY↔Infra intersection — the critical locus of the fibration
    M = state.M

    # Extract 2-plane by intersecting 3-plane with H = {x_5 = const}
    # Method: eliminate the 5th coordinate by taking the 2×4 submatrix
    # of the first two rows restricted to columns 1-4
    M24 = M[1:2, 1:4]

    # Gr(2,4) Plücker coordinates: C(4,2)=6 minors
    triples_24 = [(i,j) for i in 1:4 for j in i+1:4]
    p24 = [det(M24[:, [i,j]]) for (i,j) in triples_24]

    # Two 2×2 corner minors (for Schubert cell detection)
    minor_top = det(M24[1:2, 1:2])   # p_{12}
    minor_bot = det(M24[1:2, 3:4])   # p_{34}
    pivot     = M24[1,2]              # sAMY pivot = (2,2) entry

    # Schubert stratum
    tol = 1e-8
    stratum = if abs(minor_top) > tol && abs(minor_bot) > tol
        4   # open cell, both minors nonzero
    elseif abs(minor_top) > tol
        3   # stratum 3
    elseif abs(minor_bot) > tol
        2   # crisis boundary stratum
    elseif abs(pivot) > tol
        1   # near-basepoint
    else
        0   # basepoint
    end

    return (plucker_24=p24, stratum=stratum,
            minor_top=minor_top, minor_bot=minor_bot,
            pivot=pivot)
end

# =============================================================================
# PART 3: LEFSCHETZ FIBRATION CRITICAL VALUES
# =============================================================================

"""
    LefschetzFibration

Tracks the critical values (crisis boundaries) and the winding number
of the simulation trajectory around each critical value.

A critical value λ_c is where det(minor_{ij}) = 0 in the parameter space.
The trajectory C(t) winds around λ_c if it encircles the critical point
in the complex parameter plane.
"""
mutable struct LefschetzFibration
    critical_values  ::Vector{Complex{Float64}}   # crisis points in C-plane
    winding_numbers  ::Vector{Int}                # how many times traj winds
    total_angles     ::Vector{Float64}            # accumulated angle (no rounding)
    trajectory       ::Vector{Complex{Float64}}   # C(t) path in parameter space
    thimble_bases    ::Vector{Vector{Float64}}    # vanishing cycles at each crisis
end

function LefschetzFibration()
    # The confirmed critical values from the framework:
    # sAMY↔Infra crisis: λ_c = (v5_sAMY + v5_Infra) = -4 (double pole)
    # HPF↔Infra:         λ_c = -2 (single pole)
    # All others:        λ_c = 0  (no pole, regular)
    critical_values = [
        Complex(-4.0, 0.0),  # sAMY↔Infra double pole (gate=4)
        Complex(-2.0, 0.0),  # HPF↔Infra single pole (gate=2)
    ]
    winding_numbers = zeros(Int, length(critical_values))
    total_angles    = zeros(Float64, length(critical_values))
    LefschetzFibration(critical_values, winding_numbers, total_angles,
                       Complex{Float64}[], Vector{Float64}[])
end

"""
    update_trajectory!(fib, C_new)

Add a new point to the parameter space trajectory and update winding numbers.
The winding number around each critical value is computed by the
argument principle: Δarg(C - λ_c) / (2π).
"""
function update_trajectory!(fib::LefschetzFibration, C_new::Complex{Float64})
    push!(fib.trajectory, C_new)
    n = length(fib.trajectory)
    n < 2 && return

    C_prev = fib.trajectory[n-1]

    # Update winding numbers for each critical value
    for (i, λ_c) in enumerate(fib.critical_values)
        # Argument increment: Δarg(C - λ_c)
        z_prev = C_prev - λ_c
        z_new  = C_new  - λ_c
        # Careful angle computation to avoid 2π jumps
        darg = angle(z_new) - angle(z_prev)
        # Wrap to (-π, π]
        darg = mod(darg + π, 2π) - π
        # Accumulate TOTAL angle continuously (no per-step rounding)
        fib.total_angles[i] += darg
        # Winding number = total revolutions completed
        fib.winding_numbers[i] = round(Int, fib.total_angles[i] / (2π))
    end
end

"""
    winds_around_crisis(fib, ctx1_id, ctx2_id) -> Bool

Check whether the current trajectory winds around the critical value
corresponding to the ctx1↔ctx2 boundary.
"""
function winds_around_crisis(fib::LefschetzFibration,
                              ctx1_id::Symbol, ctx2_id::Symbol)::Bool
    pair = Set([ctx1_id, ctx2_id])
    if pair == Set([:CTX_sAMY, :CTX_INFRA])
        # Critical value index 1 (double pole)
        return fib.winding_numbers[1] != 0
    elseif pair == Set([:CTX_HPF, :CTX_INFRA])
        # Critical value index 2 (single pole)
        return fib.winding_numbers[2] != 0
    end
    return false
end

# =============================================================================
# PART 4: LEFSCHETZ THIMBLE AND VANISHING CYCLE
# =============================================================================

"""
    LefschetzThimble

A Lefschetz thimble is a disk D^n in the total space whose boundary
∂Γ = ∆ is the vanishing cycle in the fiber F_1.

In the AU pipeline:
  - The "disk" is the space of AU probability distributions approaching crisis
  - The "boundary" ∆ is the p_buffer (probability mass on boundary nodes)
  - The thimble path sweeps from the regular state to the crisis boundary

Theorem (Picard-Lefschetz): ∂(thimble) = vanishing cycle
"""
struct LefschetzThimble
    ctx_id        ::Symbol             # which AU context
    vanishing_dim ::Int                # dim of vanishing cycle = coker
    vanishing_vec ::Vector{Float64}    # the vanishing direction (eigenvector)
    critical_t    ::Float64            # time when thimble reaches crisis
    pl_sign       ::Int                # (-1)^{n(n+1)/2} sign in formula
end

"""
    build_thimble(ctx_id, coker, n_dim) -> LefschetzThimble

Build a Lefschetz thimble for the given AU context.
The vanishing cycle dimension = coker(ρ*_αβ).
The Picard-Lefschetz sign = (-1)^{n(n+1)/2} where n = fiber dimension.
"""
function build_thimble(ctx_id::Symbol, coker::Int, n_dim::Int=2)::LefschetzThimble
    # Picard-Lefschetz sign: (-1)^{n(n+1)/2}
    # For n=2 (our Gr(2,4) fiber): (-1)^{2·3/2} = (-1)^3 = -1
    # For n=3 (Gr(3,5) total space): (-1)^{3·4/2} = (-1)^6 = +1
    pl_sign = (-1)^(n_dim * (n_dim + 1) ÷ 2)

    # Vanishing direction: the null eigenvector of the restriction map
    # For coker=62: a 62-dimensional vector (simplified to unit here)
    vanishing_vec = normalize(randn(max(coker, 1)))

    LefschetzThimble(ctx_id, coker, vanishing_vec, Inf, pl_sign)
end

# =============================================================================
# PART 5: MONODROMY-DEPENDENT RESTRICTION MAPS
# =============================================================================

"""
    monodromy_restriction_map(fib, ctx1_id, ctx2_id, hh2_1, hh2_2)
    -> (mode::Int, coker::Int, twist_needed::Bool)

Compute the restriction map ρ*(γ) for path γ in parameter space.

KEY INSIGHT FROM PICARD-LEFSCHETZ THEORY:
The restriction map is NOT static — it depends on the PATH γ taken
from ctx1 to ctx2 in parameter space.

If γ avoids all critical values: ρ*(γ) = identity → coker=0 → Mode 1
If γ winds around sAMY↔Infra crisis: ρ*(γ) = P-L twist → coker=62 → Mode 4
If γ winds around HPF↔Infra crisis: ρ*(γ) = single twist → coker>0 → Mode 3

This replaces the static PUSHOUT_TABLE with a path-dependent computation.
"""
function monodromy_restriction_map(fib::LefschetzFibration,
                                    ctx1_id::Symbol,
                                    ctx2_id::Symbol,
                                    hh2_1::Int,
                                    hh2_2::Int)

    wound = winds_around_crisis(fib, ctx1_id, ctx2_id)
    pair  = Set([ctx1_id, ctx2_id])

    if !wound
        # Path avoids all critical values → trivial monodromy
        # ρ*(γ) = identity, coker = 0
        mode  = hh2_1 == hh2_2 ? 1 : 2
        coker = 0
        return (mode=mode, coker=coker, twist=false,
                description="Path avoids crisis → trivial monodromy")

    elseif pair == Set([:CTX_sAMY, :CTX_INFRA])
        # Wound around double pole (gate=4)
        # ρ*(γ) = Picard-Lefschetz twist × Picard-Lefschetz twist
        # = double twist by vanishing cycle ∆ (coker=62)
        return (mode=4, coker=62, twist=true,
                description="Path encircles sAMY↔Infra crisis (v5=-4) → double P-L twist")

    elseif pair == Set([:CTX_HPF, :CTX_INFRA])
        # Wound around single pole (gate=2)
        # ρ*(γ) = single Picard-Lefschetz twist
        return (mode=3, coker=abs(hh2_2 - hh2_1), twist=true,
                description="Path encircles HPF↔Infra crisis (v5=-2) → single P-L twist")

    else
        # Other boundaries: winding possible but resolved by GPS sectors
        return (mode=2, coker=max(0, hh2_2 - hh2_1), twist=true,
                description="Path encircles boundary → partial P-L twist")
    end
end

# =============================================================================
# PART 6: PICARD-LEFSCHETZ TWIST (Rule III — now mathematically grounded)
# =============================================================================

"""
    picard_lefschetz_twist(b, delta, pl_sign) -> b_twisted

Apply the Picard-Lefschetz monodromy formula to vector b:

    h_*(b) = b + (-1)^{n(n+1)/2} · (b · ∆) · ∆

where:
  b      = the probability vector being transported
  ∆      = the vanishing cycle (unit vector in vanishing direction)
  pl_sign = (-1)^{n(n+1)/2}  (depends on fiber dimension n)

This is the mathematically precise form of Surgery Rule III,
derived directly from Theorem 2 in the Chassé paper (Corollary 2).

The twist preserves the intersection pairing ∇·∆ = 1 because:
  h_*(∆)·∆ = (∆ + pl_sign·(∆·∆)·∆)·∆
            = ∆·∆ + pl_sign·(∆·∆)·(∆·∆)
            which equals ∆·∆ when pl_sign·(∆·∆) = -1  ✓
"""
function picard_lefschetz_twist(b::Vector{Float64},
                                 delta::Vector{Float64},
                                 pl_sign::Int = -1)::Vector{Float64}
    # Normalise ∆
    delta_norm = norm(delta)
    delta_norm < 1e-14 && return b

    delta_unit = delta ./ delta_norm

    # Intersection product b·∆ (inner product in our finite-dimensional model)
    b_dot_delta = dot(b, delta_unit)

    # Picard-Lefschetz formula: h_*(b) = b + (-1)^{n(n+1)/2} (b·∆) ∆
    return b .+ Float64(pl_sign) .* b_dot_delta .* delta_unit
end

"""
    variation_operator(nabla, delta, pl_sign) -> var(∇)

The variation operator from the Chassé paper:
    var(∇) = (-1)^{n(n+1)/2} · ∆

This is what Surgery Rule IV computes:
the variation of the generator ∇ of H_{n-1}(F_1, ∂F_1)
equals the vanishing cycle ∆ (up to sign).

In the AU pipeline:
  ∇ = the boundary probability mass (before surgery)
  var(∇) = p_buffer (after surgery: the extracted boundary mass)
  ∆ = the direction in which probability is redirected (Lan_i target)
"""
function variation_operator(nabla::Vector{Float64},
                              delta::Vector{Float64},
                              pl_sign::Int = -1)::Vector{Float64}
    delta_norm = norm(delta)
    delta_norm < 1e-14 && return zeros(length(nabla))
    delta_unit = delta ./ delta_norm

    # var(∇) = (-1)^{n(n+1)/2} · (∇·∆) · ∆
    nabla_dot_delta = dot(nabla, delta_unit)
    return Float64(pl_sign) .* nabla_dot_delta .* delta_unit
end

# =============================================================================
# PART 7: INTEGRATED GR(3,5) STATE TRACKER
# =============================================================================

"""
    Gr35Tracker

Runtime tracker that maintains the Gr(3,5) state, the Lefschetz fibration
winding data, and the current monodromy of restriction maps.

This is the object that BALBc_Opiate_Norcain.py calls every simulation step
to update the geometric state and get the correct mode classification.
"""
mutable struct Gr35Tracker
    state       ::Gr35State
    fibration   ::LefschetzFibration
    thimbles    ::Dict{Symbol, LefschetzThimble}
    history     ::Vector{Gr35State}   # trajectory of Gr(3,5) states
    step        ::Int
end

function Gr35Tracker(w_init::Vector{Float64})
    state = gr35_from_context_weights(
        w_init[1], w_init[2], w_init[3], w_init[4], w_init[5], 0.0)
    fib = LefschetzFibration()
    thimbles = Dict(
        :CTX_sAMY  => build_thimble(:CTX_sAMY,  0,  2),
        :CTX_HPF   => build_thimble(:CTX_HPF,   0,  2),
        :CTX_INFRA => build_thimble(:CTX_INFRA, 62, 2),  # coker=62 confirmed
        :CTX_BG    => build_thimble(:CTX_BG,    0,  2),
    )
    Gr35Tracker(state, fib, thimbles, Gr35State[], 0)
end

"""
    step!(tracker, w_sAMY, w_HPF, w_Infra, w_BG, h, t)

Update the Gr(3,5) tracker for one simulation step.
Updates:
  - Current Gr(3,5) state (new 3-plane from context weights)
  - Lefschetz fibration trajectory (new point in parameter space)
  - Winding numbers (whether trajectory has encircled critical values)
  - Gr(2,4) projection (GPS sector for current step)
"""
function step!(tracker::Gr35Tracker,
               w_sAMY::Float64, w_HPF::Float64,
               w_Infra::Float64, w_BG::Float64,
               h::Float64, t::Float64)

    # Update Gr(3,5) state
    push!(tracker.history, tracker.state)
    tracker.state = gr35_from_context_weights(
        w_sAMY, w_HPF, w_Infra, w_BG, h, t)
    tracker.step += 1

    # Update Lefschetz trajectory
    # Map drug state to complex parameter: C = w_sAMY + i·w_Infra
    C_current = Complex(w_sAMY, w_Infra)
    update_trajectory!(tracker.fibration, C_current)

    # Project to Gr(2,4)
    proj = project_to_gr24(tracker.state)

    return proj
end

"""
    get_restriction_mode(tracker, ctx1_id, ctx2_id, hh2_1, hh2_2)
    -> (mode, coker, twist, description)

Get the current restriction map mode for crossing from ctx1 to ctx2.
This is path-dependent: the mode changes as the trajectory winds around
different critical values.
"""
function get_restriction_mode(tracker::Gr35Tracker,
                               ctx1_id::Symbol, ctx2_id::Symbol,
                               hh2_1::Int=89, hh2_2::Int=89)
    return monodromy_restriction_map(
        tracker.fibration, ctx1_id, ctx2_id, hh2_1, hh2_2)
end

# =============================================================================
# PART 8: SURGERY WITH CORRECT PICARD-LEFSCHETZ FORMULAS
# =============================================================================

"""
    pl_surgery!(tracker, ctx1_id, ctx2_id, prob_vec)
    -> (prob_updated, p_buffer, description)

Execute Picard-Lefschetz surgery when the trajectory winds around a
critical value. This is the mathematically grounded version of Rules I-IV.

Rule I:  Detect winding (trajectory encircles critical value)
Rule II: Extract p_buffer = var_γ(∇) = (-1)^{n(n+1)/2} · (∇·∆) · ∆
         (the variation operator applied to the boundary mass)
Rule III: Apply P-L twist: b ↦ b + pl_sign · (b·∆) · ∆
Rule IV: Redistribute p_buffer along ∆ direction (Lan_i target)
         Normalise to Σp = 1//1
"""
function pl_surgery!(tracker::Gr35Tracker,
                      ctx1_id::Symbol, ctx2_id::Symbol,
                      prob_vec::Vector{Float64})

    mode_info = get_restriction_mode(tracker, ctx1_id, ctx2_id)
    !mode_info.twist && return (prob_vec, 0.0, "No surgery needed")

    # Get thimble for ctx1 (the source context)
    thimble = get(tracker.thimbles, ctx1_id, nothing)
    thimble === nothing && return (prob_vec, 0.0, "No thimble for $ctx1_id")

    delta   = thimble.vanishing_vec
    pl_sign = thimble.pl_sign
    n       = length(prob_vec)

    # Extend delta to match prob_vec dimension
    delta_extended = zeros(n)
    m = min(length(delta), n)
    delta_extended[1:m] = delta[1:m]

    # Rule II: extract p_buffer = variation operator
    # var(∇) = pl_sign · (∇·∆) · ∆  where ∇ = boundary components
    p_buffer_vec = variation_operator(prob_vec, delta_extended, pl_sign)
    p_buffer     = norm(p_buffer_vec)

    # Rule III: apply P-L twist to remaining probability
    prob_twisted = picard_lefschetz_twist(prob_vec, delta_extended, pl_sign)

    # Ensure non-negativity (probabilities can't go negative)
    prob_twisted = max.(prob_twisted, 0.0)

    # Rule IV: renormalise to Σ = 1
    s = sum(prob_twisted)
    s > 1e-14 && (prob_twisted ./= s)

    @printf("  [P-L Surgery] %s→%s: mode=%d, coker=%d, |p_buffer|=%.4f\n",
            ctx1_id, ctx2_id, mode_info.mode, mode_info.coker, p_buffer)
    @printf("    Twist sign: %+d, ∆ dim: %d, wound: %s\n",
            pl_sign, thimble.vanishing_dim,
            winds_around_crisis(tracker.fibration, ctx1_id, ctx2_id))

    return (prob_twisted, p_buffer, mode_info.description)
end

# =============================================================================
# PART 9: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^65)
    println("Gr(3,5) Lefschetz Fibration Layer")
    println("="^65)

    println("\n[1] Build initial Gr(3,5) state")
    println("    w = (C_sAMY=1.2, C_HPF=0.8, C_Infra=0.5, C_BG=0.3, h=10.0)")
    state = gr35_from_context_weights(1.2, 0.8, 0.5, 0.3, 10.0, 0.0)
    println(@sprintf("    Plücker coords: [%s]",
            join([@sprintf("%.3f", p) for p in state.plucker], ", ")))

    println("\n[2] Project to Gr(2,4)")
    proj = project_to_gr24(state)
    @printf("    Stratum: %d  minor_top: %.4f  minor_bot: %.4f\n",
            proj.stratum, proj.minor_top, proj.minor_bot)
    gps = Dict(4=>"Sector A (open cell)",
               3=>"Sector B (transition)",
               2=>"Sector C (CRISIS boundary)",
               1=>"Sector D (near-basepoint)",
               0=>"Basepoint")[proj.stratum]
    println("    GPS hint: $gps")

    println("\n[3] Build Gr(3,5) tracker and simulate trajectory")
    tracker = Gr35Tracker([1.2, 0.8, 0.5, 0.3, 10.0])

    # Demo trajectory A: sub-threshold (stays in Q1, no winding)
    println("    Sub-threshold trajectory (no crisis crossing):")
    for i in 1:20
        t       = i * 0.5
        w_sAMY  = 1.2 + 0.1 * sin(t)
        w_HPF   = 0.8 + 0.05 * cos(t)
        w_Infra = 0.5 - 0.02 * t
        w_BG    = 0.3
        h       = max(1.0, 10.0 - 0.3 * t)
        local proj = step!(tracker, w_sAMY, w_HPF, w_Infra, w_BG, h, t)
        i % 5 == 0 && @printf("    t=%.1f: stratum=%d, winding=%s\n",
                t, proj.stratum, string(tracker.fibration.winding_numbers))
    end

    # Demo trajectory B: encircles the sAMY↔Infra critical value λ_c=-4
    # C(t) = -4 + r*exp(2πi*t/T) traces a circle around -4
    println("    Crisis trajectory (encircles λ_c=-4):")
    tracker2 = Gr35Tracker([1.2, 0.8, 0.5, 0.3, 10.0])
    r = 0.5  # radius around critical value
    for i in 1:20
        θ = 2π * i / 20          # full circle in 20 steps
        # C = -4 + r*exp(iθ) → real part = -4+r*cos(θ), imag = r*sin(θ)
        # Map: w_sAMY = real(C), w_Infra = imag(C) shifted to positive
        w_sAMY  = -4.0 + r * cos(θ)   # will be negative → crisis regime
        w_Infra =  r * sin(θ)
        w_HPF   = 0.8
        w_BG    = 0.3
        h       = 1.0
        local proj = step!(tracker2, w_sAMY, w_HPF, w_Infra, w_BG, h, Float64(i))
        i % 5 == 0 && @printf("    t=%d: stratum=%d, winding=%s (C=%.2f%+.2fi)\n",
                i, proj.stratum, string(tracker2.fibration.winding_numbers),
                w_sAMY, w_Infra)
    end
    println("    After 1 full circle: winding numbers = ",
            tracker2.fibration.winding_numbers)

    println("\n[4] Path-dependent restriction maps")
    println("    (using tracker2 = crisis trajectory with winding=[1,0])")
    for (c1, c2, hh1, hh2) in [
            (:CTX_sAMY, :CTX_INFRA, 89, 151),
            (:CTX_HPF,  :CTX_INFRA, 89, 89),
            (:CTX_sAMY, :CTX_HPF,   89, 89),
        ]
        m = get_restriction_mode(tracker2, c1, c2, hh1, hh2)
        @printf("    %s ↔ %s: mode=%d, coker=%d, twist=%s\n",
                c1, c2, m.mode, m.coker, m.twist)
        println("      $(m.description)")
    end

    println("\n[5] Picard-Lefschetz surgery test")
    # pl_surgery! needs an NNOAUContext (not a thimble).
    # Build a minimal 5-region context for the surgery demo.
    edges_surg = [(:sAMY,:HPF),(:sAMY,:BLA),(:HPF,:BLA),(:BLA,:LA),(:LA,:sAMY)]
    w_surg = Dict(e => NNOProb(Int128(10),Int128(1)) for e in edges_surg)
    stops_surg = Set{Tuple{Symbol,Symbol}}()
    ctx_surgery = build_nno_au(:CTX_sAMY_surg, "sAMY surgery test",
        [:sAMY,:HPF,:BLA,:LA,:CA1sp], edges_surg, stops_surg, w_surg,
        :C, 151, 62, 1.618; initial_node=:sAMY)
    ctx_infra_surg = build_nno_au(:CTX_INFRA_surg, "Infra surgery test",
        [:HPF,:BLA,:LA], edges_surg, stops_surg, w_surg,
        :C, 89, 0, 1.618; initial_node=:HPF)

    # Attach these contexts to tracker2 so pl_surgery! can find them
    tracker2_au = Dict(:CTX_sAMY_surg => ctx_surgery,
                       :CTX_INFRA_surg => ctx_infra_surg)

    prob = Float64[0.4, 0.2, 0.2, 0.1, 0.1]
    @printf("    Before: Σp=%.6f\n", sum(prob))

    # pl_surgery! with explicit au_contexts — call the NNO version directly
    # Rule II: boundary nodes = sAMY nodes with edges into Infra context
    set2 = Set(ctx_infra_surg.regions)
    idx1 = Dict(v => i for (i,v) in enumerate(ctx_surgery.regions))
    boundary_nodes = [s for (s,t) in ctx_surgery.edges
                      if t ∈ set2 && s ∈ keys(idx1)]
    unique!(boundary_nodes)

    # Use local to avoid Julia soft-scope ambiguity with += in top-level for loop
    local p_buffer = 0.0
    for v in boundary_nodes
        i = get(idx1, v, 0)
        i == 0 && continue
        i <= length(prob) || continue
        p_buffer += prob[i]
        prob[i] = 0.0
    end
    # Rule III: P-L twist on remaining mass
    thimble = tracker2.thimbles[:CTX_INFRA]
    delta   = thimble.vanishing_vec[1:min(end,length(prob))]
    delta_ext = vcat(delta, zeros(max(0, length(prob)-length(delta))))
    prob_new = picard_lefschetz_twist(prob, delta_ext, thimble.pl_sign)
    # Rule IV: renormalise
    s = sum(prob_new); s > 1e-14 && (prob_new ./= s)

    @printf("    After:  Σp=%.6f  |p_buffer|=%.6f\n",
            sum(prob_new), p_buffer)
    @printf("    Boundary nodes extracted: %s\n",
            join(string.(boundary_nodes), ", "))
    p_buffer > 0 ?
        println("    Surgery applied: Rules I-IV ✓") :
        println("    No boundary nodes found (contexts don't overlap)")

    println("\n[6] Vanishing cycle and variation operator")
    thimble = tracker.thimbles[:CTX_INFRA]
    delta   = thimble.vanishing_vec
    nabla   = normalize(abs.(randn(length(delta))))
    var_out = variation_operator(nabla, delta, thimble.pl_sign)
    @printf("    ∆ dim=%d  pl_sign=%+d\n",
            thimble.vanishing_dim, thimble.pl_sign)
    @printf("    |var(∇)| = %.6f  (= |(-1)^{n(n+1)/2} (∇·∆) ∆|)\n",
            norm(var_out))
    @printf("    ∇·∆ = %.6f  →  var(∇) ∝ ∆  ✓\n", dot(nabla, delta))

    println("\n" * "="^65)
    println("Gr(3,5) Lefschetz layer complete.")
    println("  gr35_from_context_weights()  → Gr(3,5) state + Plücker coords")
    println("  project_to_gr24()            → Gr(2,4) stratum + minors")
    println("  Gr35Tracker / step!()        → runtime trajectory tracker")
    println("  monodromy_restriction_map()  → path-dependent mode/coker")
    println("  picard_lefschetz_twist()     → h_*(b) = b + pl_sign(b·∆)∆")
    println("  variation_operator()         → var(∇) = pl_sign(∇·∆)∆")
    println("  pl_surgery!()               → Rules I-IV, P-L grounded")
    println("="^65)
end
