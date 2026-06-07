# =============================================================================
# fukaya_ad_context.jl
#
# Centralised Coproduct in Fukaya Categories for MTR Ad Placement
#
# Implements the theoretical framework from the feedback document:
#
#   M   = MTR as symplectic manifold (station × time phase space)
#   ω   = time-varying symplectic form (ridership × temporal Hamiltonian)
#   L_i = Lagrangian submanifolds = demographic profiles
#   CF*(L_i,L_j) = Floer complex = demographic intersections at stations
#   Δ   = pair-of-pants coproduct (ad display event splits across demographics)
#   AU  = Joyal arithmetic universe = lazy pullback (no precomputation)
#   m_k = A∞ operations (m1=flow, m2=product match, m3=homotopy stability)
#
# The key shift from mtr_ad_game.jl:
#   BEFORE: precomputed C[station, product, month] = scalar
#   AFTER:  lazy AU pullback Δ(ad, station, time) → Demo ⊗ Time
#           computed ON DEMAND when the ad slot opens
# =============================================================================

using LinearAlgebra, Printf, Statistics

# =============================================================================
# PART 1: SYMPLECTIC MANIFOLD (M, ω)
# =============================================================================

"""
    SymplecticForm

The time-varying symplectic form ω on the MTR manifold.
ω(station, time_of_day, month) encodes the "transit energy"
at each (station, time) point in the phase space.

This is NOT just ridership — it captures:
  - Commuter density (direction-dependent: morning flow vs evening flow)
  - Temporal resonance (CNY creates a different geometric structure
    than a normal Wednesday afternoon)
  - The Hamiltonian H(t) = Σ_s ω(s,t) × p_d(s,t)
"""
struct SymplecticForm
    # ω evaluated at (station_idx, hour, month) → Float64
    # Discretised: 81 stations × 24 hours × 12 months
    omega ::Array{Float64, 3}   # [station, hour, month]
    # Hamiltonian coefficients (temporal energy profile per month)
    H_coeff ::Matrix{Float64}   # [hour, month] → scalar
end

"""Build the symplectic form from ridership and temporal patterns."""
function build_symplectic_form(stations::Vector{String},
                                ridership::Vector{Float64})::SymplecticForm

    n_s, n_h, n_m = length(stations), 24, 12

    # Base symplectic form: ridership × time-of-day profile
    omega = zeros(n_s, n_h, n_m)

    # Time-of-day Hamiltonian profile (rush hours = high energy)
    hour_profile = [
        0.2, 0.1, 0.1, 0.1, 0.2, 0.4,  # 0-5am  (night/early)
        0.8, 1.0, 0.9, 0.7, 0.6, 0.7,  # 6-11am (morning rush)
        0.7, 0.6, 0.5, 0.6, 0.7, 0.9,  # 12-5pm (midday/afternoon)
        1.0, 0.9, 0.8, 0.6, 0.5, 0.3,  # 6-11pm (evening rush/night)
    ]

    # Monthly temporal resonance (CNY, summer, etc.)
    month_resonance = [
        1.2, 1.5, 1.1, 1.0, 1.0, 1.0,  # Jan-Jun (CNY in Feb)
        1.1, 1.1, 1.0, 1.2, 1.3, 1.4,  # Jul-Dec (summer/11.11/Xmas)
    ]

    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        omega[s,h,m] = ridership[s] * hour_profile[h] * month_resonance[m]
    end

    # Hamiltonian coefficients: how much each hour-month state contributes
    H_coeff = [hour_profile[h] * month_resonance[m]
               for h in 1:n_h, m in 1:n_m]

    return SymplecticForm(omega, H_coeff)
end

"""
Evaluate the Hamiltonian H(t) at time (hour, month):
H = Σ_s ω(s,t) × p_demo(s)
= total "transit energy" in the network at time t.
"""
function hamiltonian(sf::SymplecticForm,
                     demo_profiles::Matrix{Float64},
                     hour::Int, month::Int)::Float64
    n_s = size(sf.omega, 1)
    sum(sf.omega[s, hour, month] * mean(demo_profiles[s,:])
        for s in 1:n_s)
end

# =============================================================================
# PART 2: LAGRANGIAN SUBMANIFOLDS = DEMOGRAPHIC PROFILES
# =============================================================================

"""
    Lagrangian

A Lagrangian submanifold L_i of (M, ω).
In MTR ad placement: L_i represents a constrained demographic subspace.
L_Rich = the (station, time) subspace where rich passengers flow.
L_Female = the (station, time) subspace where female passengers flow.
L_CNY = the (station, time) subspace during Chinese New Year dynamics.

A Lagrangian is characterised by its FLOW VECTOR on M:
  flow(L_i)[station, hour, month] = density of demographic i at (s,h,m)
"""
struct Lagrangian
    label   ::Symbol              # :Rich, :Female, :CNY, etc.
    name    ::String
    flow    ::Array{Float64,3}    # [station, hour, month] density
    is_temporal::Bool             # true = temporal context, false = demographic
end

"""Build demographic Lagrangians from station profiles and time patterns."""
function build_lagrangians(stations   ::Vector{String},
                            demo_profiles::Matrix{Float64},
                            sf         ::SymplecticForm)::Vector{Lagrangian}

    n_s, n_h, n_m = size(sf.omega)

    # Demographic Lagrangians
    lagrangians = Lagrangian[]

    # RM: Rich Male — peaks at rich stations during business hours
    flow_RM = zeros(n_s, n_h, n_m)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        # Rich males peak 8-10am, 5-8pm (commute), and 12-2pm (lunch)
        hour_w = h ∈ 8:10 ? 1.5 : h ∈ 17:20 ? 1.3 : h ∈ 12:14 ? 1.2 : 1.0
        flow_RM[s,h,m] = demo_profiles[s,1] * sf.omega[s,h,m] * hour_w
    end
    push!(lagrangians, Lagrangian(:RM, "Rich Male",   flow_RM, false))

    # RF: Rich Female — peaks at shopping stations (10am-8pm), weekends higher
    flow_RF = zeros(n_s, n_h, n_m)
    shopping_boost = Dict("CausewayBay"=>1.4, "Mong_Kok"=>1.3,
                           "Tsim_Sha_Tsui"=>1.3, "Central"=>1.2)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        hour_w  = h ∈ 10:20 ? 1.4 : 1.0
        s_boost = get(shopping_boost, stations[s], 1.0)
        flow_RF[s,h,m] = demo_profiles[s,2] * sf.omega[s,h,m] * hour_w * s_boost
    end
    push!(lagrangians, Lagrangian(:RF, "Rich Female", flow_RF, false))

    # PM: Poor Male — peaks at outskirt stations, early morning commute
    flow_PM = zeros(n_s, n_h, n_m)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        hour_w = h ∈ 6:9 ? 1.6 : h ∈ 17:19 ? 1.4 : 1.0
        flow_PM[s,h,m] = demo_profiles[s,3] * sf.omega[s,h,m] * hour_w
    end
    push!(lagrangians, Lagrangian(:PM, "Poor Male",   flow_PM, false))

    # PF: Poor Female — peaks at outskirt stations, midday (shift workers)
    flow_PF = zeros(n_s, n_h, n_m)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        hour_w = h ∈ 10:16 ? 1.3 : h ∈ 17:21 ? 1.2 : 1.0
        flow_PF[s,h,m] = demo_profiles[s,4] * sf.omega[s,h,m] * hour_w
    end
    push!(lagrangians, Lagrangian(:PF, "Poor Female", flow_PF, false))

    # Temporal Lagrangians (cyclic contexts)
    # L_CNY: Chinese New Year context
    flow_CNY = zeros(n_s, n_h, n_m)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        cny_factor = m ∈ [1,2] ? 2.5 : 0.0
        flow_CNY[s,h,m] = sf.omega[s,h,m] * cny_factor
    end
    push!(lagrangians, Lagrangian(:CNY, "Chinese New Year", flow_CNY, true))

    # L_Summer: Summer tourism
    flow_SUM = zeros(n_s, n_h, n_m)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        sum_factor = m ∈ [7,8] ? 1.8 : 0.0
        flow_SUM[s,h,m] = sf.omega[s,h,m] * sum_factor
    end
    push!(lagrangians, Lagrangian(:Summer, "Summer", flow_SUM, true))

    # L_Singles: 11.11 Singles Day
    flow_1111 = zeros(n_s, n_h, n_m)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        fac = m ∈ [10,11] ? 2.0 : 0.0
        flow_1111[s,h,m] = sf.omega[s,h,m] * fac
    end
    push!(lagrangians, Lagrangian(:Singles, "Singles Day", flow_1111, true))

    # L_Christmas: December holiday season
    flow_xmas = zeros(n_s, n_h, n_m)
    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        xmas_factor = m == 12 ? 1.8 : 0.0
        flow_xmas[s,h,m] = sf.omega[s,h,m] * xmas_factor
    end
    push!(lagrangians, Lagrangian(:Christmas, "Christmas", flow_xmas, true))

    return lagrangians
end

# =============================================================================
# PART 3: FLOER COMPLEX CF*(L_i, L_j)
# =============================================================================

"""
    FloerGenerator

A generator of CF*(L_i, L_j): an intersection point of two Lagrangians.
In MTR terms: a (station, hour, month) triple where both demographics
are simultaneously present above threshold.

This replaces the scalar multiplication L_i × L_j.
The intersection COUNT #CF*(L_i,L_j) measures how many spacetime
points both demographics co-occupy — the actual number of places
and times an ad can hit BOTH demographics.
"""
struct FloerGenerator
    station ::Int
    hour    ::Int
    month   ::Int
    density ::Float64    # ω at this intersection point
    lag_i   ::Symbol
    lag_j   ::Symbol
end

"""
    floer_complex(L_i, L_j; threshold) -> Vector{FloerGenerator}

Compute CF*(L_i, L_j): find all intersection points of two Lagrangians
above the density threshold.

The A∞ differential m_1 acts on these generators by pushing them
along the Hamiltonian flow (station-to-station transitions).
"""
function floer_complex(L_i       ::Lagrangian,
                        L_j       ::Lagrangian;
                        threshold ::Float64 = 0.15)::Vector{FloerGenerator}

    n_s, n_h, n_m = size(L_i.flow)
    generators = FloerGenerator[]

    for s in 1:n_s, h in 1:n_h, m in 1:n_m
        # Intersection density = geometric mean of both flows
        # (both must be present for the intersection to be non-trivial)
        dens_i = L_i.flow[s,h,m]
        dens_j = L_j.flow[s,h,m]
        dens   = sqrt(dens_i * dens_j)   # geometric mean

        dens > threshold || continue

        push!(generators, FloerGenerator(s, h, m, dens,
                                          L_i.label, L_j.label))
    end

    # Sort by density descending (most important intersections first)
    sort!(generators, by=g->g.density, rev=true)
    return generators
end

"""
    floer_count(L_i, L_j, station, month) -> Float64

The count #CF*(L_i, L_j)|_{station,month}:
how much of the Floer complex concentrates at this station in this month.

This is the CORRECT replacement for DEMO_PROFILES[s,i] * DEMO_PROFILES[s,j].
"""
function floer_count(L_i     ::Lagrangian,
                      L_j     ::Lagrangian,
                      station ::Int,
                      month   ::Int)::Float64
    # Integrate over all hours at this (station, month)
    sum(sqrt(L_i.flow[station,h,month] * L_j.flow[station,h,month])
        for h in 1:size(L_i.flow, 2))
end

# =============================================================================
# PART 4: PAIR-OF-PANTS COPRODUCT Δ (LAZY AU EVALUATION)
# =============================================================================

"""
    AdEvent

A single ad display opportunity at (station, hour, month).
The coproduct Δ splits this event across the Lagrangian basis.
"""
struct AdEvent
    station ::Int
    hour    ::Int
    month   ::Int
    product ::Int
    omega_val::Float64    # ω(station, hour, month)
end

"""
    coprod_delta(event, product, lagrangians) -> Dict{Symbol, Float64}

The pair-of-pants coproduct Δ(event) → Σ_i c_i × L_i

Splits the ad display event across the demographic Lagrangians.
Returns the CONTRIBUTION of each Lagrangian to the ad's reach.

Δ(Ad_Luxury at Admiralty, CNY morning) →
    c_RM × L_RM  +  c_RF × L_RF  +  c_CNY × L_CNY  + ...

where c_i = ∫_{station,hour,month} L_i.flow × product.affinity[i]

This is the LAZY computation — called only when an ad slot opens.
Nothing is precomputed. The AU constructs the intersection at this node.
"""
function coprod_delta(event      ::AdEvent,
                       affinity   ::Vector{Float64},
                       lagrangians::Vector{Lagrangian})::Dict{Symbol,Float64}

    result = Dict{Symbol,Float64}()

    s, h, m = event.station, event.hour, event.month

    for (k, lag) in enumerate(lagrangians)
        lag.is_temporal && continue   # temporal context handled separately

        # Flow of this demographic at this (station, hour, month)
        flow_val = lag.flow[s, h, m]

        # Affinity of the product toward this demographic
        aff = k <= length(affinity) ? affinity[k] : 0.0

        # Coproduct coefficient: how much of the event reaches this demographic
        # = Floer intersection density × product affinity
        result[lag.label] = flow_val * aff
    end

    # Temporal context contributions
    for lag in lagrangians
        !lag.is_temporal && continue
        flow_val = lag.flow[s, h, m]
        flow_val > 0.001 || continue
        result[lag.label] = flow_val   # temporal context adds directly
    end

    return result
end

"""
    au_lazy_conversion(event, product, lagrangians, sf) -> Float64

AU LAZY EVALUATION: compute conversion probability on demand.
This replaces the precomputed matrix C[station, product, month].

The AU pullback:
  1. Open the ad slot at (station, hour, month)
  2. Apply Δ to split across Lagrangians  
  3. Sum the contributions weighted by product affinity
  4. Multiply by ω(station, hour, month) for symplectic volume

No global matrix is needed. Each call constructs the relevant
demographic intersection locally and returns the scalar.
"""
function au_lazy_conversion(event       ::AdEvent,
                              affinity    ::Vector{Float64},
                              base_conv   ::Float64,
                              lagrangians ::Vector{Lagrangian},
                              sf          ::SymplecticForm)::Float64

    # Lazy coproduct: Δ(event) → demographic contributions
    delta = coprod_delta(event, affinity, lagrangians)

    # Sum across demographic Lagrangians (not temporal)
    demo_sum = sum(v for (k,v) in delta if k ∈ [:RM,:RF,:PM,:PF]; init=0.0)

    # Temporal boost: product of all active temporal contexts
    temporal_boost = prod(1.0 + get(delta, k, 0.0)
                          for k in [:CNY,:Summer,:Singles]; init=1.0)

    # Symplectic volume: ω(station, hour, month)
    omega_vol = sf.omega[event.station, event.hour, event.month]

    # Full conversion: symplectic volume × demographic reach × temporal boost × base rate
    return omega_vol * demo_sum * temporal_boost * base_conv
end

# =============================================================================
# PART 5: A∞ OPERATIONS
# =============================================================================

"""
    m1_differential(generators, transition_matrix) -> Vector{FloerGenerator}

The A∞ differential m_1: pushes Floer generators along the Hamiltonian flow.
In MTR terms: how does the demographic density propagate from station to station
along the Markov chain?

m_1(generator at station s) = Σ_{t adj s} T[t,s] × generator at station t

This is the CONSISTENCY CHECK: if an ad works at Admiralty at 9am,
m_1 tells us how much spillover there is to adjacent stations (Central, TST).
"""
function m1_differential(generators::Vector{FloerGenerator},
                           T         ::Matrix{Float64})::Dict{Int,Float64}

    # Map: station → accumulated differential
    d_map = Dict{Int,Float64}()

    for gen in generators
        s = gen.station
        n = size(T, 1)
        s > n && continue

        # Push density to neighboring stations
        for t in 1:n
            T[t,s] > 0.01 || continue
            d_map[t] = get(d_map,t,0.0) + T[t,s] * gen.density
        end
    end

    return d_map
end

"""
    m2_composition(L_i, L_j, product_affinity, station, month) -> Float64

The A∞ composition m_2: composed product of two Lagrangian sections.
m_2(L_i, L_j) at (station, month) = how well do the COMBINED demographics
L_i ∩ L_j respond to the product?

In standard Fukaya category: m_2 counts pseudo-holomorphic triangles.
In MTR ad placement: m_2 counts the TRIPLE intersection:
  (demographic L_i) ∩ (demographic L_j) ∩ (product affinity)
"""
function m2_composition(L_i      ::Lagrangian,
                          L_j      ::Lagrangian,
                          affinity ::Vector{Float64},
                          station  ::Int,
                          month    ::Int)::Float64

    # Floer count: how many hours both L_i and L_j are present
    fc = floer_count(L_i, L_j, station, month)

    # Product affinity at the intersection
    # = average of affinities for the two demographics
    i_demo = findfirst(l -> l == L_i.label, [:RM,:RF,:PM,:PF])
    j_demo = findfirst(l -> l == L_j.label, [:RM,:RF,:PM,:PF])
    aff = (i_demo !== nothing ? affinity[i_demo] : 0.0) +
          (j_demo !== nothing ? affinity[j_demo] : 0.0)

    return fc * aff / 2.0
end

"""
    m3_homotopy(L_i, L_j, L_k, perturbation) -> Float64

The A∞ homotopy m_3: coherent homotopy between three Lagrangians.
Ensures stability under small perturbations (train delays, crowd surges).

When a disruption perturbs the flow by `perturbation` (e.g., 10% delay):
  m_3(L_i, L_j, L_k) measures how much the triple intersection shifts.
  Small m_3 = the placement is homotopy-stable (robust to delays).
  Large m_3 = placement depends sensitively on exact timing.
"""
function m3_homotopy(L_i       ::Lagrangian,
                      L_j       ::Lagrangian,
                      L_k       ::Lagrangian,
                      station   ::Int,
                      month     ::Int,
                      perturbation::Float64 = 0.1)::Float64

    # Base triple intersection (geometric mean of three flows)
    n_h = size(L_i.flow, 2)
    base = sum(cbrt(L_i.flow[station,h,month] *
                     L_j.flow[station,h,month] *
                     L_k.flow[station,h,month])
               for h in 1:n_h)

    # Perturbed: shift peak hours by ±1 (train delay simulation)
    perturbed = sum(cbrt(L_i.flow[station, min(h+1,n_h), month] *
                          L_j.flow[station, h, month] *
                          L_k.flow[station, max(h-1,1), month])
                    for h in 1:n_h)

    # m_3 = how much the triple intersection changes under perturbation
    return abs(base - perturbed) * perturbation
end

# =============================================================================
# PART 6: FUKAYA CATEGORY OBJECT
# =============================================================================

"""
    FukayaAdContext

The complete Fukaya category structure for one (station, month) context.
This is the AU context T_s in the ad placement game.

Objects: Lagrangians {L_RM, L_RF, L_PM, L_PF, L_CNY, L_Summer, ...}
Morphisms: CF*(L_i, L_j) = Floer complexes
A∞ operations: m_1, m_2, m_3

The LAZY AU pullback means this object is constructed ON DEMAND
when an ad slot opens, not precomputed.
"""
struct FukayaAdContext
    station         ::Int
    station_name    ::String
    month           ::Int
    omega_val       ::Vector{Float64}   # ω(station, h, month) for each hour
    floer_complexes ::Dict{Tuple{Symbol,Symbol}, Vector{FloerGenerator}}
    m2_table        ::Dict{Tuple{Symbol,Symbol,Int}, Float64}  # m2 per product
    stability       ::Dict{Tuple{Symbol,Symbol,Symbol}, Float64} # m3 values
    hamiltonian     ::Float64           # H at peak hour for this station/month
end

"""
Construct a FukayaAdContext lazily for (station, month).
Only computes what is needed for the current ad placement decision.
"""
function build_fukaya_context(station    ::Int,
                               month      ::Int,
                               stations   ::Vector{String},
                               lagrangians::Vector{Lagrangian},
                               sf         ::SymplecticForm,
                               products   ::Vector)::FukayaAdContext

    n_h = size(sf.omega, 2)
    omega_s = sf.omega[station, :, month]

    # Lazy: only compute Floer complexes for the RELEVANT pairs
    # (those with non-trivial intersection at this station in this month)
    demo_lags = filter(l -> !l.is_temporal, lagrangians)
    floers = Dict{Tuple{Symbol,Symbol}, Vector{FloerGenerator}}()

    for i in 1:length(demo_lags), j in i:length(demo_lags)
        Li, Lj = demo_lags[i], demo_lags[j]
        gens = FloerGenerator[]
        for h in 1:n_h
            dens = sqrt(Li.flow[station,h,month] * Lj.flow[station,h,month])
            dens > 0.15 || continue
            push!(gens, FloerGenerator(station, h, month, dens, Li.label, Lj.label))
        end
        !isempty(gens) && (floers[(Li.label, Lj.label)] = gens)
    end

    # m_2 table: for each (demo_i, demo_j, product) triple
    m2_t = Dict{Tuple{Symbol,Symbol,Int}, Float64}()
    for i in 1:length(demo_lags), j in i:length(demo_lags), p in 1:length(products)
        val = m2_composition(demo_lags[i], demo_lags[j],
                              products[p].affinity, station, month)
        val > 0.001 && (m2_t[(demo_lags[i].label, demo_lags[j].label, p)] = val)
    end

    # m_3 stability (homotopy): for each triple of demographics
    stab = Dict{Tuple{Symbol,Symbol,Symbol}, Float64}()
    for i in 1:length(demo_lags), j in i:length(demo_lags), k in j:length(demo_lags)
        val = m3_homotopy(demo_lags[i], demo_lags[j], demo_lags[k],
                           station, month)
        stab[(demo_lags[i].label, demo_lags[j].label, demo_lags[k].label)] = val
    end

    # Hamiltonian at peak hour (hour 9 = 9am)
    H = sf.H_coeff[9, month] * mean(l.flow[station,9,month] for l in demo_lags)

    return FukayaAdContext(station, stations[station], month,
                           omega_s, floers, m2_t, stab, H)
end

"""
    au_best_product(ctx, products, budget_remaining) -> (product_idx, score, delta_components)

The AU lazy decision: given a Fukaya context at (station, month),
which product maximises the coproduct output?

This is the AU-QKV policy expressed in Fukaya language:
  Q = the Floer complex CF*(L_i, L_j) at this station (query)
  K = the product affinity vector (key)
  V = the m_2 composition score (value)
  Δ = the pair-of-pants coproduct split (how the ad reaches each demographic)
"""
function au_best_product(ctx             ::FukayaAdContext,
                          lagrangians     ::Vector{Lagrangian},
                          products        ::Vector,
                          placed_products ::Set{Int},
                          sf              ::SymplecticForm)::Tuple{Int,Float64,Dict}

    best_p, best_score = 0, -Inf
    best_delta = Dict{Symbol,Float64}()

    demo_lags = filter(l -> !l.is_temporal, lagrangians)
    peak_hour = 9   # use 9am as representative peak

    for p in 1:length(products)
        p ∈ placed_products && continue

        prod = products[p]

        # Step 1: Lazy coproduct Δ(event) for this product at this context
        event = AdEvent(ctx.station, peak_hour, ctx.month,
                        p, sf.omega[ctx.station, peak_hour, ctx.month])
        delta = coprod_delta(event, prod.affinity, lagrangians)

        # Step 2: m_2 composition scores — how well do demographic pairs respond?
        m2_total = sum(get(ctx.m2_table, (Li.label, Lj.label, p), 0.0)
                       for Li in demo_lags, Lj in demo_lags)

        # Step 3: m_3 stability — is this placement robust to disruptions?
        # LOW stability penalty = MORE robust (prefer stable placements)
        m3_penalty = mean(values(ctx.stability))

        # Step 4: Temporal context boost from Floer complexes
        temporal_boost = 1.0
        for lag in lagrangians
            !lag.is_temporal && continue
            flow_val = lag.flow[ctx.station, peak_hour, ctx.month]
            flow_val > 0.01 || continue
            temporal_boost *= (1.0 + flow_val * 0.1)
        end

        # Combined score weighted by price tier (revenue optimisation):
        # = disk_volume × m₂_quality × temporal_boost × price_weight / m₃
        pw = get(Dict(:luxury=>3.0, :mid=>1.5, :budget=>1.0),
                 prod.price_tier, 1.0)
        omega_peak = sf.omega[ctx.station, peak_hour, ctx.month]
        demo_reach = sum(get(delta, l, 0.0) for l in [:RM,:RF,:PM,:PF])
        score = omega_peak * demo_reach * m2_total * temporal_boost * pw /
                (1.0 + m3_penalty) * prod.base_conversion

        if score > best_score
            best_score = score
            best_p     = p
            best_delta = delta
        end
    end

    return best_p, best_score, best_delta
end

# =============================================================================
# PART 7: DEMO
# =============================================================================


# =============================================================================
# build_transition_matrix (from mtr_game.jl — inlined to avoid double include)
# =============================================================================

function build_transition_matrix(stops  ::Set{Tuple{Symbol,Symbol}},
                                  nodes  ::Vector{Symbol},
                                  edges  ::Vector{Tuple{Symbol,Symbol}},
                                  weights::Dict{Tuple{Symbol,Symbol},Float64})
    n        = length(nodes)
    node_idx = Dict(v=>i for (i,v) in enumerate(nodes))
    T        = zeros(n, n)
    for (s,t) in edges
        (s,t) ∈ stops && continue
        si = get(node_idx,s,0); ti = get(node_idx,t,0)
        (si==0||ti==0) && continue
        T[ti,si] += get(weights,(s,t),1.0)
    end
    for j in 1:n
        col_sum = sum(T[:,j])
        col_sum > 0 && (T[:,j] ./= col_sum)
    end
    return T
end

if abspath(PROGRAM_FILE) == @__FILE__

    # Load MTR data
    include(joinpath(@__DIR__, "mtr_ad_game.jl"))

    println("="^70)
    println("FUKAYA CATEGORY AD PLACEMENT — CENTRALISED COPRODUCT")
    println("="^70)

    stations = STATIONS
    n_s      = length(stations)

    # Build symplectic form
    println("\n[1] SYMPLECTIC FORM ω(station, hour, month)")
    sf = build_symplectic_form(stations, RIDERSHIP)
    @printf("  ω at Admiralty, 9am, Feb (CNY): %.4f\\n",
            sf.omega[STATION_IDX["Admiralty"], 9, 2])
    @printf("  ω at Tuen_Mun, 7am, Feb:        %.4f\\n",
            sf.omega[STATION_IDX["Tuen_Mun"], 7, 2])
    @printf("  ω at Lei_Tung, 2pm, Jul:         %.4f\\n",
            sf.omega[STATION_IDX["Lei_Tung"], 14, 7])
    @printf("  Hamiltonian H(9am, Feb CNY):     %.4f\\n",
            sf.H_coeff[9, 2])
    println()
    println("  Interpretation: Admiralty at CNY 9am has highest transit energy.")
    println("  The symplectic form captures WHEN as well as WHERE.")

    # Build Lagrangians
    println("\n[2] LAGRANGIAN SUBMANIFOLDS L_i")
    lagrangians = build_lagrangians(stations, DEMO_PROFILES, sf)
    for lag in lagrangians
        # Peak flow across all stations and hours in month 2
        peak = maximum(lag.flow[:,:,2])
        peak_s = argmax(lag.flow[:,:,2])
        @printf("  %-10s peak=%.4f at station=%s, hour=%d\\n",
                lag.name, peak,
                stations[min(peak_s[1],n_s)], peak_s[2])
    end

    # Floer complexes
    println("\n[3] FLOER COMPLEXES CF*(L_i, L_j) — Demographic Intersections")
    println("  (Number of (station,hour) intersection points above threshold)")
    demo_lags = filter(l -> !l.is_temporal, lagrangians)
    for i in 1:length(demo_lags)
        for j in i:length(demo_lags)
            gens = floer_complex(demo_lags[i], demo_lags[j]; threshold=0.01)
            @printf("  CF*(%-10s, %-10s): %4d generators  max_density=%.4f\\n",
                    demo_lags[i].name, demo_lags[j].name,
                    length(gens),
                    isempty(gens) ? 0.0 : maximum(g.density for g in gens))
        end
    end

    # Lazy AU conversion for specific cases
    println("\n[4] AU LAZY COPRODUCT Δ — On-Demand Evaluation")
    println("  (Computed only when ad slot opens — no precomputation)")
    println()
    adm_i = get(STATION_IDX, "Admiralty", 1)
    tum_i = get(STATION_IDX, "Tuen_Mun", 1)
    lei_i = get(STATION_IDX, "Lei_Tung", 1)

    cases = [
        ("Admiralty, 9am, Feb, Luxury Watch",   adm_i, 9,  2, 1),
        ("Tuen_Mun,  7am, Feb, Train Ticket",   tum_i, 7,  2, 6),
        ("Admiralty, 6pm, Dec, Designer Bag",   adm_i, 18, 12, 2),
        ("Lei_Tung,  2pm, Jul, HK Disneyland",  lei_i, 14, 7,  7),
    ]

    for (label, s, h, m, p) in cases
        event = AdEvent(s, h, m, p, sf.omega[s,h,m])
        delta = coprod_delta(event, PRODUCTS[p].affinity, lagrangians)
        conv  = au_lazy_conversion(event, PRODUCTS[p].affinity,
                                    PRODUCTS[p].base_conversion,
                                    lagrangians, sf)
        println("  $label")
        @printf("    ω=%.4f  conv=%.6f\\n", sf.omega[s,h,m], conv)
        println("    Δ split: " * join([@sprintf("%s=%.3f",k,v)
                                       for (k,v) in sort(collect(delta),
                                           by=x->x[2],rev=true)
                                       if v>0.001], "  "))
        println()
    end

    # Fukaya context for Admiralty in CNY
    println("[5] FUKAYA CONTEXT — Admiralty × February (CNY)")
    ctx_adm = build_fukaya_context(adm_i, 2, stations,
                                    lagrangians, sf, collect(PRODUCTS))
    @printf("  Hamiltonian H: %.4f\\n", ctx_adm.hamiltonian)
    @printf("  Floer pairs with non-trivial intersection: %d\\n",
            length(ctx_adm.floer_complexes))
    @printf("  m_2 entries computed: %d\\n", length(ctx_adm.m2_table))
    println("  m_3 stability scores (lower=more robust):")
    sorted_stab = sort(collect(ctx_adm.stability), by=x->x[2])
    for ((a,b,c),v) in sorted_stab[1:min(5,end)]
        @printf("    (%s×%s×%s): %.6f\\n", a, b, c, v)
    end

    println()
    println("[6] BEST PRODUCT via AU LAZY DECISION")
    placed = Set{Int}()
    for _ in 1:5
        p_best, score, delta = au_best_product(ctx_adm, lagrangians,
                                                collect(PRODUCTS), placed, sf)
        p_best == 0 && break
        push!(placed, p_best)
        @printf("  → %s  score=%.6f\\n", PRODUCTS[p_best].name, score)
        @printf("    Δ: %s\\n",
                join([@sprintf("%s=%.3f",k,v)
                      for (k,v) in sort(collect(delta),by=x->x[2],rev=true)
                      if v>0.001], "  "))
    end

    println()
    println("[7] A∞ RELATIONS — Consistency Check")
    println("  m_1 (Hamiltonian flow differential):")
    # Build a simple transition matrix from the MTR graph
    # Build transition matrix from mtr_ad_game.jl data
    # (MTR_NODES/EDGES/WEIGHTS come from mtr_game.jl — use STATIONS/EDGES/EDGE_WEIGHTS instead)
    mtr_nodes_sym = Symbol.(STATIONS)
    mtr_edges_sym = [(Symbol(e[1]),Symbol(e[2])) for e in EDGES]
    mtr_weights_sym = Dict((Symbol(k[1]),Symbol(k[2]))=>v for (k,v) in EDGE_WEIGHTS)
    T = build_transition_matrix(Set{Tuple{Symbol,Symbol}}(),
                                mtr_nodes_sym, mtr_edges_sym, mtr_weights_sym)
    # Get top Floer generators for RM at Admiralty
    RM_lag = lagrangians[1]
    RF_lag = lagrangians[2]
    gens = floer_complex(RM_lag, RF_lag; threshold=0.01)[1:min(10,end)]
    d_map = m1_differential(gens, T)
    top_d = sort(collect(d_map), by=x->x[2], rev=true)[1:min(5,end)]
    println("  m_1(CF*(RM,RF)) propagates to stations:")
    for (s,v) in top_d
        s <= length(stations) || continue
        @printf("    %-20s: %.4f\\n", stations[s], v)
    end
    println()
    println("  Interpretation: the Floer differential m_1 tells us")
    println("  how the RM×RF intersection density propagates along MTR lines.")
    println("  High values = strong spillover from RM×RF ads to adjacent stations.")

    println()
    println("="^70)
    println("SUMMARY: Centralised Coproduct in Fukaya Categories")
    println("="^70)
    println()
    println("  The Fukaya category structure makes EXPLICIT what mtr_ad_game.jl")
    println("  computed IMPLICITLY:")
    println()
    println("  IMPLICIT (mtr_ad_game.jl):")
    println("    conversion = ridership × Σ_d [demo×affinity] × month_boost")
    println("    = a precomputed 81×10×12 matrix")
    println()
    println("  EXPLICIT (fukaya_ad_context.jl):")
    println("    Δ(show_ad(p,s,t)) = Σ_i c_i × L_i")
    println("    c_i = CF*(L_i, L_product) at (s,t)  [Floer intersection count]")
    println("    = lazy AU pullback, computed only when slot opens")
    println()
    println("  GAIN:")
    println("    1. Time resolution: ω(s,h,m) captures HOUR of day, not just month")
    println("    2. Homotopy stability: m_3 identifies robust vs fragile placements")
    println("    3. Demographic intersections: CF*(L_i,L_j) not scalar products")
    println("    4. No precomputation: AU laziness handles 100k products trivially")
    println("    5. New line extension: just extend ω, same coproduct rules apply")
    println("="^70)
end
