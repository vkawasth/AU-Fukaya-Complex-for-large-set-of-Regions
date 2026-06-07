# =============================================================================
# au_fukaya_engine.jl
#
# The Complete AU-Fukaya Lazy Demographic Engine
# Implements the architecture diagram exactly:
#
#   TOP:    AU Attic — populations as unmaterialised coproducts
#   MIDDLE: Fukaya category over MTR manifold — Hamiltonian flow
#   BOTTOM: Pair-of-pants coproduct — ad slot splits across demographics
#
# The three layers communicate via:
#   AU pullback surgery → Floer complex → moduli space count → Δ
#
# No precomputation. Everything evaluated lazily at the surgery node.
# =============================================================================

using LinearAlgebra, Printf, Statistics

# =============================================================================
# LAYER 1: THE AU ATTIC — Unmaterialised Coproducts
# =============================================================================
# Populations sit as abstract coproduct objects. No memory allocated.
# They are FORMAL SYMBOLS until a surgery node demands their realisation.

"""
    AUObject

An unmaterialised population object in Joyal's Arithmetic Universe.
Exists as a formal coproduct symbol: Male ⊔ Female, Rich ⊔ Poor, etc.
NOT evaluated until pulled back through a surgery node.

The AU guarantees: strict constructivism — objects exist only when
an algebraic operation DEMANDS them at a specific (station, time) point.
"""
struct AUObject
    label    ::Symbol                    # :Male, :Female, :Rich, :Poor, ...
    axes     ::Vector{Symbol}            # dimensions it spans
    pullback ::Function                  # lazy evaluator: (station,hour,month) → Float64
    # The pullback function is NOT called here — only when surgery fires
end

"""
    AUCoproduct

A formal coproduct A ⊔ B in the AU.
Unmaterialised — holds the two objects and the injection maps.
Evaluated lazily via the pullback functor when a surgery node fires.
"""
struct AUCoproduct
    left     ::AUObject
    right    ::AUObject
    label    ::String    # "Male ⊔ Female" etc.
end

"""
    au_pullback(coprod, station, hour, month) -> (left_val, right_val)

The AU surgery: materialise the coproduct at a specific (station,hour,month).
Called ONLY when the Hamiltonian flow reaches this node.
This is the "AU pullback surgery" in the diagram.
"""
function au_pullback(coprod::AUCoproduct,
                     station::Int, hour::Int, month::Int)
    l = coprod.left.pullback(station, hour, month)
    r = coprod.right.pullback(station, hour, month)
    return l, r
end

# =============================================================================
# BUILD THE AU ATTIC
# =============================================================================

"""
Build all population coproducts in the AU attic.
These are FORMAL objects — no computation happens here.
The pullback functions are closures that will only execute
when a surgery node fires.
"""
function build_au_attic(stations      ::Vector{String},
                         demo_profiles ::Matrix{Float64},   # [station, 4]
                         ridership     ::Vector{Float64},
                         sf_omega      ::Array{Float64,3})  # [station,hour,month]

    n_s = length(stations)

    # ── Gender axis ───────────────────────────────────────────────────────────
    Male = AUObject(:Male, [:gender],
        (s,h,m) -> begin
            # Males = RM + PM fractions, modulated by hour
            hour_w = h ∈ 7:10 ? 1.4 : h ∈ 17:20 ? 1.3 : 1.0
            (demo_profiles[min(s,n_s),1] + demo_profiles[min(s,n_s),3]) *
            sf_omega[min(s,n_s),h,m] * hour_w
        end)

    Female = AUObject(:Female, [:gender],
        (s,h,m) -> begin
            hour_w = h ∈ 10:20 ? 1.3 : 1.0
            (demo_profiles[min(s,n_s),2] + demo_profiles[min(s,n_s),4]) *
            sf_omega[min(s,n_s),h,m] * hour_w
        end)

    # ── Income axis ───────────────────────────────────────────────────────────
    Rich = AUObject(:Rich, [:income],
        (s,h,m) -> begin
            (demo_profiles[min(s,n_s),1] + demo_profiles[min(s,n_s),2]) *
            sf_omega[min(s,n_s),h,m]
        end)

    Poor = AUObject(:Poor, [:income],
        (s,h,m) -> begin
            (demo_profiles[min(s,n_s),3] + demo_profiles[min(s,n_s),4]) *
            sf_omega[min(s,n_s),h,m]
        end)

    # ── Education axis (proxy: station type) ─────────────────────────────────
    business_stations = Set(["Central","Admiralty","Sheung_Wan","Austin",
                              "Kowloon","Tsim_Sha_Tsui"])
    Educated = AUObject(:Educated, [:education],
        (s,h,m) -> begin
            boost = stations[min(s,n_s)] ∈ business_stations ? 1.4 : 0.8
            demo_profiles[min(s,n_s),1] * boost *    # RM dominantly educated
            sf_omega[min(s,n_s),h,m]
        end)

    Uneducated = AUObject(:Uneducated, [:education],
        (s,h,m) -> begin
            boost = stations[min(s,n_s)] ∈ business_stations ? 0.6 : 1.2
            demo_profiles[min(s,n_s),3] * boost *
            sf_omega[min(s,n_s),h,m]
        end)

    # ── Temporal objects (seasonal contexts) ─────────────────────────────────
    CNY = AUObject(:CNY, [:time],
        (s,h,m) -> m ∈ [1,2] ? sf_omega[min(s,n_s),h,m] * 2.5 : 0.0)

    Summer = AUObject(:Summer, [:time],
        (s,h,m) -> m ∈ [7,8] ? sf_omega[min(s,n_s),h,m] * 1.8 : 0.0)

    Singles = AUObject(:Singles, [:time],
        (s,h,m) -> m ∈ [10,11] ? sf_omega[min(s,n_s),h,m] * 2.0 : 0.0)

    Christmas = AUObject(:Christmas, [:time],
        (s,h,m) -> m == 12 ? sf_omega[min(s,n_s),h,m] * 1.8 : 0.0)

    # ── Formal coproducts (AU Attic objects) ──────────────────────────────────
    attic = Dict{String, AUCoproduct}(
        "gender"    => AUCoproduct(Male,      Female,    "Male ⊔ Female"),
        "income"    => AUCoproduct(Rich,      Poor,      "Rich ⊔ Poor"),
        "education" => AUCoproduct(Educated,  Uneducated,"Educated ⊔ Uneducated"),
        "cny"       => AUCoproduct(CNY,       Summer,    "CNY ⊔ Summer"),
        "seasonal"  => AUCoproduct(Singles,   Christmas, "Singles ⊔ Christmas"),
    )

    primitives = Dict{Symbol,AUObject}(
        :Male=>Male, :Female=>Female, :Rich=>Rich, :Poor=>Poor,
        :Educated=>Educated, :Uneducated=>Uneducated,
        :CNY=>CNY, :Summer=>Summer, :Singles=>Singles, :Christmas=>Christmas,
    )

    return attic, primitives
end

# =============================================================================
# LAYER 2: HAMILTONIAN FLOW ON THE MTR MANIFOLD
# =============================================================================

"""
    HamiltonianState

The current state of the Hamiltonian flow on the MTR manifold.
Tracks: which station the flow is at, the current time (hour, month),
and the accumulated flow history (m₁ differential chain).
"""
mutable struct HamiltonianState
    station     ::Int
    hour        ::Int
    month       ::Int
    flow_history::Vector{Tuple{Int,Int,Int,Float64}}  # (s,h,m,ω) visited
    chain       ::Vector{Float64}    # m₁ differential chain values
    omega_val   ::Float64            # ω at current (s,h,m)
end

"""
    hamiltonian_step!(state, T, omega) -> HamiltonianState

Advance the Hamiltonian flow one step: dx/dt = ∂H/∂p
In discrete terms: apply T (transition matrix) to propagate
the flow from current station to adjacent stations.
This is the m₁ differential in the Floer complex.
"""
function hamiltonian_step!(state ::HamiltonianState,
                             T     ::Matrix{Float64},
                             omega ::Array{Float64,3})

    s, h, m = state.station, state.hour, state.month
    n       = size(T, 1)
    s > n   && return state

    # Find next station: argmax T[:,s] (highest flow destination)
    probs = T[:,s]
    next_s = argmax(probs)

    # Advance hour (modular)
    next_h = mod(h, 24) + 1
    next_ω = omega[min(next_s,size(omega,1)), next_h, m]

    push!(state.flow_history, (s, h, m, state.omega_val))
    push!(state.chain, state.omega_val)

    state.station  = next_s
    state.hour     = next_h
    state.omega_val = next_ω

    return state
end

# =============================================================================
# LAYER 3: THE SURGERY NODE — Lazy AU Pullback
# =============================================================================

"""
    SurgeryNode

An AU Surgery Node at a specific (station, hour, month).
When the Hamiltonian flow reaches this node, it fires the lazy pullback:
  - Materialises the relevant AU coproducts
  - Computes the local demographic intersection
  - Returns the instantiated Floer complex generators
  
This is the "Admiralty Surgery Node" in the diagram.
Nothing computed until the flow arrives here.
"""
struct SurgeryNode
    station  ::Int
    station_name::String
    hour     ::Int
    month    ::Int
    omega    ::Float64
end

"""
    fire_surgery!(node, attic, primitives, product_affinity)
        -> (floer_generators, coproduct_split)

Execute the AU surgery at this node:
  1. Pull back all AU coproducts to (station, hour, month)
  2. Compute Floer intersection at this local point
  3. Return the materialised demographic split

This is the core lazy evaluation — happens only at surgery nodes,
not globally. This is how 100k products scale: each product's
interaction with demographics is computed ON DEMAND here,
not stored in a precomputed table.
"""
function fire_surgery!(node            ::SurgeryNode,
                        attic           ::Dict{String,AUCoproduct},
                        primitives      ::Dict{Symbol,AUObject},
                        product_affinity::Vector{Float64},  # [RM,RF,PM,PF]
                        base_conversion ::Float64)

    s, h, m = node.station, node.hour, node.month

    # ── Step 1: Materialise AU coproducts at this node ───────────────────────
    materialised = Dict{String, Tuple{Float64,Float64}}()
    for (axis, coprod) in attic
        l, r = au_pullback(coprod, s, h, m)
        materialised[axis] = (l, r)
    end

    # ── Step 2: Floer intersection — demographic co-presence ─────────────────
    # For each pair of primitive objects, compute their intersection at (s,h,m)
    # CF*(L_i, L_j) at this node = √(L_i(s,h,m) × L_j(s,h,m))
    floer_local = Dict{Tuple{Symbol,Symbol}, Float64}()
    demo_keys = [:Male, :Female, :Rich, :Poor, :Educated]
    for i in 1:length(demo_keys), j in i:length(demo_keys)
        ki, kj = demo_keys[i], demo_keys[j]
        vi = get(primitives, ki, nothing)
        vj = get(primitives, kj, nothing)
        (vi === nothing || vj === nothing) && continue
        dens = sqrt(vi.pullback(s,h,m) * vj.pullback(s,h,m))
        dens > 0.001 && (floer_local[(ki,kj)] = dens)
    end

    # ── Step 3: Temporal context at this node ────────────────────────────────
    temporal_active = Symbol[]
    cny_val  = get(primitives,:CNY,nothing)
    sum_val  = get(primitives,:Summer,nothing)
    sin_val  = get(primitives,:Singles,nothing)
    xms_val  = get(primitives,:Christmas,nothing)
    cny_val  !== nothing && cny_val.pullback(s,h,m)  > 0.01 && push!(temporal_active,:CNY)
    sum_val  !== nothing && sum_val.pullback(s,h,m)  > 0.01 && push!(temporal_active,:Summer)
    sin_val  !== nothing && sin_val.pullback(s,h,m)  > 0.01 && push!(temporal_active,:Singles)
    xms_val  !== nothing && xms_val.pullback(s,h,m)  > 0.01 && push!(temporal_active,:Christmas)

    # ── Step 4: Pair-of-pants coproduct Δ ─────────────────────────────────────
    # Δ(ad_slot) → Σ_i c_i × L_i
    # c_i = Floer density × product affinity toward demographic i
    demo_map = Dict(:RM=>1, :RF=>2, :PM=>3, :PF=>4)
    # Map primitives to product affinity dimensions
    prim_to_aff = Dict(
        :Male      => (product_affinity[1] + product_affinity[3]) / 2,
        :Female    => (product_affinity[2] + product_affinity[4]) / 2,
        :Rich      => (product_affinity[1] + product_affinity[2]) / 2,
        :Poor      => (product_affinity[3] + product_affinity[4]) / 2,
        :Educated  =>  product_affinity[1],
    )

    # The coproduct output: how much of the ad reaches each demographic
    delta = Dict{Symbol,Float64}()
    for ((ki,kj), dens) in floer_local
        aff_i = get(prim_to_aff, ki, 0.0)
        aff_j = get(prim_to_aff, kj, 0.0)
        combined_aff = (aff_i + aff_j) / 2.0
        key = Symbol("$(ki)×$(kj)")
        delta[key] = dens * combined_aff * node.omega
    end

    # ── Step 5: Moduli space count — disk volume ──────────────────────────────
    # #𝔐(Ad; L_i, L_j) ≈ Σ_{i,j} Floer_density(i,j) × affinity_match × ω
    disk_volume = sum(values(delta))

    # Temporal boost: seasonal context multiplies the disk volume
    temporal_mult = prod(temporal_active; init=1.0) do t
        obj = get(primitives, t, nothing)
        obj === nothing ? 1.0 : 1.0 + obj.pullback(s,h,m) * 0.2
    end

    # Final conversion: disk_volume × temporal × base_rate
    conversion = disk_volume * temporal_mult * base_conversion

    return (
        floer_generators = floer_local,
        coproduct_split  = delta,
        temporal_active  = temporal_active,
        disk_volume      = disk_volume,
        temporal_mult    = temporal_mult,
        conversion       = conversion,
        materialised_au  = materialised,
    )
end

# =============================================================================
# THE ENGINE: Flowing Through the MTR, Firing Surgery Nodes
# =============================================================================

"""
    AUFukayaEngine

The complete lazy engine.
Passengers flow through the MTR Hamiltonian system.
At each major station, an AU surgery node fires.
The coproduct Δ assigns ads to demographics on demand.
"""
struct AUFukayaEngine
    attic      ::Dict{String,AUCoproduct}
    primitives ::Dict{Symbol,AUObject}
    omega      ::Array{Float64,3}
    stations   ::Vector{String}
    T          ::Matrix{Float64}   # Markov transition matrix
end

"""
    flow_and_place!(engine, source, month, products; n_steps, budget)

Run the Hamiltonian flow from `source` station for `n_steps`,
firing surgery nodes at each major station and placing the best ad.

This is the complete pipeline:
  1. Initialise Hamiltonian flow at source
  2. Advance flow (m₁ differential: station-to-station)
  3. At each interchange station: fire AU surgery
  4. The surgery returns Δ(ad_slot) → demographics
  5. Select the product that maximises disk volume
  6. Record placement
"""
function flow_and_place!(engine   ::AUFukayaEngine,
                          source   ::Int,
                          month    ::Int,
                          products ::Vector;
                          n_steps  ::Int  = 20,
                          budget   ::Int  = 10,
                          verbose  ::Bool = true)

    # Interchange stations: surgery fires here (high demographic mixing)
    major = Set(["Central","Admiralty","Tsim_Sha_Tsui","Hung_Hom","Austin",
                 "Ho_Man_Tin","Diamond_Hill","Kowloon_Tong","Prince_Edward",
                 "Mong_Kok","North_Point","Quarry_Bay"])

    state = HamiltonianState(source, 9, month, [], [], 
                              engine.omega[min(source,size(engine.omega,1)),9,month])
    placements  = NamedTuple[]
    placed_prod = Set{Int}()
    budget_used = 0

    verbose && begin
        println("  Hamiltonian flow: $(engine.stations[min(source,end)]) → network")
        println("  ─"^52)
        @printf("  %-20s %-22s %-8s %-6s\n", "Surgery Node","Best Ad","Conv","Δ-keys")
        println("  " * "─"^60)
    end

    for step in 1:n_steps
        budget_used >= budget && break

        s    = state.station
        s_nm = engine.stations[min(s,end)]

        # Fire surgery only at major interchange stations
        if s_nm ∈ major
            node = SurgeryNode(s, s_nm, state.hour, month,
                               engine.omega[min(s,size(engine.omega,1)),
                                            state.hour, month])

            # Find best unmaterialised product for this node
            best_p, best_conv, best_result = 0, -Inf, nothing

            for p in 1:length(products)
                p ∈ placed_prod && continue
                prod = products[p]

                result = fire_surgery!(node, engine.attic, engine.primitives,
                                       prod.affinity, prod.base_conversion)

                if result.conversion > best_conv
                    best_conv   = result.conversion
                    best_p      = p
                    best_result = result
                end
            end

            if best_p > 0 && best_conv > 0.001
                push!(placed_prod, best_p)
                budget_used += 1
                push!(placements, (
                    step         = step,
                    station      = s_nm,
                    product      = products[best_p].name,
                    conversion   = best_conv,
                    disk_volume  = best_result.disk_volume,
                    temporal     = best_result.temporal_active,
                    n_delta      = length(best_result.coproduct_split),
                    n_floer      = length(best_result.floer_generators),
                ))

                verbose && begin
                    temp_s = isempty(best_result.temporal_active) ? "—" :
                             join(string.(best_result.temporal_active), ",")
                    @printf("  %-20s %-22s %-8.4f %d Δ-terms [%s]\n",
                            first(s_nm,20), first(products[best_p].name,22),
                            best_conv, length(best_result.coproduct_split), temp_s)
                end
            end
        end

        # Advance Hamiltonian flow (m₁ differential)
        hamiltonian_step!(state, engine.T, engine.omega)
    end

    verbose && println("  " * "─"^60)
    return placements
end

# =============================================================================
# DEMO
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

    include(joinpath(@__DIR__, "mtr_ad_game.jl"))

    println()
    println("╔" * "═"^68 * "╗")
    println("║  AU-FUKAYA LAZY DEMOGRAPHIC ENGINE                                  ║")
    println("║  Joyal Arithmetic Universe × Fukaya Categories × MTR               ║")
    println("╚" * "═"^68 * "╝")

    # Build symplectic form
    line_ridership_flat = Dict{String,Float64}()
    for (line,(seq,w)) in LINE_SEQ
        for s in seq; line_ridership_flat[s] = max(get(line_ridership_flat,s,0.0),w); end
    end
    R_vec = Float64[get(line_ridership_flat,s,50.0) for s in STATIONS]
    R_vec ./= maximum(R_vec)

    # 81 × 24 × 12 symplectic form
    n_s = length(STATIONS)
    hour_profile = [0.2,0.1,0.1,0.1,0.2,0.4,0.8,1.0,0.9,0.7,0.6,0.7,
                    0.7,0.6,0.5,0.6,0.7,0.9,1.0,0.9,0.8,0.6,0.5,0.3]
    month_res    = [1.2,1.5,1.1,1.0,1.0,1.0,1.1,1.1,1.0,1.2,1.3,1.4]
    omega = zeros(n_s,24,12)
    for s in 1:n_s, h in 1:24, m in 1:12
        omega[s,h,m] = R_vec[s] * hour_profile[h] * month_res[m]
    end

    # Build AU attic
    println("\n[AU ATTIC] Building unmaterialised coproducts...")
    attic, primitives = build_au_attic(STATIONS, DEMO_PROFILES, R_vec, omega)
    println("  Coproducts in attic (unmaterialised, zero memory cost):")
    for (k,v) in attic
        println("    $(v.label)  [formal — not evaluated yet]")
    end

    # Build transition matrix for Hamiltonian flow
    mtr_nodes_sym   = Symbol.(STATIONS)
    mtr_edges_sym   = [(Symbol(e[1]),Symbol(e[2])) for e in EDGES]
    mtr_weights_sym = Dict((Symbol(k[1]),Symbol(k[2]))=>v for (k,v) in EDGE_WEIGHTS)
    T = build_transition_matrix(Set{Tuple{Symbol,Symbol}}(),
                                mtr_nodes_sym, mtr_edges_sym, mtr_weights_sym)

    engine = AUFukayaEngine(attic, primitives, omega, STATIONS, T)

    println()
    println("[HAMILTONIAN FLOW] Passenger flow from Tuen_Mun (outskirts)")
    println("─"^70)

    for (month_name, month_idx) in [("February (CNY)",2),("July (Summer)",7),
                                     ("November (11.11)",11)]
        println("\n  Season: $month_name")
        src = get(STATION_IDX,"Tuen_Mun",1)
        placements = flow_and_place!(engine, src, month_idx,
                                      collect(PRODUCTS);
                                      n_steps=25, budget=6)
        println()
        @printf("  Placed %d ads via lazy surgery. Total conv = %.4f\n",
                length(placements),
                sum(p.conversion for p in placements; init=0.0))
    end

    println()
    println("[SURGERY NODE DEMO] Admiralty at CNY 9am — full materialisation")
    println("─"^70)
    adm_i = get(STATION_IDX,"Admiralty",1)
    node  = SurgeryNode(adm_i,"Admiralty",9,2,omega[adm_i,9,2])

    println("\n  AU Attic objects being materialised at this node:")
    for (axis,coprod) in attic
        l, r = au_pullback(coprod, adm_i, 9, 2)
        @printf("    %-12s: %s=%.4f  %s=%.4f\n",
                coprod.label, coprod.left.label, l, coprod.right.label, r)
    end

    println("\n  Firing surgery for each product:")
    println("  ─"^60)
    @printf("  %-22s %-8s %-6s %-20s %-10s\n",
            "Product","Conv","Floer","Top Δ-term","Temporal")
    println("  " * "─"^68)

    results_all = []
    for prod in PRODUCTS
        res = fire_surgery!(node, attic, primitives,
                            prod.affinity, prod.base_conversion)
        top_delta = isempty(res.coproduct_split) ? ("—",0.0) :
                    sort(collect(res.coproduct_split),by=x->x[2],rev=true)[1]
        push!(results_all, (prod.name, res.conversion, res, top_delta))
    end
    sort!(results_all, by=x->x[2], rev=true)

    for (name, conv, res, top_d) in results_all
        temp_s = isempty(res.temporal_active) ? "—" :
                 join(string.(res.temporal_active),",")
        @printf("  %-22s %-8.4f %-6d %-20s %-10s\n",
                first(name,22), conv,
                length(res.floer_generators),
                first(string(top_d[1]),20),
                temp_s)
    end

    println()
    println("  The pair-of-pants coproduct Δ at Admiralty CNY 9am:")
    println("  Δ(ad_slot) → demographic components:")
    best_res = results_all[1][3]
    for (k,v) in sort(collect(best_res.coproduct_split),by=x->x[2],rev=true)[1:min(5,end)]
        bar = "█"^Int(round(v/maximum(values(best_res.coproduct_split))*15))
        @printf("    %-25s %.4f  %s\n", k, v, bar)
    end
    @printf("  Disk volume ∫𝔐 = %.4f  × temporal %.2fx = conv %.4f\n",
            best_res.disk_volume, best_res.temporal_mult, best_res.conversion)

    println()
    println("╔" * "═"^68 * "╗")
    println("║  ARCHITECTURAL SUMMARY                                              ║")
    println("╠" * "═"^68 * "╣")
    println("║  AU Attic:     Coproducts exist as FORMAL SYMBOLS                  ║")
    println("║                Male⊔Female, Rich⊔Poor — no memory until surgery    ║")
    println("║  Hamiltonian:  Passengers flow dx/dt=∂H/∂p through MTR manifold    ║")
    println("║                H(t) changes with hour, month, season                ║")
    println("║  Surgery Node: AU pullback fires at interchange stations            ║")
    println("║                Materialises ONLY the needed demographic slice       ║")
    println("║  Floer CF*:    Intersection of L_Rich ∩ L_Female at (s,h,m)        ║")
    println("║                Replaces scalar multiply with geometric count        ║")
    println("║  Δ (pants):    Ad slot splits → Σ c_i × L_i via disk moduli        ║")
    println("║                #𝔐(Ad; L_i,L_j) maximised by product selection     ║")
    println("║  100k products: Each evaluated LAZILY at surgery node               ║")
    println("║                 Zero precomputation. AU constructs on demand.       ║")
    println("╚" * "═"^68 * "╝")
end
