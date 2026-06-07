# =============================================================================
# mtr_ad_game.jl
#
# MTR Advertising Placement as AU-Fukaya MARL Game
#
# Three contexts:
#   C_people  = demographic flow (Markov chain on 81 stations × 4 demographics)
#   C_product = product affinity (10 products × 4 demographics)
#   C_month   = cyclic seasonal context (12 months)
#
# Mapping to pharmacodynamics:
#   Opiate    = advertiser (maximises P(product seen by right person))
#   Norcain   = budget/competitor (removes low-performing placements)
#   sAMY      = purchase decision moment at high-traffic station
#   HPF       = major interchange (relay hub)
#   Renkin-Crone w = ridership × demographic match score
#   Stops Λ   = ad slots removed (budget exhausted / bad performance)
#   coker     = irreducible demographic mismatch at a station pair
#   Mode 4    = seasonal campaign override (CNY, Christmas)
#   Feedback  = NEGATIVE COPRODUCT (bad ads actively harm brand)
#
# 5000 simulated passengers: 4 blocks
#   Rich Male (RM), Rich Female (RF), Poor Male (PM), Poor Female (PF)
# =============================================================================

using LinearAlgebra, Printf, Random, Statistics

Random.seed!(42)

# =============================================================================
# PART 1: CONSTANTS AND DATA
# =============================================================================

const N_STATIONS    = 81
const N_DEMO        = 4    # RM, RF, PM, PF
const N_PRODUCTS    = 10
const N_MONTHS      = 12
const N_PEOPLE      = 5000
const DEMO_LABELS   = [:RM, :RF, :PM, :PF]
const DEMO_NAMES    = Dict(:RM=>"Rich Male", :RF=>"Rich Female",
                             :PM=>"Poor Male", :PF=>"Poor Female")
const MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
const MONTH_EVENTS  = Dict(1=>"CNY", 2=>"CNY", 7=>"Summer",
                            8=>"Summer", 10=>"Singles Day",
                            11=>"Singles Day", 12=>"Christmas")

# MTR line sequences (same as mtr_game.jl)
const LINE_SEQ = Dict(
    :island      => (["Kennedy","HKU","Sai_Ying_Pun","Sheung_Wan","Central",
                      "Admiralty","WanChai","CausewayBay","TinHau","Fortress_Hill",
                      "North_Point","Quarry_Bay","Tai_Koo","Sai_Wan_Ho",
                      "Shau_Kei_Wan","Heng_Fa_Chuen","Chai_Wan"], 180),
    :tsuen_wan   => (["Tsuen_Wan","Tai_Wo_Hau","Kwai_Hing","Kwai_Fong",
                      "Lai_King","Mei_Foo","Lai_Chi_Kok","Cheung_Sha_Wan",
                      "Sham_Shui_Po","Prince_Edward","Mong_Kok","Yau_Ma_Tei",
                      "Jordan","Tsim_Sha_Tsui","Admiralty","Central"], 160),
    :kwun_tong   => (["Tiu_Keng_Leng","Yau_Tong","Lam_Tin","Kwun_Tong",
                      "Kowloon_Bay","Ngau_Tau_Kok","Choi_Hung","Diamond_Hill",
                      "Wong_Tai_Sin","Lok_Fu","Wang_Tau_Hom","Kowloon_Tong",
                      "Prince_Edward","Mong_Kok","Whampoa","Ho_Man_Tin","Hung_Hom"], 140),
    :east_rail   => (["Lo_Wu","Sheung_Shui","Fanling","Tai_Po_Market",
                      "University","Fo_Tan","Sha_Tin","Tai_Wai","Kowloon_Tong",
                      "Mong_Kok_East","Hung_Hom","Kowloon","Austin",
                      "Admiralty","Tsim_Sha_Tsui","Ho_Man_Tin",
                      "Diamond_Hill","Wong_Chuk_Hang"], 130),
    :west_rail   => (["Tuen_Mun","Siu_Hong","Tin_Shui_Wai","Long_Ping",
                      "Yuen_Long","Kam_Sheung_Road","Kwu_Tung","Ping_Shan",
                      "Nam_Cheong","Austin","Hung_Hom"], 75),
    :south_island=> (["South_Horizons","Lei_Tung","Wong_Chuk_Hang",
                      "Ocean_Park","Admiralty"], 40),
    :tseung_kwan_o=>(["LOHAS_Park","Tseung_Kwan_O","Hang_Hau","Po_Lam",
                       "Tiu_Keng_Leng","Yau_Tong","North_Point","Quarry_Bay"], 70),
    :tung_chung  => (["Tung_Chung","Tsing_Yi","Sunny_Bay","Lai_King",
                      "Nam_Cheong","Olympic","Mei_Foo"], 55),
    :ma_on_shan  => (["Wu_Kai_Sha","Ma_On_Shan","Heng_On","Tai_Wai"], 45),
)

"""Build station list and adjacency."""
function build_mtr_stations()
    stations = String[]
    edges    = Tuple{Int,Int}[]
    weights  = Dict{Tuple{Int,Int},Float64}()

    for (line,(seq,w)) in LINE_SEQ
        for s in seq
            s ∉ stations && push!(stations,s)
        end
    end
    sort!(stations)
    idx = Dict(s=>i for (i,s) in enumerate(stations))

    for (line,(seq,w)) in LINE_SEQ
        for i in 1:length(seq)-1
            si,ti = idx[seq[i]], idx[seq[i+1]]
            e=(min(si,ti),max(si,ti))
            e ∉ edges && push!(edges,e)
            weights[e] = max(get(weights,e,0.0), w)
        end
    end
    return stations, idx, edges, weights
end

const STATIONS, STATION_IDX, EDGES, EDGE_WEIGHTS = build_mtr_stations()

# =============================================================================
# PART 2: DEMOGRAPHIC CONTEXT C_PEOPLE
# =============================================================================

"""
Station demographic profile: fraction of each demographic at each station.
Based on geographic income distribution in Hong Kong.
Rich areas: Central, Admiralty, Tsim_Sha_Tsui, CausewayBay, Mong_Kok
Poor areas: Tuen_Mun, Yuen_Long, Fanling, Tin_Shui_Wai, Tiu_Keng_Leng
"""
function build_demographic_profiles()
    n = length(STATIONS)
    # D[i, demo] = fraction of demographic `demo` at station i
    # demo order: RM=1, RF=2, PM=3, PF=4
    D = fill(0.25, n, N_DEMO)   # uniform baseline

    # Rich areas: higher RM+RF fractions
    rich_stations = ["Central","Admiralty","Tsim_Sha_Tsui","CausewayBay",
                     "Sheung_Wan","HKU","Kennedy","Mong_Kok"]
    # Financial/business: more RM
    business = ["Central","Admiralty","Sheung_Wan","Austin","Kowloon"]
    # Shopping: more RF
    shopping = ["CausewayBay","Mong_Kok","Tsim_Sha_Tsui","Jordan"]
    # Poor/distant: higher PM+PF
    poor_stations = ["Tuen_Mun","Siu_Hong","Tin_Shui_Wai","Long_Ping",
                     "Yuen_Long","Lo_Wu","Fanling","Sheung_Shui",
                     "Tiu_Keng_Leng","LOHAS_Park","Tseung_Kwan_O"]

    for (i,s) in enumerate(STATIONS)
        if s ∈ rich_stations
            # Rich areas: 35% RM, 35% RF, 15% PM, 15% PF
            D[i,:] = [0.35, 0.35, 0.15, 0.15]
        elseif s ∈ poor_stations
            # Poor areas: 15% RM, 10% RF, 40% PM, 35% PF
            D[i,:] = [0.15, 0.10, 0.40, 0.35]
        end
        if s ∈ business; D[i,1] += 0.10; D[i,2] -= 0.05; D[i,3] -= 0.03; D[i,4] -= 0.02; end
        if s ∈ shopping; D[i,2] += 0.10; D[i,1] -= 0.05; end
        # Renormalise
        D[i,:] ./= sum(D[i,:])
    end

    # Ridership weights (total people through station per day)
    ridership = Dict{String,Float64}()
    for (line,(seq,w)) in LINE_SEQ
        for s in seq
            ridership[s] = get(ridership,s,0.0) + w
        end
    end
    R = [get(ridership,s,50.0) for s in STATIONS]
    R ./= maximum(R)   # normalise to [0,1]

    return D, R
end

const DEMO_PROFILES, RIDERSHIP = build_demographic_profiles()

# =============================================================================
# PART 3: PRODUCT CONTEXT C_PRODUCT
# =============================================================================

"""
10 products with affinity vectors over 4 demographics.
Affinity[p,d] = P(demographic d buys product p | sees the ad).
"""
struct Product
    id          ::Int
    name        ::String
    price_tier  ::Symbol    # :luxury, :mid, :budget
    affinity    ::Vector{Float64}   # [RM, RF, PM, PF]
    month_boost ::Dict{Int,Float64} # month → multiplier
    base_conversion::Float64        # baseline conversion rate
end

function build_products()::Vector{Product}
    [
     Product(1, "Luxury Watch",    :luxury,
             [0.55, 0.35, 0.05, 0.05],
             Dict(12=>2.0, 2=>1.8, 1=>1.8),   # Christmas, CNY
             0.02),
     Product(2, "Designer Handbag",:luxury,
             [0.10, 0.70, 0.05, 0.15],
             Dict(12=>2.0, 2=>1.5, 8=>1.3),
             0.025),
     Product(3, "Business Laptop", :mid,
             [0.65, 0.20, 0.10, 0.05],
             Dict(9=>1.5, 10=>1.8, 11=>1.8),  # Back to school/11.11
             0.03),
     Product(4, "Smartphone",      :mid,
             [0.30, 0.30, 0.25, 0.15],
             Dict(10=>2.0, 11=>2.0, 12=>1.5),
             0.04),
     Product(5, "Fashion Jewelry", :mid,
             [0.05, 0.55, 0.05, 0.35],
             Dict(2=>1.8, 12=>1.5, 5=>1.3),   # CNY, Valentine
             0.035),
     Product(6, "Train Ticket (Shenzhen)", :budget,
             [0.10, 0.10, 0.40, 0.40],
             Dict(1=>2.0, 2=>2.5, 10=>1.5),   # CNY travel
             0.08),
     Product(7, "HK Disneyland",   :mid,
             [0.20, 0.25, 0.30, 0.25],
             Dict(7=>2.0, 8=>2.0, 12=>1.8),   # Summer, Christmas
             0.05),
     Product(8, "Fitness App",     :budget,
             [0.35, 0.30, 0.20, 0.15],
             Dict(1=>2.0, 2=>1.5),             # New Year resolutions
             0.06),
     Product(9, "Grocery Delivery",:budget,
             [0.15, 0.25, 0.30, 0.30],
             Dict{Int,Float64}(),              # no strong season
             0.10),
     Product(10,"CNY Gift Set",    :mid,
             [0.25, 0.30, 0.25, 0.20],
             Dict(1=>3.0, 2=>4.0, 12=>1.5),   # CNY dominant
             0.04),
    ]
end

const PRODUCTS = build_products()

# =============================================================================
# PART 4: CONVERSION PROBABILITY
# =============================================================================

"""
P(conversion | station s, product p, month m)
= ridership(s) × Σ_d [demo_profile(s,d) × affinity(p,d)] × month_factor(p,m)
"""
function conversion_prob(s::Int, p::Int, month::Int)::Float64
    prod     = PRODUCTS[p]
    affinity = sum(DEMO_PROFILES[s,d] * prod.affinity[d] for d in 1:N_DEMO)
    m_boost  = get(prod.month_boost, month, 1.0)
    return RIDERSHIP[s] * affinity * m_boost * prod.base_conversion
end

"""Full conversion matrix: stations × products for a given month."""
function conversion_matrix(month::Int)::Matrix{Float64}
    n = length(STATIONS)
    C = zeros(n, N_PRODUCTS)
    for s in 1:n, p in 1:N_PRODUCTS
        C[s,p] = conversion_prob(s, p, month)
    end
    return C
end

# =============================================================================
# PART 5: AU CONTEXTS FOR AD PLACEMENT
# =============================================================================

"""
AdContext: AU context for a station or group of stations.
Stores the demographic mix and which products are currently placed.
"""
mutable struct AdContext
    station_ids ::Vector{Int}           # stations in this AU
    demo_mix    ::Vector{Float64}       # aggregate demographic [RM,RF,PM,PF]
    placed_ads  ::Dict{Int,Bool}        # product_id → is placed here?
    conversion  ::Vector{Float64}       # current conversion rate per product
    feedback    ::Vector{Float64}       # running feedback per product (signed)
    coker_score ::Float64               # demographic diversity (high=diverse)
end

"""Build an AU context for a single station."""
function station_context(s::Int, month::Int)::AdContext
    dm     = DEMO_PROFILES[s,:]
    conv   = [conversion_prob(s, p, month) for p in 1:N_PRODUCTS]
    placed = Dict(p=>false for p in 1:N_PRODUCTS)
    # Cokernel score = demographic entropy (high diversity = high coker)
    coker  = -sum(d > 0 ? d*log(d) : 0.0 for d in dm) / log(N_DEMO)
    return AdContext([s], Float64.(dm), placed, conv, zeros(N_PRODUCTS), coker)
end

"""
Coproduct of two AdContexts: T_s ⊔ T_t.
Represents advertising the same products at both stations.
Combined demographic = weighted average by ridership.
"""
function coproduct(ctx_s::AdContext, ctx_t::AdContext)::AdContext
    r_s = mean(RIDERSHIP[ctx_s.station_ids])
    r_t = mean(RIDERSHIP[ctx_t.station_ids])
    w_s = r_s / (r_s + r_t + 1e-10)
    w_t = r_t / (r_s + r_t + 1e-10)

    new_dm    = w_s .* ctx_s.demo_mix .+ w_t .* ctx_t.demo_mix
    new_conv  = w_s .* ctx_s.conversion .+ w_t .* ctx_t.conversion
    new_sids  = vcat(ctx_s.station_ids, ctx_t.station_ids)
    new_feed  = ctx_s.feedback .+ ctx_t.feedback
    new_placed= Dict(p => (ctx_s.placed_ads[p] || ctx_t.placed_ads[p])
                     for p in 1:N_PRODUCTS)

    # Cokernel of the coproduct: how different are the two demographics?
    # High difference → high coker → placing ads here adds value
    coker_new = sum(abs.(ctx_s.demo_mix .- ctx_t.demo_mix)) / 2.0

    return AdContext(new_sids, new_dm, new_placed, new_conv, new_feed, coker_new)
end

"""
NEGATIVE COPRODUCT: feedback correction.
When feedback F[p] < 0, the ad is actively harming the brand.
Apply as: ctx_corrected = ctx ⊓ feedback (remove bad placements).
When F[p] > 0, reinforce (keep/expand).
"""
function apply_feedback!(ctx::AdContext, feedback::Vector{Float64},
                          threshold::Float64 = -0.01)
    for p in 1:N_PRODUCTS
        if feedback[p] < threshold && ctx.placed_ads[p]
            # NEGATIVE COPRODUCT: pull the ad
            ctx.placed_ads[p] = false
            ctx.feedback[p]  += feedback[p]
        elseif feedback[p] > abs(threshold) && !ctx.placed_ads[p]
            # POSITIVE COPRODUCT: reinforce
            ctx.placed_ads[p] = true
            ctx.feedback[p]  += feedback[p]
        end
    end
end

# =============================================================================
# PART 6: PEOPLE SIMULATION (5000 passengers)
# =============================================================================

"""
Simulate 5000 passengers on the MTR.
Each person has a demographic, a home station, and a destination.
Returns: array of (person_id, demo, home_station, dest_station, journey_stations)
"""
function simulate_passengers(n::Int = N_PEOPLE, month::Int = 1)

    # Demographic distribution: 5000 people split by income/gender
    # Proportions roughly matching Hong Kong demographics
    demo_counts = [750, 750, 1850, 1650]  # RM, RF, PM, PF (total=5000)

    # Home station distributions by demographic
    rich_homes = ["Central","Admiralty","Tsim_Sha_Tsui","CausewayBay",
                  "Sheung_Wan","Fortress_Hill","TinHau","Tai_Koo"]
    poor_homes = ["Tuen_Mun","Tin_Shui_Wai","Yuen_Long","Long_Ping",
                  "Fanling","Sheung_Shui","Lo_Wu","LOHAS_Park",
                  "Tseung_Kwan_O","Hang_Hau","Tiu_Keng_Leng"]

    passengers = NamedTuple[]
    pid        = 0

    for (di, demo) in enumerate(DEMO_LABELS)
        home_pool = di <= 2 ? rich_homes : poor_homes
        dest_pool = di <= 2 ?
            ["Central","Admiralty","Tsim_Sha_Tsui","Mong_Kok"] :
            ["Central","Admiralty","Hung_Hom","Kowloon_Tong","Diamond_Hill"]

        for _ in 1:demo_counts[di]
            pid += 1
            home_s  = home_pool[rand(1:length(home_pool))]
            dest_s  = dest_pool[rand(1:length(dest_pool))]
            home_i  = get(STATION_IDX, home_s, 1)
            dest_i  = get(STATION_IDX, dest_s, 1)

            # Journey stations (simplified: home + intermediate hubs + dest)
            journey = [home_i]
            # Rich people transfer at most once; poor people transfer more
            if di <= 2
                # Short journey: home → one transfer → dest
                hub_options = [get(STATION_IDX,"Admiralty",1),
                               get(STATION_IDX,"Central",1)]
                push!(journey, hub_options[rand(1:end)])
            else
                # Long journey: home → major interchange → dest
                hub_options = [get(STATION_IDX,"Hung_Hom",1),
                               get(STATION_IDX,"Kowloon_Tong",1),
                               get(STATION_IDX,"Admiralty",1)]
                push!(journey, hub_options[rand(1:end)])
                push!(journey, get(STATION_IDX,"Central",1))
            end
            push!(journey, dest_i)
            unique!(journey)

            push!(passengers, (id=pid, demo=demo, demo_idx=di,
                                home=home_i, dest=dest_i,
                                journey=journey))
        end
    end
    return passengers
end

# =============================================================================
# PART 7: MARL GAME — ADVERTISER VS BUDGET
# =============================================================================

"""
AdPlacementState: game state for the ad placement MARL game.
"""
mutable struct AdPlacementState
    month       ::Int
    placed      ::Matrix{Bool}   # [station, product] → is ad placed?
    conversions ::Matrix{Float64}# [station, product] → conversion prob
    feedback    ::Matrix{Float64}# [station, product] → running feedback
    budget_used ::Int            # total ad slots used
    budget_max  ::Int            # maximum allowed ad slots
    total_conversions::Float64   # cumulative conversions
    round       ::Int
end

"""Initialise game state for a given month."""
function init_game(month::Int; budget::Int = 50)::AdPlacementState
    n      = length(STATIONS)
    C      = conversion_matrix(month)
    placed = falses(n, N_PRODUCTS)
    feed   = zeros(n, N_PRODUCTS)
    AdPlacementState(month, placed, C, feed, 0, budget, 0.0, 0)
end

"""
Total expected conversions given current placement.
P(sale) = Σ_{s,p} placed[s,p] × conversions[s,p] × ridership[s]
"""
function total_expected_conversions(state::AdPlacementState)::Float64
    sum(state.placed[s,p] * state.conversions[s,p] * RIDERSHIP[s]
        for s in 1:length(STATIONS), p in 1:N_PRODUCTS)
end

"""
ADVERTISER (opiate) action: place product p at station s.
Chooses greedily by conversion_prob × ridership.
"""
function advertiser_action(state::AdPlacementState)
    best_val = -Inf
    best     = (0, 0)
    n        = length(STATIONS)

    for s in 1:n, p in 1:N_PRODUCTS
        state.placed[s,p] && continue  # already placed
        val = state.conversions[s,p] * RIDERSHIP[s]
        if val > best_val
            best_val = val
            best     = (s, p)
        end
    end
    return best
end

"""
AU-QKV ADVERTISER action: attention-weighted placement.
Q = station demographic profile (which types of people are here?)
K = product affinity vector (which types of people does this target?)
V = expected conversion × ridership (what value does this placement create?)
"""
function au_qkv_advertiser_action(state::AdPlacementState)
    n        = length(STATIONS)
    best_val = -Inf
    best     = (0, 0)

    for s in 1:n
        Q = DEMO_PROFILES[s,:]   # Query: station demographic

        for p in 1:N_PRODUCTS
            state.placed[s,p] && continue

            K = PRODUCTS[p].affinity     # Key: product demographic target
            V = state.conversions[s,p] * RIDERSHIP[s]   # Value: expected return

            # Attention score: demographic alignment × expected value
            alignment = dot(Q, K)        # how well does product match station?
            feedback_pen = max(0.0, -state.feedback[s,p])  # penalise bad history
            score = alignment * V * (1.0 - feedback_pen)

            if score > best_val
                best_val = score
                best     = (s, p)
            end
        end
    end
    return best
end

"""
BUDGET CONSTRAINT (norcain) action: remove lowest-performing placement.
Simulates budget pressure removing inefficient placements.
"""
function budget_action(state::AdPlacementState)
    worst_val = Inf
    worst     = (0, 0)
    n         = length(STATIONS)

    for s in 1:n, p in 1:N_PRODUCTS
        !state.placed[s,p] && continue
        val = state.conversions[s,p] * RIDERSHIP[s] + state.feedback[s,p]
        if val < worst_val
            worst_val = val
            worst     = (s, p)
        end
    end
    return worst
end

"""
Simulate passenger feedback: observe actual conversions vs expected.
Returns feedback matrix (positive = working, negative = not working).
"""
function simulate_feedback!(state::AdPlacementState,
                              passengers::Vector)::Matrix{Float64}
    n     = length(STATIONS)
    obs   = zeros(n, N_PRODUCTS)  # observed conversions

    for pax in passengers
        for s in pax.journey
            s == 0 && continue
            for p in 1:N_PRODUCTS
                !state.placed[s,p] && continue
                # Simulate purchase: Bernoulli with conversion prob
                # adjusted by demographic match
                d   = pax.demo_idx
                aff = PRODUCTS[p].affinity[d]
                m_b = get(PRODUCTS[p].month_boost, state.month, 1.0)
                p_buy = aff * m_b * PRODUCTS[p].base_conversion * RIDERSHIP[s]
                rand() < p_buy && (obs[s,p] += 1.0)
            end
        end
    end

    # Feedback = (observed - expected) / expected
    feedback = zeros(n, N_PRODUCTS)
    for s in 1:n, p in 1:N_PRODUCTS
        expected = state.conversions[s,p] * RIDERSHIP[s] * length(passengers) / 1000.0
        expected > 0.001 || continue
        feedback[s,p] = (obs[s,p] - expected) / expected
    end

    # Apply to running feedback (exponential moving average)
    state.feedback .= 0.7 .* state.feedback .+ 0.3 .* feedback
    return feedback
end

# =============================================================================
# PART 8: FULL GAME LOOP
# =============================================================================

"""Run the full ad placement MARL game for one month."""
function run_ad_game(month::Int;
                     n_rounds  ::Int  = 20,
                     budget    ::Int  = 50,
                     policy    ::Symbol = :au_qkv,
                     verbose   ::Bool  = true)

    state      = init_game(month; budget=budget)
    passengers = simulate_passengers(N_PEOPLE, month)
    history    = NamedTuple[]
    event      = get(MONTH_EVENTS, month, "")

    verbose && begin
        println("="^70)
        println("MTR AD PLACEMENT GAME — $(MONTH_NAMES[month])$(isempty(event) ? "" : " ($event)")")
        println("="^70)
        @printf("  Products: %d   Stations: %d   Budget: %d slots   Policy: %s\n",
                N_PRODUCTS, length(STATIONS), budget, policy)
        println("─"^70)
        @printf("  %-5s %-30s %-10s %-10s %-8s\n",
                "Round","Action (station→product)","Conv","Budget","Feedback")
        println("  " * "─"^63)
    end

    for rnd in 1:n_rounds
        state.round = rnd

        # ADVERTISER places an ad
        s, p = policy == :au_qkv ?
               au_qkv_advertiser_action(state) :
               advertiser_action(state)

        (s == 0 || p == 0 || state.budget_used >= budget) && break

        state.placed[s,p] = true
        state.budget_used += 1

        # BUDGET removes a bad placement every 3 rounds
        if rnd % 3 == 0 && state.budget_used > 5
            ws, wp = budget_action(state)
            if ws > 0 && wp > 0
                state.placed[ws,wp] = false
                state.budget_used  -= 1
            end
        end

        # Simulate feedback every 5 rounds
        if rnd % 5 == 0
            simulate_feedback!(state, passengers)
        end

        # Track conversions
        total_conv = total_expected_conversions(state)
        push!(history, (round=rnd, station=s, product=p,
                         station_name=STATIONS[s],
                         product_name=PRODUCTS[p].name,
                         conversion=total_conv,
                         budget_used=state.budget_used))

        verbose && begin
            act_s = "$(first(STATIONS[s], 15))→$(first(PRODUCTS[p].name, 12))"
            @printf("  R%-4d %-30s %-10.4f %-10d\n",
                    rnd, act_s, total_conv, state.budget_used)
        end
    end

    # Final feedback pass
    simulate_feedback!(state, passengers)

    verbose && begin
        println("  " * "─"^63)
        println()
        println("  TOP 5 PLACEMENTS BY CONVERSION:")
        placements = [(STATIONS[s], PRODUCTS[p].name,
                       state.conversions[s,p]*RIDERSHIP[s],
                       state.feedback[s,p])
                      for s in 1:length(STATIONS), p in 1:N_PRODUCTS
                      if state.placed[s,p]]
        sort!(placements, by=x->x[3], rev=true)
        for (sn,pn,cv,fb) in placements[1:min(5,end)]
            fb_s = fb > 0.05 ? "✓" : fb < -0.05 ? "✗" : "~"
            @printf("    %-20s × %-20s  conv=%.4f  %s\n", sn, pn, cv, fb_s)
        end
        println()
        println("  NEGATIVE COPRODUCT (pulled by feedback):")
        bad = [(STATIONS[s], PRODUCTS[p].name, state.feedback[s,p])
               for s in 1:length(STATIONS), p in 1:N_PRODUCTS
               if !state.placed[s,p] && state.feedback[s,p] < -0.1]
        isempty(bad) && println("    None — all placements performing")
        for (sn,pn,fb) in sort(bad, by=x->x[3])[1:min(3,end)]
            @printf("    %-20s × %-20s  feedback=%.3f (pulled)\n", sn, pn, fb)
        end
    end

    return state, history
end

"""Compare AU-QKV vs greedy policy across months."""
function policy_comparison(months::Vector{Int} = [2,7,10,12])
    println("="^70)
    println("POLICY COMPARISON: Greedy vs AU-QKV across seasons")
    println("="^70)
    @printf("  %-12s %-12s %-14s %-14s %-10s\n",
            "Month","Event","Greedy conv","AU-QKV conv","Δ")
    println("  " * "─"^60)

    for month in months
        s_g, _ = run_ad_game(month; policy=:greedy,  verbose=false)
        s_a, _ = run_ad_game(month; policy=:au_qkv,  verbose=false)
        c_g    = total_expected_conversions(s_g)
        c_a    = total_expected_conversions(s_a)
        event  = get(MONTH_EVENTS, month, "—")
        @printf("  %-12s %-12s %-14.4f %-14.4f %+.4f\n",
                MONTH_NAMES[month], event, c_g, c_a, c_a-c_g)
    end
    println("  " * "─"^60)
    println("  AU-QKV advantage: aligns demographic Q vector with product K vector")
    println("  Greedy takes highest raw conversion without demographic alignment")
end

"""Show Postnikov tower for ad placement."""
function print_ad_postnikov(month::Int = 2)
    println("\n" * "="^70)
    println("POSTNIKOV TOWER OF AD REWARDS — $(MONTH_NAMES[month])")
    println("="^70)

    C = conversion_matrix(month)
    n = length(STATIONS)

    max_conv = maximum(C)
    avg_conv = mean(C[C .> 0])

    println()
    println("  L0 — VISIBILITY: product is seen at all")
    @printf("     Bracket: [0, %.4f]  (max conversion rate)\n", max_conv)
    println("     k-invariant = 0  (any product can be placed anywhere)")
    println()
    println("  L1 — DEMOGRAPHIC FIT: right person sees the product")
    for p in 1:N_PRODUCTS
        best_s = argmax(C[:,p])
        best_d = argmax(PRODUCTS[p].affinity)
        @printf("     %s → best at %-20s (demo: %s)\n",
                first(PRODUCTS[p].name, 20),
                first(STATIONS[best_s], 20),
                DEMO_LABELS[best_d])
    end
    println()
    println("  L2 — PURCHASE INTENT: person actually buys")
    println("     k-invariant = irreducible mismatch (purchase intent gap)")
    println("     Some stations have NO product fit (6am commuters = no purchase intent)")

    # Find stations with lowest max conversion (high mismatch)
    max_per_station = [maximum(C[s,:]) for s in 1:n]
    worst = sortperm(max_per_station)[1:5]
    println()
    println("     Highest mismatch stations (k-invariant analogue):")
    for s in worst
        @printf("       %-20s  max_conv=%.4f  (demographic: %s)\n",
                first(STATIONS[s], 20), max_per_station[s],
                DEMO_LABELS[argmax(DEMO_PROFILES[s,:])])
    end

    println()
    println("  NEGATIVE COPRODUCT stations (feedback would pull any ad):")
    println("  = stations where purchase intent is structurally absent")
    println("  = the 62-class obstruction analogue in ad placement")
    println("  = no amount of targeting overcomes the barrier")
    println("="^70)
end

# =============================================================================
# PART 9: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

    println("="^70)
    println("MTR ADVERTISING PLACEMENT — AU-FUKAYA MARL GAME")
    println("10 products × 81 stations × 4 demographics × 12 months")
    println("="^70)

    # [1] CNY campaign — expensive gifts + train tickets
    println("\n[1] CHINESE NEW YEAR CAMPAIGN (February)")
    run_ad_game(2; n_rounds=20, budget=50, policy=:au_qkv)

    # [2] Summer campaign — tourism, Disneyland
    println("\n[2] SUMMER CAMPAIGN (July)")
    run_ad_game(7; n_rounds=20, budget=50, policy=:au_qkv)

    # [3] Policy comparison across seasons
    println("\n[3] CROSS-SEASON POLICY COMPARISON")
    policy_comparison([1,2,7,10,12])

    # [4] Postnikov tower
    print_ad_postnikov(2)

    # [5] Demographic breakdown
    println("\n[5] STATION DEMOGRAPHIC PROFILES (top 10 by ridership)")
    println("─"^70)
    top_r = sortperm(RIDERSHIP, rev=true)[1:10]
    @printf("  %-22s  %6s  %6s  %6s  %6s  %6s\n",
            "Station", "Riders", "RM%","RF%","PM%","PF%")
    println("  " * "─"^60)
    for s in top_r
        @printf("  %-22s  %6.3f  %5.0f%%  %5.0f%%  %5.0f%%  %5.0f%%\n",
                first(STATIONS[s], 22), RIDERSHIP[s],
                100*DEMO_PROFILES[s,1], 100*DEMO_PROFILES[s,2],
                100*DEMO_PROFILES[s,3], 100*DEMO_PROFILES[s,4])
    end

    println("\n[6] PRODUCT × DEMOGRAPHIC AFFINITY MATRIX")
    println("─"^70)
    @printf("  %-22s  %8s %8s %8s %8s  %s\n",
            "Product", "RM", "RF", "PM", "PF", "Price")
    println("  " * "─"^65)
    for prod in PRODUCTS
        @printf("  %-22s  %7.0f%% %7.0f%% %7.0f%% %7.0f%%  %s\n",
                first(prod.name, 22),
                100*prod.affinity[1], 100*prod.affinity[2],
                100*prod.affinity[3], 100*prod.affinity[4],
                prod.price_tier)
    end
    println("="^70)
end
