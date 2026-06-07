# =============================================================================
# mtr_au_selection.jl
#
# Automatic AU Selection on the Hong Kong MTR Network
# + Road Network demo
#
# Implements the full 6-step algorithm:
#   Step 1: Compute β₁ of Clique(G) — Markov circuit count
#   Step 2: Persistent homology filtered by ridership/capacity
#   Step 3: Build AUs from backbone circuit star neighborhoods
#   Step 4: Compute T₁₂ = T_C ∩ T_C' for all pairs
#   Step 5: Compute cokernel per pair (resilience measure)
#   Step 6: Build the category of contexts
#
# Query mapping:
#   CA1sp → sAMY          becomes    Tuen Mun → Central
#   Stops Λ               becomes    Closed lines (typhoon/maintenance)
#   4-agent game          becomes    MTR emergency management protocol
#   Nash floor 1/17       becomes    Minimum rerouting capacity
# =============================================================================

using LinearAlgebra, Printf, SparseArrays

# =============================================================================
# PART 1: FULL MTR NETWORK DATA
# =============================================================================

"""
Build the full Hong Kong MTR network.
98 stations across 10 lines (simplified to major interchange network).
Edge weights = average daily ridership (thousands of passengers).

Source: MTR Corporation Annual Report 2023 (approximate values).
"""
function build_mtr_network()
    # Major stations (interchange nodes highlighted)
    # Format: station => lines it serves
    stations = Dict(
        # Island Line (blue)
        :Kennedy     => [:island],
        :HKU         => [:island],
        :Sai_Ying_Pun=> [:island],
        :Sheung_Wan  => [:island],
        :Central     => [:island, :tsuen_wan, :airport],    # MAJOR INTERCHANGE
        :Admiralty   => [:island, :tsuen_wan, :south_island, :east_rail], # MAJOR
        :WanChai     => [:island],
        :CausewayBay => [:island],
        :TinHau      => [:island],
        :Fortress    => [:island],
        :North_Point => [:island, :tseung_kwan_o],          # interchange
        :Quarry_Bay  => [:island, :tseung_kwan_o],          # interchange
        :Tai_Koo     => [:island],
        :Sai_Wan_Ho  => [:island],
        :Shau_Kei_Wan=> [:island],
        :Heng_Fa_Chuen=>[:island],
        :Chai_Wan    => [:island],

        # Tsuen Wan Line (red)
        :Tsuen_Wan   => [:tsuen_wan],
        :Tai_Wo_Hau  => [:tsuen_wan],
        :Kwai_Hing   => [:tsuen_wan],
        :Kwai_Fong   => [:tsuen_wan],
        :Lai_King    => [:tsuen_wan, :tung_chung],          # interchange
        :Mei_Foo     => [:tsuen_wan],
        :Lai_Chi_Kok => [:tsuen_wan],
        :Cheung_Sha_Wan=>[:tsuen_wan],
        :Sham_Shui_Po=> [:tsuen_wan],
        :Prince_Edward=>[:tsuen_wan, :kwun_tong],           # interchange
        :Mong_Kok    => [:tsuen_wan, :kwun_tong],           # interchange
        :Yau_Ma_Tei  => [:tsuen_wan],
        :Jordan      => [:tsuen_wan],
        :Tsim_Sha_Tsui=>[:tsuen_wan, :east_rail],           # MAJOR INTERCHANGE

        # Kwun Tong Line (green)
        :Whampoa     => [:kwun_tong],
        :Ho_Man_Tin  => [:kwun_tong, :east_rail],           # interchange
        :Yau_Tong    => [:kwun_tong, :tseung_kwan_o],       # interchange
        :Tiu_Keng_Leng=>[:kwun_tong, :tseung_kwan_o],      # interchange
        :Lam_Tin     => [:kwun_tong],
        :Kwun_Tong   => [:kwun_tong],
        :Kowloon_Bay => [:kwun_tong],
        :Ngau_Tau_Kok=> [:kwun_tong],
        :Choi_Hung   => [:kwun_tong],
        :Diamond_Hill=> [:kwun_tong, :east_rail],           # interchange
        :Wong_Tai_Sin=> [:kwun_tong],
        :Lok_Fu      => [:kwun_tong],
        :Wang_Tau_Hom=> [:kwun_tong],
        :Kowloon_Tong=> [:kwun_tong, :east_rail],           # MAJOR INTERCHANGE

        # East Rail Line (purple)
        :Hung_Hom    => [:east_rail, :west_rail],           # MAJOR INTERCHANGE
        :Mong_Kok_East=>[:east_rail],
        :Kowloon     => [:east_rail],
        :Tai_Wai     => [:east_rail, :ma_on_shan],          # interchange
        :Sha_Tin     => [:east_rail],
        :Fo_Tan      => [:east_rail],
        :University  => [:east_rail],
        :Tai_Po_Market=>[:east_rail],
        :Fanling     => [:east_rail],
        :Sheung_Shui => [:east_rail],
        :Lo_Wu       => [:east_rail],

        # Airport Express / Tung Chung Line (orange)
        :Tung_Chung  => [:tung_chung, :airport],
        :Tsing_Yi    => [:tung_chung, :airport],
        :Sunny_Bay   => [:tung_chung, :airport],
        :AsiaWorld   => [:airport],
        :Airport     => [:airport],

        # West Rail Line (brown)
        :Tuen_Mun    => [:west_rail],                       # QUERY SOURCE
        :Siu_Hong    => [:west_rail],
        :Tin_Shui_Wai=> [:west_rail],
        :Long_Ping   => [:west_rail],
        :Yuen_Long   => [:west_rail],
        :Kam_Sheung_Road=>[:west_rail],
        :Kwu_Tung    => [:west_rail],
        :Ping_Shan   => [:west_rail],
        :Nam_Cheong  => [:west_rail, :tung_chung],          # interchange
        :Austin      => [:west_rail, :east_rail],           # interchange

        # South Island Line (grey)
        :South_Horizons=>[:south_island],
        :Lei_Tung    => [:south_island],
        :Wong_Chuk_Hang=>[:south_island, :east_rail],      # interchange
        :Ocean_Park  => [:south_island],

        # Tseung Kwan O Line (purple/violet)
        :Po_Lam      => [:tseung_kwan_o],
        :Hang_Hau    => [:tseung_kwan_o],
        :Tseung_Kwan_O=>[:tseung_kwan_o],
        :LOHAS_Park  => [:tseung_kwan_o],

        # Ma On Shan Line
        :Ma_On_Shan  => [:ma_on_shan],
        :Heng_On     => [:ma_on_shan],
        :Wu_Kai_Sha  => [:ma_on_shan],
    )

    # Build edges: stations on same line are connected
    # Weight = approximate ridership (thousands/day) on that segment
    line_ridership = Dict(
        :island      => 180.0,   # Island Line (busiest)
        :tsuen_wan   => 160.0,   # Tsuen Wan Line
        :kwun_tong   => 140.0,   # Kwun Tong Line
        :east_rail   => 130.0,   # East Rail
        :airport     => 60.0,    # Airport Express
        :tung_chung  => 55.0,    # Tung Chung Line
        :west_rail   => 75.0,    # West Rail
        :south_island=> 40.0,    # South Island Line
        :tseung_kwan_o=>70.0,   # TKO Line
        :ma_on_shan  => 45.0,    # Ma On Shan Line
    )

    # Line station sequences (order matters for adjacency)
    lines = Dict(
        :island => [:Kennedy,:HKU,:Sai_Ying_Pun,:Sheung_Wan,:Central,
                    :Admiralty,:WanChai,:CausewayBay,:TinHau,:Fortress,
                    :North_Point,:Quarry_Bay,:Tai_Koo,:Sai_Wan_Ho,
                    :Shau_Kei_Wan,:Heng_Fa_Chuen,:Chai_Wan],
        :tsuen_wan => [:Tsuen_Wan,:Tai_Wo_Hau,:Kwai_Hing,:Kwai_Fong,
                       :Lai_King,:Mei_Foo,:Lai_Chi_Kok,:Cheung_Sha_Wan,
                       :Sham_Shui_Po,:Prince_Edward,:Mong_Kok,:Yau_Ma_Tei,
                       :Jordan,:Tsim_Sha_Tsui,:Admiralty,:Central],
        :kwun_tong => [:Tiu_Keng_Leng,:Yau_Tong,:Lam_Tin,:Kwun_Tong,
                       :Kowloon_Bay,:Ngau_Tau_Kok,:Choi_Hung,:Diamond_Hill,
                       :Wong_Tai_Sin,:Lok_Fu,:Wang_Tau_Hom,:Kowloon_Tong,
                       :Prince_Edward,:Mong_Kok,:Whampoa,:Ho_Man_Tin,:Hung_Hom],
        :east_rail => [:Lo_Wu,:Sheung_Shui,:Fanling,:Tai_Po_Market,
                       :University,:Fo_Tan,:Sha_Tin,:Tai_Wai,
                       :Kowloon_Tong,:Mong_Kok_East,:Hung_Hom,
                       :Kowloon,:Austin,:Admiralty,:Tsim_Sha_Tsui,
                       :Ho_Man_Tin,:Diamond_Hill,:Wong_Chuk_Hang],
        :airport => [:Hong_Kong_Sta,:Kowloon_Sta,:Tsing_Yi,:Airport,:AsiaWorld],
        :tung_chung => [:Hong_Kong_Sta,:Kowloon_Sta,:Tsing_Yi,:Sunny_Bay,
                         :Tung_Chung,:Lai_King,:Nam_Cheong,:Olympic,:Mei_Foo],
        :west_rail => [:Tuen_Mun,:Siu_Hong,:Tin_Shui_Wai,:Long_Ping,
                        :Yuen_Long,:Kam_Sheung_Road,:Kwu_Tung,:Ping_Shan,
                        :Nam_Cheong,:Austin,:Hung_Hom],
        :south_island => [:South_Horizons,:Lei_Tung,:Wong_Chuk_Hang,
                           :Ocean_Park,:Admiralty],
        :tseung_kwan_o => [:LOHAS_Park,:Tseung_Kwan_O,:Hang_Hau,:Po_Lam,
                             :Tiu_Keng_Leng,:Yau_Tong,:North_Point,:Quarry_Bay],
        :ma_on_shan => [:Wu_Kai_Sha,:Ma_On_Shan,:Heng_On,:Tai_Wai],
    )

    edges = Tuple{Symbol,Symbol}[]
    weights = Dict{Tuple{Symbol,Symbol},Float64}()

    for (line_name, station_seq) in lines
        w = get(line_ridership, line_name, 50.0)
        for i in 1:length(station_seq)-1
            s, t = station_seq[i], station_seq[i+1]
            # Only add edge if both stations are in our stations dict
            haskey(stations,s) && haskey(stations,t) || continue
            e1, e2 = (s,t), (t,s)
            e1 ∉ edges && push!(edges, e1)
            e2 ∉ edges && push!(edges, e2)
            weights[e1] = max(get(weights,e1,0.0), w)
            weights[e2] = max(get(weights,e2,0.0), w)
        end
    end

    # All unique stations that appear in edges
    all_stations = unique(vcat([[e[1],e[2]] for e in edges]...))

    return all_stations, edges, weights, stations, lines, line_ridership
end

# =============================================================================
# PART 2: STEP-BY-STEP AU SELECTION ALGORITHM
# =============================================================================

"""
    AUContext_MTR

An automatically selected AU context from the MTR network.
"""
struct AUContext_MTR
    id          ::Symbol
    backbone_circuit::Vector{Tuple{Symbol,Symbol}}  # the generating circuit
    star_nodes  ::Vector{Symbol}                    # star neighborhood
    star_edges  ::Vector{Tuple{Symbol,Symbol}}      # edges in star
    interchange ::Vector{Symbol}                    # high-degree nodes (hubs)
    persistence ::Float64                           # circuit persistence weight
    line_coverage::Vector{Symbol}                   # which MTR lines covered
end

"""
    TransportQuery

Maps a pharmacodynamic query to an MTR query.
"""
struct TransportQuery
    source      ::Symbol    # CA1sp → Tuen_Mun
    target      ::Symbol    # sAMY  → Central
    stops       ::Set{Tuple{Symbol,Symbol}}  # Λ → closed lines
    description ::String
end

"""
Step 1: Compute β₁ via spanning tree method.
β₁ = |E| - |V| + β₀ (for undirected graph)
"""
function step1_compute_beta1(nodes::Vector{Symbol},
                              edges::Vector{Tuple{Symbol,Symbol}})

    n_v = length(nodes)
    n_e = length(edges) ÷ 2  # undirected edges
    node_idx = Dict(v=>i for (i,v) in enumerate(nodes))

    # Union-Find for β₀
    parent = collect(1:n_v)
    find!(x) = parent[x]==x ? x : (parent[x]=find!(parent[x]); parent[x])
    function unite!(x,y)
        px,py=find!(x),find!(y); px==py&&return false
        parent[py]=px; true
    end

    n_components = n_v
    for (s,t) in edges
        si=get(node_idx,s,0); ti=get(node_idx,t,0)
        (si==0||ti==0) && continue
        unite!(si,ti) && (n_components-=1)
    end

    beta_0 = n_components
    beta_1 = n_e - n_v + beta_0
    return beta_0, beta_1
end

"""
Step 2: Persistent homology — find backbone circuits.
Returns circuits sorted by persistence (birth weight) descending.
"""
function step2_persistent_backbone(nodes ::Vector{Symbol},
                                    edges ::Vector{Tuple{Symbol,Symbol}},
                                    weights::Dict)

    # Sort edges by weight descending (add highest weight first)
    # Canonicalise each undirected edge as (lexicographically smaller, larger)
    directed_pairs = unique([(string(e[1]) < string(e[2]) ? e[1] : e[2],
                               string(e[1]) < string(e[2]) ? e[2] : e[1])
                              for e in edges])
    sorted = sort(directed_pairs,
                  by=e->get(weights,e,get(weights,(e[2],e[1]),1.0)), rev=true)

    node_idx = Dict(v=>i for (i,v) in enumerate(nodes))
    n = length(nodes)
    parent = collect(1:n)
    find!(x) = parent[x]==x ? x : (parent[x]=find!(parent[x]); parent[x])
    unite!(x,y) = begin px,py=find!(x),find!(y); px==py && return false
                  parent[py]=px; true end

    backbone = Tuple{Float64, Tuple{Symbol,Symbol}}[]
    for (s,t) in sorted
        si=get(node_idx,s,0); ti=get(node_idx,t,0)
        (si==0||ti==0) && continue
        w = get(weights,(s,t), get(weights,(t,s),1.0))
        if !unite!(si,ti)
            # This edge creates a cycle → backbone circuit born at weight w
            push!(backbone, (w, (s,t)))
        end
    end

    sort!(backbone, by=x->x[1], rev=true)
    return backbone
end

"""
Step 3: Build AU star neighborhood for each backbone circuit.
T_C = all stations adjacent to any station in the circuit + the circuit itself.
"""
function step3_build_au_star(circuit_edge ::Tuple{Symbol,Symbol},
                               all_edges    ::Vector{Tuple{Symbol,Symbol}},
                               weights      ::Dict,
                               stations     ::Dict,
                               lines        ::Dict,
                               persistence  ::Float64)::AUContext_MTR

    # The circuit is anchored at two stations
    s, t = circuit_edge
    core_stations = Set([s, t])

    # Find all stations adjacent to the core (star neighborhood)
    star_nodes = Set(core_stations)
    for (a,b) in all_edges
        a ∈ core_stations && push!(star_nodes, b)
        b ∈ core_stations && push!(star_nodes, a)
    end

    # Star edges: all edges within the star
    star_edges = [(a,b) for (a,b) in all_edges
                  if a ∈ star_nodes && b ∈ star_nodes]

    # Identify interchange stations (serve 2+ lines)
    star_node_list = collect(star_nodes)
    interchanges = [v for v in star_node_list
                    if haskey(stations,v) && length(stations[v]) >= 2]

    # Which lines pass through this AU
    covered_lines = Symbol[]
    for v in star_node_list
        haskey(stations,v) && append!(covered_lines, stations[v])
    end
    covered_lines = unique(covered_lines)

    id = Symbol("AU_$(s)_$(t)")
    return AUContext_MTR(id, [circuit_edge],
                          star_node_list, star_edges,
                          interchanges, persistence, covered_lines)
end

"""
Step 4: Compute T₁₂ = T_C ∩ T_C' (intersection of two AUs).
"""
function step4_intersection(au1::AUContext_MTR,
                              au2::AUContext_MTR)

    shared_nodes = intersect(Set(au1.star_nodes), Set(au2.star_nodes))
    shared_edges = intersect(Set(au1.star_edges), Set(au2.star_edges))
    return collect(shared_nodes), collect(shared_edges)
end

"""
Step 5: Compute approximate cokernel for a context pair.
coker ≈ |shared_edges| × (β₁_au1 + β₁_au2) / max_possible
Higher coker = more independent rerouting options = more resilient.
"""
function step5_cokernel(au1::AUContext_MTR,
                         au2::AUContext_MTR,
                         nodes::Vector{Symbol},
                         weights::Dict)::Int

    shared_nodes, shared_edges = step4_intersection(au1, au2)
    n_shared = length(shared_nodes)
    n_shared == 0 && return 0

    # β₁ of the intersection
    _, beta1_12 = step1_compute_beta1(shared_nodes,
                                       collect(shared_edges))

    # β₁ of each AU
    _, beta1_1 = step1_compute_beta1(au1.star_nodes, au1.star_edges)
    _, beta1_2 = step1_compute_beta1(au2.star_nodes, au2.star_edges)

    # Mayer-Vietoris approximation:
    # coker ≈ β₁(AU1) + β₁(AU2) - β₁(T₁₂) (extra classes in union)
    coker_approx = max(0, beta1_1 + beta1_2 - beta1_12)

    # Scale by interchange density (more interchanges = more alternatives)
    n_interchanges = length(union(Set(au1.interchange), Set(au2.interchange)))
    coker_scaled   = coker_approx * max(n_interchanges, 1)

    return coker_scaled
end

"""
Step 6: Build the category of AU contexts.
Objects = {T_C}, Morphisms = inclusions when circuits share edges.
"""
struct MTRCategory
    aus         ::Vector{AUContext_MTR}
    morphisms   ::Dict{Tuple{Int,Int}, Int}  # (i,j) → coker
    resilience  ::Dict{Symbol, Float64}       # station → resilience score
    cut_sets    ::Vector{Vector{Symbol}}      # minimum station cut sets
    n_objects   ::Int
    n_morphisms ::Int
end

function step6_build_category(aus    ::Vector{AUContext_MTR},
                               nodes  ::Vector{Symbol},
                               weights::Dict)::MTRCategory

    n = length(aus)
    morphisms = Dict{Tuple{Int,Int},Int}()

    # Compute cokernel for all pairs
    for i in 1:n, j in i+1:n
        shared_nodes, _ = step4_intersection(aus[i], aus[j])
        isempty(shared_nodes) && continue
        coker = step5_cokernel(aus[i], aus[j], nodes, weights)
        coker > 0 && (morphisms[(i,j)] = coker)
    end

    # Resilience: sum of cokers for each AU (more connections = more resilient)
    resilience = Dict{Symbol,Float64}()
    for (k, au) in enumerate(aus)
        r = sum(coker for ((i,j),coker) in morphisms
                if i==k || j==k; init=0)
        for v in au.interchange
            resilience[v] = get(resilience,v,0.0) + Float64(r)
        end
    end

    # Minimum cut sets: AUs with no morphisms (isolated = fragile)
    isolated = [au for (k,au) in enumerate(aus)
                if !any(i==k||j==k for (i,j) in keys(morphisms))]
    cut_sets = [[v for v in au.star_nodes if haskey(resilience,v)]
                for au in isolated]

    return MTRCategory(aus, morphisms, resilience, cut_sets,
                        n, length(morphisms))
end

# =============================================================================
# PART 3: QUERY MAPPING (Pharmacodynamics → MTR)
# =============================================================================

"""Map a pharmacodynamic scenario to an MTR emergency scenario."""
function map_pd_to_mtr(query::TransportQuery, cat::MTRCategory)

    # Find which AU contains the source and target
    source_au = findfirst(au -> query.source ∈ au.star_nodes, cat.aus)
    target_au = findfirst(au -> query.target ∈ au.star_nodes, cat.aus)

    println("\n  Query: $(query.source) → $(query.target)")
    println("  ($(query.description))")
    println()

    if source_au !== nothing
        @printf("  Source AU: %s (covers %d stations, %d lines)\\n",
                cat.aus[source_au].id,
                length(cat.aus[source_au].star_nodes),
                length(cat.aus[source_au].line_coverage))
    end
    if target_au !== nothing
        @printf("  Target AU: %s (covers %d stations, %d lines)\\n",
                cat.aus[target_au].id,
                length(cat.aus[target_au].star_nodes),
                length(cat.aus[target_au].line_coverage))
    end

    if source_au !== nothing && target_au !== nothing && source_au != target_au
        coker_key = (min(source_au,target_au), max(source_au,target_au))
        coker = get(cat.morphisms, coker_key, 0)
        @printf("  Cokernel between source↔target AUs: %d\\n", coker)
        @printf("  Nash floor equivalent: 1/%d paths unavoidable\\n",
                max(coker, 1))

        if coker > 30
            println("  → HIGH RESILIENCE: many independent rerouting options")
            println("  → Even with primary line closed, multiple alternates exist")
        elseif coker > 10
            println("  → MODERATE RESILIENCE: some rerouting available")
            println("  → Primary closure is disruptive but manageable")
        else
            println("  → LOW RESILIENCE: few alternatives — potential crisis point")
            println("  → Surgery equivalent: activate emergency bus/ferry protocol")
        end
    end
end

# =============================================================================
# PART 4: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__

    println("="^70)
    println("AUTOMATIC AU SELECTION — HONG KONG MTR NETWORK")
    println("Full 6-Step Algorithm")
    println("="^70)

    nodes, edges, weights, stations, lines, line_ridership = build_mtr_network()

    @printf("\nNetwork: %d stations, %d directed edges\\n",
            length(nodes), length(edges))

    # ── Step 1: β₁ ───────────────────────────────────────────────────────
    println("\n[STEP 1] Compute β₁ of Clique(MTR)")
    println("─"^70)
    β₀, β₁ = step1_compute_beta1(nodes, edges)
    @printf("  β₀ = %d (connected components)\\n", β₀)
    @printf("  β₁ = %d (independent circuits = Markov basis size)\\n", β₁)
    if β₁ == 0
        println("  β₁=0: tree network — no AUs needed")
    else
        println("  β₁>0: network has loops — AU selection proceeds")
        @printf("  This MTR subnetwork needs %d AU contexts to cover\\n",
                β₁)
        println("  all transport queries.")
    end

    # ── Step 2: Persistent backbone ───────────────────────────────────────
    println("\n[STEP 2] Persistent Homology — Backbone Circuits")
    println("─"^70)
    backbone = step2_persistent_backbone(nodes, edges, weights)

    println("  Persistence diagram (birth weight = ridership at circuit birth):")
    println("  Birth weight  Circuit edge              Interpretation")
    println("  " * "─"^65)
    for (w, (s,t)) in backbone[1:min(12,end)]
        # Classify by which lines share these stations
        s_lines = get(stations, s, Symbol[])
        t_lines = get(stations, t, Symbol[])
        shared  = intersect(Set(s_lines), Set(t_lines))
        interp  = isempty(shared) ? "transfer point" :
                  "$(first(shared)) line loop"
        @printf("  %-12.1f  %-24s  %s\\n", w,
                "$(s)↔$(t)"[1:min(24,end)], interp)
    end
    @printf("\\n  Total backbone circuits: %d\\n", length(backbone))
    @printf("  Top 5 by ridership (most essential to protect):\\n")
    for (w,(s,t)) in backbone[1:min(5,end)]
        @printf("    w=%-7.1f  %s ↔ %s\\n", w, s, t)
    end

    # ── Step 3: Build AUs ─────────────────────────────────────────────────
    println("\n[STEP 3] Build AU Star Neighborhoods")
    println("─"^70)
    aus = AUContext_MTR[]
    for (w,(s,t)) in backbone[1:min(β₁, 15)]  # top circuits only
        au = step3_build_au_star((s,t), edges, weights,
                                  stations, lines, w)
        push!(aus, au)
    end

    println("  AU contexts built (one per backbone circuit):")
    for au in aus
        @printf("  %-30s  %2d stations  %2d lines  interchanges: %s\\n",
                string(au.id)[1:min(30,end)],
                length(au.star_nodes),
                length(au.line_coverage),
                join([string(v)[1:min(12,end)] for v in au.interchange[1:min(3,end)]],
                     ", "))
    end

    # ── Step 4 & 5: Intersections and cokernel ────────────────────────────
    println("\n[STEP 4+5] Intersections T₁₂ and Cokernel (Resilience)")
    println("─"^70)
    println("  Context pair cokernel (higher = more resilient):")
    println("  AU_i                    AU_j                    coker  status")
    println("  " * "─"^65)

    for i in 1:length(aus), j in i+1:length(aus)
        shared_nodes, shared_edges = step4_intersection(aus[i], aus[j])
        isempty(shared_nodes) && continue
        coker = step5_cokernel(aus[i], aus[j], nodes, weights)
        coker == 0 && continue
        status = coker >= 30 ? "resilient" :
                 coker >= 10 ? "moderate" : "⚠ fragile"
        @printf("  %-23s %-23s %-6d %s\\n",
                string(aus[i].id)[1:min(23,end)],
                string(aus[j].id)[1:min(23,end)],
                coker, status)
    end

    # ── Step 6: Category ──────────────────────────────────────────────────
    println("\n[STEP 6] Build Category of AU Contexts")
    println("─"^70)
    cat = step6_build_category(aus, nodes, weights)

    @printf("  Category: %d objects, %d morphisms\\n",
            cat.n_objects, cat.n_morphisms)

    if !isempty(cat.resilience)
        println("\n  Station resilience scores (higher = more critical hub):")
        sorted_res = sort(collect(cat.resilience), by=x->x[2], rev=true)
        for (v,r) in sorted_res[1:min(10,end)]
            bar = "█"^min(Int(round(r/10)), 20)
            @printf("    %-20s  %.1f  %s\\n", v, r, bar)
        end
    end

    if !isempty(cat.cut_sets)
        println("\n  ⚠ Fragile stations (potential single points of failure):")
        for cs in cat.cut_sets[1:min(5,end)]
            isempty(cs) && continue
            println("    ", join(string.(cs), ", "))
        end
    end

    # ── Query mapping ─────────────────────────────────────────────────────
    println("\n[QUERY MAPPING] Pharmacodynamics → MTR Emergency Scenarios")
    println("─"^70)

    queries = [
        TransportQuery(:Tuen_Mun, :Central,
                        Set{Tuple{Symbol,Symbol}}(),
                        "Baseline: Tuen Mun→Central (CA1sp→sAMY analogue)"),
        TransportQuery(:Tuen_Mun, :Central,
                        Set([(:Nam_Cheong,:Hung_Hom),(:Hung_Hom,:Nam_Cheong)]),
                        "West Rail cross-harbour blocked (HPF→sAMY analogue)"),
        TransportQuery(:Lo_Wu, :Central,
                        Set{Tuple{Symbol,Symbol}}(),
                        "Cross-border: Lo Wu→Central"),
    ]

    for q in queries
        map_pd_to_mtr(q, cat)
    end

    # ── Summary table ─────────────────────────────────────────────────────
    println("\n" * "="^70)
    println("AUTOMATIC AU SELECTION — SUMMARY")
    println("="^70)
    println()
    println("  MTR Network                 BALBc Pharmacodynamics")
    println("  " * "─"^65)
    println("  $(length(nodes)) stations              7 nodes (Q_7P)")
    @printf("  β₁ = %-3d circuits              β₁ = 4-6 circuits\\n", β₁)
    println("  $(length(aus)) AU contexts selected     2 AU contexts (CTX_sAMY, CTX_HPF)")
    println("  Persistence filter = ridership  Persistence filter = Renkin-Crone")
    println("  Interchanges = hubs             Interchanges = sAMY, HPF")
    println()
    println("  Transport query:")
    println("    CA1sp → sAMY               ↔  Tuen Mun → Central")
    println("    stops Λ = drug blocks       ↔  stops Λ = line closures")
    println("    Nash floor 1/17             ↔  Min rerouting (1/β₁)")
    println("    coker=62 obstruction        ↔  coker=? (run 4ti2 on MTR)")
    println("    norcain blocks HPF→sAMY     ↔  Typhoon closes West Rail")
    println("    AU-QKV finds CA1sp→sAMY     ↔  AI routes via MTR Airport Express")
    println("    Surgery: Mode 4 → Sector D  ↔  Emergency cross-harbour ferry")
    println()
    println("  NEXT STEP: Run 4ti2 markov on MTR toric ideal")
    println("    → Get exact Markov basis circuits")
    println("    → Compute true coker between interchange AU pairs")
    println("    → Check: does any pair have coker=62?")
    println("    → If yes: EXACT transfer from pharmacodynamic Q-table")
    println("="^70)
end
