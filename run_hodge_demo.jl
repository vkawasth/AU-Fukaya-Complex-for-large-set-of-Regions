# =============================================================================
# run_hodge_demo.jl
#
# Runs Pass 0 (Hodge decomposition) and compares to existing scalar gates.
#
#   julia run_hodge_demo.jl
#
# Expected output:
#   - Hodge gate for every notable station (dead/unstable/live)
#   - Comparison: Hodge harm_ratio vs. existing k-inv
#   - Comparison: Hodge curl_ratio vs. existing stability score
#   - serve_ad comparison: old vs Hodge-gated
# =============================================================================

using LinearAlgebra, Printf

const DIR = @__DIR__

println("Loading pipeline...")
include(joinpath(DIR, "mtr_ad_game.jl"))
include(joinpath(DIR, "fukaya_ad_context.jl"))
include(joinpath(DIR, "au_fukaya_engine.jl"))
include(joinpath(DIR, "au_compiler.jl"))
include(joinpath(DIR, "hodge_pass.jl"))

# ── Build omega ────────────────────────────────────────────────────────────────
line_ridership_flat = Dict{String,Float64}()
for (line,(seq,w)) in LINE_SEQ
    for s in seq
        line_ridership_flat[s] = max(get(line_ridership_flat,s,0.0),w)
    end
end
R_vec = Float64[get(line_ridership_flat,s,50.0) for s in STATIONS]
R_vec ./= maximum(R_vec)

hour_profile = [0.2,0.1,0.1,0.1,0.2,0.4,0.8,1.0,0.9,0.7,0.6,0.7,
                0.7,0.6,0.5,0.6,0.7,0.9,1.0,0.9,0.8,0.6,0.5,0.3]
month_res    = [1.2,1.5,1.1,1.0,1.0,1.0,1.1,1.1,1.0,1.2,1.3,1.4]
nS, nH, nM  = length(STATIONS), 24, 12
omega = zeros(nS, nH, nM)
for s in 1:nS, h in 1:nH, m in 1:nM
    omega[s,h,m] = R_vec[s] * hour_profile[h] * month_res[m]
end

sf          = build_symplectic_form(STATIONS, R_vec)
lagrangians = build_lagrangians(STATIONS, DEMO_PROFILES, sf)
demo_lags   = filter(l -> !l.is_temporal, lagrangians)
products    = collect(PRODUCTS)

# ── Run existing passes 1-5 ───────────────────────────────────────────────────
println("Running existing passes 1-5...")
embeddings  = compile_product_embeddings(products, lagrangians, omega)
stab_table  = compile_stability_table(STATIONS, demo_lags, omega)
edges_named = [(STATIONS[e[1]], STATIONS[e[2]]) for e in EDGES]
weights_named = Dict((STATIONS[k[1]], STATIONS[k[2]])=>v
                     for (k,v) in EDGE_WEIGHTS)
neighbors   = compile_neighborhood_table(STATIONS, edges_named, weights_named)
routes_vec  = compile_routing_table(STATIONS, products, embeddings,
                                     demo_lags, omega, stab_table;
                                     top_n=10, hours=[9,18], months=[2,7,12])
route_dict  = Dict{Tuple{Int,Int,Int}, Vector{AdRoute}}()
for r in routes_vec
    key = (r.station_idx, r.hour, findfirst(==(r.month), MONTH_NAMES))
    push!(get!(route_dict, key, AdRoute[]), r)
end
brackets    = compile_hmm_brackets(STATIONS, products, demo_lags, omega,
                                    embeddings, stab_table;
                                    months=[2,7,12])
bracket_idx = build_bracket_index(brackets)

ctx = RuntimeContext(
    route_dict, stab_table, neighbors,
    Dict(i=>true for i in 1:length(products)),
    Dict{Tuple{Int,Int,Int},Float64}(),
    embeddings,
)

# ── Run Pass 0: Hodge decomposition ──────────────────────────────────────────
println("\nRunning Pass 0: Hodge decomposition...")
hodge_table = compile_hodge_table(STATIONS, EDGES, omega)

# ── Report ────────────────────────────────────────────────────────────────────
print_hodge_report(hodge_table, STATIONS; hours=[9,18], months=[2,7])

# ── Comparison: Hodge harm_ratio vs k-inv ─────────────────────────────────────
println("="^70)
println("COMPARISON: Hodge classifications for all 81 stations (9am Feb)")
println("  Gate: LIVE/DEAD  |  topo: LOOP/tree  |  harm=loop signal  |  omega=ridership")
println("="^70)

omega_max_val = maximum(omega)
@printf("  %-22s  %s  %6s  %6s  %6s  %7s  %s\n",
        "Station", "Gate    ", "grad", "harm", "curl", "omega", "topo")
println("  " * "─"^74)

# Sort by omega descending so busiest stations appear first
order = sortperm([omega[s,9,2] for s in 1:length(STATIONS)], rev=true)
for s in order
    sname = STATIONS[s]
    slot  = hodge_table[s, 9, 2]
    gate_str = slot.gate == HODGE_DEAD ? "DEAD    " :
               slot.gate == HODGE_UNSTABLE ? "UNSTABLE" : "LIVE    "
    topo = slot.harm_ratio > 0.05 ? "LOOP" : "tree"
    @printf("  %-22s  %s  %6.3f  %6.3f  %6.3f  %7.4f  %s\n",
            sname, gate_str,
            slot.grad_ratio, slot.harm_ratio, slot.curl_ratio,
            omega[s,9,2], topo)
end

# Summary: loop stations
loop_stations = [STATIONS[s] for s in 1:length(STATIONS)
                 if hodge_table[s,9,2].harm_ratio > 0.05]
dead_stations = [STATIONS[s] for s in 1:length(STATIONS)
                 if hodge_table[s,9,2].gate == HODGE_DEAD]

println("\nLoop stations (β₁=9 cycles, harm_ratio > 0.05):")
for sn in loop_stations
    s = findfirst(==(sn), STATIONS)
    @printf("  %-22s  harm=%.3f  omega=%.4f\n",
            sn, hodge_table[s,9,2].harm_ratio, omega[s,9,2])
end

println("\nDead zones (omega < $(round(0.02*omega_max_val,digits=4))):")
if isempty(dead_stations)
    println("  (none at 9am Feb — all stations have ridership above threshold)")
    println("  Try off-peak: check hour=3, month=7 for lower-ridership slots")
    # Show lowest-omega stations
    println("\nLowest-omega stations at 9am Feb (most likely dead off-peak):")
    bottom = sortperm([omega[s,9,2] for s in 1:length(STATIONS)])[1:8]
    for s in bottom
        @printf("  %-22s  omega=%.4f\n", STATIONS[s], omega[s,9,2])
    end
else
    for sn in dead_stations; println("  $sn"); end
end

# ── Comparison: old serve_ad vs Hodge serve_ad — all stations ──────────────
println("\n" * "="^70)
println("COMPARISON: serve_ad (old) vs hodge_serve_ad (new) — all 81 stations")
println("  Only mismatches shown. Summary counts at the bottom.")
println("="^70)

# Discover which hours/months the routing table was compiled for
month_keys = sort(unique(k[3] for k in keys(ctx.routes)))
hour_keys  = sort(unique(k[2] for k in keys(ctx.routes)))
println("  Route dict: hours=$hour_keys  months=$month_keys\n")

@printf("  %-22s  %4s %3s  %-18s  %-18s  %s\n",
        "Station", "hour", "mo", "old serve_ad", "hodge_serve_ad", "gate")
println("  " * "-"^82)

n_match = 0; n_dead_agree = 0; n_mismatch = 0

order = sortperm([omega[s, 9, first(month_keys)] for s in 1:length(STATIONS)], rev=true)

let n_match = 0, n_dead_agree = 0, n_mismatch = 0
    for s in order
        sname = STATIONS[s]
        for h in hour_keys, m in month_keys
            old_r         = serve_ad(ctx, s, h, m)
            hodge_r, gate = hodge_serve_ad(ctx, hodge_table, s, h, m)

            old_str   = old_r   !== nothing ? old_r.product[1:min(16,end)]   : "(none)"
            hodge_str = hodge_r !== nothing ? hodge_r.product[1:min(16,end)] : "(none)"
            gate_str  = gate == HODGE_DEAD ? "DEAD  " :
                        gate == HODGE_UNSTABLE ? "UNSTBL" : "LIVE  "

            if old_str == hodge_str
                n_match += 1
            elseif old_r === nothing && hodge_r === nothing
                n_dead_agree += 1
            else
                n_mismatch += 1
                @printf("  %-22s  %4d %3d  %-18s  %-18s  [%s] ✗\n",
                        sname, h, m, old_str, hodge_str, gate_str)
            end
        end
    end

    total = n_match + n_dead_agree + n_mismatch
    println()
    @printf("  Slots tested   : %d (%d stations x %d hours x %d months)\n",
            total, length(STATIONS), length(hour_keys), length(month_keys))
    @printf("  Exact match    : %d  (%.1f%%)\n", n_match,      100*n_match/total)
    @printf("  Both (none)    : %d  (%.1f%%)  dead-zone agreement\n",
            n_dead_agree, 100*n_dead_agree/total)
    @printf("  Mismatch       : %d  (%.1f%%)\n", n_mismatch,   100*n_mismatch/total)
end

println("""
Interpretation:
  harm_ratio > 0  station is on one of the 9 network loops (LOOP)
  harm_ratio = 0  tree station, pure gradient routing, still live
  gate = DEAD     omega < 2pct of peak (off-peak terminus)
  gate = LIVE     normal serve path (all stations at 9am Feb are LIVE)
""")
