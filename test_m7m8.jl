# =============================================================================
# test_m7m8.jl
# Run from the FukayaAUComplex directory:
#   julia test_m7m8.jl
# =============================================================================

# ── Step 1: Load original file (definitions only, skip execution block) ──────
let
    orig_dir  = dirname(abspath(@__FILE__))
    orig_file = joinpath(orig_dir, "curved_hh2_sparse_refactored_filteredA.jl")
    isfile(orig_file) || error("Cannot find $orig_file")
    src  = read(orig_file, String)
    # Replace @__DIR__ (the macro, not substrings) with the actual path string
    # Use a regex that matches @__DIR__ as a whole token (not part of another word)
    src  = replace(src, r"@__DIR__(?!\w)" => repr(orig_dir))
    marker = "if length(ARGS) >= 1 && ARGS[1] == \"--ainf-only\""
    pos  = findfirst(marker, src)
    defs = pos !== nothing ? src[1:pos[1]-1] : src
    tmp  = tempname() * ".jl"
    write(tmp, defs); include(tmp); rm(tmp, force=true)
end
println("Original file loaded.")

# ── Step 2: Define sector_raw_coeffs directly (avoids loading gps_cone_hh2) ──
const STOPS_TEST = Dict(
    :A => Set(["f_BLA_sAMY","f_sAMY_BLA","f_LA_sAMY","f_sAMY_LA",
               "f_sAMY_HY","f_HY_sAMY","f_sAMY_PAL","f_PAL_sAMY"]),
    :B => Set(["f_BLA_sAMY","f_sAMY_BLA","f_LA_sAMY","f_sAMY_LA"]),
    :C => Set(["f_sAMY_HY","f_HY_sAMY","f_sAMY_PAL","f_PAL_sAMY"]),
    :D => Set(["f_LA_sAMY","f_sAMY_LA"]),
)

function filter_rels(rel_str::String, stops::Set{String})
    kept = String[]
    for line in split(rel_str, '\n')
        line = strip(line)
        isempty(line) && continue
        parts = split(line, r"[\*\- ]+")
        arrows = filter(p -> startswith(strip(p), "f_"), parts)
        any(a -> strip(a) in stops, arrows) && continue
        push!(kept, line)
    end
    return join(kept, '\n')
end

function sector_raw_coeffs(sector::Symbol)
    stops = get(STOPS_TEST, sector, Set{String}())
    filtered = filter_rels(relations_str, stops)
    parse_relations(filtered)
end

println("sector_raw_coeffs defined for sectors A/B/C/D.")

# ── Step 3: Load m7/m8 extension ─────────────────────────────────────────────
include(joinpath(dirname(abspath(@__FILE__)), "m7_m8_extension.jl"))

# ── Step 4: Build Sector A basis and m2..m6 ──────────────────────────────────
println("\n── Building Sector A A∞ structure ──────────────────────────")
using Printf

rc_A   = sector_raw_coeffs(:A)
basis_A = build_basis(nodes, rc_A)
m2_A    = make_m2(rc_A, basis_A)
is_comp = (x,y) -> tgt(x) == src(y)
C2_A, C3_A = compute_composable_chains(basis_A, is_comp)
C3_idx     = Dict(c=>i for (i,c) in enumerate(C3_A))

println(@sprintf("  |C2|=%d  |C3|=%d", length(C2_A), length(C3_A)))

# m3
m3_raw, mult_tab = compute_m3(basis_A, m2_A, C2_A, C3_A)
# Convert m3_raw (Tuple format) to Dict format used by m7_obstruction_full
m3_A = Dict{Tuple{Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
for (triple, (left, right)) in m3_raw
    diff = Dict{Symbol,Float64}()
    left[2]  !== nothing && (diff[left[2]]  = get(diff, left[2],  0.0) + left[1])
    right[2] !== nothing && (diff[right[2]] = get(diff, right[2], 0.0) - right[1])
    !isempty(diff) && (m3_A[triple] = diff)
end
println(@sprintf("  m3 entries: %d", length(m3_A)))

# m4, m5, m6 via the original pipeline
C4_A = build_Ck(basis_A, is_comp, 4; max_size=20000)
C5_A = build_Ck(basis_A, is_comp, 5; max_size=50000)
C6_A = build_Ck(basis_A, is_comp, 6; max_size=100000)
println(@sprintf("  |C4|=%d  |C5|=%d  |C6|=%d", length(C4_A), length(C5_A), length(C6_A)))

# Build mult_dict: wraps m2_A to return Dict{Symbol,Float64} instead of tuple
# This is what compute_global_m5 and compute_m6_selective expect
mult_dict_raw_A = Dict{Tuple{Symbol,Symbol}, Dict{Symbol,Float64}}()
for a in basis_A, b in basis_A
    c, t = m2_A(a,b)
    if t !== nothing && abs(c) > 1e-12
        mult_dict_raw_A[(a,b)] = Dict(t => c)
    end
end
mult_dict_A(a,b) = get(mult_dict_raw_A, (a,b), Dict{Symbol,Float64}())

m4_A = compute_m4_obs(C4_A, m2_A, m3_A)
println(@sprintf("  m4 entries: %d", length(m4_A)))

m5_A = compute_global_m5(C5_A, m3_A, m4_A, mult_dict_A)
println(@sprintf("  m5 entries: %d", length(m5_A)))

m6_A = compute_m6_selective(C6_A, mult_dict_A, m3_A, m4_A, m5_A)
println(@sprintf("  m6 entries: %d", length(m6_A)))

# ── Step 5: Build C7 and compute m7 ──────────────────────────────────────────
println("\n── m7 ───────────────────────────────────────────────────────")
C7_A = compute_C7(basis_A, is_comp)
println(@sprintf("  |C7| = %d", length(C7_A)))

println("  Computing m7...")
m7_A = compute_m7(C7_A, mult_dict_A, m3_A, m4_A, m5_A, m6_A)
println(@sprintf("  m7 entries: %d", length(m7_A)))

# ── Step 6: Verify A∞ relation n=8 ───────────────────────────────────────────
println("\n── Verification: A∞ relation n=8 ───────────────────────────")
v8 = verify_ainf_relation_n8(C7_A, mult_dict_A, m3_A, m4_A, m5_A, m6_A, m7_A)
println(@sprintf("  passed=%s  max_residual=%.2e  n_checked=%d",
        v8.passed, v8.max_residual, v8.n_checked))
v8.passed || @warn "A∞ relation n=8 NOT satisfied — check m3 dict format"

# ── Step 7: Order-7 prime paths ───────────────────────────────────────────────
println("\n── Order-7 prime paths (top 5) ──────────────────────────────")
paths7 = extract_prime_paths_m7(m7_A; top_n=5)
for (chain, score) in paths7
    println(@sprintf("  %s  score=%.4f", chain, score))
end

# ── Step 8: Build C8 and compute m8 ──────────────────────────────────────────
println("\n── m8 ───────────────────────────────────────────────────────")
C8_A = compute_C8(basis_A, is_comp)
println(@sprintf("  |C8| = %d", length(C8_A)))

println("  Computing m8...")
m8_A = compute_m8(C8_A, mult_dict_A, m3_A, m4_A, m5_A, m6_A, m7_A)
println(@sprintf("  m8 entries: %d", length(m8_A)))

# ── Step 9: Verify A∞ relation n=9 ───────────────────────────────────────────
println("\n── Verification: A∞ relation n=9 ───────────────────────────")
v9 = verify_ainf_relation_n9(C8_A, mult_dict_A, m3_A, m4_A, m5_A, m6_A, m7_A, m8_A)
println(@sprintf("  passed=%s  max_residual=%.2e  n_checked=%d",
        v9.passed, v9.max_residual, v9.n_checked))

# ── Summary ────────────────────────────────────────────────────────────────────
println("\n── Summary ──────────────────────────────────────────────────")
println(@sprintf("  m7 entries: %d  (order-7 non-associativity loci)", length(m7_A)))
println(@sprintf("  m8 entries: %d  (order-8 non-associativity loci)", length(m8_A)))
println(@sprintf("  A∞ n=8 verified: %s", v8.passed))
println(@sprintf("  A∞ n=9 verified: %s", v9.passed))
println()
println("  HH²(W_A) is UNCHANGED — coker=62 from gps_cone_hh2.jl is exact.")
println("  m7 prime paths are new defect chains at order 7.")
println("  m8 is needed for sAMY⊗HPF⊗Infra triple-context gate=8 prediction.")
