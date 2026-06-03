# =============================================================================
# compute_hh3.jl  (v6 — Complete SVD Pipeline + Synchronized m3 Corrections)
# =============================================================================

using LinearAlgebra, SparseArrays, Printf

let
    orig_dir  = dirname(abspath(@__FILE__))
    orig_file = joinpath(orig_dir, "curved_hh2_sparse_refactored_filteredA.jl")
    isfile(orig_file) || error("Cannot find $orig_file")
    src = read(orig_file, String)
    src = replace(src, r"@__DIR__(?!\w)" => repr(orig_dir))
    marker = "if length(ARGS) >= 1 && ARGS[1] == \"--ainf-only\""
    pos = findfirst(marker, src)
    defs = pos !== nothing ? src[1:pos[1]-1] : src
    tmp = tempname() * ".jl"; write(tmp, defs); include(tmp); rm(tmp, force=true)
end
println("Definitions loaded.")
import Random; Random.seed!(42)

const STOPS_HH3 = Dict(
    :A => Set(["f_BLA_sAMY","f_sAMY_BLA","f_LA_sAMY","f_sAMY_LA",
               "f_sAMY_HY","f_HY_sAMY","f_sAMY_PAL","f_PAL_sAMY"]),
    :C => Set(["f_sAMY_HY","f_HY_sAMY","f_sAMY_PAL","f_PAL_sAMY"]),
)

function filter_rels_hh3(rel_str, stops)
    kept = String[]
    for line in split(rel_str, '\n')
        line = strip(line); isempty(line) && continue
        parts = split(line, r"[\*\- ]+")
        arrows = filter(p->startswith(strip(p),"f_"), parts)
        any(a->strip(a) in stops, arrows) && continue
        push!(kept, line)
    end
    join(kept, '\n')
end

sector_raw_coeffs(s::Symbol) = parse_relations(
    filter_rels_hh3(relations_str, get(STOPS_HH3, s, Set{String}())))

function sp_rank(A; rtol=1e-6)
    isempty(A.nzval) && return 0
    D = Matrix(A); F = svd(D)
    isempty(F.S) && return 0
    atol = rtol * maximum(F.S)
    sum(F.S .> atol)
end

function check_zero(A, B; label="")
    prod = A * B
    isempty(prod.nzval) && (println(@sprintf("    max|%s| = 0  ✓", label)); return 0.0)
    maxval = maximum(abs.(prod.nzval))
    maxA = isempty(A.nzval) ? 1.0 : maximum(abs.(A.nzval))
    maxB = isempty(B.nzval) ? 1.0 : maximum(abs.(B.nzval))
    rel = maxval / (maxA * maxB)
    println(@sprintf("    max|%s| = %.2e  relative = %.2e  %s",
            label, maxval, rel, rel < 1e-6 ? "✓" : "✗"))
    return rel
end

function convert_m3(m3_raw)
    m3 = Dict{Tuple{Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
    for (triple, (left, right)) in m3_raw
        diff = Dict{Symbol,Float64}()
        left[2]  !== nothing && (diff[left[2]]  = get(diff, left[2], 0.0) + left[1])
        right[2] !== nothing && (diff[right[2]] = get(diff, right[2], 0.0) - right[1])
        for (k,v) in diff; abs(v) < 1e-12 && delete!(diff, k); end
        !isempty(diff) && (m3[triple] = diff)
    end
    return m3
end

# =============================================================================
# build_d2_with_m3: Fully synchronized with both left and right m3 actions
# =============================================================================
function build_d2_with_m3(C2, C3, mult_tab, m2, C3_idx, m3_dict)
    nC2 = length(C2); nC3 = length(C3)
    rows, cols, vals = Int[], Int[], Float64[]

    C2_idx = Dict(pair => idx for (idx, pair) in enumerate(C2))

    # 1. Standard inner terms
    for (j, (a,b)) in enumerate(C2)
        for (c, _) in mult_tab[b]
            c1, t1 = m2(a,b)
            if t1 !== nothing
                c2, t2 = m2(t1, c)
                if t2 !== nothing
                    push!(rows, C3_idx[(a,b,c)]); push!(cols, j); push!(vals, c1*c2)
                end
            end
            c3, t3 = m2(b,c)
            if t3 !== nothing
                c4, t4 = m2(a, t3)
                if t4 !== nothing
                    push!(rows, C3_idx[(a,b,c)]); push!(cols, j); push!(vals, -c3*c4)
                end
            end
        end
    end

    # 2. m3 homotopy defect terms (Left and Right boundary corrections)
    for (i, (a,b,c)) in enumerate(C3)
        haskey(m3_dict, (a,b,c)) || continue
        for (sym_out, coeff) in m3_dict[(a,b,c)]
            abs(coeff) < 1e-12 && continue

            lp = (sym_out, c)
            if haskey(C2_idx, lp)
                push!(rows, i); push!(cols, C2_idx[lp]); push!(vals,  coeff)
            end

            rp = (a, sym_out)
            if haskey(C2_idx, rp)
                push!(rows, i); push!(cols, C2_idx[rp]); push!(vals, -coeff)
            end
        end
    end

    return sparse(rows, cols, vals, nC3, nC2)
end

function build_d3_full_corrected(C3, C4, C4_idx, basis, m2_fn, m3_dict)
    nC3 = length(C3); nC4 = length(C4)
    rows, cols, vals = Int[], Int[], Float64[]

    m3_inv = Dict{Symbol, Vector{Tuple{NTuple{3,Symbol}, Float64}}}()
    for (triple, outputs) in m3_dict
        for (sym_out, coeff) in outputs
            abs(coeff) < 1e-12 && continue
            if !haskey(m3_inv, sym_out)
                m3_inv[sym_out] = Vector{Tuple{NTuple{3,Symbol}, Float64}}()
            end
            push!(m3_inv[sym_out], (triple, coeff))
        end
    end

    for (j, (e,f,g)) in enumerate(C3)
        # Inner terms
        for a in basis, b in basis
            tgt(a)==src(b) || continue
            c_ab, t_ab = m2_fn(a,b); t_ab == e || continue
            rk = (a,b,f,g); haskey(C4_idx,rk) || continue
            abs(c_ab)>1e-12 && (push!(rows,C4_idx[rk]); push!(cols,j); push!(vals,-c_ab))
        end
        for b in basis, c in basis
            tgt(b)==src(c) || continue
            c_bc, t_bc = m2_fn(b,c); t_bc == f || continue
            rk = (e,b,c,g); haskey(C4_idx,rk) || continue
            abs(c_bc)>1e-12 && (push!(rows,C4_idx[rk]); push!(cols,j); push!(vals,c_bc))
        end
        for c in basis, d in basis
            tgt(c)==src(d) || continue
            c_cd, t_cd = m2_fn(c,d); t_cd == g || continue
            rk = (e,f,c,d); haskey(C4_idx,rk) || continue
            abs(c_cd)>1e-12 && (push!(rows,C4_idx[rk]); push!(cols,j); push!(vals,-c_cd))
        end

        # Outer terms
        for a in basis
            tgt(a)==src(e) || continue
            rk = (a,e,f,g); haskey(C4_idx,rk) || continue
            push!(rows,C4_idx[rk]); push!(cols,j); push!(vals, 1.0)
        end
        for d in basis
            tgt(g)==src(d) || continue
            rk = (e,f,g,d); haskey(C4_idx,rk) || continue
            push!(rows,C4_idx[rk]); push!(cols,j); push!(vals,-1.0)
        end

        # m3 Homotopy terms via inverted lookup
        if haskey(m3_inv, e)
            for (triple, coeff) in m3_inv[e]
                rk = (triple[1], triple[2], triple[3], g)
                haskey(C4_idx,rk) || continue
                push!(rows,C4_idx[rk]); push!(cols,j); push!(vals,-coeff)
            end
        end
        if haskey(m3_inv, g)
            for (triple, coeff) in m3_inv[g]
                rk = (e, triple[1], triple[2], triple[3])
                haskey(C4_idx,rk) || continue
                push!(rows,C4_idx[rk]); push!(cols,j); push!(vals, coeff)
            end
        end
    end
    return sparse(rows, cols, vals, nC4, nC3)
end

# =============================================================================
function compute_sector(sector::Symbol)
    rc     = sector_raw_coeffs(sector)
    basis  = build_basis(nodes, rc)
    m2     = make_m2(rc, basis)
    is_c   = (x,y) -> tgt(x) == src(y)
    C2, C3 = compute_composable_chains(basis, is_c)
    C3_idx = Dict(c=>i for (i,c) in enumerate(C3))
    C4     = build_Ck(basis, is_c, 4; max_size=50000)
    C4_idx = Dict(c=>i for (i,c) in enumerate(C4))
    m3_raw, mult_tab = compute_m3(basis, m2, C2, C3)
    m3_dict = convert_m3(m3_raw)

    println(@sprintf("  Sector %s: |C2|=%d |C3|=%d |C4|=%d  |m3|=%d",
            sector, length(C2), length(C3), length(C4), length(m3_dict)))

    d0  = build_d0(C2, basis, m2)
    d1  = build_d1(C2, basis, m2)
    
    # FIX: Call the corrected operator containing m3 homotopy terms!
    d2  = build_d2_with_m3(C2, C3, mult_tab, m2, C3_idx, m3_dict)
    d3  = build_d3_full_corrected(C3, C4, C4_idx, basis, m2, m3_dict)

    # Cross-check against the original file's baseline structure
    hh2_check = compute_HH2(d0, d1, d2)

    println(@sprintf("    d1:%d×%d nnz=%d  d2:%d×%d nnz=%d  d3:%d×%d nnz=%d",
            size(d1)..., nnz(d1), size(d2)..., nnz(d2), size(d3)..., nnz(d3)))

    rel21 = check_zero(d2, d1; label="d2∘d1")
    rel32 = check_zero(d3, d2; label="d3∘d2")

    # Precise rank assessment kernel matching gps_cone_hh2 SVD thresholding
    function hh_dim(d_in, d_out; atol=1e-8)
        function arank(A)
            m, n = size(A)
            k = min(m, n)
            k < 1 && return 0
            F = svd(Matrix(Float64.(A)))
            sum(F.S .> atol)
        end
        nullity_out = size(d_out, 2) - arank(d_out)
        rank_in     = arank(d_in)
        return max(0, nullity_out - rank_in)
    end

    hh2 = hh_dim(d1, d2)
    hh3 = hh_dim(d2, d3)

    r1 = sp_rank(d1); r2 = sp_rank(d2); r3 = sp_rank(d3)
    println(@sprintf("    rank: d1=%d d2=%d d3=%d", r1, r2, r3))
    println(@sprintf("    HH²(W_%s) = %d  (compute_HH2 cross-check=%d)  %s",
            sector, hh2, hh2_check, hh2 == hh2_check ? "✓ consistent" : "✗ mismatch"))
    println(@sprintf("    HH³(W_%s) = %d  (SVD projection method)", sector, hh3))

    return (hh2=hh2, hh3=hh3, valid=rel32<1e-6,
            provisional=rel32<1e-2, rel32=rel32, rel21=rel21)
end

println("\n── Sector A ─────────────────────────────────────────────────")
rA = compute_sector(:A)
println("\n── Sector C ─────────────────────────────────────────────────")
rC = compute_sector(:C)

println("\n── Summary ──────────────────────────────────────────────────")
println(@sprintf("  HH²(W_A)=%d  HH²(W_C)=%d  (SVD projection, matching gps_cone_hh2)", rA.hh2, rC.hh2))
println(@sprintf("  HH³(W_A)=%d  HH³(W_C)=%d  (SVD projection via corrected differential)", rA.hh3, rC.hh3))
println()
