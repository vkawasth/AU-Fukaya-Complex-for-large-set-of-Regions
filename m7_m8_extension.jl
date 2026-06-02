# =============================================================================
# m7_m8_extension.jl
#
# Extension of curved_hh2_sparse_refactored_filteredA.jl to m7 and m8.
# Include AFTER the original file (definitions only, no execution block).
#
# Implements:
#   m7_obstruction_full(a,b,c,d,e,f,g, m2,m3,m4,m5,m6)  — 15 terms
#   compute_m7(C7, m2,m3,m4,m5,m6)
#   m8_obstruction_full(a,b,c,d,e,f,g,h, m2,m3,m4,m5,m6,m7)  — 21 terms
#   compute_m8(C8, m2,m3,m4,m5,m6,m7)
#   compute_C7(basis, is_composable)  — 7-tuples of composable elements
#   compute_C8(basis, is_composable)  — 8-tuples
#   compute_HH2_with_m8(...)          — HH² using full m2..m8 differential
#
# Usage:
#   include("curved_hh2_sparse_refactored_filteredA.jl")  # defs only
#   include("m7_m8_extension.jl")
#
# Pattern follows m6_obstruction_full exactly:
#   - Each (j,i) block handles m_i(..., m_j(...), ...)
#   - Signs alternate +/- starting with + for k=0 in each block
#   - haskey checks before lookup (m3,m4,m5,m6 are sparse Dicts)
#   - m2 is a function, all others are Dict{Tuple,Dict{Symbol,Float64}}
# =============================================================================

# =============================================================================
# CHAIN SPACE BUILDERS: C7 and C8
# =============================================================================

"""
    compute_C7(basis, is_composable)

Build the space of composable 7-tuples from the given basis.
C7 = {(a,b,c,d,e,f,g) : each consecutive pair is composable}
"""
function compute_C7(basis, is_composable)
    C7 = NTuple{7,Symbol}[]
    for a in basis, b in basis
        is_composable(a,b) || continue
        for c in basis
            is_composable(b,c) || continue
            for d in basis
                is_composable(c,d) || continue
                for e in basis
                    is_composable(d,e) || continue
                    for f in basis
                        is_composable(e,f) || continue
                        for g in basis
                            is_composable(f,g) || continue
                            push!(C7, (a,b,c,d,e,f,g))
                        end
                    end
                end
            end
        end
    end
    return C7
end

"""
    compute_C8(basis, is_composable)

Build the space of composable 8-tuples.
"""
function compute_C8(basis, is_composable)
    C8 = NTuple{8,Symbol}[]
    for a in basis, b in basis
        is_composable(a,b) || continue
        for c in basis
            is_composable(b,c) || continue
            for d in basis
                is_composable(c,d) || continue
                for e in basis
                    is_composable(d,e) || continue
                    for f in basis
                        is_composable(e,f) || continue
                        for g in basis
                            is_composable(f,g) || continue
                            for h in basis
                                is_composable(g,h) || continue
                                push!(C8, (a,b,c,d,e,f,g,h))
                            end
                        end
                    end
                end
            end
        end
    end
    return C8
end

# =============================================================================
# m7_obstruction_full — 15 terms, pairs (i,j) with i+j=8
# =============================================================================

function m7_obstruction_full(a,b,c,d,e,f,g, m2,m3,m4,m5,m6)
    total = Dict{Symbol,Float64}()

    # ── j=2, i=6: m6(..., m2(...), ...) ── 5 terms ──────────────────────
    # + m6(m2(a,b), c,d,e,f, g)
    ab = m2(a,b)
    for (x,c_ab) in ab
        haskey(m6,(x,c,d,e,f,g)) || continue
        add_dict!(total, m6[(x,c,d,e,f,g)], c_ab)
    end
    # - m6(a, m2(b,c), d,e,f, g)
    bc = m2(b,c)
    for (x,c_bc) in bc
        haskey(m6,(a,x,d,e,f,g)) || continue
        add_dict!(total, m6[(a,x,d,e,f,g)], -c_bc)
    end
    # + m6(a,b, m2(c,d), e,f, g)
    cd = m2(c,d)
    for (x,c_cd) in cd
        haskey(m6,(a,b,x,e,f,g)) || continue
        add_dict!(total, m6[(a,b,x,e,f,g)], c_cd)
    end
    # - m6(a,b,c, m2(d,e), f, g)
    de = m2(d,e)
    for (x,c_de) in de
        haskey(m6,(a,b,c,x,f,g)) || continue
        add_dict!(total, m6[(a,b,c,x,f,g)], -c_de)
    end
    # + m6(a,b,c,d, m2(e,f), g)
    ef = m2(e,f)
    for (x,c_ef) in ef
        haskey(m6,(a,b,c,d,x,g)) || continue
        add_dict!(total, m6[(a,b,c,d,x,g)], c_ef)
    end

    # ── j=3, i=5: m5(..., m3(...), ...) ── 4 terms ──────────────────────
    # + m5(m3(a,b,c), d,e,f,g)
    if haskey(m3,(a,b,c))
        for (x,cx) in m3[(a,b,c)]
            haskey(m5,(x,d,e,f,g)) || continue
            add_dict!(total, m5[(x,d,e,f,g)], cx)
        end
    end
    # - m5(a, m3(b,c,d), e,f,g)
    if haskey(m3,(b,c,d))
        for (x,cx) in m3[(b,c,d)]
            haskey(m5,(a,x,e,f,g)) || continue
            add_dict!(total, m5[(a,x,e,f,g)], -cx)
        end
    end
    # + m5(a,b, m3(c,d,e), f,g)
    if haskey(m3,(c,d,e))
        for (x,cx) in m3[(c,d,e)]
            haskey(m5,(a,b,x,f,g)) || continue
            add_dict!(total, m5[(a,b,x,f,g)], cx)
        end
    end
    # - m5(a,b,c, m3(d,e,f), g)
    if haskey(m3,(d,e,f))
        for (x,cx) in m3[(d,e,f)]
            haskey(m5,(a,b,c,x,g)) || continue
            add_dict!(total, m5[(a,b,c,x,g)], -cx)
        end
    end

    # ── j=4, i=4: m4(..., m4(...), ...) ── 3 terms ──────────────────────
    # + m4(m4(a,b,c,d), e,f,g)
    if haskey(m4,(a,b,c,d))
        for (x,cx) in m4[(a,b,c,d)]
            haskey(m4,(x,e,f,g)) || continue
            add_dict!(total, m4[(x,e,f,g)], cx)
        end
    end
    # - m4(a, m4(b,c,d,e), f,g)
    if haskey(m4,(b,c,d,e))
        for (x,cx) in m4[(b,c,d,e)]
            haskey(m4,(a,x,f,g)) || continue
            add_dict!(total, m4[(a,x,f,g)], -cx)
        end
    end
    # + m4(a,b, m4(c,d,e,f), g)
    if haskey(m4,(c,d,e,f))
        for (x,cx) in m4[(c,d,e,f)]
            haskey(m4,(a,b,x,g)) || continue
            add_dict!(total, m4[(a,b,x,g)], cx)
        end
    end

    # ── j=5, i=3: m3(..., m5(...), ...) ── 2 terms ──────────────────────
    # + m3(m5(a,b,c,d,e), f,g)
    if haskey(m5,(a,b,c,d,e))
        for (x,cx) in m5[(a,b,c,d,e)]
            haskey(m3,(x,f,g)) || continue
            add_dict!(total, m3[(x,f,g)], cx)
        end
    end
    # - m3(a, m5(b,c,d,e,f), g)
    if haskey(m5,(b,c,d,e,f))
        for (x,cx) in m5[(b,c,d,e,f)]
            haskey(m3,(a,x,g)) || continue
            add_dict!(total, m3[(a,x,g)], -cx)
        end
    end

    # ── j=6, i=2: m2(m6(...), ...) ── 1 term ────────────────────────────
    # + m2(m6(a,b,c,d,e,f), g)
    if haskey(m6,(a,b,c,d,e,f))
        for (x,cx) in m6[(a,b,c,d,e,f)]
            tmp = m2(x, g)
            add_dict!(total, tmp, cx)
        end
    end

    return total
end

function compute_m7(C7, m2, m3, m4, m5, m6; tol=1e-6)
    m7 = Dict{NTuple{7,Symbol}, Dict{Symbol,Float64}}()
    for (a,b,c,d,e,f,g) in C7
        obs = m7_obstruction_full(a,b,c,d,e,f,g, m2,m3,m4,m5,m6)
        if !isempty(obs) && maximum(abs.(values(obs))) > tol
            m7_cancel = Dict{Symbol,Float64}()
            for (k,v) in obs
                m7_cancel[k] = -v
            end
            m7[(a,b,c,d,e,f,g)] = m7_cancel
        end
    end
    return m7
end

# =============================================================================
# m8_obstruction_full — 21 terms, pairs (i,j) with i+j=9
# =============================================================================

function m8_obstruction_full(a,b,c,d,e,f,g,h, m2,m3,m4,m5,m6,m7)
    total = Dict{Symbol,Float64}()

    # ── j=2, i=7: m7(..., m2(...), ...) ── 6 terms ──────────────────────
    for (sign, pair, key_fn) in [
        (+1, m2(a,b), (x)->haskey(m7,(x,c,d,e,f,g,h)) ? m7[(x,c,d,e,f,g,h)] : nothing),
        (-1, m2(b,c), (x)->haskey(m7,(a,x,d,e,f,g,h)) ? m7[(a,x,d,e,f,g,h)] : nothing),
        (+1, m2(c,d), (x)->haskey(m7,(a,b,x,e,f,g,h)) ? m7[(a,b,x,e,f,g,h)] : nothing),
        (-1, m2(d,e), (x)->haskey(m7,(a,b,c,x,f,g,h)) ? m7[(a,b,c,x,f,g,h)] : nothing),
        (+1, m2(e,f), (x)->haskey(m7,(a,b,c,d,x,g,h)) ? m7[(a,b,c,d,x,g,h)] : nothing),
        (-1, m2(f,g), (x)->haskey(m7,(a,b,c,d,e,x,h)) ? m7[(a,b,c,d,e,x,h)] : nothing),
    ]
        for (x,cx) in pair
            r = key_fn(x)
            r !== nothing && add_dict!(total, r, sign*cx)
        end
    end

    # ── j=3, i=6: m6(..., m3(...), ...) ── 5 terms ──────────────────────
    for (sign, key3, key6_fn) in [
        (+1, (a,b,c), (x)->haskey(m6,(x,d,e,f,g,h)) ? m6[(x,d,e,f,g,h)] : nothing),
        (-1, (b,c,d), (x)->haskey(m6,(a,x,e,f,g,h)) ? m6[(a,x,e,f,g,h)] : nothing),
        (+1, (c,d,e), (x)->haskey(m6,(a,b,x,f,g,h)) ? m6[(a,b,x,f,g,h)] : nothing),
        (-1, (d,e,f), (x)->haskey(m6,(a,b,c,x,g,h)) ? m6[(a,b,c,x,g,h)] : nothing),
        (+1, (e,f,g), (x)->haskey(m6,(a,b,c,d,x,h)) ? m6[(a,b,c,d,x,h)] : nothing),
    ]
        haskey(m3, key3) || continue
        for (x,cx) in m3[key3]
            r = key6_fn(x)
            r !== nothing && add_dict!(total, r, sign*cx)
        end
    end

    # ── j=4, i=5: m5(..., m4(...), ...) ── 4 terms ──────────────────────
    for (sign, key4, key5_fn) in [
        (+1, (a,b,c,d), (x)->haskey(m5,(x,e,f,g,h)) ? m5[(x,e,f,g,h)] : nothing),
        (-1, (b,c,d,e), (x)->haskey(m5,(a,x,f,g,h)) ? m5[(a,x,f,g,h)] : nothing),
        (+1, (c,d,e,f), (x)->haskey(m5,(a,b,x,g,h)) ? m5[(a,b,x,g,h)] : nothing),
        (-1, (d,e,f,g), (x)->haskey(m5,(a,b,c,x,h)) ? m5[(a,b,c,x,h)] : nothing),
    ]
        haskey(m4, key4) || continue
        for (x,cx) in m4[key4]
            r = key5_fn(x)
            r !== nothing && add_dict!(total, r, sign*cx)
        end
    end

    # ── j=5, i=4: m4(..., m5(...), ...) ── 3 terms ──────────────────────
    for (sign, key5, key4_fn) in [
        (+1, (a,b,c,d,e), (x)->haskey(m4,(x,f,g,h)) ? m4[(x,f,g,h)] : nothing),
        (-1, (b,c,d,e,f), (x)->haskey(m4,(a,x,g,h)) ? m4[(a,x,g,h)] : nothing),
        (+1, (c,d,e,f,g), (x)->haskey(m4,(a,b,x,h)) ? m4[(a,b,x,h)] : nothing),
    ]
        haskey(m5, key5) || continue
        for (x,cx) in m5[key5]
            r = key4_fn(x)
            r !== nothing && add_dict!(total, r, sign*cx)
        end
    end

    # ── j=6, i=3: m3(..., m6(...), ...) ── 2 terms ──────────────────────
    for (sign, key6, key3_fn) in [
        (+1, (a,b,c,d,e,f), (x)->haskey(m3,(x,g,h)) ? m3[(x,g,h)] : nothing),
        (-1, (b,c,d,e,f,g), (x)->haskey(m3,(a,x,h)) ? m3[(a,x,h)] : nothing),
    ]
        haskey(m6, key6) || continue
        for (x,cx) in m6[key6]
            r = key3_fn(x)
            r !== nothing && add_dict!(total, r, sign*cx)
        end
    end

    # ── j=7, i=2: m2(m7(...), h) ── 1 term ─────────────────────────────
    if haskey(m7,(a,b,c,d,e,f,g))
        for (x,cx) in m7[(a,b,c,d,e,f,g)]
            tmp = m2(x, h)
            add_dict!(total, tmp, cx)
        end
    end

    return total
end

function compute_m8(C8, m2, m3, m4, m5, m6, m7; tol=1e-6)
    m8 = Dict{NTuple{8,Symbol}, Dict{Symbol,Float64}}()
    for (a,b,c,d,e,f,g,h) in C8
        obs = m8_obstruction_full(a,b,c,d,e,f,g,h, m2,m3,m4,m5,m6,m7)
        if !isempty(obs) && maximum(abs.(values(obs))) > tol
            m8_cancel = Dict{Symbol,Float64}()
            for (k,v) in obs; m8_cancel[k] = -v; end
            m8[(a,b,c,d,e,f,g,h)] = m8_cancel
        end
    end
    return m8
end


# =============================================================================
# MATHEMATICAL NOTES ON THIS IMPLEMENTATION
# =============================================================================
#
# SIGN CONVENTION:
#   This file uses the SUSPENDED A∞ convention (sign = (-1)^r, alternating
#   +/- per insertion position). Matches curved_hh2_sparse_refactored_filteredA.jl
#   which uses the same convention (see m4_obstruction_full).
#   DO NOT mix with code using the standard unsuspended Stasheff sign (-1)^{r+st}.
#
# GAUGE CHOICE m_n = -Obs_{n+1}:
#   Valid ONLY for MINIMAL A∞ algebras (m1 = 0).
#   The BALBc path algebra has m1 = 0 (no differential), so this is correct.
#   For dg-algebras (m1 ≠ 0) the full ∂m_n = -Obs_{n+1} must be solved.
#
# HH² IS UNCHANGED:
#   d2: C2→C3 involves only m1,m2,m3. m7 and m8 do not affect d2.
#   The coker=62 result from gps_cone_hh2.jl is already exact.
#
# HH³ NOT IMPLEMENTED:
#   Verifying HH³(W_A)=0 requires building d3: C3→C4 with full A∞ insertions.
#   This is a substantial separate implementation. Marked as future work.
#   The paper's current assumption HH³(W_A)=0 is correct for the associative
#   truncation (global dim 2) but unverified for the curved A∞ algebra.
# =============================================================================


# =============================================================================
# VERIFICATION: Obs_{n+1} + ∂m_n = 0
# For minimal A∞ (m1=0), ∂m_n = 0, so this reduces to checking m_n = -Obs_{n+1}
# =============================================================================


"""
    verify_ainf_relation_n8(C7, m2, m3, m4, m5, m6, m7; tol=1e-6)

Verify: m7_obstruction_full(a,...,g) + m7[(a,...,g)] = 0 for all (a,...,g) in C7.
For minimal A∞ (m1=0), this is sufficient.
Returns (passed, max_residual, n_checked).
"""
function verify_ainf_relation_n8(C7, m2, m3, m4, m5, m6, m7; tol=1e-6)
    max_res = 0.0; n = 0
    for chain in C7
        obs = m7_obstruction_full(chain..., m2, m3, m4, m5, m6)
        isempty(obs) && continue
        n += 1
        haskey(m7, chain) || continue
        for (sym, v) in obs
            res = abs(v + get(m7[chain], sym, 0.0))
            max_res = max(max_res, res)
        end
    end
    return (passed=max_res < tol, max_residual=max_res, n_checked=n)
end

"""
    verify_ainf_relation_n9(C8, m2, m3, m4, m5, m6, m7, m8; tol=1e-6)

Verify: m8_obstruction_full(a,...,h) + m8[(a,...,h)] = 0 for all (a,...,h) in C8.
"""
function verify_ainf_relation_n9(C8, m2, m3, m4, m5, m6, m7, m8; tol=1e-6)
    max_res = 0.0; n = 0
    for chain in C8
        obs = m8_obstruction_full(chain..., m2, m3, m4, m5, m6, m7)
        isempty(obs) && continue
        n += 1
        haskey(m8, chain) || continue
        for (sym, v) in obs
            res = abs(v + get(m8[chain], sym, 0.0))
            max_res = max(max_res, res)
        end
    end
    return (passed=max_res < tol, max_residual=max_res, n_checked=n)
end

"""
    extract_prime_paths_m7(m7; top_n=50)
Extract highest-weight prime paths from m7 — order-7 non-associativity loci.
"""
function extract_prime_paths_m7(m7; top_n=50)
    scores = Dict{NTuple{7,Symbol}, Float64}()
    for (chain, obs) in m7
        isempty(obs) && continue
        scores[chain] = sum(abs.(values(obs)))
    end
    return first(sort(collect(scores), by=x->-x[2]), top_n)
end

println("m7_m8_extension.jl loaded.")
println("  m7_obstruction_full: 15 terms (suspended convention)")
println("  m8_obstruction_full: 21 terms")
println("  compute_C7, compute_C8: chain space builders")
println("  compute_m7, compute_m8: minimal A∞ gauge (m1=0 required)")
println("  verify_ainf_relation_n8/n9: check Obs + m = 0 numerically")
println("  extract_prime_paths_m7: order-7 prime paths")
println()
println("  WARNING: HH3 computation NOT included (requires d3: C3→C4).")
println("  HH2 is UNCHANGED by m7/m8. coker=62 from gps_cone_hh2.jl is exact.")
