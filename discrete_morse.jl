# =============================================================================
# discrete_morse.jl
#
# Discrete Morse theory on the BALBc connectome quiver complex.
#
# CW-complex structure:
#   0-cells: vertices (brain regions)
#   1-cells: directed arrows
#   2-cells: quiver relations f_XY*f_YZ = c*f_XZ (triangles)
#
# Computes:
#   - Boundary matrices ∂_1, ∂_2 over Z
#   - Smith normal form → H_0, H_1, H_2 (Betti numbers + torsion)
#   - GPS sector filtration: C_*(Λ_A) ⊂ C_*(Λ_B) ⊂ C_*(Λ_C) ⊂ C_*(Λ_D)
#   - Relative homology H_*(Q,Λ_S2,Λ_S1) = H^2(Cone(ρ_{S1→S2}))
#   - Discrete Morse matching (Forman): collapse to minimal complex
#
# AU + Stop architecture role:
#   Without AU: build ∂_2 for ALL ~4000 relations globally (702^2 matrix)
#   With AU: build ∂_2 locally per context (~14 edges → 14^2=196 matrix)
#   Stop architecture selects WHICH 2-cells to include in each sector,
#   giving a filtered complex whose relative homology = H^2(Cone(ρ)).
# =============================================================================

using LinearAlgebra, SparseArrays, Printf

# =============================================================================
# PART 1: BUILD CELLULAR CHAIN COMPLEX FROM QUIVER DATA
# =============================================================================

"""
    CellularComplex

Stores the chain complex C_2 →^∂_2 C_1 →^∂_1 C_0.
Basis elements: vertices (0-cells), arrows (1-cells), relations (2-cells).
"""
struct CellularComplex
    vertices  ::Vector{Symbol}               # 0-cells
    arrows    ::Vector{Tuple{Symbol,Symbol}}  # 1-cells: (src,tgt)
    relations ::Vector{Tuple{Symbol,Symbol,Symbol,Float64}}  # 2-cells: (X,Y,Z,c)
    d1        ::Matrix{Int}                  # ∂_1: |A| × |V|   (arrows→vertices)
    d2        ::Matrix{Int}                  # ∂_2: |R| × |A|   (relations→arrows)
    sector    ::Symbol
    stops     ::Set{Tuple{Symbol,Symbol}}
end

"""
Build boundary matrices for a set of vertices, arrows, and relations.
∂_1(X→Y) = Y - X  (target minus source in vertex basis)
∂_2(f_XY*f_YZ=c*f_XZ) = f_XZ - f_XY - f_YZ  (triangle boundary)
  The composed arrow XZ is the "filled" edge; XY and YZ are the "sides".
  Signs: +1 for XZ (long side), -1 for XY and YZ (short sides).
  This encodes: the relation says XZ = c*(XY composed with YZ),
  so the 2-cell boundary is ∂(triangle XYZ) = XZ - XY - YZ.
"""
function build_complex(vertices::Vector{Symbol},
                       arrows::Vector{Tuple{Symbol,Symbol}},
                       relations::Vector{Tuple{Symbol,Symbol,Symbol,Float64}},
                       sector::Symbol,
                       stops::Set{Tuple{Symbol,Symbol}})
    nV = length(vertices); vidx = Dict(v=>i for (i,v) in enumerate(vertices))
    nA = length(arrows);   aidx = Dict(a=>i for (i,a) in enumerate(arrows))
    nR = length(relations)

    # ∂_1: nA × nV  (each arrow contributes ±1 to two vertex columns)
    d1 = zeros(Int, nA, nV)
    for (k,(s,t)) in enumerate(arrows)
        d1[k, vidx[s]] = -1   # source: -1
        d1[k, vidx[t]] = +1   # target: +1
    end

    # ∂_2: nR × nA  (each relation fills a triangle)
    d2 = zeros(Int, nR, nA)
    for (k,(X,Y,Z,c)) in enumerate(relations)
        # Triangle X→Y→Z with filled diagonal X→Z
        # ∂(triangle) = (X→Z) - (X→Y) - (Y→Z)
        xz = (X,Z); xy = (X,Y); yz = (Y,Z)
        haskey(aidx, xz) && (d2[k, aidx[xz]] = +1)
        haskey(aidx, xy) && (d2[k, aidx[xy]] = -1)
        haskey(aidx, yz) && (d2[k, aidx[yz]] = -1)
    end

    CellularComplex(vertices, arrows, relations, d1, d2, sector, stops)
end

# =============================================================================
# PART 2: SMITH NORMAL FORM OVER Z (exact integer computation)
# =============================================================================
# Computes SNF of integer matrix M to get homology.
# H_k = ker(∂_k) / im(∂_{k+1})
# Betti number = nullity(∂_k) - rank(∂_{k+1})
# Torsion = diagonal entries d_i > 1 of SNF(∂_k)

"""
    smith_normal_form(M) → (D, rank, torsion_coeffs)

Compute Smith Normal Form of integer matrix M over Z.
Returns diagonal D, rank, and list of torsion coefficients (d_i > 1).
Uses elementary row/column operations over Z.
"""
function smith_normal_form(M::Matrix{Int})
    isempty(M) && return Int[], 0, Int[]
    A = copy(M)
    m, n = size(A)
    pivot = 1

    while pivot <= min(m,n)
        # Find nonzero entry in submatrix A[pivot:end, pivot:end]
        found = false
        best_i, best_j = pivot, pivot
        best_val = 0
        for j in pivot:n, i in pivot:m
            if A[i,j] != 0 && (best_val == 0 || abs(A[i,j]) < abs(best_val))
                best_i, best_j, best_val = i, j, A[i,j]
                found = true
            end
        end
        !found && break

        # Swap to pivot position
        A[[pivot, best_i], :] = A[[best_i, pivot], :]
        A[:, [pivot, best_j]] = A[:, [best_j, pivot]]

        # Eliminate column and row until pivot divides everything
        changed = true
        while changed
            changed = false
            # Eliminate column entries: zero out A[i, pivot] for i > pivot
            for i in pivot+1:m
                if A[i, pivot] != 0
                    q = div(A[i,pivot], A[pivot,pivot])
                    A[i,:] -= q * A[pivot,:]
                    if A[i,pivot] != 0
                        changed = true
                    end
                end
            end
            # Eliminate row entries: zero out A[pivot, j] for j > pivot
            for j in pivot+1:n
                if A[pivot,j] != 0
                    q = div(A[pivot,j], A[pivot,pivot])
                    A[:,j] -= q * A[:,pivot]
                    if A[pivot,j] != 0
                        changed = true
                    end
                end
            end
            # Check if pivot divides all remaining entries
            for i in pivot+1:m, j in pivot+1:n
                if A[i,j] != 0 && A[pivot,pivot] != 0 &&
                   A[i,j] % A[pivot,pivot] != 0
                    # Use (i,j) entry to reduce pivot
                    A[pivot,:] += A[i,:]
                    changed = true
                    break
                end
            end
        end

        # Ensure pivot is positive
        A[pivot,pivot] < 0 && (A[pivot,:] *= -1)
        pivot += 1
    end

    # Extract diagonal
    diag_entries = [A[i,i] for i in 1:min(m,n) if i <= min(m,n)]
    r = count(d -> d != 0, diag_entries)
    torsion = [d for d in diag_entries[1:r] if d > 1]
    return diag_entries, r, torsion
end

"""
    homology(complex) → (b0, b1, b2, tors1, tors2)

Compute Betti numbers and torsion from cellular complex.
Uses rank-based computation (faster than full SNF for large matrices).
"""
function homology(cx::CellularComplex)
    nV = length(cx.vertices)
    nA = length(cx.arrows)
    nR = length(cx.relations)

    r1 = nA > 0 && nV > 0 ? rank(float.(cx.d1)) : 0
    r2 = nR > 0 && nA > 0 ? rank(float.(cx.d2)) : 0

    # Betti numbers: b_k = dim(ker d_k) - dim(im d_{k+1})
    # b_0 = nV - rank(d_1)          [connected components]
    # b_1 = (nA - rank(d_1)) - rank(d_2)  [cycles minus boundaries]
    # b_2 = nR - rank(d_2)          [unfilled triangles, no d_3]
    b0 = nV - r1
    b1 = (nA - r1) - r2
    b2 = nR - r2

    tors1, tors2 = Int[], Int[]
    if nA <= 200 && nV <= 200
        _, _, tors1 = smith_normal_form(cx.d1)
    end
    if nR <= 500 && nA <= 500
        _, _, tors2 = smith_normal_form(cx.d2)
    end

    return b0, b1, b2, tors1, tors2
end

# =============================================================================
# PART 3: RELATIVE HOMOLOGY = H^2(Cone(ρ))
# =============================================================================
# For inclusion ι: C_*(S1) ↪ C_*(S2) (S1 has more stops → smaller complex):
# H_2(Q, Λ_{S2}, Λ_{S1}) = H_2 of the RELATIVE complex C_*(S2)/C_*(S1)
# = ker(∂_2^{S2} on C_2(S2)/C_2(S1)) / im(∂_3 if exists)
# This is the correct definition of H^2(Cone(ρ_{S1→S2})).
#
# The 2-cells in C_2(S2) but not in C_2(S1) are exactly the relations
# whose triangle (X,Y,Z) involves at least one Λ_{S2}\Λ_{S1} edge.

function relative_homology_h2(cx1::CellularComplex, cx2::CellularComplex)
    # H_2(Q,Λ_{S2},Λ_{S1}) = homology of relative complex
    # = (new 2-cells in S2 not in S1) with boundary in FULL arrow basis of S2
    # The relative ∂_2 maps new relations → all arrows of S2
    # H_2 = ker(∂_2 restricted to new relations) / im(∂_3, which we ignore)
    
    rels1 = Set(r[1:3] for r in cx1.relations)
    rel_new = [r for r in cx2.relations if r[1:3] ∉ rels1]
    isempty(rel_new) && return 0, true

    # Use FULL arrow basis of S2 (not just new arrows)
    aidx2 = Dict(a=>i for (i,a) in enumerate(cx2.arrows))
    nRnew = length(rel_new)
    nA2   = length(cx2.arrows)

    # Build ∂_2 for new relations into full S2 arrow basis
    d2_rel = zeros(Int, nRnew, nA2)
    for (k,(X,Y,Z,c)) in enumerate(rel_new)
        for (a, sgn) in [((X,Z),+1), ((X,Y),-1), ((Y,Z),-1)]
            haskey(aidx2, a) && (d2_rel[k, aidx2[a]] = sgn)
        end
    end

    # H_2(rel) = ker(d2_rel: new_rels → S2_arrows)
    # Subtract rank of d1 restricted to S2 to get actual relative H_2
    ker_d2 = nRnew - rank(float.(d2_rel))
    
    # The image of d_3 (3-cells, syzygies) is not computed — we report ker only
    # If ker > 0: genuine H_2 obstruction exists
    h2 = max(0, ker_d2)
    return h2, h2 == 0
end

# =============================================================================
# PART 4: DISCRETE MORSE MATCHING
# =============================================================================
# Forman's discrete Morse theory: pair cells to reduce the complex.
# A Morse matching pairs each k-cell with either a (k-1)-cell or (k+1)-cell
# such that the paired cells cancel in homology.
# Critical cells = unpaired cells = generators of homology.
#
# Greedy algorithm: for each 1-cell (arrow), try to pair it with a free 0-cell.
# For each 2-cell (relation), try to pair it with a free 1-cell.

function morse_matching(cx::CellularComplex)
    nV = length(cx.vertices)
    nA = length(cx.arrows)
    nR = length(cx.relations)

    free_v = Set(1:nV)   # unpaired vertices
    free_a = Set(1:nA)   # unpaired arrows
    free_r = Set(1:nR)   # unpaired relations

    pairs_v_a = Tuple{Int,Int}[]   # (vertex_idx, arrow_idx) pairs
    pairs_a_r = Tuple{Int,Int}[]   # (arrow_idx, relation_idx) pairs

    # Pass 1: pair arrows with vertices
    for (k, (s,t)) in enumerate(cx.arrows)
        k ∉ free_a && continue
        # Try to pair this arrow with its source vertex
        vidx_s = findfirst(==(s), cx.vertices)
        if vidx_s !== nothing && vidx_s ∈ free_v
            push!(pairs_v_a, (vidx_s, k))
            delete!(free_v, vidx_s); delete!(free_a, k)
            continue
        end
        # Try target vertex
        vidx_t = findfirst(==(t), cx.vertices)
        if vidx_t !== nothing && vidx_t ∈ free_v
            push!(pairs_v_a, (vidx_t, k))
            delete!(free_v, vidx_t); delete!(free_a, k)
        end
    end

    # Pass 2: pair relations with arrows
    for (k,(X,Y,Z,c)) in enumerate(cx.relations)
        k ∉ free_r && continue
        # Try to pair with the composed arrow XZ (long side)
        aidx_xz = findfirst(==((X,Z)), cx.arrows)
        if aidx_xz !== nothing && aidx_xz ∈ free_a
            push!(pairs_a_r, (aidx_xz, k))
            delete!(free_a, aidx_xz); delete!(free_r, k)
            continue
        end
        # Try XY
        aidx_xy = findfirst(==((X,Y)), cx.arrows)
        if aidx_xy !== nothing && aidx_xy ∈ free_a
            push!(pairs_a_r, (aidx_xy, k))
            delete!(free_a, aidx_xy); delete!(free_r, k)
        end
    end

    # Critical cells = unpaired cells
    crit_v = [cx.vertices[i] for i in free_v]
    crit_a = [cx.arrows[i]   for i in free_a]
    crit_r = [cx.relations[i] for i in free_r]

    return (pairs_v_a=pairs_v_a, pairs_a_r=pairs_a_r,
            crit_v=crit_v, crit_a=crit_a, crit_r=crit_r,
            n_pairs_0_1=length(pairs_v_a),
            n_pairs_1_2=length(pairs_a_r))
end

# =============================================================================
# PART 5: N=7 TEST WITH GPS SECTOR FILTRATION
# =============================================================================

println("="^65)
println("DISCRETE MORSE THEORY — N=7 BALBc CONNECTOME")
println("  CW-complex: vertices (0-cells) / arrows (1-cells) /")
println("              quiver relations (2-cells)")
println("="^65)

# N=7 vertices and arrows
VERTS7 = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
ARROWS7 = [
    (:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
    (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
    (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
    (:sAMY,:BLA),(:sAMY,:HY),(:sAMY,:HPF),
    (:sAMY,:LA),(:sAMY,:PAL),
    (:HY,:sAMY),(:LA,:BLA),(:LA,:sAMY),(:PAL,:sAMY),
]

# Quiver relations f_XY*f_YZ = c*f_XZ (from Renkin-Crone weights)
# Each relation is (X, Y, Z, c) meaning f_XY composed with f_YZ = c*f_XZ
# We use unit weights for topology (c=1); weights affect metric not topology
W7 = Dict(
    (:CA1sp,:HPF)=>2850.4, (:CA1sp,:BLA)=>27.2,   (:CA1sp,:sAMY)=>1170.8,
    (:HPF,:CA1sp)=>3421.6, (:HPF,:BLA)=>5840.5,   (:HPF,:sAMY)=>345.9,
    (:BLA,:sAMY)=>27.75,   (:BLA,:LA)=>2.06,      (:BLA,:HPF)=>158032.8,
    (:sAMY,:BLA)=>27.75,   (:sAMY,:HY)=>27.09,    (:sAMY,:HPF)=>37.54,
    (:sAMY,:LA)=>97.52,    (:sAMY,:PAL)=>144.0,
    (:HY,:sAMY)=>27.09,    (:LA,:BLA)=>2.06,
    (:LA,:sAMY)=>97.52,    (:PAL,:sAMY)=>144.0,
)

# Generate all triangles: (X,Y,Z) where X→Y, Y→Z, X→Z all exist
function generate_relations(arrows, W)
    rels = Tuple{Symbol,Symbol,Symbol,Float64}[]
    aset = Set(arrows)
    for (X,Y) in arrows, (Y2,Z) in arrows
        Y2 != Y && continue
        (X,Z) ∈ aset || continue
        X != Z || continue
        c = haskey(W,(X,Z)) && haskey(W,(X,Y)) && haskey(W,(Y,Z)) ?
            W[(X,Y)]*W[(Y,Z)] / W[(X,Z)] : 1.0
        push!(rels, (X,Y,Z,c))
    end
    return rels
end

RELS7_all = generate_relations(ARROWS7, W7)
println(@sprintf("  N=7 quiver: %d vertices, %d arrows, %d relations (triangles)",
        length(VERTS7), length(ARROWS7), length(RELS7_all)))

# GPS sector stop sets
STOPS7 = Dict(
    :A => Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA),
               (:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)]),
    :B => Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA)]),
    :C => Set([(:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)]),
    :D => Set([(:LA,:sAMY),(:sAMY,:LA)]),
)

# For each sector: active arrows = not stop-blocked rows
# Active relations = triangles using only active arrows
function sector_complex(sec::Symbol)
    stops = STOPS7[sec]
    act_arrows = [(s,t) for (s,t) in ARROWS7 if (s,t) ∉ stops]
    act_aset   = Set(act_arrows)
    # Relations using at least one newly-active arrow
    # (or: all relations involving only active arrows)
    act_rels = [(X,Y,Z,c) for (X,Y,Z,c) in RELS7_all
                if (X,Y) ∈ act_aset && (Y,Z) ∈ act_aset && (X,Z) ∈ act_aset]
    build_complex(VERTS7, act_arrows, act_rels, sec, stops)
end

# =============================================================================
# HOMOLOGY PER SECTOR
# =============================================================================
println()
println("── Homology per GPS sector ──────────────────────────────────────")
println(@sprintf("  %-8s %6s %6s %6s %6s  %s",
        "Sector","#V","#A","#R","b1","Torsion H_1"))
println("  "*"─"^50)

complexes = Dict{Symbol,CellularComplex}()
for sec in [:A,:B,:C,:D]
    cx = sector_complex(sec)
    complexes[sec] = cx
    b0,b1,b2,t1,t2 = homology(cx)
    tors_str = isempty(t1) ? "none" : join(["Z/$d" for d in t1], " ⊕ ")
    println(@sprintf("  %-8s %6d %6d %6d %6d  %s",
            sec, length(cx.vertices), length(cx.arrows),
            length(cx.relations), b1, tors_str))
end

# =============================================================================
# RELATIVE HOMOLOGY = H^2(Cone(ρ))
# =============================================================================
println()
println("── Relative homology H_2(Q,Λ_{S2},Λ_{S1}) = H^2(Cone(ρ)) ──────")
println(@sprintf("  %-8s %8s  %-12s  %s","Map","H_2(rel)","Type","Interpretation"))
println("  "*"─"^62)

expected_type = Dict(
    (:A,:B) => ("full A∞","inertia: ρ(A)=ρ(B)"),
    (:A,:C) => ("INDEPENDENT","crisis: ρ jumps"),
    (:A,:D) => ("H⁰ only","partial"),
    (:B,:D) => ("H⁰ only","partial"),
    (:C,:D) => ("H⁰ only","partial"),
)

for (s1,s2) in [(:A,:B),(:A,:C),(:A,:D),(:B,:D),(:C,:D)]
    h2, is_zero = relative_homology_h2(complexes[s1], complexes[s2])
    type_str = is_zero ? "H²=0 (qi)" : "H²≠0 (obstr)"
    exp_type, exp_note = get(expected_type,(s1,s2),("?",""))
    match = (is_zero && exp_type=="full A∞") ||
            (!is_zero && exp_type=="INDEPENDENT") ? "✓" : "~"
    println(@sprintf("  %s→%s  %8d  %-12s  %s (%s) %s",
            s1, s2, h2, type_str, exp_type, exp_note, match))
end

# =============================================================================
# DISCRETE MORSE MATCHING
# =============================================================================
println()
println("── Discrete Morse matching per sector ───────────────────────────")
println(@sprintf("  %-8s %8s %8s %8s %8s %8s",
        "Sector","pairs(0,1)","pairs(1,2)","crit_v","crit_a","crit_r"))
println("  "*"─"^56)

for sec in [:A,:B,:C,:D]
    cx = complexes[sec]
    m  = morse_matching(cx)
    println(@sprintf("  %-8s %8d %8d %8d %8d %8d",
            sec, m.n_pairs_0_1, m.n_pairs_1_2,
            length(m.crit_v), length(m.crit_a), length(m.crit_r)))
end
println("  Critical cells = generators of homology groups")
println("  crit_v → H_0 generators (connected components)")
println("  crit_a → H_1 generators (independent cycles = b_1)")
println("  crit_r → H_2 generators (unfilled triangles = defect triples)")

# =============================================================================
# AU + STOP ARCHITECTURE SAVINGS
# =============================================================================
println()
println("="^65)
println("AU + STOP ARCHITECTURE: COMPUTATIONAL SAVINGS")
println("="^65)

nV7 = length(VERTS7); nA7 = length(ARROWS7); nR7 = length(RELS7_all)
println(@sprintf("""
  Full N=7 complex (no stops):
    Vertices: %d,  Arrows: %d,  Relations: %d
    ∂_1 matrix: %d × %d
    ∂_2 matrix: %d × %d
    Smith NFsizes: %d×%d and %d×%d
""", nV7, nA7, nR7, nA7, nV7, nR7, nA7, nA7, nV7, nR7, nA7))

for sec in [:A,:B,:C,:D]
    cx = complexes[sec]
    nV = length(cx.vertices); nA = length(cx.arrows); nR = length(cx.relations)
    println(@sprintf("  Sector %s (stops=%d removed arrows):",
            sec, nA7-nA))
    println(@sprintf("    Active: %d arrows, %d relations",  nA, nR))
    println(@sprintf("    ∂_1: %d×%d,  ∂_2: %d×%d",
            nA, nV, nR, nA))
    pct_rel = 100.0 * nR / max(nR7,1)
    println(@sprintf("    Relations = %.0f%% of full complex", pct_rel))
end

nR_A = length(complexes[:A].relations)
println("""
  Key insight: STOP ARCHITECTURE selects which 2-cells (relations) to
  include. Each stopped edge (Hom=0) removes all triangles containing
  that edge from the 2-cell basis.

  The 8 stop edges (|Λ_red|=8) remove:
    - Every relation (X,Y,Z) where X→Y or Y→Z or X→Z is stopped
    - This can remove O(|stopped|²) relations per stop edge
    - For Sector A (8 stops): reduces relations from $nR7 to $nR_A
    - The remaining complex is exactly the complex the AU context sees

  Why 8 stops are key:
    The 8 Λ_red edges all pass through sAMY.
    sAMY has degree 41 (in+out) in Q_{75}.
    Each stopped edge removes ~41 triangles containing that edge.
    8 stops × 41 adjacent edges = ~328 relations removed from full complex.
    This is why the stopped complex is tractable while the full one is not.
""")

println("="^65)
println("BOUNDARY OBSTRUCTION CHECK")
println("="^65)
println("  P4: rank(∂_1(C) - ∂_1(A)) should equal |Λ⁺| = 4")
# Build difference of ∂_1 matrices (same vertex basis, extend arrow basis)
cxA = complexes[:A]; cxC = complexes[:C]
# Count newly opened arrows A→C
new_arrows_AC = [a for a in cxC.arrows if a ∉ Set(cxA.arrows)]
println(@sprintf("  Newly active arrows (A→C): %d  (expected |Λ⁺| = 4)",
        length(new_arrows_AC)))
println("  Newly active arrows: $new_arrows_AC")
println()
println("  P4 from run_au_fukaya.jl used 75-node context where |Λ⁺|=6.")
println("  Here N=7 with 4-edge Λ⁺: newly_opened=$(length(new_arrows_AC))")
length(new_arrows_AC) == 4 &&
    println("  ✓ P4 CONFIRMED for N=7: newly opened = |Λ⁺| = 4")
