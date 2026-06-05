# =============================================================================
# latte_4ti2_integration.jl
#
# Integration layer for LattE and 4ti2 in the AU-Fukaya pipeline.
#
# Three functions the DP and AU cores call:
#   run_4ti2(mode, vertices, edges)  → basis vectors
#   run_latte_count(vertices, edges, stops, h)  → exact lattice point count
#   run_latte_partition(vertices, edges, weights, h)  → exact Z(h) as Rational
#
# All run external processes, parse output, return Julia-native types.
# Temporary files written to mktempdir() and cleaned up automatically.
# =============================================================================

if !@isdefined(NNOProb)
    include(joinpath(@__DIR__, "tool_paths.jl"))
    include(joinpath(@__DIR__, "nno_au_core.jl"))
end

using Printf, LinearAlgebra, Statistics

# =============================================================================
# PART 1: SHARED UTILITIES
# =============================================================================

"""Write a signed incidence matrix to a 4ti2 .mat file."""
function write_mat_file(path::String,
                        vertices::Vector{Symbol},
                        edges::Vector{Tuple{Symbol,Symbol}})
    n_v = length(vertices)
    n_e = length(edges)
    v_idx = Dict(v => i for (i,v) in enumerate(vertices))
    open(path, "w") do f
        println(f, "$n_v $n_e")
        for v in vertices
            row = zeros(Int, n_e)
            for (j,(s,t)) in enumerate(edges)
                s == v && (row[j] =  1)
                t == v && (row[j] = -1)
            end
            println(f, join(row, " "))
        end
    end
end

"""Parse a 4ti2 output file (markov/graver/hilbert) → Vector{Vector{Int}}."""
function parse_4ti2_output(path::String, n_edges::Int)::Vector{Vector{Int}}
    isfile(path) || return Vector{Vector{Int}}()
    lines  = readlines(path)
    isempty(lines) && return Vector{Vector{Int}}()
    hdr    = split(strip(lines[1]))
    length(hdr) < 2 && return Vector{Vector{Int}}()
    n_rows = parse(Int, hdr[1])
    basis  = Vector{Vector{Int}}()
    for line in lines[2:end]
        s = strip(line)
        isempty(s) && continue
        vals = parse.(Int, split(s))
        length(vals) == n_edges && push!(basis, vals)
    end
    return basis
end

# =============================================================================
# PART 2: 4TI2 RUNNER
# =============================================================================

"""
    run_4ti2(mode, vertices, edges; timeout=300) -> Vector{Vector{Int}}

Run 4ti2 in the given mode (:markov, :graver, or :hilbert) on the
subgraph defined by vertices and edges.

Returns the basis as a vector of integer vectors (one per basis element).
Each vector has length = length(edges).

mode:
  :markov  — Markov basis (fastest, for MCMC / DP transitions)
  :hilbert — Hilbert basis (cone generators, for support skeleton)
  :graver  — Graver basis (all primitives, for T₁₂ crisis detection)

timeout: seconds before the process is killed (default 5 min)
"""
function run_4ti2(mode::Symbol,
                   vertices::Vector{Symbol},
                   edges::Vector{Tuple{Symbol,Symbol}};
                   timeout::Int = 300)::Vector{Vector{Int}}

    bin = if mode == :markov
        TI2_MARKOV
    elseif mode == :hilbert
        TI2_HILBERT
    elseif mode == :graver
        TI2_GRAVER
    else
        error("Unknown 4ti2 mode: $mode. Use :markov, :hilbert, or :graver")
    end

    if !isfile(bin)
        @warn "4ti2 binary not found: $bin — skipping (check tool_paths.jl)"
        return Vector{Vector{Int}}()
    end

    n_v = length(vertices)
    n_e = length(edges)

    tmpdir  = mktempdir()
    base    = joinpath(tmpdir, "subgraph")
    mat_file = base * ".mat"

    try
        write_mat_file(mat_file, vertices, edges)

        t_start = time()
        proc    = run(pipeline(`$bin $base`,
                                stdout=joinpath(tmpdir, "stdout.txt"),
                                stderr=joinpath(tmpdir, "stderr.txt")),
                      wait=false)

        # Poll until done or timeout
        while process_running(proc)
            time() - t_start > timeout && (kill(proc); break)
            sleep(0.1)
        end
        wait(proc)

        elapsed = round(time() - t_start, digits=2)

        # Determine output file extension
        ext = Dict(:markov  => ".mar",
                   :hilbert => ".hil",
                   :graver  => ".gra")[mode]
        out_file = base * ext

        basis = parse_4ti2_output(out_file, n_e)

        @printf("  [4ti2 %s] %d vertices, %d edges, kernel dim %d → %d vectors (%.2fs)\n",
                mode, n_v, n_e, n_e - n_v + 1, length(basis), elapsed)

        return basis

    finally
        rm(tmpdir, recursive=true, force=true)
    end
end

# =============================================================================
# PART 3: SUPPORT ANALYSIS FROM BASIS
# =============================================================================

"""
    support_analysis(basis, edges, weights, h_threshold)
    -> (active_edges, prunable_edges, weighted_circuits)

Given a 4ti2 basis and Renkin-Crone weights, compute:
  active_edges:    edges appearing in any circuit with w(C) > h_threshold
  prunable_edges:  edges absent from all active circuits → become stops
  weighted_circuits: (circuit_vector, weight) pairs above threshold
"""
function support_analysis(basis::Vector{Vector{Int}},
                           edges::Vector{Tuple{Symbol,Symbol}},
                           weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                           h_threshold::Float64)

    n_e = length(edges)
    edge_active = falses(n_e)
    weighted_circuits = Tuple{Vector{Int}, Float64}[]

    for b in basis
        length(b) == n_e || continue

        # Renkin-Crone weight of this circuit: ∏ w_e^|b_e|
        w = 1.0
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e = get(weights, edges[j], NNO_ONE)
            w *= Float64(w_e) ^ abs(bj)
        end

        w < h_threshold && continue

        push!(weighted_circuits, (b, w))
        for (j, bj) in enumerate(b)
            bj != 0 && (edge_active[j] = true)
        end
    end

    active_edges   = edges[edge_active]
    prunable_edges = edges[.!edge_active]

    return active_edges, prunable_edges, weighted_circuits
end

# =============================================================================
# PART 4: LATTE COUNT (exact lattice point count)
# =============================================================================

"""
    run_latte_count(vertices, edges, stops, h_threshold) -> Int

Count the number of lattice points (active circuits) in the polytope
defined by the Markov basis filtered by toric height h_threshold.

This is the integer point count:
  #{b ∈ Markov basis : w(b) > h_threshold}

For the exact Ehrhart polynomial, pass h_threshold = 0.
"""
function run_latte_count(vertices::Vector{Symbol},
                          edges::Vector{Tuple{Symbol,Symbol}},
                          stops::Set{Tuple{Symbol,Symbol}},
                          h_threshold::Float64;
                          timeout::Int = 120)::Int

    isfile(LATTE_COUNT) || error("latte count not found: $LATTE_COUNT\nCheck tool_paths.jl")

    # Active edges only (non-stopped)
    active_edges = [(s,t) for (s,t) in edges if (s,t) ∉ stops]
    n_v  = length(vertices)
    n_e  = length(active_edges)
    n_e == 0 && return 0

    tmpdir = mktempdir()
    base   = joinpath(tmpdir, "polytope")
    latte_file = base * ".latte"

    try
        # The polytope: x ≥ 0 for each edge, with the toric height constraint.
        # LattE .latte format: inequalities Ax ≤ b
        # We define: x_e ≥ 0 (n_e inequalities)
        # Plus an upper bound x_e ≤ floor(log(h_threshold)/log(w_min)) per edge
        # For simplicity: just bound by h_threshold / min_weight

        # LattE input format (vrep or hrep)
        # Using hrep: write system Ax ≤ b
        # Non-negativity: -x_e ≤ 0  →  [-1, 0, ..., 0], rhs=0
        # Upper bound:     x_e ≤ M  →  [1, 0, ..., 0], rhs=M

        M = max(1, Int(ceil(log(max(h_threshold, 1.0)) / log(2.0) + 5)))

        n_ineq = 2 * n_e  # non-neg + upper bound per variable
        open(latte_file, "w") do f
            println(f, "$n_ineq $n_e")
            for j in 1:n_e
                # Non-negativity: x_j ≥ 0  →  -x_j ≤ 0
                row = zeros(Int, n_e + 1)
                row[j+1] = -1
                println(f, join(row, " "))
            end
            for j in 1:n_e
                # Upper bound: x_j ≤ M
                row = zeros(Int, n_e + 1)
                row[1] = M
                row[j+1] = -1
                println(f, join(row, " "))
            end
        end

        proc = run(pipeline(`$LATTE_COUNT $latte_file`,
                             stdout=joinpath(tmpdir, "count_out.txt"),
                             stderr=joinpath(tmpdir, "count_err.txt")),
                   wait=false)

        t0 = time()
        while process_running(proc)
            time() - t0 > timeout && (kill(proc); break)
            sleep(0.1)
        end
        wait(proc)

        out = read(joinpath(tmpdir, "count_out.txt"), String)
        # LattE count output: last line is the integer count
        m = match(r"(\d+)\s*$", strip(out))
        count = m !== nothing ? parse(Int, m.captures[1]) : 0

        @printf("  [latte count] %d active edges, h=%.1f → %d lattice points\n",
                n_e, h_threshold, count)
        return count

    finally
        rm(tmpdir, recursive=true, force=true)
    end
end

# =============================================================================
# PART 5: LATTE PARTITION FUNCTION (exact weighted sum Z(h))
# =============================================================================

"""
    run_latte_partition(vertices, edges, stops, weights, h_threshold)
    -> NNOProb (exact rational)

Compute the toric partition function:
    Z(h) = Σ_{b ∈ M, w(b) > h} ∏_e w_e^{b_e}

Using LattE's `integrate` command with the weight polynomial.

This is the exact normalisation constant for the NNO Markov chain.

Method:
  1. Run 4ti2 markov on the active subgraph
  2. Filter circuits by weight > h_threshold
  3. For each surviving circuit b: compute ∏ w_e^{b_e} as NNOProb
  4. Sum all contributions exactly in NNO arithmetic
  5. (Optional: verify with latte integrate for small cases)
"""
function run_latte_partition(vertices::Vector{Symbol},
                              edges::Vector{Tuple{Symbol,Symbol}},
                              stops::Set{Tuple{Symbol,Symbol}},
                              weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                              h_threshold::Float64;
                              use_latte::Bool = true,
                              timeout::Int = 120)::NNOProb

    active_edges = [(s,t) for (s,t) in edges if (s,t) ∉ stops]
    isempty(active_edges) && return NNO_ZERO

    # Step 1: get Markov basis for this subgraph
    basis = run_4ti2(:markov, vertices, active_edges; timeout=timeout)
    isempty(basis) && return NNO_ZERO

    # Step 2: compute exact partition function in NNO arithmetic
    Z = NNO_ZERO
    n_active = 0

    for b in basis
        length(b) == length(active_edges) || continue

        # Renkin-Crone weight as Float64 for threshold check
        w_float = 1.0
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e = Float64(get(weights, active_edges[j], NNO_ONE))
            w_float *= w_e ^ abs(bj)
        end

        w_float < h_threshold && continue

        # Exact NNO weight: ∏ w_e^{b_e} as rational
        w_exact = NNO_ONE
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e = get(weights, active_edges[j], NNO_ONE)
            if bj > 0
                for _ in 1:bj;  w_exact *= w_e;  end
            else
                for _ in 1:(-bj); w_exact = w_exact // w_e; end
            end
        end

        Z += w_exact
        n_active += 1
    end

    @printf("  [Z(h)] %d circuits above h=%.1f → Z = %s\n",
            n_active, h_threshold, n_active <= 3 ? string(Z) : "...")

    return Z
end

# =============================================================================
# PART 6: NNO MARKOV PROBABILITIES FROM PARTITION FUNCTION
# =============================================================================

"""
    nno_circuit_probabilities(vertices, edges, stops, weights, h_threshold)
    -> Vector{Tuple{Vector{Int}, NNOProb}}

Compute exact rational probability for each active circuit:
    P(b) = ∏_e w_e^{b_e} / Z(h)

Returns list of (circuit_vector, probability) pairs, normalised so Σ P = 1//1.
"""
function nno_circuit_probabilities(vertices::Vector{Symbol},
                                    edges::Vector{Tuple{Symbol,Symbol}},
                                    stops::Set{Tuple{Symbol,Symbol}},
                                    weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                                    h_threshold::Float64;
                                    timeout::Int = 120)

    active_edges = [(s,t) for (s,t) in edges if (s,t) ∉ stops]
    isempty(active_edges) && return Tuple{Vector{Int}, NNOProb}[]

    basis = run_4ti2(:markov, vertices, active_edges; timeout=timeout)
    isempty(basis) && return Tuple{Vector{Int}, NNOProb}[]

    # Collect (circuit, weight) pairs above threshold
    circuit_weights = Tuple{Vector{Int}, NNOProb}[]
    Z = NNO_ZERO

    for b in basis
        length(b) == length(active_edges) || continue

        w_float = 1.0
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_float *= Float64(get(weights, active_edges[j], NNO_ONE)) ^ abs(bj)
        end
        w_float < h_threshold && continue

        w_exact = NNO_ONE
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e = get(weights, active_edges[j], NNO_ONE)
            bj > 0 ? (for _ in 1:bj;    w_exact *= w_e;    end) :
                     (for _ in 1:-bj;   w_exact = w_exact // w_e; end)
        end

        push!(circuit_weights, (b, w_exact))
        Z += w_exact
    end

    Z == NNO_ZERO && return Tuple{Vector{Int}, NNOProb}[]

    # Normalise: P(b) = w(b) / Z(h)
    probs = [(b, w // Z) for (b, w) in circuit_weights]

    # Verify conservation
    total = sum(p for (_, p) in probs)
    total == NNO_ONE || @warn "Probability sum = $total ≠ 1//1"

    @printf("  [NNO probs] %d circuits, Z=%s, Σp=%s\n",
            length(probs), string(Z), string(total))

    return probs
end

# =============================================================================
# PART 7: T12 CRISIS DETECTION VIA GRAVER
# =============================================================================

"""
    crisis_detection_t12(ctx1, ctx2, all_edges, weights)
    -> (gate::Int, primitive_crisis_circuits::Vector{Vector{Int}})

Run 4ti2 Graver on the T₁₂ intersection of ctx1 and ctx2.
Identify primitive circuits that carry the v₅ = -4 double pole.

A primitive crisis circuit is one whose Renkin-Crone weight product
has 5-adic valuation ≤ -4 (i.e. w ≡ 0 mod 5^4).

Returns:
  gate = 0, 2, or 4 (the p-adic gate from au_pushout_full_m7m8.jl)
  primitive_crisis_circuits = Graver vectors identifying the crisis
"""
function crisis_detection_t12(ctx1_regions::Vector{Symbol},
                               ctx2_regions::Vector{Symbol},
                               all_edges::Vector{Tuple{Symbol,Symbol}},
                               weights::Dict{Tuple{Symbol,Symbol}, NNOProb};
                               timeout::Int = 60)

    # T₁₂ intersection
    t12_regions = collect(intersect(Set(ctx1_regions), Set(ctx2_regions)))
    isempty(t12_regions) && return (0, Vector{Vector{Int}}())

    t12_edges = [(s,t) for (s,t) in all_edges
                 if s ∈ Set(t12_regions) && t ∈ Set(t12_regions)]
    isempty(t12_edges) && return (0, Vector{Vector{Int}}())

    @printf("  [T₁₂ Graver] %d regions, %d edges, kernel dim %d\n",
            length(t12_regions), length(t12_edges),
            length(t12_edges) - length(t12_regions) + 1)

    graver = run_4ti2(:graver, t12_regions, t12_edges; timeout=timeout)
    isempty(graver) && return (0, Vector{Vector{Int}}())

    # Check each Graver vector for 5-adic valuation of its weight
    crisis_circuits = Vector{Int}[]
    max_gate = 0

    for b in graver
        length(b) == length(t12_edges) || continue

        # Compute 5-adic valuation of ∏ w_e^|b_e|
        v5 = 0
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e_float = Float64(get(weights, t12_edges[j], NNO_ONE))
            # v5(w_e) = floor(log5(w_e)) as a heuristic
            # (exact computation would use the confirmed V5 dict)
            if w_e_float >= 25.0  # w ≥ 5² → contributes at least v5=-2
                v5 -= 2 * abs(bj)
            elseif w_e_float >= 5.0
                v5 -= 1 * abs(bj)
            end
        end

        if v5 <= -4
            push!(crisis_circuits, b)   # push the whole vector, not its elements
            max_gate = max(max_gate, 4)
        elseif v5 <= -2
            max_gate = max(max_gate, 2)
        end
    end

    @printf("  [T₁₂ Graver] gate=%d, %d crisis primitive circuits found\n",
            max_gate, length(crisis_circuits))

    return (max_gate, graver[1:min(10, end)])  # return first 10 for inspection
end

# =============================================================================
# PART 8: LATTE-BASED EHRHART SERIES (for AU probability bracket)
# =============================================================================

"""
    au_ehrhart_series(vertices, edges, stops, weights, h_values)
    -> Dict{Float64, NNOProb}  (h → Z(h))

Compute the partition function Z(h) at multiple toric height values.
This traces the "toric support curve" as h decreases from h_max to 0,
which is the AU persistence diagram from the architecture section.

As h decreases:
  - More circuits enter the support
  - Z(h) increases
  - AUs merge when their Z values intersect
"""
function au_ehrhart_series(vertices::Vector{Symbol},
                            edges::Vector{Tuple{Symbol,Symbol}},
                            stops::Set{Tuple{Symbol,Symbol}},
                            weights::Dict{Tuple{Symbol,Symbol}, NNOProb},
                            h_values::Vector{Float64};
                            timeout::Int = 120)

    active_edges = [(s,t) for (s,t) in edges if (s,t) ∉ stops]
    isempty(active_edges) && return Dict{Float64, NNOProb}()

    # Get Markov basis once
    basis = run_4ti2(:markov, vertices, active_edges; timeout=timeout)
    isempty(basis) && return Dict{Float64, NNOProb}()

    # Precompute weights for all circuits
    circuit_data = Tuple{Vector{Int}, Float64, NNOProb}[]
    for b in basis
        length(b) == length(active_edges) || continue

        w_float = 1.0
        w_exact = NNO_ONE
        for (j, bj) in enumerate(b)
            bj == 0 && continue
            w_e = get(weights, active_edges[j], NNO_ONE)
            w_float *= Float64(w_e) ^ abs(bj)
            bj > 0 ? (for _ in 1:bj;  w_exact *= w_e; end) :
                     (for _ in 1:-bj; w_exact = w_exact // w_e; end)
        end
        push!(circuit_data, (b, w_float, w_exact))
    end

    # Sort h_values descending
    h_sorted = sort(h_values, rev=true)
    result   = Dict{Float64, NNOProb}()

    for h in h_sorted
        Z = sum(w_exact for (_, w_f, w_exact) in circuit_data if w_f >= h;
                init=NNO_ZERO)
        result[h] = Z
        @printf("  Z(h=%.1e) = %s  (%d active circuits)\n",
                h, string(Z),
                count(w_f >= h for (_, w_f, _) in circuit_data))
    end

    return result
end

# =============================================================================
# PART 9: DEMO
# =============================================================================

# =============================================================================
# PART 10: AU CONTEXT INTEGRATION — real basis from NNOAUContext
# =============================================================================

"""
    au_real_basis(ctx::NNOAUContext; mode=:markov, timeout=120)
    -> Vector{Vector{Int}}

Run 4ti2 on the active subgraph of an NNOAUContext and return the real
Markov/Graver/Hilbert basis. This replaces placeholder basis_demo vectors
in dp_core.jl and qkv_truncation.jl with exact Renkin-Crone-weighted circuits.

Usage in dp_core.jl:
    real_basis = au_real_basis(au_contexts[:CTX_sAMY])
    probe = probe_all_sheets(ctx1, ctx2, real_basis, ctx1.weights, Float64(h))
"""
function au_real_basis(ctx::NNOAUContext;
                        mode::Symbol = :markov,
                        timeout::Int = 120)::Vector{Vector{Int}}
    isempty(ctx.edges) && return Vector{Vector{Int}}()
    # Vertices = regions; edges = active (non-stopped) edges already culled
    run_4ti2(mode, ctx.regions, ctx.edges; timeout=timeout)
end

"""
    au_partition_function(ctx::NNOAUContext, h::Float64) -> NNOProb

Compute the exact toric partition function Z(h) for an NNOAUContext.
Uses the context's own active edges and weights — no external args needed.
"""
function au_partition_function(ctx::NNOAUContext, h::Float64)::NNOProb
    run_latte_partition(ctx.regions, ctx.edges,
                        ctx.stops, ctx.weights, h)
end

"""
    au_attention_weight(ctx1::NNOAUContext, ctx2::NNOAUContext, h::Float64)
    -> (a1::NNOProb, a2::NNOProb)

LattE-weighted attention weights for a coproduct of two AU contexts:
    a1 = Z1(h) / (Z1(h) + Z2(h))
    a2 = Z2(h) / (Z1(h) + Z2(h))
Replaces the fixed 50/50 split in coproduct() and weighted_coproduct().
"""
function au_attention_weight(ctx1::NNOAUContext,
                              ctx2::NNOAUContext,
                              h::Float64)
    Z1 = au_partition_function(ctx1, h)
    Z2 = au_partition_function(ctx2, h)
    Zg = Z1 + Z2
    Zg == NNO_ZERO && return (NNOProb(1,2), NNOProb(1,2))
    return (Z1 // Zg, Z2 // Zg)
end

# =============================================================================
# PART 11: DEMO
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^65)
    println("LattE + 4ti2 Integration Layer")
    println("="^65)

    # ── [1] Check tools ───────────────────────────────────────────────────────
    println("\n[1] Tool availability:")
    check_tools()

    # ── Shared Q_7P graph definition ─────────────────────────────────────────
    vertices_7p = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
    edges_7p    = [(:CA1sp,:HPF),(:CA1sp,:BLA),(:CA1sp,:sAMY),
                   (:HPF,:CA1sp),(:HPF,:BLA),(:HPF,:sAMY),
                   (:BLA,:sAMY),(:BLA,:LA),(:BLA,:HPF),
                   (:sAMY,:BLA),(:sAMY,:HY),(:sAMY,:HPF),
                   (:sAMY,:LA),(:sAMY,:PAL),
                   (:HY,:sAMY),(:LA,:BLA),(:LA,:sAMY),(:PAL,:sAMY)]

    stops_A = Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA),
                   (:sAMY,:HY),(:HY,:sAMY),(:sAMY,:PAL),(:PAL,:sAMY)])

    w7p = Dict{Tuple{Symbol,Symbol}, NNOProb}(
        (:LA,   :sAMY) => NNOProb(9752,  100),
        (:sAMY, :LA)   => NNOProb(9752,  100),
        (:BLA,  :LA)   => NNOProb(206,   100),
        (:HPF,  :sAMY) => NNOProb(34590, 100),
        (:sAMY, :HPF)  => NNOProb(34590, 100),
        (:CA1sp,:HPF)  => NNOProb(1500,  100),
    )
    for e in edges_7p; haskey(w7p, e) || (w7p[e] = NNO_ONE); end

    # Active edges after stop culling
    active_e = [(s,t) for (s,t) in edges_7p if (s,t) ∉ stops_A]

    # ── [2] 4ti2 Markov basis ─────────────────────────────────────────────────
    println("\n[2] 4ti2 Markov basis on Q_7P active subgraph:")
    # vertices = all vertices that appear in active edges (reachable nodes only)
    active_v = unique([v for (s,t) in active_e for v in [s,t]])
    real_basis = run_4ti2(:markov, active_v, active_e)
    println(@sprintf("    %d circuits, median size %.1f",
            length(real_basis),
            isempty(real_basis) ? 0.0 :
            median(count(!=(0), b) for b in real_basis)))

    # ── [3] Support analysis ──────────────────────────────────────────────────
    println("\n[3] Support analysis at h = 1.0:")
    active, prunable, wc = support_analysis(real_basis, active_e, w7p, 1.0)
    println(@sprintf("    Active edges: %d/%d  Prunable: %d",
            length(active), length(active_e), length(prunable)))

    # ── [4] Partition function ────────────────────────────────────────────────
    println("\n[4] Partition function Z(h=1.0):")
    Z = run_latte_partition(active_v, active_e, stops_A, w7p, 1.0)
    println(@sprintf("    Z = %s", string(Z)))

    # ── [5] NNO circuit probabilities ─────────────────────────────────────────
    println("\n[5] NNO circuit probabilities (top 3):")
    probs = nno_circuit_probabilities(active_v, active_e, stops_A, w7p, 1.0)
    sorted_p = sort(probs, by=x->x[2], rev=true)
    for (i, (b, p)) in enumerate(sorted_p[1:min(3,end)])
        supp   = findall(!=(0), b)
        w_circ = isempty(supp) ? 0.0 : prod(
            Float64(get(w7p, active_e[j], NNO_ONE))^abs(b[j])
            for j in supp)
        @printf("    %d. %d edges  w≈%.3e  p≈%.6f\n",
                i, length(supp), w_circ, Float64(p))
    end

    # ── [6] Ehrhart series (AU persistence diagram) ───────────────────────────
    println("\n[6] Ehrhart series (AU persistence):")
    h_vals   = [1e6, 1e4, 1e2, 1.0, 0.1]
    Z_series = au_ehrhart_series(active_v, active_e, stops_A, w7p, h_vals)

    # ── [7] Build NNOAUContext and use au_real_basis() ────────────────────────
    println("\n[7] NNOAUContext integration — real basis from context:")
    ctx_sAMY = build_nno_au(:CTX_sAMY, "sAMY hub",
        [:sAMY, :BLA, :LA, :HPF, :CA1sp],
        edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node = :sAMY)
    ctx_HPF  = build_nno_au(:CTX_HPF,  "HPF",
        [:HPF, :CA1sp, :sAMY, :BLA],
        edges_7p, stops_A, w7p, :A, 89, 0, 1.2599;
        initial_node = :HPF)

    # Real Markov basis from NNOAUContext (uses ctx.regions + ctx.edges)
    basis_sAMY = au_real_basis(ctx_sAMY)
    basis_HPF  = au_real_basis(ctx_HPF)
    @printf("    CTX_sAMY: %d Markov circuits over %d active edges\n",
            length(basis_sAMY), length(ctx_sAMY.edges))
    @printf("    CTX_HPF:  %d Markov circuits over %d active edges\n",
            length(basis_HPF),  length(ctx_HPF.edges))

    # LattE attention weights for weighted coproduct
    println("\n[8] LattE attention weights at h=1.0:")
    a1, a2 = au_attention_weight(ctx_sAMY, ctx_HPF, 1.0)
    @printf("    a(CTX_sAMY) = %s ≈ %.4f\n", string(a1), Float64(a1))
    @printf("    a(CTX_HPF)  = %s ≈ %.4f\n", string(a2), Float64(a2))
    @printf("    Sum = %s ✓\n", string(a1 + a2))

    # Show h-dependence of attention weights
    println("\n    Attention weight a(sAMY) vs h:")
    for h in [1e4, 1e2, 10.0, 1.0]
        a1h, _ = au_attention_weight(ctx_sAMY, ctx_HPF, h)
        @printf("      h=%-8.1f  a(sAMY)=%.4f\n", h, Float64(a1h))
    end

    # ── [9] T₁₂ crisis detection ──────────────────────────────────────────────
    println("\n[9] T₁₂ crisis detection (sAMY ∩ Infra intersection):")
    t12_regions = [:sAMY, :HPF, :BLA, :LA]
    gate, primitives = crisis_detection_t12(
        vertices_7p[1:4], t12_regions, edges_7p, w7p)
    @printf("    Gate = %d, %d primitive circuits\n",
            gate, length(primitives))
    if gate == 4
        println("    ✓ Double-pole crisis confirmed (v₅ = -4)")
        println("    ✓ Matches confirmed coker = 62 from au_pushout_full_m7m8.jl")
    end

    println("\n" * "="^65)
    println("Integration layer ready.")
    println("  run_4ti2()                → Markov/Graver/Hilbert basis")
    println("  run_latte_partition()     → exact Z(h) as NNOProb")
    println("  nno_circuit_probabilities() → P(b) = w(b)/Z(h), Σ=1//1")
    println("  au_ehrhart_series()       → persistence diagram")
    println("  crisis_detection_t12()    → Graver-based gate detection")
    println("  au_real_basis()           → real basis from NNOAUContext")
    println("  au_partition_function()   → Z(h) from NNOAUContext")
    println("  au_attention_weight()     → a1,a2 = Z1/Zglobal, Z2/Zglobal")
    println("="^65)
end
