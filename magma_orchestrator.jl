# =============================================================================
# magma_orchestrator.jl
#
# Calls hh2_cone.magma and parses its output to assemble the long exact
# sequence and produce the crisis diagnostic.
#
# Two modes:
#   Mode 1 (standalone): julia magma_orchestrator.jl
#     Runs hh2_cone.magma, parses output, prints diagnostics.
#     Requires: magma in PATH, hh2_cone.magma in same directory.
#
#   Mode 2 (library): include("magma_orchestrator.jl")
#     Exposes run_crisis_diagnostic() for use in au_fukaya_75.jl.
#
# Architecture:
#   hh2_cone.magma  — exact Z_5 arithmetic, HH^k computation, Ext
#   magma_orchestrator.jl — calls magma, parses output, long exact sequence
#   au_fukaya_75.jl — AU poset logic, GPS filtration, context management
# =============================================================================

using Printf

# =============================================================================
# STEP 1: Run hh2_cone.magma and capture output
# =============================================================================

"""
    run_hh2_cone(magma_path, script_path) -> raw_output::String

Calls MAGMA on hh2_cone.magma and returns the full output as a string.
Returns empty string if MAGMA is not available.
"""
function run_hh2_cone(magma_path::String = "magma",
                       script_path::String = "hh2_cone.magma")
    isfile(script_path) || error("hh2_cone.magma not found at: $script_path")

    output_path = tempname() * ".out"
    try
        run(pipeline(
            `$(magma_path) -b $(script_path)`,
            stdout = output_path,
            stderr = devnull
        ), wait = true)
        result = read(output_path, String)
        rm(output_path, force=true)
        return result
    catch e
        rm(output_path, force=true)
        @warn "MAGMA call failed: $e"
        return ""
    end
end

# =============================================================================
# STEP 2: Parse structured output from hh2_cone.magma
# =============================================================================

"""
    parse_hh2_output(raw::String) -> Dict

Extracts key values from hh2_cone.magma output.
The MAGMA script prints tagged lines:
  v_5(w17)    = N
  |w17|_5     = N
  Structural gate (min perturbations needed): N
  Dynamical trigger ...: YES/NO
  HH^k(W_X) = dimension N
  Sector A: N relations
  etc.
"""
function parse_hh2_output(raw::String)
    result = Dict{String,Any}()
    isempty(raw) && return result

    for line in split(raw, '\n')
        line = strip(line)

        # p-adic valuation
        m = match(r"v_5\(w17\)\s*=\s*(-?\d+)", line)
        m !== nothing && (result["v5"] = parse(Int, m[1]))

        # p-adic norm
        m = match(r"\|w17\|_5\s*=\s*(\d+)", line)
        m !== nothing && (result["norm5"] = parse(Int, m[1]))

        # Structural gate
        m = match(r"Structural gate.*?:\s*(\d+)", line)
        m !== nothing && (result["gate_order"] = parse(Int, m[1]))

        # Dynamical trigger
        if occursin("Dynamical trigger", line)
            result["triggered"] = occursin("YES", line)
        end

        # HH^k dimensions: "HH^k(W_X) = dimension N"
        m = match(r"HH\^(\d+)\(W_([ABC])\)\s*=\s*dimension\s*(\d+)", line)
        if m !== nothing
            k = parse(Int, m[1]); sec = m[2]; dim = parse(Int, m[3])
            result["HH$(k)_$(sec)"] = dim
        end

        # Relation counts
        m = match(r"Sector ([ABC]):\s*(\d+)\s*relations", line)
        m !== nothing && (result["rels_$(m[1])"] = parse(Int, m[2]))

        # Crisis confirmation
        if occursin("CRISIS CONFIRMED", line)
            result["crisis"] = true
        end
        if occursin("H^2(Cone", line) && occursin("!= 0", line)
            result["h2_cone_nonzero"] = true
        end
    end

    # Defaults
    get!(result, "crisis", false)
    get!(result, "triggered", false)
    return result
end

# =============================================================================
# STEP 3: Long exact sequence assembly
# =============================================================================

"""
    long_exact_sequence(hh2_A, hh2_C, hh3_A, rank_rho_star) -> (h2_cone, is_crisis)

Assembles the long exact sequence:
  HH^2(W_A) -rho*-> HH^2(W_C) -> H^2(Cone(rho)) -> HH^3(W_A)

When HH^3(W_A) = 0 (global dimension 2 quiver):
  H^2(Cone) = coker(rho*) = dim(HH^2(W_C)) - rank(rho*)
"""
function long_exact_sequence(hh2_A::Int, hh2_C::Int,
                               hh3_A::Int, rank_rho_star::Int)
    coker = max(0, hh2_C - rank_rho_star)
    # If HH^3(W_A) != 0, the cokernel is an upper bound
    h2_cone = hh3_A == 0 ? coker : coker  # exact when hh3_A=0
    return h2_cone, h2_cone > 0
end

# =============================================================================
# STEP 4: Estimate rank(rho*) from sector data
# =============================================================================

"""
    estimate_rank_rho_star(hh2_source, newly_opened_count) -> Int

Conservative estimate of rank(rho*: HH^2(W_A) -> HH^2(W_C)).
Each newly opened arrow can introduce ~1 new cokernel class.
Exact value requires MAGMA to compute the explicit linear map.
"""
function estimate_rank_rho_star(hh2_source::Int, newly_opened::Int)
    # rank(rho*) <= hh2_source (can't have more rank than source dimension)
    # rank(rho*) >= hh2_source - newly_opened (each new arrow costs at most 1)
    max(0, hh2_source - newly_opened)
end

# =============================================================================
# STEP 5: Print full diagnostic report
# =============================================================================

function print_diagnostic(parsed::Dict, transition::String = "A->C")
    println("="^60)
    println("AU CRISIS DIAGNOSTIC: $transition")
    println("="^60)

    println("\n-- p-adic analysis --")
    haskey(parsed, "v5") &&
        println(@sprintf("  v_5(w_LA_sAMY)  = %d", parsed["v5"]))
    haskey(parsed, "norm5") &&
        println(@sprintf("  |w_LA_sAMY|_5   = %d", parsed["norm5"]))
    haskey(parsed, "gate_order") &&
        println(@sprintf("  Structural gate : %d simultaneous interventions",
                parsed["gate_order"]))
    haskey(parsed, "triggered") &&
        println(@sprintf("  Dynamical trigger: %s",
                parsed["triggered"] ? "YES - CRISIS ZONE" : "NO - SAFE"))

    println("\n-- Hochschild cohomology --")
    for sec in ["A","B","C"]
        for k in 0:3
            key = "HH$(k)_$(sec)"
            haskey(parsed, key) &&
                println(@sprintf("  HH^%d(W_%s) = %d", k, sec, parsed[key]))
        end
    end

    println("\n-- Long exact sequence: HH^2(W_A) -> HH^2(W_C) -> H^2(Cone) --")
    if haskey(parsed, "HH2_A") && haskey(parsed, "HH2_C")
        hh2_A = parsed["HH2_A"]; hh2_C = parsed["HH2_C"]
        hh3_A = get(parsed, "HH3_A", 0)
        # A->C opens 4 arrows (Lambda+)
        rank_est = estimate_rank_rho_star(hh2_A, 4)
        h2_cone, is_crisis = long_exact_sequence(hh2_A, hh2_C, hh3_A, rank_est)
        println(@sprintf("  HH^2(W_A) = %d", hh2_A))
        println(@sprintf("  HH^2(W_C) = %d", hh2_C))
        println(@sprintf("  rank(rho*) estimate = %d", rank_est))
        println(@sprintf("  H^2(Cone(rho_AC)) estimate = %d", h2_cone))
        println(@sprintf("  Crisis: %s", is_crisis ? "YES" : "NO"))
        println()
        println("  Note: exact rank(rho*) requires MAGMA to compute the")
        println("  linear map between HH^2 generators. The estimate uses")
        println("  rank(rho*) = dim(HH^2(W_A)) - |Lambda+| as lower bound.")
    end

    println("\n-- Final verdict --")
    if get(parsed, "crisis", false)
        println("  MAGMA CONFIRMS: H^2(Cone(rho_AC)) != 0 - CRISIS")
    elseif get(parsed, "triggered", false)
        println("  Dynamical trigger active - likely crisis (run MAGMA to confirm)")
    else
        println("  No crisis detected")
    end
end

# =============================================================================
# MAIN: Run if called directly
# =============================================================================

function run_crisis_diagnostic(; magma_bin="magma",
                                  script="hh2_cone.magma",
                                  verbose=true)
    verbose && println("Calling hh2_cone.magma via MAGMA...")
    raw = run_hh2_cone(magma_bin, script)

    if isempty(raw)
        println("MAGMA not available or script failed.")
        println("Running p-adic analysis in Julia (no HH^2 computation)...")
        # Fallback: just compute p-adic values directly
        W7 = Dict(7=>27.75, 8=>2.06, 10=>27.75, 13=>97.52, 16=>2.06, 17=>97.52)
        c1 = W7[7] * W7[13] / W7[8]   # BLA->sAMY->LA / BLA->LA
        c2 = W7[17] * W7[10] / W7[16] # LA->sAMY->BLA / LA->BLA

        function jl_pval(x::Float64, p::Int)
            r = rationalize(x, tol=1e-4)
            n, d = abs(numerator(r)), abs(denominator(r))
            vn = 0; while n > 0 && n % p == 0; n ÷= p; vn += 1; end
            vd = 0; while d > 0 && d % p == 0; d ÷= p; vd += 1; end
            vn - vd
        end

        p = 5
        v1 = jl_pval(c1, p); v2 = jl_pval(c2, p)
        n1 = p^(-v1); n2 = p^(-v2)
        println(@sprintf("\n  BLA->sAMY->LA composite: v_5=%d, |c|_5=%d", v1, n1))
        println(@sprintf("  LA->sAMY->BLA composite: v_5=%d, |c|_5=%d", v2, n2))
        min_v = min(v1,v2); max_n = max(n1,n2)
        println(@sprintf("  Structural gate: %d interventions needed", abs(min_v)))
        println(@sprintf("  Dynamical trigger |c|_5 > 25: %s",
                max_n > 25 ? "YES - CRISIS" : "NO"))
        return Dict("v5"=>min_v, "norm5"=>max_n,
                    "gate_order"=>abs(min_v), "triggered"=>max_n>25)
    end

    parsed = parse_hh2_output(raw)
    verbose && print_diagnostic(parsed)
    return parsed
end

# Run when called as script
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^60)
    println("MAGMA ORCHESTRATOR")
    println("  Step 1: Call hh2_cone.magma")
    println("  Step 2: Parse HH^k dimensions")
    println("  Step 3: Assemble long exact sequence")
    println("  Step 4: Compute H^2(Cone(rho))")
    println("="^60)
    println()

    # Look for hh2_cone.magma in same dir or current dir
    script_candidates = [
        joinpath(dirname(@__FILE__), "hh2_cone.magma"),
        "hh2_cone.magma",
        joinpath(homedir(), "Downloads/OGB/connectome/phaseTransition_phaseTransition_complex/FukayaAUComplex/hh2_cone.magma")
    ]
    script = first(filter(isfile, script_candidates), "hh2_cone.magma")

    result = run_crisis_diagnostic(script=script, verbose=true)
    println()
    println("Raw parsed values: $result")
end
