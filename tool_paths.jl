# =============================================================================
# tool_paths.jl
# =============================================================================
# Centralised tool path configuration for the AU-Fukaya pipeline.
# Include this file in any script that calls external tools.
# =============================================================================

using Printf

# ── External tool paths ───────────────────────────────────────────────────────

const LATTE_BIN    = "/Users/vaw1/latte/bin"
const LATTE_HB_BIN = "/Users/vaw1/Downloads/latte/homebrew/bin"  # markov lives here
const FOURTITI_BIN = "/Users/vaw1/4ti2/bin"
const R_BIN        = "/usr/local/bin/R"

# ── LattE binaries ────────────────────────────────────────────────────────────
const LATTE_COUNT        = joinpath(LATTE_BIN, "count")
const LATTE_INTEGRATE    = joinpath(LATTE_BIN, "integrate")
const LATTE_EHRHART      = joinpath(LATTE_BIN, "ehrhart")
const LATTE_MAXIMIZE     = joinpath(LATTE_BIN, "latte-maximize")
const LATTE_TRIANGULATE  = joinpath(LATTE_BIN, "triangulate")

# ── 4ti2 / markov binaries ────────────────────────────────────────────────────
# markov is bundled with LattE homebrew, not in the 4ti2 bin.
# graver and hilbert are in the 4ti2 bin.
const TI2_MARKOV  = joinpath(LATTE_HB_BIN, "markov")   # ← LattE homebrew
const TI2_GRAVER  = joinpath(FOURTITI_BIN, "graver")
const TI2_HILBERT = joinpath(FOURTITI_BIN, "hilbert")
const TI2_4TI2    = joinpath(FOURTITI_BIN, "4ti2")

# ── Validation ────────────────────────────────────────────────────────────────
function check_tools(; verbose=true)
    tools = [
        ("latte count",     LATTE_COUNT),
        ("latte integrate", LATTE_INTEGRATE),
        ("latte ehrhart",   LATTE_EHRHART),
        ("4ti2 markov",     TI2_MARKOV),
        ("4ti2 graver",     TI2_GRAVER),
        ("4ti2 hilbert",    TI2_HILBERT),
        ("R",               R_BIN),
    ]
    all_ok = true
    for (name, path) in tools
        ok = isfile(path)
        all_ok &= ok
        verbose && @printf("  %-22s %s  %s\n", name,
            ok ? "✓" : "✗", path)
    end
    return all_ok
end
