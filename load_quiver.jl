# =============================================================================
# load_quiver.jl
#
# Parses brain_complex_quiver_FIXED_ALL.txt (or .g) and extracts:
#   - Vertex list
#   - Directed arrow list
#   - Renkin-Crone path weights from relations f_X_Y*f_Y_Z - c*f_X_Z = 0
#
# Usage:
#   include("load_quiver.jl")
#   W, arrows = load_quiver("brain_complex_quiver_FIXED_ALL.txt")
#
# Returns:
#   W      ::Dict{Tuple{Symbol,Symbol}, Float64}
#            Direct edge weights: W[(X,Y)] = geometric mean of all
#            Renkin-Crone coefficients c where (X,*,Y) appears
#   arrows ::Vector{Tuple{Symbol,Symbol}}
#            All directed edges in the quiver
# =============================================================================

using Printf

"""
    parse_vertex_name(s) → Symbol

Clean a vertex name string to a valid Julia Symbol.
Handles names with spaces ("fiber tracts" → :fibertracts),
hyphens ("MY-mot" → :MYmot), etc.
"""
function parse_vertex_name(s::AbstractString)::Symbol
    s = strip(s)
    s = replace(s, " " => "")
    s = replace(s, "-" => "")
    s = replace(s, "." => "")
    Symbol(s)
end

"""
    parse_relation(line) → (X, Y, Z, c) or nothing

Parse a relation of the form:
  f_X_Y*f_Y_Z - c*f_X_Z    (= 0 implied)

Returns (X, Y, Z, c) as (Symbol, Symbol, Symbol, Float64)
or nothing if the line cannot be parsed.
"""
function parse_relation(line::AbstractString)
    line = strip(line)
    isempty(line) && return nothing

    # Split on " - " to separate LHS and RHS
    m = match(r"^f_(.+)\*f_(.+)\s*-\s*([0-9][0-9.eE+\-]*)\*f_(.+)$", line)
    m === nothing && return nothing

    xy_str, yz_str, c_str, xz_str = m.captures

    # Parse coefficient
    c = tryparse(Float64, c_str)
    c === nothing && return nothing

    # Each of xy_str, yz_str, xz_str is "X_Y" — split at the right underscore
    # Region names can contain underscores? No — they use hyphens and spaces.
    # Strategy: try all split points, pick the one where both parts are known vertices
    function split_edge(s::AbstractString, known_verts::Set{Symbol})
        # Try splitting at each underscore
        parts = split(s, "_")
        length(parts) < 2 && return nothing, nothing
        # Try each possible split point
        for k in 1:length(parts)-1
            v1 = parse_vertex_name(join(parts[1:k], "_"))
            v2 = parse_vertex_name(join(parts[k+1:end], "_"))
            if v1 ∈ known_verts && v2 ∈ known_verts
                return v1, v2
            end
        end
        return nothing, nothing
    end

    return xy_str, yz_str, xz_str, c
end

"""
    load_quiver(filename; verbose=true) → (W, arrows, vertices)

Load the 75-node quiver from a MAGMA/GAP .g or .txt file.

The file format expected:
  Q := Quiver(...);
  arrows := [..., "f_X_Y", ...];
  rels := [..., f_X_Y*f_Y_Z - c*f_X_Z, ...];

Returns:
  W        ::Dict{Tuple{Symbol,Symbol}, Float64}  edge weights
  arrows   ::Vector{Tuple{Symbol,Symbol}}          directed edges
  vertices ::Vector{Symbol}                        vertex list
"""
function load_quiver(filename::AbstractString; verbose::Bool=true)
    isfile(filename) || error("File not found: $filename")

    verbose && println("Loading quiver from: $filename")
    content = read(filename, String)
    lines   = split(content, r"[,\n]")  # split on comma OR newline

    # ── Pass 1: collect all vertex names from arrow declarations ─────────
    # Arrow names look like: "f_X_Y" (quoted strings in the arrows list)
    # or f_X_Y (unquoted in relations)
    vertex_set = Set{Symbol}()
    arrow_set  = Set{Tuple{Symbol,Symbol}}()

    # Extract quoted arrow names: "f_X_Y"
    for m in eachmatch(r"\"f_([^\"]+)_([^\"]+)\"", content)
        v1 = parse_vertex_name(m.captures[1])
        v2 = parse_vertex_name(m.captures[2])
        push!(vertex_set, v1, v2)
        push!(arrow_set, (v1, v2))
    end

    # Also extract from unquoted relation patterns: f_X_Y*f_Y_Z
    for m in eachmatch(r"\bf_([A-Za-z0-9 _\-]+?)_([A-Za-z0-9 _\-]+?)\b(?=[\*\-\s])", content)
        v1 = parse_vertex_name(m.captures[1])
        v2 = parse_vertex_name(m.captures[2])
        # Only add if non-empty
        isempty(string(v1)) || isempty(string(v2)) && continue
        push!(vertex_set, v1, v2)
        push!(arrow_set, (v1, v2))
    end

    verbose && println("  Vertices found: $(length(vertex_set))")
    verbose && println("  Arrows found:   $(length(arrow_set))")

    # ── Pass 2: parse relations ───────────────────────────────────────────
    # Relations: f_X_Y*f_Y_Z - c*f_X_Z = 0  (= 0 often implicit)
    weight_accum = Dict{Tuple{Symbol,Symbol}, Vector{Float64}}()

    n_relations = 0
    known = vertex_set  # use for splitting guidance

    for line in lines
        line = strip(line)
        isempty(line) && continue
        # Remove trailing semicolons, brackets
        line = replace(line, r"[;\[\]]" => "")
        line = strip(line)

        # Match: f_X_Y*f_Y_Z - c*f_X_Z
        m = match(r"f_(\S+?)\*f_(\S+?)\s*-\s*([0-9][0-9.eE+\-]*)\s*\*\s*f_(\S+)", line)
        m === nothing && continue

        xy_raw, yz_raw, c_str, xz_raw = m.captures
        c = tryparse(Float64, c_str)
        c === nothing && continue

        # Split each raw string into (vertex, vertex) pair
        function best_split(raw::AbstractString)
            parts = split(raw, "_")
            for k in 1:length(parts)-1
                v1 = parse_vertex_name(join(parts[1:k], "_"))
                v2 = parse_vertex_name(join(parts[k+1:end], "_"))
                if v1 ∈ known && v2 ∈ known
                    return v1, v2
                end
            end
            # Fallback: split at last underscore
            idx = findlast('_', raw)
            idx === nothing && return nothing, nothing
            v1 = parse_vertex_name(raw[1:idx-1])
            v2 = parse_vertex_name(raw[idx+1:end])
            return v1, v2
        end

        X, Y1 = best_split(xy_raw)
        Y2, Z  = best_split(yz_raw)
        X2, Z2 = best_split(xz_raw)

        (X === nothing || Y1 === nothing || Z === nothing) && continue
        (Y1 != Y2 || X != X2 || Z != Z2) && continue  # consistency check

        # Accumulate weight for direct edges (X→Y1) and (Y2→Z) and (X2→Z2)
        for edge in [(X,Y1), (Y2,Z), (X2,Z2)]
            push!(get!(weight_accum, edge, Float64[]), c)
        end
        push!(arrow_set, (X,Y1), (Y2,Z), (X2,Z2))
        push!(vertex_set, X, Y1, Y2, Z, X2, Z2)
        n_relations += 1
    end

    verbose && println("  Relations parsed: $n_relations")

    # ── Build weight dict: geometric mean of accumulated coefficients ─────
    W = Dict{Tuple{Symbol,Symbol}, Float64}()
    for (edge, cs) in weight_accum
        # Geometric mean of Renkin-Crone coefficients
        W[edge] = exp(mean(log.(cs .+ 1e-10)))
    end

    # Fill unit weights for arrows without relations
    for arrow in arrow_set
        haskey(W, arrow) || (W[arrow] = 1.0)
    end

    verbose && println("  Weighted edges:   $(length(W))")
    verbose && println("  Weight range:     [$(round(minimum(values(W)), sigdigits=4)), $(round(maximum(values(W)), sigdigits=4))]")

    # Sort vertices and arrows for stable ordering
    vertices = sort(collect(vertex_set), by=string)
    arrows   = sort(collect(arrow_set), by=x->string(x[1])*string(x[2]))

    return W, arrows, vertices
end

# =============================================================================
# Quick self-test if run directly
# =============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    filename = length(ARGS) > 0 ? ARGS[1] : "brain_complex_quiver_FIXED_ALL.txt"
    W, arrows, vertices = load_quiver(filename)
    println("\nTop 10 edges by weight:")
    sorted_edges = sort(collect(W), by=x->-x[2])
    for (e, w) in sorted_edges[1:min(10, end)]
        println(@sprintf("  %-20s → %-20s  w = %.4e", string(e[1]), string(e[2]), w))
    end
    println("\nEdges through sAMY:")
    samy_edges = filter(kv -> kv[1][1] == :sAMY || kv[1][2] == :sAMY, W)
    for (e, w) in sort(collect(samy_edges), by=x->-x[2])
        println(@sprintf("  %-20s → %-20s  w = %.4e", string(e[1]), string(e[2]), w))
    end
end
