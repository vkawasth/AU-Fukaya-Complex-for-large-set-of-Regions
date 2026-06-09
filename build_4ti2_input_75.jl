# build_4ti2_input_75.jl
# Generates the incidence matrix of the full 75-node BALBc connectome for 4ti2
# Run: julia build_4ti2_input_75.jl
# Output: balbc_75.mat  (ready for: hilbert balbc_75)

using CSV, DataFrames, SparseArrays

orig_dir = dirname(abspath(@__FILE__))

# Load the full connectome directly from CSV (bypassing the 7-node model)
nodes_file = joinpath(orig_dir, "node_regions_clean.csv")
edges_file = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"

isfile(nodes_file) || error("Cannot find $nodes_file")
isfile(edges_file) || error("Cannot find $edges_file")

nodes_df = CSV.read(nodes_file, DataFrame)
edges_df = CSV.read(edges_file, DataFrame)

println("Nodes: $(nrow(nodes_df))")
println("Edges: $(nrow(edges_df))")

# Build node index
node_ids = sort(unique(nodes_df[!, 1]))  # first column = node id
node_idx = Dict(id => i for (i,id) in enumerate(node_ids))
n_nodes  = length(node_ids)

# Identify edge columns: typically source, target (and weight)
# Check column names
println("Edge columns: ", names(edges_df))

# Standard Allen connectome format: source_id, target_id, ...
src_col = names(edges_df)[1]
tgt_col = names(edges_df)[2]

# Filter to edges where both endpoints are in our node set
valid = [haskey(node_idx, r[src_col]) && haskey(node_idx, r[tgt_col])
         for r in eachrow(edges_df)]
edges_filt = edges_df[valid, :]
println("Valid edges (both endpoints in node set): $(nrow(edges_filt))")

# Deduplicate directed edges
edge_pairs = unique([(r[src_col], r[tgt_col]) for r in eachrow(edges_filt)])
n_edges = length(edge_pairs)
println("Unique directed edges: $n_edges")

# Build incidence matrix: rows=nodes, cols=edges
# A[src,e] = +1, A[tgt,e] = -1
A = zeros(Int, n_nodes, n_edges)
for (j, (s, t)) in enumerate(edge_pairs)
    A[node_idx[s], j] += 1
    A[node_idx[t], j] -= 1
end

# Write .mat file
outfile = joinpath(orig_dir, "balbc_75.mat")
open(outfile, "w") do io
    println(io, "$n_nodes $n_edges")
    for i in 1:n_nodes
        println(io, join(A[i,:], " "))
    end
end

println("\nWritten: $outfile")
println("Matrix: $n_nodes × $n_edges")
println()
println("Run:")
println("  hilbert balbc_75        # Hilbert basis (may take minutes)")
println("  graver  balbc_75        # Graver basis (primitive circuits)")
println()
println("Then in Julia:")
println("  lines = readlines(\"balbc_75.hil\")")
println("  dims  = parse.(Int, split(lines[1]))")
println("  H     = [parse.(Int, split(l)) for l in lines[2:end] if !isempty(strip(l))]")
println("  H_mat = reduce(vcat, [h' for h in H])")
println("  zero_cols = findall(c -> all(H_mat[:,c] .== 0), 1:dims[2])")
println("  println(\"Prunable: \", length(zero_cols), \" of \", dims[2])")
