# build_4ti2_input.jl
# Generates the incidence matrix of the BALBc quiver for 4ti2 Hilbert basis
# Run: julia build_4ti2_input.jl
# Output: balbc_q7p.mat  (ready for: 4ti2int32 hilbert balbc_q7p)

let
    orig_dir  = dirname(abspath(@__FILE__))
    orig_file = joinpath(orig_dir, "curved_hh2_sparse_refactored_filteredA.jl")
    src = read(orig_file, String)
    src = replace(src, r"@__DIR__(?!\w)" => repr(orig_dir))
    marker = "if length(ARGS) >= 1 && ARGS[1] == \"--ainf-only\""
    pos = findfirst(marker, src)
    defs = pos !== nothing ? src[1:pos[1]-1] : src
    tmp = tempname() * ".jl"; write(tmp, defs); include(tmp); rm(tmp, force=true)
end
println("Loaded. nodes=$(length(nodes))")

# Collect all arrows from basis (all idempotents + path arrows)
# Filter to just the f_ arrows (genuine quiver arrows, not idempotents)
all_arrows = sort([b for b in build_basis(nodes, parse_relations(relations_str))
                   if startswith(string(b), "f_")])

println("Arrows: $(length(all_arrows))")
println("Nodes:  $(length(nodes))")

# Build incidence matrix: rows=nodes, cols=arrows
# Convention: +1 at src(arrow), -1 at tgt(arrow)
node_list = sort(collect(nodes))
node_idx  = Dict(n => i for (i,n) in enumerate(node_list))

nrows = length(node_list)
ncols = length(all_arrows)
A = zeros(Int, nrows, ncols)

for (j, arr) in enumerate(all_arrows)
    s = src(arr); t = tgt(arr)
    A[node_idx[s], j] += 1
    A[node_idx[t], j] -= 1
end

# Write .mat file for 4ti2
outfile = joinpath(dirname(abspath(@__FILE__)), "balbc_q7p.mat")
open(outfile, "w") do io
    println(io, "$nrows $ncols")
    for i in 1:nrows
        println(io, join(A[i,:], " "))
    end
end

println("Written: $outfile")
println("Matrix: $nrows rows (nodes) × $ncols cols (arrows)")
println()
println("Run: 4ti2int32 hilbert balbc_q7p")
println("Then: cat balbc_q7p.hil")
