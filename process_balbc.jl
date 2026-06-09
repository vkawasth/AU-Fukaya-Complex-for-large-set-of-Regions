# Parse Hilbert basis and find prunable arrows
lines = readlines("balbc_q7p.hil")
dims = parse.(Int, split(strip(lines[1])))
nrows, ncols = dims[1], dims[2]
H = zeros(Int, nrows, ncols)
for (i, line) in enumerate(lines[2:end])
    isempty(strip(line)) && continue
    H[i,:] = parse.(Int, split(strip(line)))
end
col_sums = sum(abs.(H), dims=1)[:]
zero_cols = findall(x -> x == 0, col_sums)
println("Prunable arrows (zero in all generators): ", zero_cols)
println("Active arrows: ", ncols - length(zero_cols), " of ", ncols)
