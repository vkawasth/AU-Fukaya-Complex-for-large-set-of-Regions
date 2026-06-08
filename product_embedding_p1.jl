# From au_compiler.jl, Pass 1
function compile_product_embeddings(products, lagrangians, omega, month_range)
    # This is a PROJECTION: Δ(product) → ℝ^D
    # Maps the infinite-dimensional product object onto the D-dimensional 
    # Lagrangian basis {L_RM, L_RF, L_PM, L_PF}
    for (p_idx, prod) in enumerate(products)
        for (d_idx, lag) in enumerate(demo_lags)
            pairing = Σ_{s,h,m} lag.flow[s,h,m] × prod.affinity[d_idx] × ω[s,h,m]
            embeddings[p_idx, d_idx] = pairing
        end
    end
    return embeddings  # ℝ^(n_products × D)
end