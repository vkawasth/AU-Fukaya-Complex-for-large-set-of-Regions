function observe_feedback!(ctx, station_idx, product_idx, feedback_value)
    # This is PROJECTION of feedback onto L_i ⊗ L_j basis
    for i in 1:D, j in i:D
        tensor_weight = sqrt(ei * ej)  # projection coefficient
        ctx.feedback[key] = (1-α)*old + α*feedback_value*tensor_weight
    end
end

function feedback_signal(ctx, station_idx, product_idx)
    # This is the ADJOINT projection: ℝ^(D×D) → ℝ
    score = Σ_{i≤j} fb_ij × embed(p)[i] × embed(p)[j]
    return score
end