function compile_instruction!(ctx::FkCompilerContext, inst::ProjectOntoBasis)
    source_expr = ctx.ssa_map[inst.source]
    result_var = fresh_temp(ctx, "proj")
    
    # Generate code that calls the embedding compiler
    expr = quote
        D = length($(inst.basis))
        $result_var = zeros(D)
        for (d, lag_label) in enumerate($(inst.basis))
            lag = get($(ctx.engine_var).lagrangians_by_label, lag_label, nothing)
            if lag !== nothing
                # Floer pairing ⟨source, L_i⟩
                $result_var[d] = floer_pairing($source_expr, lag, $(ctx.engine_var).omega)
            end
        end
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::ProjectToScalar)
    tensor_expr = ctx.ssa_map[inst.tensor]
    vector_expr = ctx.ssa_map[inst.vector]
    result_var = fresh_temp(ctx, "score")
    
    # Contract tensor with vector: score = vᵀ T v (for symmetric T)
    expr = quote
        D = length($vector_expr)
        $result_var = 0.0
        for i in 1:D, j in i:D
            T_ij = $tensor_expr[i,j]
            T_ij == 0.0 && continue
            $result_var += T_ij * $vector_expr[i] * $vector_expr[j]
        end
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::Pushforward)
    proj_expr = ctx.ssa_map[inst.projection]
    result_var = fresh_temp(ctx, "pushed")
    
    if inst.functor == :m1
        # Push projection through the Floer differential
        expr = quote
            # m₁ acts on the coefficients of the projection
            $result_var = m1_pushforward($proj_expr, $(ctx.engine_var).T)
        end
    elseif inst.functor == :hmm_backward
        # Push projection through the HMM backward process
        expr = quote
            $result_var = hmm_pushforward($proj_expr, $(ctx.engine_var).omega,
                                          $(ctx.engine_var).products)
        end
    else
        error("Unknown functor: $(inst.functor)")
    end
    
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end