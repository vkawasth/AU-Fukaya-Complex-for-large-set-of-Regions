# =============================================================================
# fk_ir_to_julia.jl
#
# Lowers AU Fukaya Floer IR to Julia AST that calls au_fukaya_engine functions.
# =============================================================================

"""
    FkCompilerContext

Holds the state of compilation: SSA value map, current block, etc.
"""
mutable struct FkCompilerContext
    ssa_map::Dict{Symbol, Expr}      # IR value → Julia expression
    temp_counter::Int                # for generating fresh temporaries
    engine_var::Symbol               # variable name for AUFukayaEngine instance
end

FkCompilerContext(engine_var::Symbol) = FkCompilerContext(Dict(), 0, engine_var)

function fresh_temp(ctx::FkCompilerContext, base::String="tmp")
    ctx.temp_counter += 1
    return Symbol("$(base)_$(ctx.temp_counter)")
end

"""
    compile_instruction!(ctx, inst) -> Expr

Compile a single IR instruction to a Julia expression.
"""
function compile_instruction!(ctx::FkCompilerContext, inst::AllocLagrangian)
    # Call au_fukaya_engine's Lagrangian constructor
    # In au_fukaya_engine.jl, Lagrangians are defined via build_lagrangians
    # But for runtime allocation, we need to look up from the attic
    expr = quote
        # Look up from AU attic (unmaterialised coproduct)
        get($(ctx.engine_var).primitives, $(QuoteNode(inst.demo_vec)), nothing)
    end
    ctx.ssa_map[inst.result] = expr
    return nothing
end

function compile_instruction!(ctx::FkCompilerContext, inst::FloerIntersection)
    # Call floer_complex from fukaya_ad_context.jl
    Li_expr = ctx.ssa_map[inst.L_i]
    Lj_expr = ctx.ssa_map[inst.L_j]
    result_var = fresh_temp(ctx, "cf")
    expr = quote
        $result_var = floer_complex($Li_expr, $Lj_expr; threshold=$(inst.threshold))
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())  # reference to variable
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::M1Differential)
    # Call m1_differential from fukaya_ad_context.jl
    cf_expr = ctx.ssa_map[inst.floer_complex]
    result_var = fresh_temp(ctx, "m1")
    T_expr = inst.T  # precomputed transition matrix
    expr = quote
        $result_var = m1_differential($cf_expr, $T_expr)
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::M2Composition)
    # Call m2_composition from fukaya_ad_context.jl
    Li_expr = ctx.ssa_map[inst.L_i]
    Lj_expr = ctx.ssa_map[inst.L_j]
    result_var = fresh_temp(ctx, "m2")
    expr = quote
        $result_var = m2_composition($Li_expr, $Lj_expr, 
                                      $(inst.affinity), 
                                      $(inst.product_idx))
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::CoproductDelta)
    # Call coprod_delta from fukaya_ad_context.jl
    result_var = fresh_temp(ctx, "delta")
    expr = quote
        event = AdEvent($(inst.station), $(inst.hour), $(inst.month),
                        $(inst.product_idx), 
                        $(ctx.engine_var).omega[$(inst.station), $(inst.hour), $(inst.month)])
        $result_var = coprod_delta(event, $(inst.affinity), 
                                   $(ctx.engine_var).lagrangians)
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::NNOUnrolledLoop)
    # Generate unrolled loop — this is the KEY optimization
    # We generate value(n_steps) copies of the loop body
    n = value(inst.n_steps)
    
    # Compile the loop body once (it will be duplicated)
    body_exprs = Expr[]
    sub_ctx = FkCompilerContext(ctx.engine_var)
    # Pass through existing SSA values for variables defined outside the loop
    for (k,v) in ctx.ssa_map
        sub_ctx.ssa_map[k] = v
    end
    
    # State variable that carries through iterations
    state_var = inst.start
    for i in 1:n
        # Create fresh state for this iteration
        new_state = fresh_temp(sub_ctx, "state_$(i)")
        # Compile body with current state as input
        for body_inst in inst.body
            compile_instruction!(sub_ctx, body_inst)
        end
        # Body should produce new state
        push!(body_exprs, :($new_state = $(sub_ctx.ssa_map[inst.result])))
        state_var = new_state
    end
    
    ctx.ssa_map[inst.result] = state_var
    return Expr(:block, body_exprs...)
end

function compile_instruction!(ctx::FkCompilerContext, inst::LazyEval)
    # Wrap in a thunk for lazy evaluation
    source_expr = ctx.ssa_map[inst.source]
    result_var = fresh_temp(ctx, "lazy")
    expr = quote
        $result_var = () -> $source_expr
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::ServeAd)
    # Call serve_ad from au_compiler.jl (runtime path)
    result_var = fresh_temp(ctx, "served")
    expr = quote
        $result_var = serve_ad($(ctx.engine_var).runtime_ctx,
                               $(inst.station), $(inst.hour), $(inst.month);
                               stab_floor=$(inst.stab_floor))
    end
    ctx.ssa_map[inst.result] = Expr(:ref, result_var, :())
    return expr
end

function compile_instruction!(ctx::FkCompilerContext, inst::Branch)
    cond_expr = ctx.ssa_map[inst.cond]
    
    # Compile true branch
    true_exprs = Expr[]
    true_ctx = FkCompilerContext(ctx.engine_var)
    for (k,v) in ctx.ssa_map
        true_ctx.ssa_map[k] = v
    end
    for inst in inst.true_block
        push!(true_exprs, compile_instruction!(true_ctx, inst))
    end
    
    # Compile false branch
    false_exprs = Expr[]
    false_ctx = FkCompilerContext(ctx.engine_var)
    for (k,v) in ctx.ssa_map
        false_ctx.ssa_map[k] = v
    end
    for inst in inst.false_block
        push!(false_exprs, compile_instruction!(false_ctx, inst))
    end
    
    expr = quote
        if $cond_expr
            $(Expr(:block, true_exprs...))
        else
            $(Expr(:block, false_exprs...))
        end
    end
    return expr
end

"""
    compile_function(module_name, function_name, instructions, args) -> Expr

Compile a complete IR function to a Julia function definition.
"""
function compile_function(module_name::Symbol,
                          function_name::Symbol,
                          instructions::Vector{FkInstruction},
                          args::Vector{Symbol},
                          engine_var::Symbol=:engine)::Expr
    
    ctx = FkCompilerContext(engine_var)
    
    # Pre-populate SSA map with arguments
    for (i, arg) in enumerate(args)
        ctx.ssa_map[arg] = arg
    end
    
    # Compile all instructions
    body_exprs = Expr[]
    for inst in instructions
        expr = compile_instruction!(ctx, inst)
        expr !== nothing && push!(body_exprs, expr)
    end
    
    # Return the result value (last instruction's result)
    last_inst = instructions[end]
    result_expr = ctx.ssa_map[last_inst.result]
    
    return quote
        function $(function_name)($(args...))
            $(Expr(:block, body_exprs...))
            return $result_expr
        end
    end
end