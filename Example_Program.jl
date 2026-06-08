# =============================================================================
# Example: Compile a Fukaya query for "best ad at Admiralty 9am during CNY"
# =============================================================================

# Step 1: Define the IR program
function build_ad_placement_ir(engine_var::Symbol=:engine)
    instructions = FkInstruction[]
    
    # Allocate Lagrangians (from AU attic)
    push!(instructions, AllocLagrangian(0, 0, 0, :RM, :L_RM))
    push!(instructions, AllocLagrangian(0, 0, 0, :RF, :L_RF))
    push!(instructions, AllocLagrangian(0, 0, 0, :PM, :L_PM))
    push!(instructions, AllocLagrangian(0, 0, 0, :PF, :L_PF))
    
    # Floer intersections at Admiralty (station_idx=1), Feb (month=2), 9am (hour=9)
    # But Floer complex integrates over hours — so we call with peak hour
    push!(instructions, FloerIntersection(:L_RM, :L_RF, 0.15, :CF_RM_RF))
    push!(instructions, FloerIntersection(:L_RM, :L_PM, 0.15, :CF_RM_PM))
    push!(instructions, FloerIntersection(:L_RF, :L_PF, 0.15, :CF_RF_PF))
    
    # Coproduct Δ for Luxury Watch (product 1) at Admiralty 9am CNY
    luxury_affinity = [0.8, 0.6, 0.1, 0.05]  # RM, RF, PM, PF
    push!(instructions, CoproductDelta(1, 9, 2, 1, luxury_affinity, :delta_luxury))
    
    # Disk volume from Floer pairs
    push!(instructions, DiskVolume([(:L_RM, :L_RF), (:L_RM, :L_PM)], :disk_vol))
    
    # Lazy evaluation (AU pullback — not computed yet)
    push!(instructions, LazyEval(:delta_luxury, :lazy_delta))
    
    # Serve ad (SLA path)
    push!(instructions, ServeAd(1, 9, 2, 0.7, :result))
    
    return instructions
end

# Step 2: Compile to Julia function
ir_program = build_ad_placement_ir()
compiled_func_expr = compile_function(:AdModule, :get_best_ad, ir_program, [:engine])

# Step 3: Evaluate the expression to define the function
eval(compiled_func_expr)

# Step 4: Use the compiled function
engine = AUFukayaEngine(...)  # your existing engine
best_ad = get_best_ad(engine)
println(best_ad)