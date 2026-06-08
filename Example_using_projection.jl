# Example: Ad placement with explicit projection steps
function build_ad_placement_ir_with_projection()
    instr = FkInstruction[]
    
    # Step 1: Define ad slot as an object
    push!(instr, DefineAdSlot(1, 9, 2, :ad_slot))  # Admiralty, 9am, Feb
    
    # Step 2: Project ad slot onto demographic basis (local query)
    push!(instr, ProjectOntoBasis(:ad_slot, [:RM, :RF, :PM, :PF], :q_local))
    
    # Step 3: Load product embeddings (global projections, precomputed)
    push!(instr, LoadEmbedding(1, :embed_luxury))   # Luxury Watch embedding
    push!(instr, LoadEmbedding(2, :embed_train))    # Train Ticket embedding
    
    # Step 4: Compute cosine similarity (dot product of projections)
    push!(instr, DotProduct(:q_local, :embed_luxury, :score_luxury))
    push!(instr, DotProduct(:q_local, :embed_train, :score_train))
    
    # Step 5: Push projection through m₁ (spillover to neighbors)
    push!(instr, Pushforward(:q_local, :m1, :q_neighbors))
    
    # Step 6: Load feedback tensor at station
    push!(instr, LoadFeedbackTensor(1, :fb_admiralty))  # D×D tensor
    
    # Step 7: Project feedback to scalar penalty
    push!(instr, ProjectToScalar(:fb_admiralty, :embed_luxury, :penalty_luxury))
    push!(instr, ProjectToScalar(:fb_admiralty, :embed_train, :penalty_train))
    
    # Step 8: Apply penalty to scores
    push!(instr, ApplyPenalty(:score_luxury, :penalty_luxury, :final_luxury))
    push!(instr, ApplyPenalty(:score_train, :penalty_train, :final_train))
    
    # Step 9: Select best
    push!(instr, ArgMax([:final_luxury, :final_train], :best_product))
    
    return instr
end