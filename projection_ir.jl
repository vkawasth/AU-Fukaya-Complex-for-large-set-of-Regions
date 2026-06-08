# =============================================================================
# Additional IR instructions for projection
# =============================================================================

"""
    ProjectOntoBasis(source, basis_lagrangians) -> FkValue

Project a Fukaya object (product, ad slot, or Floer complex) onto
the chosen Lagrangian basis {L_i}. This is the geometric analogue of
a linear projection in vector spaces.

In Fukaya terms: π(Obj) = Σ_i ⟨Obj, L_i⟩ × L_i
where ⟨·,·⟩ is the Floer pairing CF*(Obj, L_i).
"""
struct ProjectOntoBasis <: FkInstruction
    source::Symbol           # object to project (product, ad_slot, CF complex)
    basis::Vector{Symbol}    # list of Lagrangian labels to project onto
    result::Symbol
end

"""
    ProjectOntoTensorPair(source, basis_pairs) -> FkValue

Project onto the tensor product basis L_i ⊗ L_j.
Used for feedback, cokernel interactions, and syzygies.
"""
struct ProjectOntoTensorPair <: FkInstruction
    source::Symbol
    pairs::Vector{Tuple{Symbol,Symbol}}
    result::Symbol
end

"""
    ProjectToScalar(tensor, weights) -> FkValue

Contract a tensor (e.g., feedback tensor) with a vector (e.g., product embedding)
to produce a scalar score. This is the evaluation of a linear functional.
"""
struct ProjectToScalar <: FkInstruction
    tensor::Symbol       # D×D tensor (e.g., feedback)
    vector::Symbol       # D-dimensional vector (e.g., product embedding)
    result::Symbol
end

"""
    Pushforward(projection, functor) -> FkValue

Push a projection through a functor (e.g., m₁, m₂, or the HMM backward process).
Used to propagate compressed representations through the A∞ operations.
"""
struct Pushforward <: FkInstruction
    projection::Symbol
    functor::Symbol      # :m1, :m2, :hmm_backward
    result::Symbol
end

"""
    Pullback(functor, projection) -> FkValue

Pull a projection back along a functor.
Used to lift a low-dimensional query to the full Fukaya category.
"""
struct Pullback <: FkInstruction
    functor::Symbol
    projection::Symbol
    result::Symbol
end