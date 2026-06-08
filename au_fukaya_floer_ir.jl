# =============================================================================
# au_fukaya_floer_ir.jl
#
# AU Fukaya Floer Cohomology IR
# Stable intermediate representation for Fukaya category operations.
# This IR is target-independent and can be optimized before lowering to Julia.
# =============================================================================

abstract type FkInstruction end

# ──────────────────────────────────────────────────────────────────────────
# Type declarations (like LLVM's %T = type { ... })
# ──────────────────────────────────────────────────────────────────────────

"""
    FkType

Fukaya IR types: Lagrangian, FloerComplex, ModuliSpace, etc.
"""
abstract type FkType end
struct LagrangianType <: FkType end
struct FloerComplexType <: FkType end
struct ModuliSpaceType <: FkType end
struct CoproductType <: FkType end
struct NNOType{N} <: FkType end  # N is a type-level natural

# ──────────────────────────────────────────────────────────────────────────
# Value instructions (like LLVM's %v = add i32 %a, %b)
# ──────────────────────────────────────────────────────────────────────────

"""
    AllocLagrangian(station, hour, month, demo_vec) -> FkValue

Allocate a Lagrangian submanifold L at a specific (station, hour, month).
Pulls from the AU attic lazily.
"""
struct AllocLagrangian <: FkInstruction
    station::Int
    hour::Int
    month::Int
    demo_vec::Symbol  # :RM, :RF, :PM, :PF, :CNY, etc.
    result::Symbol    # SSA value name
end

"""
    FloerIntersection(L_i, L_j, threshold) -> FkValue

Compute CF*(L_i, L_j) — the Floer complex of two Lagrangians.
Returns a handle to a FloerComplex value.
"""
struct FloerIntersection <: FkInstruction
    L_i::Symbol
    L_j::Symbol
    threshold::Float64
    result::Symbol
end

"""
    M1Differential(CF, transition_matrix) -> FkValue

Apply the A∞ differential m₁ to a Floer complex.
Propagates generators along Hamiltonian flow.
"""
struct M1Differential <: FkInstruction
    floer_complex::Symbol
    T::Matrix{Float64}  # or reference to precomputed matrix
    result::Symbol
end

"""
    M2Composition(L_i, L_j, product_idx, affinity) -> FkValue

Apply the A∞ product m₂: CF*(L_i, L_j) ⊗ CF*(L_j, L_k) → CF*(L_i, L_k)
In our setting: demographic pair × product affinity.
"""
struct M2Composition <: FkInstruction
    L_i::Symbol
    L_j::Symbol
    product_idx::Int
    affinity::Vector{Float64}
    result::Symbol
end

"""
    M3Homotopy(L_i, L_j, L_k, perturbation) -> FkValue

Apply the A∞ homotopy m₃: measures stability under perturbation.
Returns a stability score (lower = more robust).
"""
struct M3Homotopy <: FkInstruction
    L_i::Symbol
    L_j::Symbol
    L_k::Symbol
    perturbation::Float64
    result::Symbol
end

"""
    CoproductDelta(ad_event, product_affinity) -> FkValue

The pair-of-pants coproduct Δ: splits an ad event across demographics.
Returns a dict mapping Lagrangian → coefficient.
"""
struct CoproductDelta <: FkInstruction
    station::Int
    hour::Int
    month::Int
    product_idx::Int
    affinity::Vector{Float64}
    result::Symbol
end

"""
    DiskVolume(floer_pairs) -> FkValue

Count of the moduli space #𝔐(Ad; L_i, L_j).
Computes the geometric intersection volume.
"""
struct DiskVolume <: FkInstruction
    floer_pairs::Vector{Tuple{Symbol,Symbol}}
    result::Symbol
end

"""
    NNOUnrolledLoop(start, n_steps, body) -> FkValue

LLVM-style unrolled loop where n_steps is an NNO type.
Generates value(n_steps) copies of body at compile time.
"""
struct NNOUnrolledLoop <: FkInstruction
    start::Symbol        # initial loop state
    n_steps::Type{<:NNO} # type-level natural
    body::Vector{FkInstruction}  # IR block for loop body
    result::Symbol
end

"""
    LazyEval(value) -> FkValue

Mark a value as lazily evaluated (AU pullback).
The value is only computed when forced (e.g., by a print or serve operation).
"""
struct LazyEval <: FkInstruction
    source::Symbol
    result::Symbol
end

"""
    ServeAd(candidates) -> FkValue

SLA-critical path: lookup from routing table, apply stability/feedback gates.
Returns the chosen AdRoute.
"""
struct ServeAd <: FkInstruction
    station::Int
    hour::Int
    month::Int
    stab_floor::Float64
    result::Symbol
end

# ──────────────────────────────────────────────────────────────────────────
# Control flow instructions (like LLVM's br, phi)
# ──────────────────────────────────────────────────────────────────────────

"""
    Branch(cond, true_block, false_block)

Conditional branch. Used for stability/feedback gates.
"""
struct Branch <: FkInstruction
    cond::Symbol
    true_block::Vector{FkInstruction}
    false_block::Vector{FkInstruction}
end

"""
    Phi(incoming_values)

PHI node for SSA form. Selects a value based on which predecessor block was taken.
"""
struct Phi <: FkInstruction
    incoming::Dict{Symbol, Symbol}  # block_label => value
    result::Symbol
end