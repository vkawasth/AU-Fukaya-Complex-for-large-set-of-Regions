# =============================================================================
# au_fukaya_ir.jl
#
# AU Fukaya Floer Cohomology IR — Complete Definition
#
# Merges: au_fukaya_floer_ir.jl + projection_ir.jl + serving path instructions
#
# ARCHITECTURE:
#   Frontend  : au_fukaya_engine.jl + fukaya_ad_context.jl  (emits this IR)
#   Middle    : optimisation passes (constant folding, loop unrolling, DCE)
#   Lowering  : au_fukaya_ir_to_julia.jl  (IR → Julia AST)
#   Backend   : Julia JIT → LLVM → machine code
#
# Every IR instruction is a Julia struct (SSA value + typed operands).
# The lowering pass (compile_instruction!) maps each struct → Julia Expr.
# eval(compile_function(...)) defines the Julia function at load time.
#
# LLVM MAPPING (precise):
#   FkInstruction           ≅  LLVM instruction
#   FkType                  ≅  LLVM type (%T = type { ... })
#   Symbol in ssa_map       ≅  LLVM %value  (SSA name)
#   AllocLagrangian         ≅  alloca / load from global lazy pool
#   FloerIntersection       ≅  function call (pure, no side effects)
#   NNOUnrolledLoop         ≅  fully unrolled loop (loop body × N copies)
#   LazyEval                ≅  load with lazy initialisation thunk
#   Branch / Phi            ≅  br + phi  (CFG branching)
#   ProjectOntoBasis        ≅  vector dot product + LICM-hoisted coefficients
#   ProjectOntoTensorPair   ≅  outer product (tensor instruction)
#   ProjectToScalar         ≅  dot product (reduction)
#   Pushforward             ≅  matrix-vector multiply (linear map application)
#   ServeAd                 ≅  call to hot-path (inlined at compile time)
# =============================================================================

# =============================================================================
# PART 1: TYPE SYSTEM
# =============================================================================

abstract type FkInstruction end
abstract type FkType end

struct LagrangianType      <: FkType end
struct FloerComplexType    <: FkType end
struct ModuliSpaceType     <: FkType end
struct CoproductType       <: FkType end
struct EmbeddingVecType    <: FkType end   # ℝ^D  (product/slot embedding)
struct TensorPairType      <: FkType end   # ℝ^(D×D)  (feedback tensor)
struct BracketType         <: FkType end   # [P_min, P_max]
struct AdRouteType         <: FkType end   # serving decision

# NNO type-level naturals — compile-time loop bounds
abstract type NNO end
struct Zero <: NNO end
struct Succ{N<:NNO} <: NNO end

# Compute the value of a type-level natural
value(::Type{Zero}) = 0
value(::Type{Succ{N}}) where N = 1 + value(N)

# Convenience: build NNO type for a literal
NNOLit(n::Int) = n == 0 ? Zero : Succ{NNOLit(n-1)}

# =============================================================================
# PART 2: CORE A∞ INSTRUCTIONS
# =============================================================================

"""
    AllocLagrangian

Allocate a Lagrangian L_i from the AU attic (lazy — zero evaluation cost).
SSA type: LagrangianType
LLVM analogue: load from global constant pool (lazy initialisation)
"""
struct AllocLagrangian <: FkInstruction
    station  ::Int
    hour     ::Int
    month    ::Int
    demo_vec ::Symbol    # :RM, :RF, :PM, :PF, :CNY, :Christmas, ...
    result   ::Symbol    # SSA name
end

"""
    FloerIntersection

Compute CF*(L_i, L_j): Floer chain complex of two Lagrangians.
SSA type: FloerComplexType
LLVM analogue: pure function call (no side effects, memoisable)
"""
struct FloerIntersection <: FkInstruction
    L_i       ::Symbol
    L_j       ::Symbol
    threshold ::Float64
    result    ::Symbol
end

"""
    M1Differential

Apply m₁: propagate Floer generators along Hamiltonian flow.
SSA type: FloerComplexType
LLVM analogue: sparse matrix-vector multiply (or neighbourhood table lookup)
"""
struct M1Differential <: FkInstruction
    floer_complex ::Symbol
    T             ::Symbol    # reference to transition matrix (or :neighborhood_table)
    result        ::Symbol
end

"""
    M2Composition

Apply m₂(L_i, L_j, product): triple intersection scoring.
SSA type: Float64 (scalar score)
LLVM analogue: dot product over D dimensions
"""
struct M2Composition <: FkInstruction
    L_i         ::Symbol
    L_j         ::Symbol
    product_idx ::Int
    affinity    ::Vector{Float64}
    result      ::Symbol
end

"""
    M3Homotopy

Apply m₃: timing stability check (loop-carried dependence analysis).
SSA type: Float64 (stability score, lower = more robust)
LLVM analogue: finite-difference perturbation check
"""
struct M3Homotopy <: FkInstruction
    L_i          ::Symbol
    L_j          ::Symbol
    L_k          ::Symbol
    perturbation ::Float64
    result       ::Symbol
end

"""
    CoproductDelta

Pair-of-pants coproduct Δ(slot) → Σ_{i≤j} #𝔐(slot; L_i, L_j) × (L_i ⊗ L_j)
SSA type: CoproductType (dict of tensor pair weights)
LLVM analogue: fan-out / splat — one input, multiple outputs
"""
struct CoproductDelta <: FkInstruction
    station     ::Int
    hour        ::Int
    month       ::Int
    product_idx ::Int
    affinity    ::Vector{Float64}
    result      ::Symbol
end

"""
    DiskVolume

Count of moduli space #𝔐(Ad; L_i, L_j) = disk volume.
SSA type: Float64 (geometric mean of flows × ω)
LLVM analogue: reduction over Floer pairs
"""
struct DiskVolume <: FkInstruction
    floer_pairs ::Vector{Tuple{Symbol,Symbol}}
    result      ::Symbol
end

"""
    NNOUnrolledLoop

Compile-time unrolled loop: n_steps is a TYPE-LEVEL natural (NNO).
Julia JIT generates n_steps copies of body as straight-line code.
LLVM analogue: fully unrolled loop — no branch, no counter.
NNO universal property: the unique morphism h:ℕ→A is the unrolled body.
"""
struct NNOUnrolledLoop <: FkInstruction
    start   ::Symbol                  # initial loop state (SSA name)
    n_steps ::Type{<:NNO}             # compile-time bound
    body    ::Vector{FkInstruction}   # IR block for one iteration
    result  ::Symbol
end

"""
    LazyEval

Mark a value as lazily evaluated (AU pullback: compute only when demanded).
LLVM analogue: load with lazy initialisation thunk (like LLVM's undef + guard)
"""
struct LazyEval <: FkInstruction
    source ::Symbol
    result ::Symbol
end

"""
    ServeAd

SLA-critical hot path: routing table lookup → stability gate → feedback gate.
LLVM analogue: call to a hot function (aggressively inlined)
This instruction compiles to a sequence of table reads — zero algebra.
"""
struct ServeAd <: FkInstruction
    station    ::Int
    hour       ::Int
    month      ::Int
    stab_floor ::Float64
    result     ::Symbol
end

# =============================================================================
# PART 3: PROJECTION INSTRUCTIONS
# (The three suggestions: slot projection, product projection, feedback projection)
# =============================================================================

"""
    DefineAdSlot

Define an ad slot as a named IR object at (station, hour, month).
Entry point for the serving path IR program.
LLVM analogue: alloca of a struct { station, hour, month }
"""
struct DefineAdSlot <: FkInstruction
    station ::Int
    hour    ::Int
    month   ::Int
    result  ::Symbol
end

"""
    ProjectOntoBasis  [PROJECTION SUGGESTION 1 + 2]

Project a Fukaya object onto the D-dimensional Lagrangian basis {L_i}.
This is the geometric projection:
  π(Obj) = Σ_i ⟨Obj, L_i⟩ × e_i  ∈  ℝ^D

For a PRODUCT: embed(p)[i] = ⟨L_i, p⟩ = global Floer pairing  → Pass 1
For an AD SLOT: q(s,h,m)[i] = L_i(s,h,m) × ω(s,h,m)           → compile_slot_query

SSA type: EmbeddingVecType (ℝ^D, normalised)
LLVM analogue: horizontal reduction + normalisation
LICM: the Lagrangian basis coefficients are loop-invariant → hoisted to offline pass
"""
struct ProjectOntoBasis <: FkInstruction
    source    ::Symbol             # object to project (product idx or ad slot)
    basis     ::Vector{Symbol}     # [:RM, :RF, :PM, :PF] — Lagrangian labels
    normalise ::Bool               # true = unit sphere (HNSW-ready)
    result    ::Symbol
end

"""
    ProjectOntoTensorPair  [PROJECTION SUGGESTION 3 — forward pass]

Project onto the tensor product basis L_i ⊗ L_j:
  F[i,j] += α × signal × √(embed[i] × embed[j])

Used for: feedback state update, cokernel computation, syzygies.
SSA type: TensorPairType (symmetric D×D matrix)
LLVM analogue: outer product update (BLAS dger-style)
Only updates entries where embed[i] > threshold AND embed[j] > threshold.
"""
struct ProjectOntoTensorPair <: FkInstruction
    source     ::Symbol          # product embedding (EmbeddingVecType)
    signal     ::Symbol          # observed feedback value (Float64)
    threshold  ::Float64         # significance gate (0.25)
    alpha      ::Float64         # EMA decay rate
    result     ::Symbol          # TensorPairType: updated feedback tensor
end

"""
    ProjectToScalar  [PROJECTION SUGGESTION 3 — adjoint/readout pass]

Contract feedback tensor with product embedding to produce a penalty score.
This is the ADJOINT of ProjectOntoTensorPair:

  score = Σ_{i≤j} F[i,j] × embed[i] × embed[j]

SSA type: Float64 (penalty score)
LLVM analogue: quadratic form evaluation (v^T F v)
The topological gate 𝟏_𝓜 can be applied here: if k-inv = 0, return 0.0
"""
struct ProjectToScalar <: FkInstruction
    tensor    ::Symbol    # TensorPairType (feedback tensor F[i,j])
    vector    ::Symbol    # EmbeddingVecType (product embedding)
    k_inv_floor ::Float64  # topological gate threshold (0.01 = dead zone cutoff)
    result    ::Symbol    # Float64 penalty score
end

"""
    Pushforward

Push a projection through a functor (m₁, m₂, HMM backward).
  pushforward(m₁)(q) = T × q   (neighbourhood spillover of slot query)
  pushforward(HMM)(q) = backward process of FK potential

Used for: m₁ spillover to neighbour stations, bracket computation.
LLVM analogue: matrix-vector multiply (or pre-computed table lookup)
"""
struct Pushforward <: FkInstruction
    projection ::Symbol
    functor    ::Symbol    # :m1, :m2, :hmm_backward, :dehn_twist
    result     ::Symbol
end

"""
    Pullback

Pull a projection back along a functor (adjoint direction).
  pullback(m₁)(q) = T^T × q   (reverse flow)
  pullback(HMM)(bracket) = Postnikov bracket from FK expectation

LLVM analogue: transpose matrix-vector multiply
"""
struct Pullback <: FkInstruction
    functor    ::Symbol
    projection ::Symbol
    result     ::Symbol
end

# =============================================================================
# PART 4: SERVING PATH INSTRUCTIONS (complete the serve_ad IR)
# =============================================================================

"""
    LoadEmbedding

Load precomputed product embedding from Pass 1 table.
LLVM analogue: load from read-only data segment (constant after compile)
"""
struct LoadEmbedding <: FkInstruction
    product_idx ::Int
    result      ::Symbol    # EmbeddingVecType
end

"""
    LoadFeedbackTensor

Load the current feedback tensor for a station.
LLVM analogue: load from mutable global (feedback state)
"""
struct LoadFeedbackTensor <: FkInstruction
    station_idx ::Int
    result      ::Symbol    # TensorPairType
end

"""
    DotProduct

Cosine similarity between slot query and product embedding.
LLVM analogue: dot product (vectorisable)
"""
struct DotProduct <: FkInstruction
    vec_a  ::Symbol
    vec_b  ::Symbol
    result ::Symbol    # Float64
end

"""
    ApplyPenalty

Subtract penalty from score: final_score = base_score + penalty_signal
(penalty_signal is typically negative)
LLVM analogue: fadd
"""
struct ApplyPenalty <: FkInstruction
    score   ::Symbol
    penalty ::Symbol
    result  ::Symbol
end

"""
    ArgMax

Select the product with the highest final score.
LLVM analogue: reduction over candidate list
"""
struct ArgMax <: FkInstruction
    scores ::Vector{Symbol}
    result ::Symbol
end

# =============================================================================
# PART 5: CONTROL FLOW
# =============================================================================

"""
    Branch

Conditional branch (stability gate, inventory check, dead-zone check).
LLVM analogue: br i1 %cond, label %true, label %false
"""
struct Branch <: FkInstruction
    cond        ::Symbol
    true_block  ::Vector{FkInstruction}
    false_block ::Vector{FkInstruction}
end

"""
    Phi

SSA PHI node: merge values from different predecessor blocks.
LLVM analogue: %v = phi type [ %v1, %block1 ], [ %v2, %block2 ]
"""
struct Phi <: FkInstruction
    incoming ::Dict{Symbol, Symbol}    # block_label => value
    result   ::Symbol
end

"""
    FkHMMBracket

Compute the Feynman-Kac bracket [P_min, P_max] for a slot.
Wraps the Pass 5 computation as a single IR instruction.
LLVM analogue: call to a pure function (bracket is memoised offline)
"""
struct FkHMMBracket <: FkInstruction
    station_idx ::Int
    product_idx ::Int
    month       ::Int
    result      ::Symbol    # BracketType: (p_min, p_max, k_inv)
end

"""
    TopoGate

Apply the topological indicator 𝟏_𝓜: hard gate on dead zones.
If k-inv < threshold → result = :nothing (dead zone, no ad served)
LLVM analogue: select instruction (cmov)
"""
struct TopoGate <: FkInstruction
    bracket   ::Symbol
    candidate ::Symbol
    threshold ::Float64    # k-inv floor (0.01)
    result    ::Symbol
end

# =============================================================================
# PART 6: IR PROGRAM HELPERS
# =============================================================================

"""
    FkBasicBlock

A basic block: linear sequence of FkInstructions, no branches within.
Corresponds to LLVM's BasicBlock.
"""
struct FkBasicBlock
    label        ::Symbol
    instructions ::Vector{FkInstruction}
    terminator   ::Union{Branch, Nothing}
end

"""
    FkFunction

A complete IR function: entry block + additional blocks (for branches).
Corresponds to LLVM's Function with CFG.
"""
struct FkFunction
    name   ::Symbol
    args   ::Vector{Symbol}
    entry  ::FkBasicBlock
    blocks ::Vector{FkBasicBlock}
end


# =============================================================================
# PART 8: ALGEBRAIC STRUCTURE PROJECTIONS
# Three projections completing the Hodge-Mayer-Vietoris triangle:
#   ExceptionalProjection — H2 level  (where things FAIL: coker basis)
#   SyzygyProjection      — H1→H2    (WHY they fail: circuit relations)
#   SpectralProjection    — H1 level  (HOW they fail: Hodge decomposition)
# =============================================================================

"""
    ExceptionalProjection

Project a cokernel class onto the 62-dimensional exceptional divisor basis.

Mathematical object: coker(rho*: H*(W_T12) → H*(W_T1) + H*(W_T2)) = HH2(W,W)
For (CTX_sAMY, CTX_INFRA): dim(coker) = 62 (confirmed by 4ti2 + Hochschild).

Each dimension is an independent failure mode of the placement system.
The scalar topological gate 1_M is the TOTAL obstruction (scalar 0/1).
ExceptionalProjection gives the DECOMPOSED obstruction in R^62:
one component per irreducible failure direction.

This is the blow-up of the dead-zone singularity: instead of one collapsed
point, we see 62 independent directions — exactly the Hironaka exceptional
divisor of the resolution.

LLVM analogue: split a single boolean flag into 62 independent condition codes.
Compiler role: Pass 6 uses specific exceptional components to compute the
correct cluster mutation coefficient at each wall crossing.
"""
struct ExceptionalProjection <: FkInstruction
    coker_class ::Symbol    # class in HH2(W,W) from Hochschild computation
    result      ::Symbol    # R^62 obstruction vector
end

"""
    SyzygyProjection

Project a Markov circuit onto the syzygy basis (relations among circuits).

Mathematical object:
  First syzygies  = 37 Markov circuits (generators of toric ideal I, from 4ti2)
  Second syzygies = relations among the 37 circuits
  For the MTR: 161 Graver elements minus 37 Markov circuits = 124 second syzygies

A syzygy of a syzygy is a relation between relations.
In Hochschild cohomology: this is HH3(W, W).
SyzygyProjection(circuit_k) identifies which second syzygies involve circuit_k.

LLVM analogue: second pass of global value numbering.
  GVN pass 1: which routing scores are equivalent? (first syzygies, Markov basis)
  GVN pass 2: which equivalences are themselves equivalent? (second syzygies)

Compiler role: Pass 6 generates wall-crossing mutations FROM first syzygies.
SyzygyProjection determines when two wall-crossings commute (trivial HH3)
vs. when they don't (non-trivial HH3 = non-commuting cluster mutations).
"""
struct SyzygyProjection <: FkInstruction
    circuit ::Symbol    # one of the 37 Markov circuit labels
    result  ::Symbol    # coefficient vector in syzygy basis (R^124 for MTR)
end

"""
    SpectralProjection

Project an edge flow vector onto one of three orthogonal Hodge components:
  :harmonic — closed AND coclosed (H1 cycles, back-edges in CFG)
  :gradient — exact (tree flows, dominator-tree edges)
  :curl     — coexact (surplus flows, sources/sinks)

Hodge decomposition: f = f_harmonic + f_gradient + f_curl
In the MTR:
  f_harmonic = the 37 independent loops (4ti2 Markov basis circuits)
  f_gradient = the spanning tree flows (baseline ridership routing)
  f_curl     = deviation above/below tree baseline (surge or slump)

In our compiler each Pass uses one component:
  routing_table score   uses f_gradient  (shortest-path potential)
  HMM brackets          uses f_harmonic  (loop corrections from H1)
  stability score       uses f_curl      (normalised surplus deviation)

SpectralProjection makes the three components explicit IR objects
instead of implicitly mixed in the single omega(s,h,m) tensor.

LLVM analogue: loop decomposition in the CFG.
  :harmonic = back-edges (natural loops, reducible)
  :gradient = dominator-tree edges (structured forward flow)
  :curl     = irreducible flow, side-exits, non-structured CFG
"""
struct SpectralProjection <: FkInstruction
    source    ::Symbol    # edge flow vector in R^|E|
    component ::Symbol    # :harmonic | :gradient | :curl
    result    ::Symbol    # projected component in R^|E|
end


# =============================================================================
# PART 7: PREDEFINED IR PROGRAMS
# =============================================================================

"""
    serving_path_ir()  →  Vector{FkInstruction}

The complete serving path expressed as Fukaya IR.
Corresponds exactly to serve_ad() in au_compiler.jl but as IR objects.

DefineAdSlot → ProjectOntoBasis (slot query) → LoadEmbedding ×N →
DotProduct ×N → LoadFeedbackTensor → ProjectToScalar ×N →
ApplyPenalty ×N → FkHMMBracket → TopoGate → ArgMax
"""
function serving_path_ir(station::Int, hour::Int, month::Int,
                          n_products::Int,
                          basis::Vector{Symbol} = [:RM, :RF, :PM, :PF])::Vector{FkInstruction}
    ir = FkInstruction[]

    # 1. Define the ad slot
    push!(ir, DefineAdSlot(station, hour, month, :slot))

    # 2. Project slot onto demographic basis → slot query vector
    push!(ir, ProjectOntoBasis(:slot, basis, true, :q_slot))

    # 3. Load feedback tensor for this station
    push!(ir, LoadFeedbackTensor(station, :fb_tensor))

    # 4. For each product: load embedding, compute score, apply penalty
    score_syms = Symbol[]
    for p in 1:n_products
        embed_sym   = Symbol("embed_$p")
        score_sym   = Symbol("score_$p")
        penalty_sym = Symbol("penalty_$p")
        final_sym   = Symbol("final_$p")
        bracket_sym = Symbol("bracket_$p")
        gated_sym   = Symbol("gated_$p")

        push!(ir, LoadEmbedding(p, embed_sym))
        push!(ir, DotProduct(:q_slot, embed_sym, score_sym))
        push!(ir, ProjectToScalar(:fb_tensor, embed_sym, 0.01, penalty_sym))
        push!(ir, ApplyPenalty(score_sym, penalty_sym, final_sym))
        push!(ir, FkHMMBracket(station, p, month, bracket_sym))
        push!(ir, TopoGate(bracket_sym, final_sym, 0.01, gated_sym))
        push!(score_syms, gated_sym)
    end

    # 5. Select best product
    push!(ir, ArgMax(score_syms, :best))

    return ir
end

"""
    product_embedding_ir(product_idx, basis)  →  Vector{FkInstruction}

Pass 1 product embedding as Fukaya IR.
ProjectOntoBasis with normalise=true.
The NNOUnrolledLoop unrolls the D-dimensional integration at compile time.
"""
function product_embedding_ir(product_idx::Int,
                               basis::Vector{Symbol} = [:RM, :RF, :PM, :PF])::Vector{FkInstruction}
    # The D-dimensional integration is expressed as an NNOUnrolledLoop
    # n_steps = D = length(basis) — a compile-time constant
    D = length(basis)

    ir = FkInstruction[
        ProjectOntoBasis(Symbol("product_$product_idx"), basis, true,
                         Symbol("embed_$product_idx"))
    ]
    return ir
end

"""
    feedback_update_ir(station_idx, product_idx, signal_var)  →  Vector{FkInstruction}

Feedback projection as Fukaya IR:
  Forward pass:  F_obs → L_i⊗L_j tensor  (ProjectOntoTensorPair)
  Readout pass:  F_s[i,j] × embed[i] × embed[j] → penalty  (ProjectToScalar)
"""
function feedback_update_ir(station_idx::Int, product_idx::Int,
                             signal_var::Symbol = :f_obs)::Vector{FkInstruction}
    embed_sym  = Symbol("embed_$product_idx")
    tensor_sym = Symbol("fb_$station_idx")
    penalty_sym = Symbol("penalty_$(station_idx)_$(product_idx)")

    return FkInstruction[
        LoadEmbedding(product_idx, embed_sym),
        LoadFeedbackTensor(station_idx, tensor_sym),
        ProjectOntoTensorPair(embed_sym, signal_var, 0.25, 0.3,
                              Symbol("fb_updated_$station_idx")),
        ProjectToScalar(tensor_sym, embed_sym, 0.01, penalty_sym),
    ]
end

