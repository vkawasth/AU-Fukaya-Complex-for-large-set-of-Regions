# =============================================================================
# au_fukaya_ir_to_julia.jl
#
# Lowering pass: Fukaya IR → Julia AST
#
# Merges: fk_ir_to_julia.jl + projection_ir_to_julia.jl
#
# Pipeline:
#   Fukaya IR (au_fukaya_ir.jl)
#     → compile_function(...)  → Julia Expr (AST)
#     → eval(...)              → Julia Function
#     → Julia JIT              → LLVM IR
#     → LLVM backend           → native machine code
#
# The lowering pass is the "LLVM opt" layer:
#   - Each FkInstruction → one or more Julia Expr nodes
#   - NNOUnrolledLoop → unrolled straight-line code (no runtime loop)
#   - LazyEval → closure thunk
#   - Branch → Julia if/else
#   - ProjectOntoBasis → loop over D (vectorised by Julia JIT)
#   - ProjectToScalar → quadratic form (single fused expression)
#   - ServeAd → direct call to pre-compiled hot path
#
# SSA invariant: every result symbol maps to exactly one Expr in ctx.ssa_map
# =============================================================================

include("au_fukaya_ir.jl")   # load IR definitions

# =============================================================================
# Compiler context
# =============================================================================

mutable struct FkCompilerContext
    ssa_map      ::Dict{Symbol, Any}    # IR value → Julia expression or value
    temp_counter ::Int
    engine_var   ::Symbol               # name of the AUFukayaEngine variable
    target       ::Symbol               # :au_fukaya_engine | :au_compiler | :runtime
end

FkCompilerContext(engine_var::Symbol;
                  target::Symbol=:au_compiler) =
    FkCompilerContext(Dict{Symbol,Any}(), 0, engine_var, target)

function fresh!(ctx::FkCompilerContext, base::String="t")
    ctx.temp_counter += 1
    Symbol("$(base)_$(ctx.temp_counter)")
end

# Bind a result name to a Julia expression
bind!(ctx::FkCompilerContext, name::Symbol, expr) =
    (ctx.ssa_map[name] = expr; nothing)

# Retrieve a bound expression (error if undefined)
fetch(ctx::FkCompilerContext, name::Symbol) =
    get(ctx.ssa_map, name, name)   # fall back to name itself (for args)

# =============================================================================
# PART 1: CORE A∞ INSTRUCTION LOWERING
# =============================================================================

function compile_instruction!(ctx::FkCompilerContext,
                               inst::AllocLagrangian)::Union{Expr,Nothing}
    # Lagrangians are looked up from the AU attic (lazy — no computation yet)
    expr = :(get($(ctx.engine_var).primitives,
                  $(QuoteNode(inst.demo_vec)), nothing))
    bind!(ctx, inst.result, expr)
    nothing
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::FloerIntersection)::Expr
    Li = fetch(ctx, inst.L_i)
    Lj = fetch(ctx, inst.L_j)
    v  = fresh!(ctx, "cf")
    bind!(ctx, inst.result, v)
    :($(v) = floer_complex($(Li), $(Lj); threshold=$(inst.threshold)))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::M1Differential)::Expr
    cf = fetch(ctx, inst.floer_complex)
    T  = fetch(ctx, inst.T)
    v  = fresh!(ctx, "m1")
    bind!(ctx, inst.result, v)
    :($(v) = m1_differential($(cf), $(T)))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::M2Composition)::Expr
    Li = fetch(ctx, inst.L_i)
    Lj = fetch(ctx, inst.L_j)
    v  = fresh!(ctx, "m2")
    bind!(ctx, inst.result, v)
    :($(v) = m2_composition($(Li), $(Lj),
                             $(inst.affinity), $(inst.product_idx)))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::M3Homotopy)::Expr
    Li = fetch(ctx, inst.L_i)
    Lj = fetch(ctx, inst.L_j)
    Lk = fetch(ctx, inst.L_k)
    v  = fresh!(ctx, "m3")
    bind!(ctx, inst.result, v)
    :($(v) = m3_homotopy($(Li), $(Lj), $(Lk), $(inst.perturbation)))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::CoproductDelta)::Expr
    v = fresh!(ctx, "delta")
    bind!(ctx, inst.result, v)
    quote
        _evt = AdEvent($(inst.station), $(inst.hour), $(inst.month),
                       $(inst.product_idx),
                       $(ctx.engine_var).omega[$(inst.station),
                                               $(inst.hour),
                                               $(inst.month)])
        $(v) = coprod_delta(_evt, $(inst.affinity),
                            $(ctx.engine_var).lagrangians)
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::DiskVolume)::Expr
    pairs_exprs = [:($(fetch(ctx,p[1])), $(fetch(ctx,p[2])))
                   for p in inst.floer_pairs]
    v = fresh!(ctx, "dvol")
    bind!(ctx, inst.result, v)
    :($(v) = sum(sqrt($(p[1]).density * $(p[2]).density)
                 for ($(p[1]), $(p[2])) in [$(pairs_exprs...)];
                 init=0.0))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::LazyEval)::Expr
    src = fetch(ctx, inst.source)
    v   = fresh!(ctx, "lazy")
    bind!(ctx, inst.result, v)
    :($(v) = () -> $(src))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ServeAd)::Expr
    # Hot path: direct call to serve_ad in au_compiler.jl
    v = fresh!(ctx, "served")
    bind!(ctx, inst.result, v)
    :($(v) = serve_ad($(ctx.engine_var).runtime_ctx,
                      $(inst.station), $(inst.hour), $(inst.month);
                      stab_floor=$(inst.stab_floor)))
end

# NNO unrolled loop — generates n_steps copies of body as straight-line code
function compile_instruction!(ctx::FkCompilerContext,
                               inst::NNOUnrolledLoop)::Expr
    n = value(inst.n_steps)
    stmts = Expr[]
    state = inst.start
    for i in 1:n
        sub_ctx = FkCompilerContext(ctx.engine_var; target=ctx.target)
        merge!(sub_ctx.ssa_map, ctx.ssa_map)
        for body_inst in inst.body
            push!(stmts, compile_instruction!(sub_ctx, body_inst))
        end
        # The result of the body is the new loop state
        new_state = fresh!(ctx, "s$(i)")
        push!(stmts, :($(new_state) = $(fetch(sub_ctx, inst.result))))
        state = new_state
    end
    bind!(ctx, inst.result, state)
    Expr(:block, stmts...)
end

# Branch → Julia if/else
function compile_instruction!(ctx::FkCompilerContext,
                               inst::Branch)::Expr
    cond = fetch(ctx, inst.cond)
    true_stmts = Expr[]
    for i in inst.true_block
        push!(true_stmts, compile_instruction!(ctx, i))
    end
    false_stmts = Expr[]
    for i in inst.false_block
        push!(false_stmts, compile_instruction!(ctx, i))
    end
    :(if $(cond)
          $(Expr(:block, true_stmts...))
      else
          $(Expr(:block, false_stmts...))
      end)
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::Phi)::Expr
    # PHI node — at runtime Julia handles this via variable assignment
    # The lowering just emits the merge (whichever branch executed last wins)
    v = fresh!(ctx, "phi")
    bind!(ctx, inst.result, v)
    :($(v) = nothing)   # placeholder; branches set the value
end

# =============================================================================
# PART 2: PROJECTION INSTRUCTION LOWERING
# =============================================================================

function compile_instruction!(ctx::FkCompilerContext,
                               inst::DefineAdSlot)::Expr
    v = fresh!(ctx, "slot")
    bind!(ctx, inst.result, v)
    :($(v) = (station=$(inst.station),
               hour=$(inst.hour),
               month=$(inst.month)))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ProjectOntoBasis)::Expr
    #
    # Suggestion 1 (slot query) + Suggestion 2 (product embedding):
    #
    # For a SLOT:   q[d] = lag.flow[s,h,m] × ω[s,h,m]
    # For a PRODUCT: embed[d] = Σ_{s,h,m} lag.flow[s,h,m] × aff[d] × ω[s,h,m]
    #
    # Both are instances of ProjectOntoBasis.
    # The lowering calls the appropriate backend function based on source type.
    #
    src = fetch(ctx, inst.source)
    bas = inst.basis
    v   = fresh!(ctx, "proj")
    bind!(ctx, inst.result, v)

    if inst.normalise
        quote
            $(v) = let
                _q = zeros($(length(bas)))
                for (_d, _lbl) in enumerate($(bas))
                    _lag = get($(ctx.engine_var).lagrangians_by_label, _lbl, nothing)
                    _lag === nothing && continue
                    # floer_pairing dispatches on source type:
                    # - for AdSlot: returns lag.flow[s,h,m] × ω
                    # - for Product: returns global Floer integral
                    _q[_d] = floer_pairing($(src), _lag,
                                            $(ctx.engine_var).omega)
                end
                _nrm = norm(_q); _nrm > 1e-10 && (_q ./= _nrm)
                _q
            end
        end
    else
        quote
            $(v) = let
                _q = zeros($(length(bas)))
                for (_d, _lbl) in enumerate($(bas))
                    _lag = get($(ctx.engine_var).lagrangians_by_label, _lbl, nothing)
                    _lag === nothing && continue
                    _q[_d] = floer_pairing($(src), _lag,
                                            $(ctx.engine_var).omega)
                end
                _q
            end
        end
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ProjectOntoTensorPair)::Expr
    #
    # Suggestion 3 — forward pass (observe_feedback!):
    # F[i,j] += α × signal × √(embed[i] × embed[j])
    # Gate: only update when embed[i] > threshold AND embed[j] > threshold
    #
    src    = fetch(ctx, inst.source)   # product embedding
    sig    = fetch(ctx, inst.signal)   # observed feedback value
    fb_v   = fetch(ctx, inst.result)   # will be the updated tensor key prefix
    α      = inst.alpha
    θ      = inst.threshold

    v = fresh!(ctx, "fb_updated")
    bind!(ctx, inst.result, v)

    quote
        $(v) = let
            _D = length($(src))
            for _i in 1:_D, _j in _i:_D
                _ei = $(src)[_i]; _ej = $(src)[_j]
                (_ei < $(θ) || _ej < $(θ)) && continue
                _key = (_i, _j)
                _old = get($(ctx.engine_var).runtime_ctx.feedback,
                            _key, 0.0)
                $(ctx.engine_var).runtime_ctx.feedback[_key] =
                    (1.0 - $(α)) * _old + $(α) * $(sig) * sqrt(_ei * _ej)
            end
            $(ctx.engine_var).runtime_ctx.feedback
        end
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ProjectToScalar)::Expr
    #
    # Suggestion 3 — readout pass (feedback_signal):
    # score = Σ_{i≤j} F[i,j] × embed[i] × embed[j]
    # With topological gate: if k-inv ≈ 0 → score = 0 (dead zone)
    #
    tensor = fetch(ctx, inst.tensor)
    vec    = fetch(ctx, inst.vector)
    # k_inv_floor is a literal Float64 in the IR — no symbol lookup needed
    v      = fresh!(ctx, "scalar")
    bind!(ctx, inst.result, v)

    quote
        $(v) = let
            # The feedback tensor projection with no topological gate
            # (gate is applied separately by FkHMMBracket + TopoGate)
            _s = 0.0
            _D = length($(vec))
            for _i in 1:_D, _j in _i:_D
                _fb = get($(tensor), (_i, _j), 0.0)
                _fb == 0.0 && continue
                _s += _fb * $(vec)[_i] * $(vec)[_j]
            end
            _s
        end
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::Pushforward)::Expr
    proj = fetch(ctx, inst.projection)
    v    = fresh!(ctx, "pushed")
    bind!(ctx, inst.result, v)

    if inst.functor == :m1
        # m₁ pushforward: propagate slot query along Markov transition
        :($(v) = m1_pushforward($(proj),
                                $(ctx.engine_var).runtime_ctx.neighbors))
    elseif inst.functor == :hmm_backward
        # FK backward process for bracket computation
        :($(v) = hmm_backward_push($(proj),
                                   $(ctx.engine_var).omega,
                                   $(ctx.engine_var).products))
    elseif inst.functor == :dehn_twist
        # Pass 7: seasonal Dehn twist on slot query
        :($(v) = dehn_twist($(proj),
                            $(ctx.engine_var).vanishing_cycles))
    else
        error("Unknown functor: $(inst.functor)")
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::Pullback)::Expr
    proj = fetch(ctx, inst.projection)
    v    = fresh!(ctx, "pulled")
    bind!(ctx, inst.result, v)

    if inst.functor == :m1
        :($(v) = m1_pullback($(proj),
                             $(ctx.engine_var).runtime_ctx.neighbors))
    else
        error("Unknown pullback functor: $(inst.functor)")
    end
end


# =============================================================================
# PART 2B: ALGEBRAIC STRUCTURE PROJECTION LOWERING
# ExceptionalProjection, SyzygyProjection, SpectralProjection
# =============================================================================

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ExceptionalProjection)::Expr
    #
    # Lower ExceptionalProjection: coker_class → R^62 obstruction vector.
    #
    # At compile time (Pass 6): the 62 cokernel basis vectors are
    # precomputed from the Hochschild complex and stored in the engine.
    # At runtime: project the coker_class onto this precomputed basis.
    #
    # The 62-dim vector enables FINE-GRAINED failure diagnosis:
    #   component k nonzero → failure mode k is active at this placement.
    #   topological gate 1_M = (norm(result) < threshold) ? 0 : 1
    #   but now we know WHICH mode failed, not just THAT it failed.
    #
    coker = fetch(ctx, inst.coker_class)
    v     = fresh!(ctx, "exc")
    bind!(ctx, inst.result, v)
    quote
        $(v) = let
            # Project coker class onto the 62-dim exceptional divisor basis
            # Engine stores: coker_basis::Matrix{Float64}  (62 x ambient_dim)
            _basis = $(ctx.engine_var).coker_basis   # 62 x ambient_dim
            if _basis === nothing || isempty(_basis)
                zeros(62)   # fallback: not yet compiled
            else
                # coker_class is a vector in ambient HH2 space
                # project: v = basis * coker_class / norm(coker_class)
                _c = $(coker)
                _n = norm(_c); _n > 1e-10 ? _basis * (_c ./ _n) : zeros(62)
            end
        end
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::SyzygyProjection)::Expr
    #
    # Lower SyzygyProjection: Markov circuit → R^124 syzygy coefficient vector.
    #
    # The 37 Markov circuits (first syzygies) are precomputed by 4ti2.
    # The 124 second syzygies are the relations among those 37 circuits.
    # At compile time (Pass 6): build the 37x124 syzygy matrix from the
    #   Graver basis: syzygy[i,j] = coefficient of circuit i in second syzygy j.
    # At runtime: look up the row for this circuit.
    #
    # Use: if syzygy_vec[j] != 0, this circuit participates in syzygy j.
    #   Two wall-crossings commute iff their syzygy vectors are orthogonal.
    #   Non-orthogonal syzygies = non-commuting cluster mutations = HH3 obstruction.
    #
    circuit = fetch(ctx, inst.circuit)
    v       = fresh!(ctx, "syz")
    bind!(ctx, inst.result, v)
    quote
        $(v) = let
            _syzygy_mat = $(ctx.engine_var).syzygy_matrix   # 37 x 124
            _circuit_idx = get($(ctx.engine_var).circuit_index,
                               $(circuit), nothing)
            if _syzygy_mat === nothing || _circuit_idx === nothing
                zeros(124)   # fallback
            else
                _syzygy_mat[_circuit_idx, :]   # row = coefficients for this circuit
            end
        end
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::SpectralProjection)::Expr
    #
    # Lower SpectralProjection: edge flow → one Hodge component.
    #
    # Hodge decomposition uses the boundary operators d0 and d1 of the
    # cellular chain complex (from discrete_morse.jl):
    #   gradient component  = d1^+ d1 f        (exact part)
    #   curl component      = d0 d0^+ f         (coexact part)
    #   harmonic component  = f - gradient - curl
    #
    # d1 = boundary matrix |E| x |V|  (arrows to vertices)
    # d0 = coboundary |V| x |V|
    #
    # For the MTR:
    #   gradient  = spanning tree flows (β₀ = 1 connected component)
    #   harmonic  = the 37 independent loop flows (β₁ = 37)
    #   curl      = the residuals (β₂ = 0 for planar sections)
    #
    # SpectralProjection(:harmonic) returns the exact loop contribution,
    # which is what the HMM bracket computation integrates over.
    #
    src  = fetch(ctx, inst.source)
    comp = inst.component
    v    = fresh!(ctx, "spec")
    bind!(ctx, inst.result, v)

    quote
        $(v) = let
            _f   = $(src)                                   # edge flow vector
            _d1  = $(ctx.engine_var).boundary_d1            # |E| x |V|
            _d1t = transpose(_d1)
            if _d1 === nothing
                $(comp) == :harmonic ? _f : zeros(length(_f))
            else
                # Hodge projectors (pseudo-inverses via backslash)
                # Correct discrete Hodge: d1 is |E|×|V|
                # divergence = d1 * f  (|V|-vector)
                # Laplacian  = d1' * d1  (|V|×|V|)
                # potentials = L^+ * div  (|V|-vector)
                # gradient   = d1' * potentials  (|E|-vector)
                _div       = _d1 * _f                       # |V|-vector
                _L         = _d1' * _d1                     # |V|×|V| Laplacian
                _x         = pinv(Matrix(_L)) * _div        # node potentials
                _grad_proj = _d1' * _x                      # gradient component
                _curl_proj = zeros(length(_f))              # β₂=0 for plain graph
                _harm_proj = _f .- _grad_proj               # harmonic = residual

                if $(QuoteNode(comp)) == :harmonic
                    _harm_proj
                elseif $(QuoteNode(comp)) == :gradient
                    _grad_proj
                else  # :curl
                    _curl_proj
                end
            end
        end
    end
end


# =============================================================================
# PART 3: SERVING PATH INSTRUCTION LOWERING
# =============================================================================

function compile_instruction!(ctx::FkCompilerContext,
                               inst::LoadEmbedding)::Union{Expr,Nothing}
    v = fresh!(ctx, "emb")
    bind!(ctx, inst.result, v)
    :($(v) = $(ctx.engine_var).embeddings[$(inst.product_idx), :])
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::LoadFeedbackTensor)::Union{Expr,Nothing}
    v = fresh!(ctx, "fbt")
    bind!(ctx, inst.result, v)
    :($(v) = $(ctx.engine_var).runtime_ctx.feedback)
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::DotProduct)::Expr
    a = fetch(ctx, inst.vec_a)
    b = fetch(ctx, inst.vec_b)
    v = fresh!(ctx, "dot")
    bind!(ctx, inst.result, v)
    :($(v) = LinearAlgebra.dot($(a), $(b)))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ApplyPenalty)::Expr
    score   = fetch(ctx, inst.score)
    penalty = fetch(ctx, inst.penalty)
    v       = fresh!(ctx, "final")
    bind!(ctx, inst.result, v)
    :($(v) = $(score) + $(penalty))   # penalty is negative → subtraction
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ArgMax)::Expr
    score_exprs = [fetch(ctx, s) for s in inst.scores]
    v = fresh!(ctx, "best")
    bind!(ctx, inst.result, v)
    :($(v) = argmax([$(score_exprs...)]))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::FkHMMBracket)::Expr
    v = fresh!(ctx, "bracket")
    bind!(ctx, inst.result, v)
    :($(v) = get($(ctx.engine_var).bracket_idx,
                 ($(ctx.engine_var).station_names[$(inst.station_idx)],
                  $(inst.product_idx), $(inst.month)),
                 nothing))
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::TopoGate)::Expr
    bracket   = fetch(ctx, inst.bracket)
    candidate = fetch(ctx, inst.candidate)
    v = fresh!(ctx, "gated")
    bind!(ctx, inst.result, v)
    quote
        $(v) = let _b = $(bracket)
            if _b === nothing || _b.k_invariant < $(inst.threshold)
                -Inf   # dead zone: remove from ArgMax competition
            else
                $(candidate)
            end
        end
    end
end

# =============================================================================
# PART 4: COMPILE FUNCTION (ENTRY POINT)
# =============================================================================

"""
    compile_function(name, instructions, args; engine_var, target) -> Expr

Compile a list of FkInstructions to a Julia function definition.
The returned Expr can be eval()d to define the function.

Pipeline:
  FkInstruction[] → FkCompilerContext → Julia Expr → eval → Julia Function → JIT → CPU
"""
function compile_function(name        ::Symbol,
                           instructions::Vector{FkInstruction},
                           args        ::Vector{Symbol};
                           engine_var  ::Symbol = :engine,
                           target      ::Symbol = :au_compiler)::Expr

    ctx = FkCompilerContext(engine_var; target=target)

    # Pre-populate SSA map with function arguments
    for arg in args
        ctx.ssa_map[arg] = arg
    end

    # Compile all instructions → body expressions
    body = Expr[]
    for inst in instructions
        expr = compile_instruction!(ctx, inst)
        expr !== nothing && push!(body, expr)
    end

    # Return value = result of last instruction
    ret_sym = fetch(ctx, instructions[end].result)

    quote
        function $(name)($(args...))
            $(body...)
            return $(ret_sym)
        end
    end
end

# =============================================================================
# PART 5: OPTIMISATION PASSES ON THE IR
# =============================================================================

"""
    constant_fold!(instructions)

Constant propagation pass: inline compile-time constants.
Identifies AllocLagrangian instructions that reference fixed (station,h,m)
and folds them into their uses.
LLVM analogue: instcombine + constant propagation
"""
function constant_fold!(instructions::Vector{FkInstruction})
    # Find AllocLagrangian with fixed coordinates
    constant_lags = Dict{Symbol, AllocLagrangian}()
    for inst in instructions
        if inst isa AllocLagrangian && inst.station != 0
            constant_lags[inst.result] = inst
        end
    end
    # Future: propagate through FloerIntersection, M2Composition uses
    # For now: return instructions with constants tagged
    return instructions
end

"""
    dead_code_eliminate!(instructions)

Remove instructions whose results are never used.
Also removes AllocLagrangian for zero-flow Lagrangians (dead zones).
LLVM analogue: DCE pass
"""
function dead_code_eliminate!(instructions::Vector{FkInstruction})
    used = Set{Symbol}()
    # Collect all used SSA names
    for inst in instructions
        for field in fieldnames(typeof(inst))
            val = getfield(inst, field)
            val isa Symbol && push!(used, val)
        end
    end
    # Keep: instructions whose result is used, or has side effects
    filter!(instructions) do inst
        hasproperty(inst, :result) ? inst.result ∈ used : true
    end
    return instructions
end

