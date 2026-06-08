struct SyzygyProjection <: FkInstruction
    circuit::Symbol
    result::Symbol  # coefficient vector in syzygy basis
end