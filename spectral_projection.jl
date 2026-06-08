struct SpectralProjection <: FkInstruction
    source::Symbol           # edge flow vector
    component::Symbol        # :harmonic, :gradient, :curl
    result::Symbol
end