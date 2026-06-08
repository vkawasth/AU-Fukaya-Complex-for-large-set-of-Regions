function compile_slot_query(station, hour, month, demo_lags, omega)
    # This is a PROJECTION: Δ(ad_slot) → ℝ^D
    q = zeros(D)
    for (d, lag) in enumerate(demo_lags)
        q[d] = lag.flow[station, hour, month] * omega[station, hour, month]
    end
    return q  # D-dimensional query vector
end