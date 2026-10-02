def step(dynamics, t, state, timestep, params):
    h = timestep
    y = state
    k1 = dynamics(t, y, params)
    k2 = dynamics(t + h / 2, y + k1 * h / 2, params)
    k3 = dynamics(t + h / 2, y + k2 * h / 2, params)
    k4 = dynamics(t + h, y + h * k3, params)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)