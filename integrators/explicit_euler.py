def step(dynamics, t, state, timestep, params):
    return state + timestep * dynamics(t, state, params)