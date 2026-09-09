def step(t, state, timestep, params, model):
    return state + timestep * model.dynamics(t, state, params)
