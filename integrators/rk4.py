def step(t, state, timestep, params, model):
    h = timestep
    y = state
    k1 = model.dynamics(t, y, params)
    k2 = model.dynamics(t+h/2, y+k1*h/2, params)
    k3 = model.dynamics(t+h/2, y+k2*h/2, params)
    k4 = model.dynamics(t+h, y+h*k3, params)
    return y+h/6*(k1+2*k2+2*k3+k4)
