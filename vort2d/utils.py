import numpy as np
from datetime import timedelta

def get_truth(c, t):
    model = c.models['vort2d']
    state = c.io.call_method(c, 'truth', model.read_var, name='velocity', member=None, time=t, model_src='vort2d')
    return state

def get_model_state(c, t, m):
    model = c.models['vort2d']
    if t == c.time:
        try:
            state = c.io.call_method(c, 'post', model.read_var, name='velocity', member=m, time=t, model_src='vort2d')
        except:
            state = c.io.call_method(c, 'current', model.read_var, name='velocity', member=m, time=t, model_src='vort2d')
    else:
        try:
            state = c.io.call_method(c, 'prior', model.read_var, name='velocity', member=m, time=t, model_src='vort2d')
        except:
            state = c.io.call_method(c, 'current', model.read_var, name='velocity', member=m, time=t, model_src='vort2d')
    return state

def get_model_ens(c, t):
    model = c.models['vort2d']
    ens = []
    for m in range(c.config.nens):
        ens.append(get_model_state(c, t, m))
    return np.array(ens)    

def get_times(c):
    model = c.models['vort2d']
    c.time = c.config.time_start
    times = []
    while c.time < c.config.time_end:
        t = c.time
        while t <= c.next_time:
            times.append(t)
            t += model.restart_dt * timedelta(hours=1)
        c.time = c.next_time
    return times

def get_time_series(c, func):
    model = c.models['vort2d']
    c.time = c.config.time_start
    time_series = []
    while c.time < c.config.time_end:
        t = c.time
        while t <= c.next_time:
            time_series.append(func(c, t))
            t += model.restart_dt * timedelta(hours=1)
        c.time = c.next_time
    return np.array(time_series)
