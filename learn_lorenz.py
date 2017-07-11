import nengo
import numpy as np

model = nengo.Network()
with model:
    speed = 0.5
    synapse = 0.1
    fn = 'osc'
    radius = 1
    n_neurons = 200

    if fn == 'lorenz':
        D = 3
        def func(x):
            x = x*30
            
            sigma = 10
            beta = 8.0/3
            rho = 28
            
            dx0 = -sigma * x[0] + sigma * x[1]
            dx1 = -x[0] * x[2] - x[1]
            dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
            
            return np.array([dx0 * synapse * speed + x[0],
                             dx1 * synapse * speed + x[1],
                             dx2 * synapse * speed + x[2]])/30.0
    elif fn == 'osc':
        D = 2
        def func(x):
            r = 1
            s = 6 * speed
            return [synapse * (-x[1] * s + x[0] * (r - x[0]**2 - x[1]**2)) + x[0],
                    synapse * ( x[0] * s + x[1] * (r - x[0]**2 - x[1]**2)) + x[1]]


    
    teacher = nengo.Ensemble(n_neurons=n_neurons, dimensions=D, radius=radius,
                            neuron_type=nengo.LIF())
    teach_conn = nengo.Connection(teacher, teacher, synapse=synapse, 
                                    function=func)
    ideal_teacher = nengo.Node(lambda t, x: func(x), size_in=D)



    student = nengo.Ensemble(n_neurons=n_neurons, dimensions=D, radius=radius,
                                neuron_type=nengo.LIF())
    nengo.Connection(student, ideal_teacher, synapse=synapse)
                                
    conn = nengo.Connection(student, student, synapse=synapse,
                            learning_rule_type=nengo.PES(learning_rate=1e-4,
                                                         pre_tau=0.01),
                            function=lambda x: [0]*D)

    syn2 = 0.01
    error = nengo.Node(None, size_in=D)
    nengo.Connection(student, error, synapse=syn2)
    nengo.Connection(teacher, error, synapse=syn2, transform=-1)
    #nengo.Connection(ideal_teacher, error, synapse=syn2, transform=-1)
    nengo.Connection(error, conn.learning_rule, synapse=None)
    
    
    import nengo_learning_display
    
    S = 30
    domain = np.zeros((D,S))
    domain[0,:] = np.linspace(-radius,radius, S)

    teach_x = nengo_learning_display.Plot1D(teach_conn, 
                    domain=domain.T, range=(-radius,radius))
    learn_x = nengo_learning_display.Plot1D(conn, 
                    domain=domain.T, range=(-radius,radius))
    
    
def on_step(sim):
    if sim is None: return
    if sim.n_steps < 2:
        teach_x.update(sim)
    learn_x.update(sim)