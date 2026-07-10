import nengo
import numpy as np

class Predictor(nengo.Network):
    def __init__(self, dimensions, Kp=0.5, Ki=1):
        super().__init__()
        self.Kp = Kp
        self.Ki = Ki
        with self:
            self.target = nengo.Ensemble(n_neurons=50*dimensions,
                                         dimensions=dimensions)
            self.actual = nengo.Ensemble(n_neurons=50*dimensions,
                                         dimensions=dimensions)
                                         
            self.disable = nengo.Ensemble(n_neurons=50, dimensions=1,
                                         encoders=nengo.dists.Choice([[1]]),
                                         intercepts=nengo.dists.Uniform(0,0.3))

            if Kp != 0:
                self.error = nengo.Ensemble(n_neurons=50*dimensions,
                                             dimensions=dimensions)
                nengo.Connection(self.target, self.error, transform=Kp)
                nengo.Connection(self.actual, self.error, transform=-Kp)
                nengo.Connection(self.error, self.actual)
                nengo.Connection(self.disable, self.error.neurons, transform=-5*np.ones((self.error.n_neurons,1)))
            if Ki != 0:
                self.int_error = nengo.Ensemble(n_neurons=50*dimensions,
                                             dimensions=dimensions)
                                             
                nengo.Connection(self.int_error, self.int_error, synapse=0.1)
                nengo.Connection(self.int_error, self.actual, transform=Ki)
                nengo.Connection(self.target, self.int_error, transform=0.1)
                nengo.Connection(self.actual, self.int_error, transform=-0.1)
                nengo.Connection(self.disable, self.int_error.neurons, transform=-5*np.ones((self.int_error.n_neurons,1)))
            
model = nengo.Network()
with model:
    
    x = Predictor(dimensions=2, Ki=6,Kp=2)
    y = Predictor(dimensions=1, Ki=6,Kp=2)

    stim_disable = nengo.Node(1)
    nengo.Connection(stim_disable, y.disable)

    stim_x = np.array([[1,1],[-1,-1],[-1,1],[1,-1]])
    stim_y = np.array([[1],[1],[-1],[-1]])    
    present_x = nengo.Node(nengo.processes.PresentInput(stim_x, presentation_time=0.1))
    present_y = nengo.Node(nengo.processes.PresentInput(stim_y, presentation_time=0.1))
    nengo.Connection(present_x, x.target)
    nengo.Connection(present_y, y.target)
    
    alpha=1e-3
    c_x2y = nengo.Connection(x.actual, y.actual, function=lambda x: 0, learning_rule_type=nengo.PES(learning_rate=alpha))
    if y.Kp != 0:
        nengo.Connection(y.error, c_x2y.learning_rule, synapse=None, transform=-1)
    if y.Ki != 0:
        nengo.Connection(y.int_error, c_x2y.learning_rule, synapse=None, transform=-1)

    #alpha=1e-3
    #c_y2x = nengo.Connection(y.actual, x.actual, function=lambda x: [0,0], learning_rule_type=nengo.PES(learning_rate=alpha))
    #if y.Kp is not 0:
    #    nengo.Connection(x.error, c_y2x.learning_rule, synapse=None, transform=-1)
    #if y.Ki is not 0:
    #    nengo.Connection(x.int_error, c_y2x.learning_rule, synapse=None, transform=-1)


    