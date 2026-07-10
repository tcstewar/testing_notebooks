import nengo
import numpy as np

class Predictor(nengo.Network):
    def __init__(self, dimensions, Kp=0.5, Ki=1):
        super().__init__()
        with self:
            self.target = nengo.Ensemble(n_neurons=50*dimensions,
                                         dimensions=dimensions)
            self.actual = nengo.Ensemble(n_neurons=50*dimensions,
                                         dimensions=dimensions)
                                         
            self.disable = nengo.Ensemble(n_neurons=50, dimensions=1,
                                         encoders=nengo.dists.Choice([[1]]),
                                         intercepts=nengo.dists.Uniform(0,0.3))

            if Kp is not 0:
                self.error = nengo.Ensemble(n_neurons=50*dimensions,
                                             dimensions=dimensions)
                nengo.Connection(self.target, self.error, transform=Kp)
                nengo.Connection(self.actual, self.error, transform=-Kp)
                nengo.Connection(self.error, self.actual)
                nengo.Connection(self.disable, self.error.neurons, transform=-3*np.ones((self.error.n_neurons,1)))
            if Ki is not 0:
                self.int_error = nengo.Ensemble(n_neurons=50*dimensions,
                                             dimensions=dimensions)
                                             
                nengo.Connection(self.int_error, self.int_error, synapse=0.1)
                nengo.Connection(self.int_error, self.actual, transform=Ki)
                nengo.Connection(self.target, self.int_error, transform=0.1)
                nengo.Connection(self.actual, self.int_error, transform=-0.1)
                nengo.Connection(self.disable, self.int_error.neurons, transform=-3*np.ones((self.int_error.n_neurons,1)))
            
model = nengo.Network()
with model:
    
    x = Predictor(dimensions=1, Ki=6,Kp=2)

    stim_disable = nengo.Node(0)
    stim_target = nengo.Node(0)
    nengo.Connection(stim_disable, x.disable)
    nengo.Connection(stim_target, x.target)
    
    