import nengo

class Clamp(nengo.Process):
    def __init__(self, Kp=3, size_in=1):
        self.Kp = Kp
        super().__init__(default_size_in=size_in*2, default_size_out=size_in)
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, x):
            target = x[:shape_out[0]]
            current = x[shape_out[0]:]
            return self.Kp*(target-current)
        return step

model = nengo.Network()
with model:
    
    clamp = nengo.Node(Clamp(size_in=1))
    
    a = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    target = nengo.Node(0)
    nengo.Connection(target, clamp[0])
    nengo.Connection(a, clamp[1])
    nengo.Connection(clamp, a)
    
