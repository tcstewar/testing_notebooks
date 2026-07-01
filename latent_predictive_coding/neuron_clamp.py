import nengo

class Clamp(nengo.Network):
    def __init__(self, size_in, Kp=1, Kd=0.01, Ki=1):
        super().__init__()
        with self:
            self.target = nengo.Node(None, size_in=size_in)
            self.actual = nengo.Node(None, size_in=size_in)
            self.u = nengo.Node(None, size_in=size_in)

            self.d_target = nengo.Node(None, size_in=size_in)
            nengo.Connection(self.target, self.d_target, synapse=None, transform=1000)
            nengo.Connection(self.target, self.d_target, synapse=0, transform=-1000)
    
        
            self.q_diff = nengo.Node(None, size_in=size_in)
            nengo.Connection(self.target, self.q_diff, synapse=None)
            nengo.Connection(self.actual, self.q_diff, synapse=None, transform=-1)
    
            nengo.Connection(self.q_diff, self.u, transform=Kp, synapse=None)
    
    
            self.dq_diff = nengo.Node(None, size_in=size_in)
            nengo.Connection(self.d_target, self.dq_diff, synapse=None)
            #nengo.Connection(env.dq, dq_diff, synapse=None, transform=-1)
    
            nengo.Connection(self.dq_diff, self.u, transform=Kd, synapse=None)
    
            intq_diff = nengo.Node(lambda t,x: x[1:] if x[0]>0 else x[1:]*0, size_in=size_in+1)

            nengo.Connection(self.target, intq_diff[1:], synapse=None)
            nengo.Connection(self.actual, intq_diff[1:], synapse=None, transform=-1)
            nengo.Connection(intq_diff, intq_diff[1:], synapse=0.1)
            
            nengo.Connection(intq_diff, self.u, transform=Ki, synapse=None)            
            
            self.active=nengo.Node(None, size_in=1)
            self.output = nengo.Node(lambda t,x: x[1:] if x[0]>0 else x[1:]*0, size_in=size_in+1)
            nengo.Connection(self.active, self.output[0], synapse=None)
            nengo.Connection(self.active, intq_diff[0], synapse=None)
            nengo.Connection(self.u, self.output[1:], synapse=None)
            
            

model = nengo.Network()
with model:
    
    clamp = Clamp(size_in=1)
    
    a = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    target = nengo.Node(0)
    nengo.Connection(target, clamp.target)
    nengo.Connection(a, clamp.actual)
    nengo.Connection(clamp.output, a)
    
    active = nengo.Node(0)
    nengo.Connection(active, clamp.active, synapse=None)
