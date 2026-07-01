import nengo
import numpy as np

class Clamp(nengo.Network):
    def __init__(self, size_in, Kp=0.5, Kd=0.01, Ki=1):
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
    
    clamp_x1 = Clamp(size_in=1)
    clamp_x2 = Clamp(size_in=1)
    clamp_y = Clamp(size_in=1)
    
    x = nengo.Ensemble(n_neurons=100, dimensions=2)
    y = nengo.Ensemble(n_neurons=50, dimensions=1)
    
    nengo.Connection(x[0], clamp_x1.actual)
    nengo.Connection(clamp_x1.output, x[0])
    nengo.Connection(x[1], clamp_x2.actual)
    nengo.Connection(clamp_x2.output, x[1])
    
    nengo.Connection(y, clamp_y.actual)
    nengo.Connection(clamp_y.output, y)
    
    mix = nengo.Ensemble(n_neurons=1000, dimensions=3, radius=2)
    nengo.Connection(x, mix[:2])
    nengo.Connection(y, mix[2:])
    nengo.Connection(mix[:2],x)
    nengo.Connection(mix[2:],y)

        
    x_pts = []
    y_pts = []
    pts = nengo.dists.UniformHypersphere(surface=False).sample(d=2, n=100)
    x_pts.extend(pts*0.25+[0.5,0.5])
    y_pts.extend(np.ones(100))
    x_pts.extend(pts*0.25+[-0.5,-0.5])
    y_pts.extend(np.ones(100))
    x_pts.extend(pts*0.25+[-0.5,+0.5])
    y_pts.extend(-np.ones(100))
    x_pts.extend(pts*0.25+[+0.5,-0.5])
    y_pts.extend(-np.ones(100))
    x_pts = np.array(x_pts)
    
    def relate(x):
        dists = (x[0]-x_pts[:,0])**2 + (x[1]-x_pts[:,1])**2 + (x[2]-y_pts)**2
        index = np.argmin(dists)
        return x_pts[index,0]-x[0], x_pts[index,1]-x[1], y_pts[index]-x[2]
    Km=2
    nengo.Connection(mix, mix, function=relate, transform=0.1*Km, synapse=0.1)
