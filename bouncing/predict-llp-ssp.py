import nengo
import numpy as np
import learn_dyn_sys
import nengo_spa as spa

import scipy.special

## Convert sparsity parameter to neuron bias/intercept
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

D = 64

vocab = spa.Vocabulary(D)
vocab.add('X', vocab.algebra.create_vector(D, {"positive", "unitary"}))
vocab.add('Y', vocab.algebra.create_vector(D, {"positive", "unitary"}))
vocab.add('VX', vocab.algebra.create_vector(D, {"positive", "unitary"}))
vocab.add('VY', vocab.algebra.create_vector(D, {"positive", "unitary"}))
X = vocab.parse('X')
Y = vocab.parse('Y')
VX = vocab.parse('VX')
VY = vocab.parse('VY')

class Environment(object):
    x = 0
    y = 1
    vx = 2
    vy = 2
    #vx = np.random.uniform(-1, 1)
    #vy = np.random.uniform(-1, 1)

    
    def update(self, t, predict):
        dt = 0.001
        #self.vx += np.random.normal(0, 0.01)
        #self.vy += np.random.normal(0, 0.01)
        #v = np.sqrt(self.vx**2+self.vy**2)
        #self.vx /= v
        #self.vy /= v
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        while self.x > 1:
            self.x = 1 - (self.x-1)
            self.vx = -self.vx
        while self.x < -1:
            self.x = -1 - (self.x+1)
            self.vx = -self.vx
        while self.y > 1:
            self.y = 1 - (self.y-1)
            self.vy = -self.vy
        while self.y < -1:
            self.y = -1 - (self.y+1)
            self.vy = -self.vy

        path = []
        #for i in range(0, 1000, 100):
        #    path.append('<circle cx={} cy={} r=1 style="fill:black"/>'.format(self.history[i,0]*100,self.history[i,1]*100))
        for i in range(10):
            path.append('<circle cx={} cy={} r=1 style="fill:yellow"/>'.format(predict[2*i]*100,predict[2*i+1]*100))

        Environment.update._nengo_html_ = '''
        <svg width=100% height=100% viewbox="-100 -100 200 200">
            <rect x=-100 y=-100 width=200 height=200 style="fill:green"/>
            <circle cx={} cy={} r=5 style="fill:white"/>
            {}           
        </svg>
        '''.format(self.x*100, self.y*100, ''.join(path))
            
        return self.x, self.y, self.vx, self.vy


q = 10
N = 1000

model = nengo.Network()
with model:
    env = Environment()

    
    scales = [2, 2, 5, 5]
    convert = nengo.Node(lambda t, x: (X**(x[0]*scales[0])*Y**(x[1]*scales[1])*VX**(x[2]*scales[2])*VY**(x[3]*scales[3])).v, size_in=4)


        
    llp = learn_dyn_sys.LearnDynSys(size_c=D, size_z=2, q=q, theta=0.5, 
                                    n_neurons=N, learning_rate=1e-4,
                                    intercepts=nengo.dists.CosineSimilarity(D+2),
                                    #intercepts=nengo.dists.Choice([sparsity_to_x_intercept(D, 0.15)]),
                                    #encoders=[convert.output(0,x) for x in np.random.uniform(-1,1,(N, 4))],
                                    )
    env_node = nengo.Node(env.update, size_in=20)
    
    nengo.Connection(env_node[:2], llp.z, synapse=None)
    nengo.Connection(env_node[:4], convert, synapse=None)
    nengo.Connection(convert, llp.c, synapse=None)
    
    pred_x = nengo.Node(size_in=10)
    nengo.Connection(llp.Z[:q], pred_x, transform=llp.get_weights_for_delays(np.linspace(0, 1, 10)), synapse=None)
    pred_y = nengo.Node(size_in=10)
    nengo.Connection(llp.Z[q:], pred_y, transform=llp.get_weights_for_delays(np.linspace(0, 1, 10)), synapse=None)
    
    nengo.Connection(pred_x, env_node[::2], synapse=0)
    nengo.Connection(pred_y, env_node[1::2], synapse=0)
    
    
    