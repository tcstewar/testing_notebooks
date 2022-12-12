import nengo
import numpy as np
import learn_dyn_sys

class Environment(object):
    x = 0
    y = 1
    vx = 2#np.random.uniform(0, 2)
    vy = 2#np.random.uniform(0, 2)

    
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

model = nengo.Network()
with model:
    env = Environment()


    c_lmu = nengo.Node(learn_dyn_sys.LMU(q=q, theta=0.5, size_in=2))
    
    

        
    llp = learn_dyn_sys.LearnDynSys(size_c=2*q, size_z=2, q=q, theta=0.5, 
                                    n_neurons=1000, learning_rate=1e-4,
                                    radius=2,
                                    intercepts=nengo.dists.CosineSimilarity(2*q+2))
    env_node = nengo.Node(env.update, size_in=20)
    
    nengo.Connection(env_node[:2], llp.z, synapse=None)
    nengo.Connection(env_node[:2], c_lmu, synapse=None)
    nengo.Connection(c_lmu, llp.c, synapse=None)
    
    pred_x = nengo.Node(size_in=10)
    nengo.Connection(llp.Z[:q], pred_x, transform=llp.get_weights_for_delays(np.linspace(0, 1, 10)), synapse=None)
    pred_y = nengo.Node(size_in=10)
    nengo.Connection(llp.Z[q:], pred_y, transform=llp.get_weights_for_delays(np.linspace(0, 1, 10)), synapse=None)
    
    nengo.Connection(pred_x, env_node[::2], synapse=0)
    nengo.Connection(pred_y, env_node[1::2], synapse=0)
    
    
    