import nengo
import numpy as np

np.random.seed(1)

class Environment(object):
    x = 0
    y = 0
    vx = np.random.uniform(-1, 1)
    vy = np.random.uniform(-1, 1)
    history = np.zeros((1000, 4))

    
    def update(self, t, predict):
        dt = 0.001
        self.vx += np.random.normal(0, 0.01)
        self.vy += np.random.normal(0, 0.01)
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

        self.history[1:] = self.history[:-1]
        self.history[0] = self.x, self.y, self.vx, self.vy

        path = []
        for i in range(0, 1000, 100):
            path.append('<circle cx={} cy={} r=1 style="fill:black"/>'.format(self.history[i,0]*100,self.history[i,1]*100))
        for i in range(10):
            path.append('<circle cx={} cy={} r=1 style="fill:yellow"/>'.format(predict[2*i]*100,predict[2*i+1]*100))

        Environment.update._nengo_html_ = '''
        <svg width=100% height=100% viewbox="-100 -100 200 200">
            <rect x=-100 y=-100 width=200 height=200 style="fill:green"/>
            <circle cx={} cy={} r=5 style="fill:white"/>
            {}           
        </svg>
        '''.format(self.history[-1,0]*100, self.history[-1,1]*100, ''.join(path))
            
        future = []
        for i in range(0, 1000, 100):
            future.extend(self.history[i, :2])
        return np.hstack([self.history[-1], future])


model = nengo.Network()
with model:
    env = Environment()
    for i in range(1000):
        env.update(0, np.zeros(20))
        
    env_node = nengo.Node(env.update, size_in=20)
    
    ens = nengo.Ensemble(n_neurons=1000, dimensions=4)
    nengo.Connection(env_node[:4], ens)
    
    prediction = nengo.Node(None, size_in=20)
    def predict(x):
        return np.zeros(20)
        x, y, vx, vy = x
        
        dt = 0.001
        r = np.zeros(20)
        for i in range(10):
            r[(9-i)*2:(9-i)*2+2] = [x,y]
            #r.extend([x, y])
            for j in range(100):
                x += vx*dt
                y += vy*dt
                while x>1:
                    x = 1-(x-1)
                    vx = -vx
                while x<-1:
                    x = -1-(x+1)
                    vx = -vx
                while y>1:
                    y = 1-(y-1)
                    vy = -vy
                while y<-1:
                    y = -1-(y+1)
                    vy = -vy

        
        return r
    conn = nengo.Connection(ens, prediction,
                            function=predict,
                            learning_rule_type=nengo.PES()
                            )
                            
    error = nengo.Node(None, size_in=20)
    nengo.Connection(prediction, error)
    nengo.Connection(env_node[4:], error, transform=-1)
    nengo.Connection(error, conn.learning_rule)
    
    nengo.Connection(prediction, env_node)
    
    
    