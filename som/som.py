import numpy as np
import nengo

class SelfOrganizingMap(nengo.Process):
    def __init__(self, weights, learning_rate=1e1, influence_sigma=1.5):
        self.weights = weights        
        self.learning_rate = learning_rate
        self.influence_sigma = influence_sigma
        
        super().__init__(default_size_in=weights.shape[2],
                         default_size_out=weights.shape[0]*weights.shape[1])
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        
        pos = np.array(np.meshgrid(np.arange(self.weights.shape[1]), 
                                   np.arange(self.weights.shape[0])))
        
        def step_som(t, x, w=self.weights, pos=pos, 
                     sigma=self.influence_sigma,
                     learning_rate=self.learning_rate):
            diff = np.sum((w - x[None,None,:])**2, axis=2)
            
            best = np.argmin(diff)
            best = np.array([best % diff.shape[1], best // diff.shape[1]])
            #assert diff[best[1],best[0]] == np.min(diff)
            
            dist = np.sum((pos - best[:,None,None])**2, axis=0)
            influence = np.exp(-dist/(2*sigma**2))
            
            w += learning_rate * dt * influence[:,:,None] * (x - w)
            
            return influence.flatten()
            
        return step_som

data = np.random.uniform(0, 1, (100,3))
w = np.random.uniform(0, 1, (10, 12, 3))

model = nengo.Network()
with model:
    stim = nengo.Node(nengo.processes.PresentInput(data, presentation_time=0.001))

    som = nengo.Node(SelfOrganizingMap(w))
    nengo.Connection(stim, som, synapse=None)

    import base64
    from PIL import Image
    try:
        from cStringIO import StringIO
    except ImportError:
        from io import BytesIO as StringIO
    def plot_func(t):
        values = np.clip(w*255, 0, 255)
        values = values.astype('uint8')
        png = Image.fromarray(values)
        buffer = StringIO()
        png.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue())
        plot_func._nengo_html_ = '''
        <svg width="100%%" height="100%%" viewbox="0 0 100 100">
        <image width="100%%" height="100%%" image-rendering="pixelated"
               xlink:href="data:image/png;base64,%s">
        </svg>''' % img_str.decode('utf-8')
    plot = nengo.Node(plot_func)
