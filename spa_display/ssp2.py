import nengo
import nengo_spa as spa

D = 128

vocab = spa.Vocabulary(D)
vocab.add('X', vocab.algebra.create_vector(D, {"positive", "unitary"}))
vocab.add('Y', vocab.algebra.create_vector(D, {"positive", "unitary"}))
X = vocab.parse('X')
Y = vocab.parse('Y')


import nengo
import numpy as np
import base64

from io import BytesIO as StringIO
import PIL
from PIL import Image
class PlotSSP(nengo.Node):
    def __init__(self, X, Y, x_vals, y_vals):
        self.X = X
        self.Y = Y
        self.recalc_decoder(x_vals, y_vals)

        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                {img}
                <text x="5" y="50" style="fill:yellow;font-size:5px">{x_min}</text>
                <text x="95" y="50" text-anchor="end" style="fill:yellow;font-size:5px">{x_max}</text>
                <text x="50" y="5" text-anchor="middle" style="fill:yellow;font-size:5px">{y_min}</text>
                <text x="50" y="95" text-anchor="middle" style="fill:yellow;font-size:5px">{y_max}</text>
            </svg>'''
            

        def plot(t, x):
            y = np.dot(self.decoder, x)
            y = np.clip(y * 255, 0, 255)
            y = y.astype('uint8')

            png = Image.fromarray(y[:,:])
            buffer = StringIO()
            png.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                
            img = '''<image width="100%%" height="100%%"
                      xlink:href="data:image/png;base64,%s" 
                      style="image-rendering: pixelated;"/>
                  ''' % img_str

            plot._nengo_html_ = template.format(img=img, x_min=self.x_vals[0],
                                                           x_max=self.x_vals[-1],
                                                           y_min=self.y_vals[0],
                                                           y_max=self.y_vals[-1])

        super().__init__(plot, size_in=len(X), size_out=0)
        self.output._nengo_html_ = template.format(img='', x_min=self.x_vals[0],
                                                           x_max=self.x_vals[-1],
                                                           y_min=self.y_vals[0],
                                                           y_max=self.y_vals[-1])

    def recalc_decoder(self, x_vals, y_vals):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.decoder = np.zeros((len(y_vals), len(x_vals), len(self.X)))
        Xs = [self.X**x for x in x_vals]
        Ys = [self.Y**y for y in y_vals]
        for i, x in enumerate(Xs):
            for j, y in enumerate(Ys):
                self.decoder[j,i,:] = (x*y).v

    def move_plot(self, keys_pressed):
        if 'w' in keys_pressed:
            S = plt.y_vals[-1]-plt.y_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals, y_vals=plt.y_vals-S/4)
        if 's' in keys_pressed:
            S = plt.y_vals[-1]-plt.y_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals, y_vals=plt.y_vals+S/4)
        if 'a' in keys_pressed:
            S = plt.x_vals[-1]-plt.x_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals-S/4, y_vals=plt.y_vals)
        if 'd' in keys_pressed:
            S = plt.x_vals[-1]-plt.x_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals+S/4, y_vals=plt.y_vals)
        if 'q' in keys_pressed:
            Xs = plt.x_vals*2
            Ys = plt.y_vals*2
            Xs += -Xs[len(Xs)//2] + plt.x_vals[len(Xs)//2]
            Ys += -Ys[len(Ys)//2] + plt.y_vals[len(Ys)//2]
            plt.recalc_decoder(x_vals=Xs, y_vals=Ys)
        if 'e' in keys_pressed:
            Xs = plt.x_vals/2
            Ys = plt.y_vals/2
            Xs += -Xs[len(Xs)//2] + plt.x_vals[len(Xs)//2]
            Ys += -Ys[len(Ys)//2] + plt.y_vals[len(Ys)//2]
            plt.recalc_decoder(x_vals=Xs, y_vals=Ys)
        if 'z' in keys_pressed:
            plt.recalc_decoder(x_vals=np.linspace(-5,5,R), y_vals=np.linspace(-5,5,R))        


R = 101
model = spa.Network()
with model:
    stim_xy = nengo.Node([0,0])
    scale = nengo.Node(1)
        
    stim = nengo.Node(lambda t, x: x[2]*(X**x[0] * Y**x[1]).v, size_in=3)
    nengo.Connection(stim_xy, stim[:2])
    nengo.Connection(scale, stim[2])
    
        
    #cc = nengo.networks.CircularConvolution(n_neurons=50, dimensions=D)
    state = spa.State(D, feedback=1)
    init_node = nengo.Node(lambda t: (X**0 * Y**0).v if t<0.1 else np.zeros(D))
    nengo.Connection(stim, state.input, synapse=0.1)    
    

    plt = PlotSSP(X, Y, x_vals=np.linspace(-5, 5, R),
                        y_vals=np.linspace(-5, 5, R))
    nengo.Connection(state.output, plt)
    
#for ens in model.all_ensembles:
#    ens.neuron_type=nengo.Direct()

def on_step(sim):
    plt.move_plot(__page__.keys_pressed)