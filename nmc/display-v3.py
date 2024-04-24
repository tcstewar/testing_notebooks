import nengo

import numpy as np
class Plot1D(nengo.Node):
    def __init__(self, c_forward, c_feedback, domain=np.linspace(-1, 1, 100),
                 range=[-100,100]):
        assert c_feedback.size_out == 1
        assert c_forward.size_out == 1
        assert c_feedback.post_obj == c_forward.post_obj
        assert c_feedback.post_slice == c_forward.post_slice
        assert c_feedback.synapse == c_forward.synapse
        assert isinstance(c_feedback.synapse, nengo.synapses.Lowpass) 
        
        self.tau = c_feedback.synapse.tau
        self.p_forward = nengo.Probe(c_forward)
        self.c_feedback = c_feedback
        self.c_forward = c_forward
        self.ensemble = c_feedback.post_obj
        self.p_ensemble = nengo.Probe(self.ensemble)
        self.p_feedback = nengo.Probe(c_feedback)
        self.sim = None
        self.u = 0
        self.feedback_func = c_feedback.function
        self.feedback_dec = None
        self.domain = domain
        self.range = range
        S = 500
        W = 30
        T = 3
        F = 20
        self.svg_x = np.linspace(0, S, len(domain))
        
        
        template = f'''
            <svg width="100%%" height="100%%" viewbox="-{W} -{W} {S+2*W} {S+2*W}">
                %s
                <line x1={S/2} y1=0 x2={S/2} y2={S} stroke="#aaaaaa"/>
                <line x1=0 y1={S/2} x2={S} y2={S/2} stroke="#aaaaaa"/>
                <line x1=0 y1={S/2-T} x2=0 y2={S/2+T} stroke="#aaaaaa"/>
                <text x=0 y={S/2+T*3} font-size="{F}pt" text-anchor="middle">{self.domain[0]:g}</text>
                <line x1={S} y1={S/2-T} x2={S} y2={S/2+T} stroke="#aaaaaa"/>
                <text x={S} y={S/2+T*3} font-size="{F}pt" text-anchor="middle">{self.domain[-1]:g}</text>
                <line x1={S/2-T} y1=0 x2={S/2+T} y2=0 stroke="#aaaaaa"/>
                <text x={S/2+T} y=0 font-size="{F}pt" text-anchor="start" alignment-baseline="central">{self.range[1]:g}</text>
                <line x1={S/2-T} y1={S} x2={S/2+T} y2={S} stroke="#aaaaaa"/>
                <text x={S/2+T} y={S} font-size="{F}pt" text-anchor="start" alignment-baseline="central">{self.range[0]:g}</text>
                
            </svg>'''
        
        self.palette = ["#1c73b380", "black", "#d65e00",
                        "#cd79a7", "#f0e542", "#56b4ea"]
        self.widths = ["3pt", "1pt"]
        


        def plot(t, self=self):
            
            y = (self.feedback_func(self.domain) - self.domain)/self.tau + self.u/self.tau
            if self.sim is not None:
                x_now = self.sim.data[self.p_ensemble][-1]
                pts = np.zeros((self.ensemble.dimensions, len(self.domain)))
                pts[:] = x_now[:,None]
                pts[self.c_feedback.post_slice,:] = self.domain

                _, a = nengo.utils.ensemble.tuning_curves(self.ensemble, self.sim, pts.T)
                v = (a @ self.feedback_dec.T) [:,0]
                y2 = (v - self.domain + self.u)/self.tau
                y = np.array([y, y2])
            else:
                y = np.array([y])
            

            min_y = self.range[0]
            max_y = self.range[1]
            data = (-y - min_y) * S / (max_y - min_y)
            paths = []
            for i, row in enumerate(data):
                path = []
                for j, d in enumerate(row):
                    path.append('%1.0f %1.0f' % (self.svg_x[j], d))
                paths.append('<path d="M%s" fill="none" stroke-width="%s" stroke="%s"/>' %
                             ('L'.join(path),
                              self.widths[i],
                              self.palette[i % len(self.palette)]))

            if self.sim is not None:
                cx = self.sim.data[self.p_ensemble][-1][self.c_feedback.post_slice]
                cy1 = (self.feedback_func(cx)-cx+self.u)/self.tau

                cy1 = (-cy1[0] - min_y) * S / (max_y - min_y)
                index = np.searchsorted(domain, cx[0])
                if index >= len(self.svg_x):
                    index = len(self.svg_x)-1
                cx = self.svg_x[index]
                circ = f'<circle cx="{cx}" cy="{cy1}" r="15" fill="{self.palette[0]}"/>'
                paths.append(circ)

                cx = self.sim.data[self.p_ensemble][-1]
                _, a = nengo.utils.ensemble.tuning_curves(self.ensemble, self.sim, np.array([cx]))
                v = (a @ self.feedback_dec.T) [:,0][0]
                cy2 = (v - cx[self.c_feedback.post_slice] + self.u)/self.tau
                cy2 = (-cy2[0] - min_y) * S / (max_y - min_y)
                cx = self.svg_x[index]

                circ = f'<circle cx="{cx}" cy="{cy2}" r="10" fill="{self.palette[1]}"/>'
                paths.append(circ)
                
                
            
            plot._nengo_html_ = template % ''.join(paths)
            
        
        super(Plot1D, self).__init__(plot, size_in=0, size_out=0)
        self.output._nengo_html_ = plot(0.0)
    
    def update(self, sim):
        self.sim = sim
        if sim is None:
            return
        self.u = sim.data[self.p_forward][-1]
        #del sim.data[self.p_forward][:]        
        
        self.feedback_dec = sim.data[self.c_feedback].weights
        

model = nengo.Network()
with model:
    stim = nengo.Node([1,1])
    
    tau_synapse = 0.01
    tau_desired = 0.1
    
    ens1 = nengo.Ensemble(n_neurons=30, dimensions=3, 
                         neuron_type=nengo.LIFRate(), 
                         radius=2,
                         seed=1)
    c_feedback = nengo.Connection(ens1[2], ens1[2], function=lambda x: (1-tau_synapse/tau_desired)*x, synapse=tau_synapse)
    c_forward = nengo.Connection(ens1[:2], ens1[2], function=lambda x: (tau_synapse/tau_desired)*(x[0]*x[1]), synapse=tau_synapse)
    nengo.Connection(stim, ens1[:2])
    
    plt1 = Plot1D(c_forward, c_feedback, range=[-100,100], domain=np.linspace(-1.1, 1.1, 500))
    
    
    
    xw = 0.025
    xf = 0.1
    cx = 3/13
    ax = -cx/(3*xw**2)

    ens2 = nengo.Ensemble(n_neurons=30, dimensions=3, 
                         neuron_type=nengo.LIFRate(), 
                         radius=0.17,
                         seed=1,
                         )
                         
    tau = 10.0
    tau_synapse = 0.1
    c_feedback = nengo.Connection(ens2[2], ens2[2], function=lambda x: tau*tau_synapse*(cx*x+ax*x**3)+x, synapse=tau_synapse)
    c_forward = nengo.Connection(ens2[:2], ens2[2], function=lambda x: tau*tau_synapse*((x[0])*(x[1])/(xf)), synapse=tau_synapse)
    
    stim2 = nengo.Node([0.1, 0.1])
    nengo.Connection(stim2, ens2[:2], synapse=None)
    
    plt2 = Plot1D(c_forward, c_feedback, range=[-3,3], domain=np.linspace(-0.11, 0.11, 500))

    

def on_step(sim):
    plt1.update(sim)
    plt2.update(sim)
