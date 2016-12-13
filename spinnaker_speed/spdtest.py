import nengo
import numpy as np
import pytry

class Speed(pytry.NengoTrial):
    def params(self):
        self.param('input dims', D_in=12)
        self.param('output dims', D_out=6)
        self.param('n_neurons', n_neurons=500)
        self.param('time to run', T=10)

    def model(self, p):
        model = nengo.Network()
        with model:
            def stim_func(t):
                return [np.sin(t)] * p.D_in
            stim = nengo.Node(stim_func)

            ens = nengo.Ensemble(n_neurons=p.n_neurons, dimensions=p.D_in)

            self.times = []
            def output_func(t, x):
                self.times.append(t)
            output = nengo.Node(output_func, size_in=p.D_out)

            nengo.Connection(stim, ens)
            nengo.Connection(ens, output, function=lambda x: [0]*p.D_out)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)

        freq = len(self.times) / float(p.T)

        return dict(freq=freq, first_time=self.times[0])
