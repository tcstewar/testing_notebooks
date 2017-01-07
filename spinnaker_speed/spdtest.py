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
            c = nengo.Connection(ens, output, function=lambda x: [0]*p.D_out,
                                 learning_rule_type=nengo.PES())


            def error_func(t):
                return [np.sin(t)] * p.D_out
            error = nengo.Node(error_func)
            nengo.Connection(error, c.learning_rule)

            self.error = error
            self.stim = stim
            self.output = output

        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)

        exec_freq = len(self.times) / float(p.T)

        error_send_freq = len(self.error._to_spinn_times) / float(p.T)
        stim_send_freq = len(self.stim._to_spinn_times) / float(p.T)
        output_read_freq = len(self.output._from_spinn_times) / float(p.T)

        return dict(exec_freq=exec_freq,
                    error_send_freq=error_send_freq,
                    stim_send_freq=stim_send_freq,
                    output_read_freq=output_read_freq,
                    )
