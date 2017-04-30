import nengo
import nengo_spinnaker
import numpy as np

model = nengo.Network()
with model:
    def stim_func(t):
        return np.sin(t), np.cos(t)
    stim = nengo.Node(stim_func)

    ea = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=2)
    nengo.Connection(stim, ea.input)

sim = nengo_spinnaker.Simulator(model)
sim.run(1)