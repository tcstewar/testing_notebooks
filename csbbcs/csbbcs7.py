import nengo
model = nengo.Network()
with model:
    stim = nengo.Node([0])
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    b = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(a, b, synapse=0.005)
    nengo.Connection(stim, a)
    
    def recurrent(b):
        return 1+b
    #nengo.Connection(b, b, function=recurrent)
    