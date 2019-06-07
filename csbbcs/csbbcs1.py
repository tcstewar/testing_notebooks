import nengo
model = nengo.Network()
with model:
    # create the neurons
    a = nengo.Ensemble(n_neurons=50, dimensions=1)

    # anything that's not neurons is a Node
    stim = nengo.Node([0])
    
    # make the connection
    nengo.Connection(stim, a)
