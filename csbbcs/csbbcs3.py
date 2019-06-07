import nengo
model = nengo.Network()
with model:
    # create the neurons
    a = nengo.Ensemble(n_neurons=50, dimensions=1)

    # anything that's not neurons is a Node
    stim = nengo.Node([0])
    
    # make the connection
    nengo.Connection(stim, a)

    b = nengo.Ensemble(n_neurons=100, dimensions=1)

    def my_function(x):
        return x**2
    nengo.Connection(a, b, function=my_function)