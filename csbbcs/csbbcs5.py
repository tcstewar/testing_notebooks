import nengo
model = nengo.Network()
with model:
    a1 = nengo.Ensemble(n_neurons=50, dimensions=1)
    stim1 = nengo.Node([0])
    nengo.Connection(stim1, a1)
    a2 = nengo.Ensemble(n_neurons=50, dimensions=1)
    stim2 = nengo.Node([0])
    nengo.Connection(stim2, a2)
    b = nengo.Ensemble(n_neurons=150, dimensions=2,
                       radius=1.5)
    #def func1(a1):
    #    return a1, 0
    #nengo.Connection(a1, b, function=func1)
    nengo.Connection(a1, b[0])
    
    #def func2(a2):
    #    return 0, a2
    #nengo.Connection(a2, b, function=func2)
    nengo.Connection(a2, b[1])
    
    c = nengo.Ensemble(n_neurons=50, dimensions=1)
    def product(b):
        return b[0]*b[1]
    nengo.Connection(b, c, function=product)
    

