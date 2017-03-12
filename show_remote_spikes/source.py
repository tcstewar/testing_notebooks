import nengo
import numpy as np
import redis

import struct

r = redis.StrictRedis('127.0.0.1')

model = nengo.Network()
with model:
    stim = nengo.Node(np.sin)
    
    
    ens = nengo.Ensemble(n_neurons=500, dimensions=1)
    nengo.Connection(stim, ens)
    
    
    def send_spikes(t, x):
        v = np.where(x!=0)[0]
        if len(v) > 0:
            msg = struct.pack('%dI' % len(v), *v)
        else:
            msg = ''
        r.set('spikes', msg)
    source_node = nengo.Node(send_spikes, size_in=10)
    nengo.Connection(ens.neurons[:10], source_node, synapse=None)
    