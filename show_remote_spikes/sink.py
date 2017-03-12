import nengo
import numpy as np
import redis

import struct

r = redis.StrictRedis('127.0.0.1')

model = nengo.Network()
with model:

    def receive_spikes(t):
        msg = r.get('spikes')
        v = np.zeros(10)
        if len(msg) > 0:
            ii = struct.unpack('%dI' % (len(msg)/4), msg)
            v[[ii]] = 1000.0
        return v
    sink_node = nengo.Node(receive_spikes, size_in=0)
