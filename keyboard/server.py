import nengo
import udp

if not hasattr(udp, 'hack'):
    m = nengo.Network()
    with m:
        udp.hack = udp.UDPReceiver(9999, 2)


model = nengo.Network()
with model:
    model.nodes.append(udp.hack)

