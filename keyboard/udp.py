import nengo
import socket
import time

class UDPSender(nengo.Node):
    def __init__(self, address, port, size_in, period=0.01):
        self.target = (address, port)
        self.period = period
        self.last_time = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        super(UDPSender, self).__init__(output=self.send, size_in=size_in)
    def send(self, t, x):
        now = time.time()
        if self.last_time is None or now > self.last_time + self.period:
            msg = ','.join(['%g' % xx for xx in x])
            self.socket.sendto(msg, self.target)
            self.last_time = now


class UDPReceiver(nengo.Node):
    def __init__(self, port, size_out):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('', port))
        self.port = port
        self.socket.setblocking(0)
        self.data = [0] * size_out
        super(UDPReceiver, self).__init__(self.receive,
                                          size_out=size_out)
    def receive(self, t):
        try:
            while True:  # empty the buffer
                data = self.socket.recv(4096)
                for i,d in enumerate(data.split(',')):
                    try:
                        self.data[i] = float(d)
                    except:
                        print 'Invalid UDP message (port=%d): %s' % (self.port,
                                                                     `data`)
        except socket.error:
            pass
        return self.data




if __name__ == '__main__':
    import numpy as np

    model = nengo.Network()
    with model:
        input = nengo.Node(np.sin)
        udp0 = UDPSender('localhost', 10000, 1)
        nengo.Connection(input, udp0)

        udp1 = UDPReceiver(10000, 1)
        def print_it(t, x):
            print t, x
        output = nengo.Node(print_it, size_in=1)
        nengo.Connection(udp1, output)

    sim = nengo.Simulator(model)
    sim.run(3.2)
