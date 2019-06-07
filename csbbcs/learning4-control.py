import grid

mymap="""
#########
#       #
#       #
#   ##  #
#   ##  #
#       #
#########

"""

class Cell(grid.Cell):
    def color(self):
        return 'black' if self.wall else None
    def load(self, char):
        if char == '#':
            self.wall = True

world = grid.World(Cell, map=mymap, directions=4)

body = grid.ContinuousAgent()
world.add(body, x=1, y=3, dir=2)

import nengo
import numpy as np    

def move(t, x):
    speed, rotation = x
    dt = 0.001
    max_speed = 20.0
    max_rotate = 10.0
    body.turn(rotation * dt * max_rotate)
    success = body.go_forward(speed * dt * max_speed)
    if not success: #Hit a wall
        return -1
    else:
        return speed

model = nengo.Network("Simple RL", seed=2)
with model:
    env = grid.GridNode(world, dt=0.005)
    
    #set up node to project movement commands to
    movement_node = nengo.Node(move, size_in=2, label='reward')
    movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)    
    nengo.Connection(movement, movement_node)

    def detect(t):
        #put sensors at -45 0 45 compared to facing direction
        angles = (np.linspace(-0.5, 0.5, 5) + body.dir ) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    stim_radar = nengo.Node(detect)

    #set up low fidelity sensors; noise might help exploration
    radar = nengo.Ensemble(n_neurons=50, dimensions=5, radius=4)
    nengo.Connection(stim_radar, radar)
    
    #set up BG to allow 3 actions (left/fwd/right)
    bg = nengo.networks.actionselection.BasalGanglia(3)
    thal = nengo.networks.actionselection.Thalamus(3)
    nengo.Connection(bg.output, thal.input)
    
    #start with a kind of random selection process, but like going fwd most
    def u_fwd(x):
        return 0.8
    def u_left(x):
        return 0.6
    def u_right(x):
        return 0.7

    conn_fwd = nengo.Connection(radar, bg.input[0], function=u_fwd, learning_rule_type=nengo.PES())
    conn_left = nengo.Connection(radar, bg.input[1], function=u_left, learning_rule_type=nengo.PES())
    conn_right = nengo.Connection(radar, bg.input[2], function=u_right, learning_rule_type=nengo.PES())
        
    nengo.Connection(thal.output[0], movement, transform=[[1],[0]])
    nengo.Connection(thal.output[1], movement, transform=[[0],[1]])
    nengo.Connection(thal.output[2], movement, transform=[[0],[-1]])
    
    errors = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=3)
    nengo.Connection(movement_node, errors.input, transform=-np.ones((3,1)))
    #inhibit learning for actions not currently chosen (recall BG is high for non-chosen actions)
    nengo.Connection(bg.output[0], errors.ensembles[0].neurons, transform=np.ones((50,1))*4)    
    nengo.Connection(bg.output[1], errors.ensembles[1].neurons, transform=np.ones((50,1))*4)    
    nengo.Connection(bg.output[2], errors.ensembles[2].neurons, transform=np.ones((50,1))*4)    
    nengo.Connection(bg.input, errors.input, transform=1)
    
    nengo.Connection(errors.ensembles[0], conn_fwd.learning_rule)
    nengo.Connection(errors.ensembles[1], conn_left.learning_rule)
    nengo.Connection(errors.ensembles[2], conn_right.learning_rule)