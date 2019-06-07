import nengo
model = nengo.Network()
with model:
    
    food = nengo.Ensemble(n_neurons=200, dimensions=2)
    stim_food = nengo.Node([0,0])
    nengo.Connection(stim_food, food)
    
    motor = nengo.Ensemble(n_neurons=200, dimensions=2)
    #nengo.Connection(food, motor)
    stim_light = nengo.Node([0])
    light = nengo.Ensemble(n_neurons=100, dimensions=1)
    nengo.Connection(stim_light, light)
    
    do_food = nengo.Ensemble(n_neurons=300, dimensions=3,
                             #neuron_type=nengo.Direct()
                             )
    nengo.Connection(light, do_food[0])
    nengo.Connection(food[0], do_food[1])
    nengo.Connection(food[1], do_food[2])
    
    def food_func(x):
        light, food_x, food_y = x
        if light < 0:
            return food_x, food_y
        else:
            return 0, 0
    nengo.Connection(do_food, motor,
                     function=food_func)
                     
    pos = nengo.Ensemble(n_neurons=1000, dimensions=2)
    nengo.Connection(pos, pos, synapse=0.2)
    def scale_motor(m):
        return m*0.2
    nengo.Connection(motor, pos, function=scale_motor)
    
    do_home = nengo.Ensemble(n_neurons=300, dimensions=3,
                             #neuron_type=nengo.Direct()
                             )
    nengo.Connection(light, do_home[0])
    nengo.Connection(pos[0], do_home[1])
    nengo.Connection(pos[1], do_home[2])
    
    def home_func(x):
        light, pos_x, pos_y = x
        if light < 0:
            return 0, 0
        else:
            return -pos_x, -pos_y
    nengo.Connection(do_home, motor,
                     function=home_func)
    
    