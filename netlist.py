import nengo
import numpy as np

def extract_ens_info(ens):
    return dict(
        size_in=ens.dimensions,
        n_neurons=ens.n_neurons,
        functions=[],
    )

def extract_node_info(node):
    return dict(
        offboard = node.output is not None,
        size_in = node.size_in,
        size_out = node.size_out,
        n_multiplies = 0,
        n_filters = 0,
        memory = 0,
        transforms = [],
        filters = [],
        size_outs = [],
    )

def extract_conn_info(conn):
    if conn.function is None:
        func = None
        pre_indices = np.arange(conn.pre_obj.size_out)[conn.pre_slice]
    else:
        func = id(conn.function)
        pre_indices=np.arange(conn.size_mid),
    return dict(
        pre=id(conn.pre_obj),
        post=id(conn.post_obj),
        func=func,
        pre_indices=pre_indices,
        post_indices=np.arange(conn.post_obj.size_in)[conn.post_slice],
        filter=None if conn.synapse is None else conn.synapse.tau,
        transform=conn.transform,
    
    )

def compute_graph(model):
    ensembles = {}
    nodes = {}
    connections = []

    with model:
        dummy_net = nengo.Network()
    for p in model.probes:
        with dummy_net:
            n = nengo.Node(lambda t, x: None, size_in=p.target.size_out)
            nengo.Connection(p.target, n, synapse=p.synapse)
    
    for ens in model.all_ensembles:
        ident = id(ens)
        ensembles[ident] = extract_ens_info(ens)
        
    for node in model.all_nodes:
        ident = id(node)
        nodes[ident] = extract_node_info(node)
        
    for c in model.all_connections:        
        if (isinstance(c.pre_obj, (nengo.Node, nengo.Ensemble)) and
            isinstance(c.post_obj, (nengo.Node, nengo.Ensemble))):                
                info = extract_conn_info(c)
                connections.append(info)
                if isinstance(c.pre_obj, nengo.Ensemble):
                    e = ensembles[id(c.pre_obj)]
                    f = (info['func'], c.size_mid)
                    if f not in e['functions']:
                        e['functions'].append(f)
                        
    for e in ensembles.values():
        e['size_out'] = sum(x[1] for x in e['functions'])

    model.networks.remove(dummy_net)
        
    return ensembles, nodes, connections


def simplify_conns(ensembles, nodes, conns):
    new_nodes = {}
    for c in conns[:]:
        if c['pre'] in ensembles:
            if (c['filter'] is not None or
                len(c['transform'].shape)>0 or
                c['transform'] != 1.0):
                
                    ens = ensembles[c['pre']]
                    
                    if c['pre'] not in new_nodes:
                        # need to insert a new node
                        node = dict(
                            offboard = False,
                            size_in = ens['size_out'],
                            size_out = ens['size_out'],
                            n_multiplies = 0,
                            memory = 0,
                            n_filters = 0,
                            transforms = [],
                            filters = [],
                            size_outs = [],
                        )
                        new_nodes[c['pre']] = node
                        nodes[id(node)] = node
                        conn = dict(
                                pre=c['pre'],
                                post=id(node),
                                func=None,
                                pre_indices=np.arange(ens['size_out']),
                                post_indices=np.arange(ens['size_out']),
                                filter=None,
                                transform=np.array(1.0),

                            )
                        conns.append(conn)
                    else:
                        node = new_nodes[c['pre']]
                        
                    c['pre'] = id(node)

                    
                    
    for c in conns:        
        if c['pre'] in nodes:
            nodes[c['pre']]['transforms'].append(c['transform'])
            nodes[c['pre']]['filters'].append(c['filter'])
            nodes[c['pre']]['size_outs'].append(len(c['post_indices']))
            c['transform'] = np.array(1)
            c['filter'] = None
            c['pre_indices'] = c['post_indices']
    for n in nodes.values():
        if len(n['transforms']) > 0:
            n['size_out'] = sum(n['size_outs'])            
        for i, t in enumerate(n['transforms']):
            if len(t.shape)==0:
                if t == 0.0:
                    pass
                elif t != 1.0:
                    n['memory'] += 1
                    n['n_multiplies'] += n['size_outs'][i]
            else:
                n['n_multiplies'] += t.shape[0]*t.shape[1]
                n['memory'] += t.shape[0]*t.shape[1]
        for i, f in enumerate(n['filters']):
            if f != None:
                n['n_filters'] += n['size_outs'][i]
                
        
                    


def plot_graph(ensembles, nodes, conns, size=(8,5)):
    import graphviz

    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'LR'
    dot.graph_attr['size'] = '%g,%g' % size
    
    for ident, ens in ensembles.items():
        dot.node(str(ident), '%d->%d\n%d' % (ens['size_in'], ens['size_out'], ens['n_neurons']))
    for ident, node in nodes.items():        
        if node['offboard']:
            dot.node(str(ident), '%d->%d' % (node['size_in'], node['size_out']), peripheries='2', shape='square')
        else:
            shape = 'square'
            if node['n_multiplies'] > 0 or node['n_filters'] > 0:
                label = '%d->%d' % (node['size_in'], node['size_out'])
            else:
                assert node['size_in'] == node['size_out']
                label = '%d' % node['size_in']
                shape = 'diamond'
                    
            if node['n_multiplies'] > 0:
                label = '%s\nM: %d' % (label, node['n_multiplies'])
            if node['n_filters'] > 0:
                label = '%s\nF: %d' % (label, node['n_filters'])

            dot.node(str(ident), label, shape=shape)
    for c in conns:
        label = '%d->%d' % (len(c['pre_indices']), len(c['post_indices']))
        t = c['transform']
        if len(t.shape) == 0:
            if t != 1.0:
                label += '\n*(1)'
            else:
                assert len(c['pre_indices']) == len(c['post_indices'])
                label = '%d' % len(c['pre_indices'])
        else:
            label += '\n*(%dx%d)'% t.shape
        if c['filter'] is not None:
            label += '\nh(t)'
        dot.edge(str(c['pre']), str(c['post']), label=label)
    
    return dot
    

def calc_stats(ensembles, nodes, conns):
    memory = 0
    messages = 0
    neurons = 0
    values = 0
    for e in ensembles.values():
        memory += e['n_neurons'] * e['size_in']
        memory += e['n_neurons'] * e['size_out']
        neurons += e['n_neurons']
    for n in nodes.values():
        memory += n['memory']
        values += n['size_out']
    for c in conns:
        messages += len(c['pre_indices'])
        t = c['transform']
        if len(t.shape)==0:
            if t != 1.0:
                memory += 1
        else:
            memory += t.shape[0]*t.shape[1]

        
    return dict(
        memory=memory,
        messages=messages,
        neurons=neurons,
        values=values,
    )
        
    
    
    
    
