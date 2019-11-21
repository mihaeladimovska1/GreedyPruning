import pickle

def load_weights(path_to_weights, num_layers, epoch):
    '''load weights of all the tensors for a given experiment
        each weight is of shape: [layer][epoch_num][out_channels][in_channels][Y: spatial][X: spatial]'''

    w = [[] for i in range(num_layers)]
    for i in range(1, num_layers+1):
        for j in [epoch]:
            w_exp = pickle.load(open(path_to_weights, 'rb'))
            w[i-1].append(w_exp[i-1][0].detach())
    return w


def load_biases(path_to_weights, num_layers, epoch):
    '''load weights of all the tensors for a given experiment
        each weight is of shape: [layer][epoch_num][out_channels][in_channels][Y: spatial][X: spatial]'''
    b = [[] for i in range(num_layers)]
    for i in range(1, num_layers + 1):
        for j in [epoch]:
            b_exp = pickle.load(open(path_to_weights, 'rb'))
            b[i - 1].append(b_exp[i - 1][1].detach())
    return b

