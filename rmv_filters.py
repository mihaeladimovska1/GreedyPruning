import os
print(os.environ['XDG_RUNTIME_DIR'])
import copy
import numpy as np


def delete_flattened_filter(weights, layer_ind, filter_index, last_epoch, num_out_channels):
    '''Delete the OUTPUT channel corresponding to filter_index from layer layer_ind'''

    all_filters = [i for i in range(num_out_channels)]
    staying_filters = [x for x in all_filters if x not in filter_index]
    layer_weights_mod = weights[layer_ind][last_epoch]
    layer_weights_mod = layer_weights_mod[staying_filters, :, :, :]#torch.cat([layer_weights_mod[:filter_index[i], :, :, :], layer_weights_mod[filter_index[i]+1:, :, :, :]])
    return layer_weights_mod


def propagate_deleted_filter(weights, layer_ind, filter_index, last_epoch, num_in_channels, last_layer_index, num_fully_connected):
    '''Delete the INPUT channels corresponding to filter_index from layer layer_ind'''

    #for propagating change to fully connected layer

    if layer_ind == last_layer_index+1:
        all_filters = [i for i in range(num_fully_connected)]
        to_be_removed = []
        for i in range(len(filter_index)):
            to_be_removed.append([j for j in range(filter_index[i]*25, (filter_index[i]+1)*25)])

        to_be_removed = np.asarray(to_be_removed)
        to_be_removed = np.ndarray.flatten(to_be_removed)
        staying_filters = [x for x in all_filters if x not in to_be_removed]
        layer_weights_mod = weights[layer_ind][last_epoch]
        layer_weights_mod = layer_weights_mod[:, staying_filters]
    else:
        all_filters = [i for i in range(num_in_channels)]
        staying_filters = [x for x in all_filters if x not in filter_index]
        layer_weights_mod = weights[layer_ind][last_epoch]
        layer_weights_mod = layer_weights_mod[:, staying_filters, :, :]

    return layer_weights_mod



def get_new_weights(weights, layer_ind, filter_index, last_epoch, num_in_channels, num_out_channels, last_layer_index, num_fully_connected):
    weights_new = copy.deepcopy(weights)
    #remove filters filter_index from a layer
    w_l = delete_flattened_filter(weights, layer_ind, filter_index, last_epoch, num_out_channels)
    #propagate change to next layer
    w_l_next = propagate_deleted_filter(weights, layer_ind+1, filter_index, last_epoch, num_in_channels, last_layer_index, num_fully_connected)

    #modify weights
    weights_new[layer_ind][last_epoch] = w_l
    weights_new[layer_ind+1][last_epoch] = w_l_next
    return  weights_new


def delete_biases(biases, layer_ind, filter_index, epoch, num_out_channels):
    '''Delete the OUTPUT channel corresponding to filter_index from layer layer_ind'''
    all_filters = [i for i in range(num_out_channels)]
    staying_filters = [x for x in all_filters if x not in filter_index]
    layer_weights_mod = biases[layer_ind][epoch]
    layer_weights_mod = layer_weights_mod[staying_filters]#torch.cat([layer_weights_mod[:filter_index[i], :, :, :], layer_weights_mod[filter_index[i]+1:, :, :, :]])
    return layer_weights_mod


def get_new_biases(biases, layer_to_remove_filter_from, index_of_filter_to_remove, epoch, num_output_channels):
    biases_new = biases
    # remove filters filter_index from a layer
    b_l = delete_biases(biases, layer_to_remove_filter_from, index_of_filter_to_remove, epoch, num_output_channels)
    # modify weights
    biases_new[layer_to_remove_filter_from][epoch] = b_l
    return biases_new