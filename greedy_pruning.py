from __future__ import print_function
import rmv_filters as rmv


def create_a_branch_from_original_net(weights, biases, epoch, layer_to_remove_filter_from, index_of_filter_to_remove):
    num_layers = len(weights)-1
    last_layer_index = num_layers-1
    num_fully_connected = weights[last_layer_index][0].shape[0]* 5 * 5
    number_of_filters_per_layer = []
    for i in range(num_layers):
        number_of_filters_per_layer.append(weights[i][0].shape[0])

    w_new = rmv.get_new_weights(weights, layer_to_remove_filter_from, index_of_filter_to_remove, epoch, number_of_filters_per_layer[layer_to_remove_filter_from], number_of_filters_per_layer[layer_to_remove_filter_from], last_layer_index, num_fully_connected)
    b_new = rmv.get_new_biases(biases, layer_to_remove_filter_from, index_of_filter_to_remove, epoch, number_of_filters_per_layer[layer_to_remove_filter_from])

    return w_new, b_new
