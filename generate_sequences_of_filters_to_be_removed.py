from __future__ import print_function
import pickle
import load_parameters as lp

path_to_weights = 'parameters_original_net/parameters_epoch_10'
num_layers = 4
epoch = 10
total_number_of_filters = 0
number_of_filters_per_layer = []

w = lp.load_weights(path_to_weights, num_layers, epoch)
b = lp.load_biases(path_to_weights, num_layers, epoch)

for i in range(num_layers):
    total_number_of_filters+=w[i][0].shape[0]
    number_of_filters_per_layer.append(w[i][0].shape[0])

print("The total number of filters in the network is: ", total_number_of_filters)
print(number_of_filters_per_layer)

#now generate the sequence of filters to be removed

sequences_of_filters_to_remove = [[[] for j in range(num_layers)] for i in range(total_number_of_filters)]
print(len(sequences_of_filters_to_remove))
count=0
for layer in range(num_layers):
    for i in range(number_of_filters_per_layer[layer]):
        sequences_of_filters_to_remove[count][layer].append(i)
        count+=1
print(sequences_of_filters_to_remove)
print(count)

fw = open('filters_to_remove', 'wb')
pickle.dump(sequences_of_filters_to_remove, fw)
fw.close()
