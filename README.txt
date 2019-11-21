This is a code showcasing the algorithm presented in Dimovska, Mihaela, and Travis Johnston. "A Novel Pruning Method for Convolutional Neural Networks Based off Identifying Critical Filters." Proceedings of the Practice and Experience in Advanced Research Computing on Rise of the Machines (learning). ACM, 2019.
If using any parts of this code, please cite the paper mentioned above. 

This is an example of how to remove one filter at a time from a CNN, creating a branch Bi of the original network,
and record the accuracy per class for that branch.
The file mnist_original_net_training.py is just training a network from scratch and extracting the parameters of the trained network.
The file load_parameters.py contains methods that load the saved weights and biases of the (original) network.
The file rmv_filters.py contains the main methods that remove a filter from a layer and propagate the change to the next layer,
as the next layer filters should not convolve with the removed filter.
The file greedy_pruning.py basically calls the necessary methods from rmv_filer.py to remove a given filter.

The file main.py is where the whole logic of the greedy pruning method is connected:
    In main.py we:
1) Load the original network parameters
2) Load the sequence of filters that need to be removed (note: one can specify any filter to be removed,
as long as the sequence is a list of lists of the kind:
    [ [list of filters to remove from 1st layer], [list of filters to remove from 2nd layer],  ...].
    As we are doing greedy filter pruning, an element of the filters_to_be_removed list will be a list of
    length the number of convolutional layers, and each of the elements will be a list that is either empty or contains one number
    (as we are removing one filter per layer).
    For example, filters_to_be_removed = [[[0], [], [], []], [[], [], [3], []]]
    specifies that filter 0 from the first layer and filter 3 from the 3rd layer will be removed.
    We generate the sequence of filters to be removed needed for the greedy pruning in the generate_sequences_of_filters_to_be_removed.py file.
3) Get the i-th branch which has the filters removed according to the i-th element of the sequence of filters that need to be removed
4) Feed-forward the data, grouped by class, through the i-th branch
5) Record accuracy per class from branch i

Finally, use the run_main.sh to run the main.py file and thus generate the branches of the network that are needed.
For any questions about the code, please contact mdimovsk@vols.utk.edu
