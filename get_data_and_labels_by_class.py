import os
print(os.environ['XDG_RUNTIME_DIR'])
import numpy as np

def get_all_labels(test_loader):
    labels = []
    for data, target in test_loader:
        labels.append(target.numpy().flatten())
    labels = np.asarray(labels)
    return np.hstack(labels)

def get_all_data(test_loader):
    data_all = []
    for data, target in test_loader:
        data_all.append(data)
    return np.vstack(data_all)


def get_groups_of_class_indices(true_labels, num_classes):
    groups = [[] for x in range(num_classes)]
    for i in range(num_classes):
        groups[i].append(np.where(np.equal(true_labels, i))[0])
    return groups