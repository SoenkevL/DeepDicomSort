import torch
import numpy as np
import pandas as pd
import os
def chooseDevice(verbose=False):
    #returns the gpu with most free memory currently
    if torch.cuda.is_available():
        devices = [torch.device(i) for i in range(torch.cuda.device_count())]
        memoryAvailabilitie = [torch.cuda.mem_get_info(device)[0] for device in devices]
        MostlyAvailable = np.argmax(memoryAvailabilitie)
        if verbose:
            for i in range(torch.cuda.device_count()):
                print(f"gpu {i} has {round(memoryAvailabilitie[i]/1000000)}MiB available")
            print(f"chose device {MostlyAvailable}")
        return devices[MostlyAvailable]
    else:
        return torch.device("cpu")

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def load_labels(label_file,nb_classes=None): #from DDS original paper
    labels = np.genfromtxt(label_file, dtype='str')
    label_IDs = labels[:, 0]
    label_IDs = np.asarray(label_IDs)
    label_values = labels[:, 1].astype(int)
    extra_inputs = labels[:, 2:].astype(float)
    np.round(extra_inputs, 2)

    if nb_classes:
        N_classes = nb_classes
    else:
        N_classes = len(np.unique(label_values))

    # Make sure that minimum of labels is 0
    label_values = label_values - np.min(label_values)

    one_hot_labels = get_one_hot(label_values, N_classes)

    return label_IDs, one_hot_labels, N_classes, extra_inputs