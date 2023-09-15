import torch
import numpy as np
import wandb
import monai
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

def log_to_wandb(epoch, train_loss, val_loss):#, batch_data, outputs):
    """ Function that logs ongoing training variables to W&B """

    # Create list of images that have segmentation masks for model output and ground truth
    # log_imgs = [wandb.Image(img, masks=wandb_masks(mask_output, mask_gt)) for img, mask_output,
    #             mask_gt in zip(batch_data['img'], outputs, batch_data['mask'])]

    # Send epoch, losses and images to W&B
    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})#, 'results': log_imgs})

def from_compose_to_list(transform_compose):
    """
    Transform an object monai.transforms.Compose in a list fully describing the transform.
    /!\ Random seed is not saved, then reproducibility is not enabled.
    """
    from copy import deepcopy

    if not isinstance(transform_compose, monai.transforms.Compose):
        raise TypeError("transform_compose should be a monai.transforms.Compose object.")

    output_list = list()
    for transform in transform_compose.transforms:
        kwargs = deepcopy(vars(transform))

        # Remove attributes which are not arguments
        args = list(transform.__init__.__code__.co_varnames[1: transform.__init__.__code__.co_argcount])
        for key, obj in vars(transform).items():
            if key not in args:
                del kwargs[key]

        output_list.append({"class": transform.__class__, "kwargs": kwargs})
    return output_list

def from_list_to_compose(transform_list):
    """
    Transform a list in the corresponding monai.transforms.Compose object.
    """

    if not isinstance(transform_list, list):
        raise TypeError("transform_list should be a list.")

    pre_compose_list = list()

    for transform_dict in transform_list:
        if not isinstance(transform_dict, dict) or 'class' not in transform_dict or 'kwargs' not in transform_dict:
            raise TypeError("transform_list should only contains dicts with keys ['class', 'kwargs']")

        try:
            transform = transform_dict['class'](**transform_dict['kwargs'])
        except TypeError: # Classes have been converted to str after saving
            transform = eval(transform_dict['class'].replace("__main__.", ""))(**transform_dict['kwargs'])

        pre_compose_list.append(transform)

    return monai.transforms.Compose(pre_compose_list)