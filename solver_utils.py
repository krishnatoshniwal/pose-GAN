import torch
import matplotlib.pyplot as plt
import numpy as np


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def vis_tensor(tens, denorm_ar=True):
    if denorm_ar:
        tens = denorm(tens)
    if tens.is_cuda:
        tens = tens.cpu()

    nump_arr = tens.numpy()
    if nump_arr.ndim == 4:
        for im in nump_arr:
            if im.shape[0] == 1:
                im = im.repeat(3, axis=0)
            im_ = im.transpose(1, 2, 0)

            plt.imshow(im_)
            plt.show()

    else:
        im_ = nump_arr.transpose(1, 2, 0)

        plt.imshow(im_)
        plt.show()



def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print("The number of parameters: {}".format(num_params))

    print(model)
