import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt

# from PIL import Image


def show_tensor(
    tensor,
    filename="figure.jpg",
    transpose=None,
    normalize=None,
    figsize=(10, 10),
    nrow=None,
    padding=2,
    verbose=True,
    **kwargs
):
    """Convenient function for visualizing tensors of any shape, supports batch_size."""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(np.array(tensor))
    tensor = tensor.detach().cpu().float()

    if tensor.ndim == 4 and tensor.shape[1] == 1:
        if verbose:
            print("processing as black&white")
        tensor = tensor.repeat(1, 3, 1, 1)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        if verbose:
            print("processing as black&white")
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    if normalize is None:
        if tensor.max() <= 1.0 and tensor.min() >= 0.0:
            normalize = False
        else:
            if verbose:
                print("tensor has been normalized to [0., 1.]")
            normalize = True

    if transpose is None:
        transpose = True if tensor.shape[1] != 3 else False
    if transpose:
        tensor = tensor.permute(0, 3, 1, 2)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(tensor.shape[0])))

    grid = torchvision.utils.make_grid(
        tensor, normalize=normalize, nrow=nrow, padding=padding, **kwargs
    )
    grid = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.tight_layout()
    plt.savefig(filename)
    # return Image.fromarray((grid*255).astype(np.uint8))
