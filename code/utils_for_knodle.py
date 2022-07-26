import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

def np_array_to_tensor(x: np.ndarray):
    
    """
    :rtype: object
    """
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    #x = TensorDataset(x)
    return x

# Took this from the knodle github for converting between arrays and tensors
def np_array_to_tensor_dataset(x: np.ndarray) -> TensorDataset:
    """
    :rtype: object
    """
    if isinstance(x, sp.csr_matrix):
        x = x.toarray()
    x = torch.from_numpy(x)
    x = TensorDataset(x)
    return x


def make_rules_and_y_plot(data, mapping, indices):
    
    plt.clf()
    fig = plt.figure(figsize = (2,3))
    ax1 = fig.add_subplot(121)
    ax1.imshow(data[indices], cmap='magma')
    #plt.colorbar(fraction=0.0046)
    ax1.set_xlabel('Rules')
    ax1.set_ylabel('Data row')

    ax2 = fig.add_subplot(122)
    im2 = ax2.imshow(np.dot(data[indices], mapping), cmap='magma')
    #plt.colorbar(fraction=0.0046)
    #ax2.set_xlabel('Rules')
    ax2.set_xlabel('Labels')
    plt.colorbar(im2, fraction = 0.046)
    plt.tight_layout()
    plt.show()

def make_rules_and_double_y_plot(data, mapping, y_noisy, indices):
    plt.clf()
    fig = plt.figure(figsize = (3,3))
    ax1 = fig.add_subplot(131)
    ax1.imshow(data[indices], cmap='magma')
    #plt.colorbar(fraction=0.0046)
    ax1.set_xlabel('Rules')
    ax1.set_ylabel('Data row')

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(np.dot(data[indices], mapping), cmap='magma')
    #plt.colorbar(fraction=0.0046)
    #ax2.set_xlabel('Rules')
    ax2.set_xlabel('Labels')
    plt.colorbar(im2, fraction = 0.046)

    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(y_noisy[indices], cmap='magma')
    #plt.colorbar(fraction=0.0046)
    #ax2.set_xlabel('Rules')
    ax3.set_xlabel('Labels')
    plt.colorbar(im3, fraction = 0.046)
    
    plt.show()

def make_double_rules_and_double_y_plot(rules_bf, rules_af, mapping, y_noisy, indices):
    
    plt.clf()
    fig = plt.figure(figsize = (3,3))
    ax1 = fig.add_subplot(151)
    ax1.imshow(rules_bf[indices], cmap='magma')
    #plt.colorbar(fraction=0.0046)
    ax1.set_xlabel('Rules before')
    ax1.set_ylabel('Data row')

    ax2 = fig.add_subplot(152)
    im2 = ax2.imshow(np.dot(rules_bf[indices], mapping), cmap='magma')
    #plt.colorbar(fraction=0.0046)
    #ax2.set_xlabel('Rules')
    ax2.set_xlabel('Labels before')
    plt.colorbar(im2, fraction = 0.046)

    ax3 = fig.add_subplot(153)
    im3 = ax3.imshow(rules_af[indices], cmap='magma')
    #plt.colorbar(fraction=0.0046)
    #ax2.set_xlabel('Rules')
    ax3.set_xlabel('Rules after')
    plt.colorbar(im3, fraction = 0.046)

    ax4 = fig.add_subplot(154)
    im4 = ax4.imshow(np.dot(rules_af[indices], mapping), cmap='magma')
    #plt.colorbar(fraction=0.0046)
    #ax2.set_xlabel('Rules')
    ax4.set_xlabel('Labels after')
    plt.colorbar(im3, fraction = 0.046)

    ax5 = fig.add_subplot(155)
    im5 = ax5.imshow(y_noisy[indices], cmap='magma')
    #plt.colorbar(fraction=0.0046)
    #ax2.set_xlabel('Rules')
    ax5.set_xlabel('predicted y')
    plt.colorbar(im5, fraction = 0.046)
    
    plt.show()
    