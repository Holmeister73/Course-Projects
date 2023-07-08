# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:48:22 2023

@author: USER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from model_vae import VAE
import PIL
from sklearn.manifold import TSNE
from torchvision.models import inception_v3
from scipy import linalg
import gc
import torch_fidelity
from tqdm import tqdm
from torch_fidelity.helpers import get_kwarg, vassert, vprint



with torch.no_grad():
    torch.cuda.empty_cache()
    vae = None
    gc.collect()
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 100

batch_size = 2


train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, shuffle=True)

def generate(network_name):
    
    with torch.no_grad():
        random_noise_plot = torch.randn(100, z_dim, 1, 1).to(device)
        vae = pickle.load(open(network_name, "rb")).to(device)
        vae.eval()
        generated_plot = vae.Convolutional_Decoder(random_noise_plot)
        
        plt.figure(figsize = (60, 40))
        plt.gray()
        imgs = generated_plot.cpu().detach().numpy()
      
        for i, item in enumerate(imgs):
            if i >= 101: break
            plt.subplot(10, 10, i+1)
            plt.imshow(item[0])

# generate("model_vae.pk")  generates 100 images from random noise via weights in model_vae.pk


            
""" Below is the implementations of Kernel Inception Distance and Frechet Inception Distance taken from https://github.com/toshas/torch-fidelity/ and
     https://github.com/hukkelas/pytorch-frechet-inception-distance respectively with minor tweaks. """
    


""" You can scroll down for KID, FID score calculation examples"""






KEY_METRIC_KID_MEAN = 'kernel_inception_distance_mean'
KEY_METRIC_KID_STD = 'kernel_inception_distance_std'

def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased'):
    vassert(mmd_est in ('biased', 'unbiased', 'u-statistic'), 'Invalid value of mmd_est')

    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K


def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    k_11 = polynomial_kernel(features_1, features_1, degree=degree, gamma=gamma, coef0=coef0)
    k_22 = polynomial_kernel(features_2, features_2, degree=degree, gamma=gamma, coef0=coef0)
    k_12 = polynomial_kernel(features_1, features_2, degree=degree, gamma=gamma, coef0=coef0)
    return mmd2(k_11, k_12, k_22)


def kid_features_to_metric(features_1, features_2, **kwargs):
    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2
    assert features_1.shape[1] == features_2.shape[1]

    kid_subsets = get_kwarg('kid_subsets', kwargs)
    kid_subset_size = get_kwarg('kid_subset_size', kwargs)
    verbose = get_kwarg('verbose', kwargs)

    n_samples_1, n_samples_2 = len(features_1), len(features_2)
    vassert(
        n_samples_1 >= kid_subset_size and n_samples_2 >= kid_subset_size,
        f'KID subset size {kid_subset_size} cannot be smaller than the number of samples (input_1: {n_samples_1}, '
        f'input_2: {n_samples_2}). Consider using "kid_subset_size" kwarg or "--kid-subset-size" command line key to '
        f'proceed.'
    )

    features_1 = features_1.cpu().numpy()
    features_2 = features_2.cpu().numpy()

    mmds = np.zeros(kid_subsets)
    rng = np.random.RandomState(get_kwarg('rng_seed', kwargs))

    for i in tqdm(
            range(kid_subsets), disable=not verbose, leave=False, unit='subsets',
            desc='Kernel Inception Distance'
    ):
        f1 = features_1[rng.choice(n_samples_1, kid_subset_size, replace=False)]
        f2 = features_2[rng.choice(n_samples_2, kid_subset_size, replace=False)]
        o = polynomial_mmd(
            f1,
            f2,
            get_kwarg('kid_degree', kwargs),
            get_kwarg('kid_gamma', kwargs),
            get_kwarg('kid_coef0', kwargs),
        )
        mmds[i] = o

    out = {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }


    return np.mean(mmds)


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32 """
        
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations

def get_activations(images, batch_size):
    
    """Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
   
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                              ", but got {}".format(images.shape) """

    num_images = images.shape[0]
    inception_network = PartialInceptionNetwork()
    inception_network = inception_network.to(device)
    inception_network.eval()
    n_batches = int(np.ceil(num_images  / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = ims.to(device)
        activations = inception_network(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
        inception_activations[start_idx:end_idx, :] = activations
    return inception_activations



def calculate_activation_statistics(images, batch_size):
    """ Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer 
                of the inception model. """

    
    act = get_activations(images, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance. """
    

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print( "fid calculation produces singular product; adding %s to diagonal of cov estimates" )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(images1, images2, use_multiprocessing, batch_size):
    """Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar) """
   
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def FID_Score(sample_size, network_name):
    
    with torch.no_grad():
        vae = pickle.load(open(network_name, "rb")).to(device)
        vae.eval()
        random_noise = torch.randn(sample_size, z_dim, 1, 1).to(device)
        generated = vae.Convolutional_Decoder(random_noise)
        resized = transforms.Resize((299,299),antialias = True)(generated)
        fake_rgb = resized.expand(-1, 3, -1, -1) 
        
        real_rgb=[]
        for i, (real, labels) in enumerate(train_loader):
            if(i*batch_size) >= sample_size:
                break
            real_resized = transforms.Resize((299,299), antialias = True)(real)
            rgb_real_resized = real_resized.expand(-1,3,-1,-1)
            real_rgb.append(rgb_real_resized)
            
        real_rgb = torch.cat(real_rgb, dim = 0)
        
        return calculate_fid(fake_rgb, real_rgb, use_multiprocessing = False, batch_size = 32)


def KID_Score(sample_size, subset_size, num_subsets, network_name):
    with torch.no_grad():
        vae = pickle.load(open(network_name, "rb")).to(device)
        vae.eval()
        random_noise = torch.randn(sample_size, z_dim, 1, 1).to(device)
        generated = vae.Convolutional_Decoder(random_noise)
        resized = transforms.Resize((299,299),antialias = True)(generated)
        fake_rgb = resized.expand(-1, 3, -1, -1) 
        
        real_rgb=[]
        for i, (real, labels) in enumerate(train_loader):
            if(i*batch_size) >= sample_size:
                break
            real_resized = transforms.Resize((299,299), antialias = True)(real)
            rgb_real_resized = real_resized.expand(-1,3,-1,-1)
            real_rgb.append(rgb_real_resized)
            
        real_rgb = torch.cat(real_rgb, dim = 0)
    features1 = torch.from_numpy(get_activations(real_rgb, batch_size = 32))
    features2=  torch.from_numpy(get_activations(fake_rgb, batch_size = 32))
    return kid_features_to_metric(features1, features2)

# Example uses for calculating FID_Score and KID_Score functions are commented out below.


# FID_Score(1000, "model_vae.pk") generates 1000 fake images and also uses 1000 real images from training set to calculate the FID score
# KID_Score(1000, 1000, 100, "model_vae.pk") generates 1000 fake images and uses 100 subsets of size 500 to calculate KID score.
# Note: FID score gets lower as sample size gets larger but KID score doesn't have that kind of bias. 
# However using larger sample sizes makes computation more difficult in terms of memory and time consumption. 
# I suggest sample size of 1000 if you want to run.
# I used 10000 sample size in the report since running once was enough for the report.

#generate("model_vae.pk")
#print("\nKID Score:",KID_Score(1000, 500, 100, "model_vae.pk"))
#print("FID Score:",FID_Score(1000, "model_vae.pk"))

