from scipy import linalg
import torch
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def get_activations(
    images_tensor, model, batch_size=50, dims=2048, device="cuda", num_workers=0):
    """Calculates the activations of the pool_3 layer for all images in a tensor.

    Params:
    -- images_tensor : Tensor of images with shape (num_images, 3, height, width)
    -- model         : Instance of inception model
    -- batch_size    : Batch size of images for the model to process at once.
    -- dims          : Dimensionality of features returned by Inception
    -- device        : Device to run calculations

    Returns:
    -- A numpy array of dimension (num_images, dims) containing the activations
       of the given tensor when feeding inception with the query tensor.
    """

    num_images = images_tensor.size(0)

    if batch_size > num_images:
        batch_size = num_images

    dataloader = torch.utils.data.DataLoader(
        images_tensor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    pred_arr = np.empty((num_images, dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]
        
    return pred_arr

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print(f"Adding {eps} to diagonal of covariances due to singularity")
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: Imaginary component {m} found, taking real part only")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def calculate_activation_statistics(
    images_tensor, model, batch_size=50, dims=2048, device="cuda", num_workers=0):
    """Calculation of the statistics used by the FID.
    Params:
    -- images_tensor : Tensor of images with shape (num_images, 3, height, width)
    -- model         : Instance of inception model
    -- batch_size    : The images numpy array is split into batches with
                       batch size batch_size. A reasonable batch size
                       depends on the hardware.
    -- dims          : Dimensionality of features returned by Inception
    -- device        : Device to run calculations
    -- num_workers   : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    
    act = get_activations(images_tensor, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def calculate_fid(tensor_1, tensor_2, model, device, batch_size=10, dims=2048, num_workers=0):
    """Calculates the FID of two paths"""

    m1, s1 = calculate_activation_statistics(
        tensor_1, model, batch_size, dims, device, num_workers
    )
    
    m2, s2 = calculate_activation_statistics(
        tensor_2, model, batch_size, dims, device, num_workers
    )
    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

