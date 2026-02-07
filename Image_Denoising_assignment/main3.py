import h5py
from noise_model import rician_weighted_gradient, rician_weighted_log_likelihood
from mrf_prior import q3A_prior_gradient, q3B_prior_gradient, q3C_prior_gradient
from mrf_prior import q3A_prior_value, q3B_prior_value, q3C_prior_value
from metrics import rrmse
import numpy as np
from matplotlib import pyplot as plt
from main import gradient_ascent

def plot_objective_fn(values, prior_fn, alpha, gamma):
    plt.plot([i for i in range(1, len(values)+1)], values)
    plt.title(f"alpha = {alpha}, gamma = {gamma}")
    image_path = rf"results3\obj_fn_{prior_fn.__name__}"
    plt.savefig(image_path)

if __name__=="__main__":

    print("\n\nFetching data ...............................")
    with h5py.File(r'data\assignmentImageDenoising_microscopy.mat', 'r') as f:
        
        print(list(f.keys()))

        noiseless_image = f['microscopyImageOrig'][:]
        noisy_image = f['microscopyImageNoisyScale350sigma0point06'][:]
    print("Successfully fetched noisy and noiseless images!!\n")

    assert noiseless_image.shape==noisy_image.shape, "Check shapes. They dont match"

    print(noisy_image.shape)
    C, H, W = noisy_image.shape
    noisy_image = noisy_image.reshape(H, W, C)
    noiseless_image = noiseless_image.reshape(H, W, C)
    print(noisy_image.shape)

    initial_RRMSE = rrmse(noiseless_image, noisy_image)
    print(initial_RRMSE)

    noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min())
    noiseless_image = (noiseless_image - noiseless_image.min()) / (noiseless_image.max() - noiseless_image.min())

    # A: alpha = 0.1
    # B: alpha = 0.1
    # C: alpha=0.768, gamma=0.0447

    best_params = [(0.768, 0.0447), (0.1, 1.0), (0.1, 1.0)]
    prior_fn = [(q3C_prior_value, q3C_prior_gradient),
                (q3A_prior_value, q3A_prior_gradient),
                (q3B_prior_value, q3B_prior_gradient)]
    
    for i, fn in enumerate(prior_fn):
        print(f"{fn[0].__name__}")
        estimated_image, new_rrmse, posterior_values = gradient_ascent(noiseless_image=noiseless_image,
                                                                    noisy_image=noisy_image,
                                                                    likelihood_value_fn=rician_weighted_log_likelihood,
                                                                    likelihood_gradient_fn=rician_weighted_gradient,
                                                                    prior_value_fn=fn[0],
                                                                    prior_gradient_fn=fn[1],
                                                                    alpha=best_params[i][0],
                                                                    gamma=best_params[i][1],
                                                                    patience=10)
        print(f"alpha = {best_params[i][0]}, gamma = {best_params[i][1]}, RRMSE = {new_rrmse}")
        print("________________________________________________________________________________")
        estimated_image = (estimated_image - np.min(estimated_image)) / (np.max(estimated_image) - np.min(estimated_image))
        image_path = rf"results3\{fn[0].__name__}.png"
        plt.imsave(image_path, estimated_image, cmap="jet")
        plot_objective_fn(posterior_values, fn[0], best_params[i][0], best_params[i][1])
