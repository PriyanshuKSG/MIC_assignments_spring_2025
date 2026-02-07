from scipy.io import loadmat
from noise_model import rician_weighted_gradient, rician_weighted_log_likelihood
from mrf_prior import quadratic_weighted_prior_gradient, huber_weighted_prior_gradient, third_prior_gradient
from mrf_prior import quadratic_prior_value, huber_prior_value, third_prior_value
from metrics import rrmse
import numpy as np
from matplotlib import pyplot as plt
from main import gradient_ascent

def plot_objective_fn(values, prior_fn, alpha, gamma):
    plt.plot([i for i in range(1, len(values)+1)], values)
    plt.title(f"alpha = {alpha}, gamma = {gamma}")
    image_path = rf"results2\obj_fn_{prior_fn.__name__}"
    plt.savefig(image_path)

if __name__=="__main__":

    print("\n\nFetching data ...............................")
    data = loadmat(r'data\assignmentImageDenoising_brainMRIslice.mat')
    noiseless_image = data['brainMRIsliceNoisy']
    noisy_image = data['brainMRIsliceOrig']
    print("Successfully fetched noisy and noiseless images!!\n")

    initial_RRMSE = rrmse(noiseless_image, noisy_image)
    print(initial_RRMSE)
    
    noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min())
    noiseless_image = (noiseless_image - noiseless_image.min()) / (noiseless_image.max() - noiseless_image.min())

    best_params = [(0.768, 0.0447), (0.1536, 1.0), (0.7373, 0.084)]
    list_of_priors = [(huber_weighted_prior_gradient, huber_prior_value),
                      (quadratic_weighted_prior_gradient, quadratic_prior_value),
                      (third_prior_gradient, third_prior_value)]
    
    for i, prior_fn in enumerate(list_of_priors):
        print(prior_fn[1].__name__)
        estimated_image, new_rrmse, posterior_values = gradient_ascent(noiseless_image=noiseless_image,
                                                    noisy_image=noisy_image,
                                                    likelihood_value_fn=rician_weighted_log_likelihood,
                                                    likelihood_gradient_fn=rician_weighted_gradient,
                                                    prior_value_fn=prior_fn[1],
                                                    prior_gradient_fn=prior_fn[0],
                                                    alpha=best_params[i][0],
                                                    gamma=best_params[i][1],
                                                    verbose=1)
        print(f"\nalpha = {best_params[i][0]}, gamma = {best_params[i][1]}, RRMSE = {new_rrmse: .5f}\n")
        print("_______________________________________________________________________________________")
        #print(posterior_values)
        image_path = rf"results2\{prior_fn[1].__name__}.png"
        plt.imsave(image_path, estimated_image, cmap="jet")
        plot_objective_fn(posterior_values, prior_fn[1], best_params[i][0], best_params[i][1])