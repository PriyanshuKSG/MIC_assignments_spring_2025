from scipy.io import loadmat
from noise_model import rician_weighted_gradient, rician_weighted_log_likelihood
from mrf_prior import quadratic_weighted_prior_gradient, huber_weighted_prior_gradient, third_prior_gradient
from mrf_prior import quadratic_prior_value, huber_prior_value, third_prior_value
from metrics import rrmse
import numpy as np
from matplotlib import pyplot as plt

def plot_objective_fn(values, prior_fn, alpha, gamma):
    plt.plot([i for i in range(1, len(values)+1)], values)
    plt.title(f"alpha = {alpha}, gamma = {gamma}")
    plt.show()
    #image_path = rf"results\obj_fn_{prior_fn.__name__}"
    #plt.savefig(image_path)

def gradient_ascent(noiseless_image: np.array,
                    noisy_image: np.array,
                    likelihood_value_fn,
                    likelihood_gradient_fn,
                    prior_value_fn,
                    prior_gradient_fn,
                    alpha: float,
                    gamma: float,
                    max_iters: int = 300,
                    lr: float = 0.1,
                    lambda_decay: float = 0.001,
                    verbose: int = 1,
                    patience: int = 20):
    """
    Implements the gradient ascent optimization loop.
    """
    x = noisy_image.copy()  # Start with noisy image as the initial estimate
    prev_x = None
    epsilon = 1e-5
    x_best, min_rrmse = None, 1.0
    prev_rrmse = None
    original_patience = patience
    obj_fn_values = list()

    for iter in range(1, max_iters+1):

        likelihood_value = likelihood_value_fn(noisy_image=noisy_image,
                                                noiseless_image=x, 
                                                alpha=(1-alpha))
        likelihood_grad = likelihood_gradient_fn(noisy_image=noisy_image,
                                                noiseless_image=x, 
                                                alpha=(1-alpha))
        prior_value = prior_value_fn(alpha=alpha,
                                    gamma=gamma,
                                    noiseless_image=x)
        prior_grad = prior_gradient_fn(alpha=alpha,
                                       gamma=gamma,
                                       noiseless_image=x)

        total_grad = likelihood_grad - prior_grad
        total_posterior = likelihood_value - prior_value
        obj_fn_values.append(total_posterior)

        lr = lr / (1 + lambda_decay*(iter-1))

        x_new = x + lr*total_grad
        current_rrmse = rrmse(noiseless_image, x_new)
        if min_rrmse > current_rrmse:
            min_rrmse = current_rrmse
            x_best = x_new
        
        if iter%5 == 0:
            print(f"Iteration = {iter}, lr = {lr: .3f}, RRMSE = {current_rrmse}, prev_RRMSE = {prev_rrmse}, patience = {patience}")
            
        
        if prev_x is not None:
            if np.sum(total_grad * (x_new - x)) < 0: # if the product is +ve then correct direction of optimization
                lr /= 2.0
        
        if prev_rrmse != None and abs(prev_rrmse - current_rrmse) < epsilon:
            break

        if prev_rrmse is not None and prev_rrmse < current_rrmse:
            patience -= 1
        else:
            patience = original_patience
        
        prev_rrmse = current_rrmse
        prev_x = x.copy()
        x = x_new.copy()

        if patience == 0:
            #print("Optimization now going in opposite direction. Optimal value already reached. Hence, stopped.")
            break

    return x_best, min_rrmse, obj_fn_values


if __name__=="__main__":

    print("\n\nFetching data ...............................")
    data = loadmat(r'data\assignmentImageDenoising_phantom.mat')
    noiseless_image = data['imageNoiseless']
    noisy_image = data['imageNoisy']
    print("Successfully fetched noisy and noiseless images!!\n")

    initial_rrmse = rrmse(noiseless_image, noisy_image)
    print(f"Initial RRMSE = {initial_rrmse}\n")
    
    # (0.7373, 0.084) = third correct
    #  (0.768, 0.0447) = huber correct
    # (0.1536, 1.0) = quadratic correct
    
    best_params = [(0.7373, 0.084), (0.768, 0.0447), (0.1536, 1.0)]
    
    scale_proof = [(1.0, 1.0), (1.0, 0.8), (0.8, 1.0), (1.2, 1.0), (1.0, 1.2)]
    list_of_priors = [(third_prior_gradient, third_prior_value),
                    (huber_weighted_prior_gradient, huber_prior_value),
                    (quadratic_weighted_prior_gradient, quadratic_prior_value)]

        
    for i, prior_fn in enumerate(list_of_priors):
        print(prior_fn[1].__name__)
        for j, scale in enumerate(scale_proof):
            estimated_image, new_rrmse, posterior_values = gradient_ascent(noiseless_image=noiseless_image,
                                                    noisy_image=noisy_image,
                                                    likelihood_value_fn=rician_weighted_log_likelihood,
                                                    likelihood_gradient_fn=rician_weighted_gradient,
                                                    prior_value_fn=prior_fn[1],
                                                    prior_gradient_fn=prior_fn[0],
                                                    alpha=scale[0]*best_params[i][0],
                                                    gamma=scale[1]*best_params[i][1],
                                                    verbose=1)
            
            print(f"\n{scale[0]}*alpha, {scale[1]}*gamma, RRMSE = {new_rrmse: .5f}\n")
            
            if scale == (1.0, 1.0):
                #image_path = rf"results\{prior_fn[1].__name__}.png"
                #plt.imsave(image_path, estimated_image, cmap="jet")
                plot_objective_fn(posterior_values, prior_fn[1], best_params[i][0], best_params[i][1])
        
        print("___________________________________________________________\n")
    
    

    """
    alphas = [0.1, 0.2, 0.5, 0.8, 0.9]
    gammas = [0.1, 0.5, 0.9, 1.5, 2.0]
    best_params = {}
    best_rrmse = float('inf')

    for alpha in alphas:
        estimated_image, new_rrmse = gradient_ascent(noiseless_image=noiseless_image,
                                                noisy_image=noisy_image,
                                                likelihood_gradient_fn=rician_weighted_gradient,
                                                prior_gradient_fn=quadratic_weighted_prior_gradient,
                                                alpha=alpha,
                                                gamma=1.0,
                                                verbose=1)
        print(f"alpha = {alpha}, RRMSE = {new_rrmse: 0.6f}")
        print("_________________________________________________________________")
        if new_rrmse < best_rrmse:
            best_rrmse = new_rrmse
            best_params[quadratic_weighted_prior_gradient.__name__] = alpha

    print("\n\n",best_params,"\n")

    #alpha = 0.1 best, RRMSE = 0.266531
     {'huber_weighted_prior_gradient': (0.8, 0.1)}, RRMSE =  0.248830
     {'third_prior_gradient': (0.8, 0.5)}, RRMSE =  0.257238
    """
    
    """
    for prior_fn in [quadratic_weighted_prior_gradient]:
        best_rrmse = float('inf')
        for alpha in alphas:
            for gamma in gammas:
                estimated_image, new_rrmse = gradient_ascent(noiseless_image=noiseless_image,
                                                noisy_image=noisy_image,
                                                likelihood_gradient_fn=rician_weighted_gradient,
                                                prior_gradient_fn=prior_fn,
                                                alpha=alpha,
                                                gamma=gamma,
                                                verbose=1)
                print(f"\nPrior function = {prior_fn.__name__}")
                print(f"alpha = {alpha}, gamma = {gamma}, RRMSE = {new_rrmse: 0.6f}")
                print("_________________________________________________________________")

                if new_rrmse < best_rrmse:
                    best_rrmse = new_rrmse
                    best_params[prior_fn.__name__] = (alpha, gamma)
    print("\n\n",best_params,"\n")
    """