import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture 
from scipy.stats import multivariate_normal

def display(image, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis("off")

    plt.show()

def initialize_GMM(brain_pixels: np.array, mask: np.array):
    brain_pixels = brain_pixels.reshape(-1, 1) 
    #print(brain_pixels.shape)
    num_classes = 3  # WM, GM, CSF
    gmm = GaussianMixture(n_components=num_classes, covariance_type="diag", random_state=97)
    gmm.fit(brain_pixels.reshape(-1, 1))

    mu = gmm.means_.flatten()  # Mean intensity for each class
    sigma = gmm.covariances_.flatten()  # Variance for each class
    pi = gmm.weights_  # Mixing weights

    posterior_probs = gmm.predict_proba(brain_pixels)
    #print(posterior_probs.shape)
    segmentation = np.argmax(posterior_probs, axis=1) # labels
    #print(segmentation.shape)

    segmented_image = np.zeros_like(mask)  # Start with an empty image
    segmented_image[mask > 0] = segmentation

    return segmented_image, segmentation, mu, sigma, pi

def compute_memberships(image, mu, sigma, beta, labels, num_classes):
    # image is flattened brain pixels 
    assert image.shape == labels.shape

    N = labels.shape[0]
    gamma = np.zeros((N, num_classes)) # Every pixel has 'num_classes' memberships ('num_classes' probabilities)

    image = image.reshape(-1)  # Ensure image is 1D
    likelihoods = np.zeros((N, num_classes)) 
    for l in range(num_classes):
        likelihoods[:, l] = multivariate_normal.pdf(image, mean=mu[l], cov=sigma[l]) 
    
    #print(likelihoods.shape)

    # Compute MRF priors using neighbor labels
    mrf_prior = np.zeros((N, num_classes))

    for i in range(N):
        for l in range(num_classes):
            penalty = 0
            if i > 0 and labels[i - 1] != l:  # Left neighbor is different
                penalty += 1
            if i < N - 1 and labels[i + 1] != l:  # Right neighbor is different
                penalty += 1

            mrf_prior[i, l] = np.exp(-beta * penalty)
    
    # Normalize priors
    mrf_prior /= np.sum(mrf_prior, axis=1, keepdims=True)

    # Compute final gamma (posterior probability)
    gamma = likelihoods * mrf_prior
    gamma /= np.sum(gamma, axis=1, keepdims=True)
    #print(gamma.shape)

    return gamma

def update_parameters(image, gamma, num_classes):
    N = image.shape[0]  # Number of pixels
    image = image.reshape(-1)  # Ensure image is 1D
    
    mu = np.zeros(num_classes)
    sigma = np.zeros(num_classes)

    for l in range(num_classes):
        gamma_l = gamma[:, l]  # Memberships for class l
        gamma_sum = np.sum(gamma_l)  # Normalization factor

        mu[l] = np.sum(gamma_l * image) / gamma_sum
        sigma[l] = np.sum(gamma_l * (image - mu[l])**2) / gamma_sum

    return mu, sigma

def compute_log_posterior(image, labels, mu, sigma, beta, num_classes):
    N = image.shape[0]
    
    # Compute Likelihood (GMM term)
    likelihoods = np.zeros((N, num_classes))
    for l in range(num_classes):
        likelihoods[:, l] = multivariate_normal.pdf(image, mean=mu[l], cov=sigma[l])
    
    log_likelihood = np.sum(np.log(likelihoods[np.arange(N), labels] + 1e-8))  # Avoid log(0)
    
    # Compute Prior (MRF term)
    mrf_prior = 0
    for i in range(N):
        if i > 0:
            mrf_prior -= beta * (labels[i] != labels[i-1])
        if i < N - 1:
            mrf_prior -= beta * (labels[i] != labels[i+1])
    
    return log_likelihood + mrf_prior

def map_mrf_em(image, num_classes, beta, max_iters=10, tol = 1e-4, patience = 2):
    # Initialize parameters
    N = image.shape[0]
    image = image.reshape(-1)  # Ensure image is 1D
    decreased = patience
    log_post_prob = list()
    
    initial_label_image, labels, mu, sigma, _ = initialize_GMM(image, mask)

    """
    # random initialization of lables, mu and sigma
    labels = np.random.randint(0, num_classes, size=N)  # Random initialization 
    initial_label_image = np.zeros_like(mask)  # Start with an empty image
    initial_label_image[mask > 0] = labels
    # Initialize mu and sigma randomly from the image
    mu = np.random.choice(image, num_classes, replace=False)
    sigma = np.full(num_classes, np.var(image))
    """

    print(f"Inital Class means\n{mu}")
    print(f"Inital Class STD\n{sigma}")
    print()

    plt.imshow(initial_label_image, cmap="jet")
    plt.colorbar()
    plt.title("Initial Segmentation")
    plt.show()

    for iteration in range(max_iters):
        print(f"Iteration {iteration + 1}")
        before_iter = compute_log_posterior(image, labels, mu, sigma, beta, num_classes)
        log_post_prob.append(before_iter)
        print(f"Before iteration {iteration}, log posterior prob = {before_iter}")

        # E-step: Compute membership probabilities
        gamma = compute_memberships(image, mu, sigma, beta, labels, num_classes)

        # M-step: Update parameters
        mu, sigma = update_parameters(image, gamma, num_classes)

        # MAP Label Assignment (for visualization)
        labels = np.argmax(gamma, axis=1)

        after_iter = compute_log_posterior(image, labels, mu, sigma, beta, num_classes)
        print(f"After iteration {iteration}, log posterior prob = {after_iter}")
        print("_______________________________________________________________________________")
    
        # Check for convergence
        if after_iter < before_iter:  # If very few pixels change, stop
            decreased -= 1
        
        if decreased == 0:
            return labels, mu, sigma, gamma, log_post_prob
        
    return labels, mu, sigma, gamma, log_post_prob

def plot_results(original, gamma_opt, labels_opt, gamma_no_mrf, labels_no_mrf):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 3 rows, 2 columns
    original_height, original_width = original.shape

    # (i) Corrupted Image (First row, spanning both columns)
    axes[0, 0].imshow(original.reshape(original_height, original_width), cmap='gray')
    axes[0, 0].set_title("Corrupted Image")
    axes[0, 1].axis("off")  # Empty space

    # (ii) Class-Membership Images (Second row)
    max_gamma_opt = np.max(gamma_opt, axis=1)  # Shape (N,)
    full_gamma_opt = np.zeros((original_height, original_width))  
    full_gamma_opt[mask > 0] = max_gamma_opt
    axes[1, 0].imshow(full_gamma_opt, cmap='jet')
    axes[1, 0].set_title("Class-Membership (Optimal β)")

    max_gamma_no_mrf = np.max(gamma_no_mrf, axis=1)  # Shape (N,)
    full_gamma_no_mrf = np.zeros((original_height, original_width))  
    full_gamma_no_mrf[mask > 0] = max_gamma_no_mrf
    axes[1, 1].imshow(full_gamma_no_mrf, cmap='jet')
    axes[1, 1].set_title("Class-Membership (β = 0)")

    # (iii) Label Images (Third row)
    labels_image_opt = np.zeros_like(mask)  
    labels_image_opt[mask > 0] = labels_opt
    axes[2, 0].imshow(labels_image_opt, cmap='jet')
    axes[2, 0].set_title("Label Image (Optimal β)")

    labels_image_no_mrf = np.zeros_like(mask)  
    labels_image_no_mrf[mask > 0] = labels_no_mrf
    axes[2, 1].imshow(labels_image_no_mrf, cmap='jet')
    axes[2, 1].set_title("Label Image (β = 0)")

    plt.tight_layout()
    plt.show()



if __name__=="__main__":
    print("\n\nFetching data ...............................")
    filepath = r'data\assignmentSegmentBrainGmmEmMrf.mat'
    arrays = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)
    print("Successfully fetched!!\n")

    corrupted_image = arrays['imageData']
    mask = arrays['imageMask']

    log_post_prob_list = list()
    log_post_prob_list0 = list()

    corrupted_image = (corrupted_image - corrupted_image.min()) / (corrupted_image.max() - corrupted_image.min()) 
    mask = mask.astype(np.uint8)

    brain_pixels = corrupted_image[mask > 0]

    beta, num_classes = 0.1, 3
    final_labels, final_mu, final_sigma, final_gamma, log_post_prob_list =  map_mrf_em(brain_pixels, num_classes, beta)
    print()
    print(f"Final Class means\n{final_mu}")
    print(f"Final Class STD\n{final_sigma}")
    print()

    plt.plot([i for i in range(len(log_post_prob_list))], log_post_prob_list)
    plt.scatter([i for i in range(len(log_post_prob_list))], log_post_prob_list)
    plt.title("Log posterior probabilities for optimal beta")
    plt.show()

    labels0, mu0, sigma0, gamma0, log_post_prob_list0 =  map_mrf_em(brain_pixels, num_classes, beta=0)
    plot_results(corrupted_image, final_gamma, final_labels, gamma0, labels0)

    plt.plot([i for i in range(len(log_post_prob_list0))], log_post_prob_list0)
    plt.scatter([i for i in range(len(log_post_prob_list0))], log_post_prob_list0)
    plt.title("Log posterior probabilities for beta = 0")
    plt.show()