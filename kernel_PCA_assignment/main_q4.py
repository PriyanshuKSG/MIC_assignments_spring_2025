import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from scipy.optimize import minimize
import time

def load_dataset(folder_path):
    image_vectors = []

    for filename in os.listdir(folder_path): 
        if filename.endswith(".png") or filename.endswith(".jpg"):  
            img_path = os.path.join(folder_path, filename)

            img = Image.open(img_path).convert("L")  # "L" ensures grayscale

            img = img.resize((64, 64))
            
            img_array = np.array(img)
            max_pixel = np.max(img_array)
            min_pixel = np.min(img_array)
            img_array = (img_array - min_pixel) / (max_pixel - min_pixel)
            
            img_vector = img_array.flatten() # (4096, )

            image_vectors.append(img_vector)

    X = np.array(image_vectors)
    return X

def apply_pca(X):

    pca = PCA(n_components=X.shape[0])  
    X_pca = pca.fit_transform(X)

    mean_image = np.mean(X, axis=0).reshape(64, 64)

    eigenvalues = pca.explained_variance_
    pc1 = pca.components_[0].reshape(64, 64)  # 1st mode of variation
    pc2 = pca.components_[1].reshape(64, 64)  # 2nd mode of variation

    return eigenvalues, mean_image, pc1, pc2

def plot_partA(eigenvalues, mean_image, pc1, pc2):

    plt.scatter([i+1 for i in range(eigenvalues.shape[0])], eigenvalues, color='red')
    plt.plot([i+1 for i in range(eigenvalues.shape[0])], eigenvalues, color='blue')
    plt.title("Eigen spectrum")
    plt.xlabel("Component")
    plt.ylabel("Eigen value")
    plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(mean_image, cmap='gray')
    ax[0].set_title("Mean Image")

    ax[1].imshow(pc1, cmap='bwr')  # Blue-White-Red colormap to show variations
    ax[1].set_title("First Mode of Variation")

    ax[2].imshow(pc2, cmap='bwr')
    ax[2].set_title("Second Mode of Variation")

    plt.show()

def rbf_kernel(X, gamma = 0.5):
    pairwise_sq_dists = np.sum(X**2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X**2, axis=1)
    return np.exp(-gamma * pairwise_sq_dists)

def center_kernel_matrix(K):
    """Center the kernel matrix K."""
    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    return K - one_N @ K - K @ one_N + one_N @ K @ one_N

def kernel_pca(X, n_components, gamma=0.5):
    """Perform Kernel PCA with Gaussian (RBF) Kernel."""
    K = rbf_kernel(X, gamma) 
    
    K_centered = center_kernel_matrix(K)
    
    eigvals, eigvecs = eigh(K_centered)
    
    # Sort Eigenvalues and Eigenvectors in descending order
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    
    eigvals, eigvecs = eigvals[:n_components], eigvecs[:, :n_components]
    
    return eigvals, eigvecs, K

def plot_eigen_spectrum(eigvals):
    
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(eigvals) + 1), eigvals, 'bo-', markersize=5)
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigen Spectrum of Kernel PCA")
    plt.grid()
    plt.show()

def compute_preimage(mean_proj, X, gamma):

    def rbf_kernel_part2(x1, x2, gamma):
        """ Compute RBF kernel between two vectors """
        return np.exp(gamma*(-np.linalg.norm(x1 - x2) ** 2))

    def objective(x_hat):
        """ Objective: minimize squared difference in RKHS """
        return np.linalg.norm(
            mean_proj - np.array([rbf_kernel_part2(x_hat, x_i, gamma) for x_i in X])
        ) ** 2

    x0 = np.mean(X, axis=0)

    res = minimize(objective, x0, method="L-BFGS-B")

    return res.x 

def get_pre_image(eigenvecs, X, gamma):
    alpha = eigenvecs[:, 0]  # First principal component
    mean_proj = np.mean(alpha)  # Mean projection in RKHS
    pre_image = compute_preimage(mean_proj, X, gamma = gamma)
    pre_image = pre_image.reshape(64, 64)

    return pre_image

def plot_pre_image(pre_image):
    plt.figure(figsize=(5, 5))
    plt.imshow(pre_image, cmap='gray')
    plt.title("Estimated Pre-Image in Input Space")
    plt.axis('off')
    plt.show()

def pca_reconstruct(X, X_distorted, n_components=3):

    pca = PCA(n_components=n_components)
    pca.fit(X)

    projected = pca.transform(X_distorted)

    reconstructed = pca.inverse_transform(projected)
    
    reconstructed_images = reconstructed.reshape(X.shape[0], 64, 64)
    
    return reconstructed_images

def compute_preimage_part_c(x_proj, alphas, lambdas, train_X, gamma):
    def objective(x):
        k_x = np.exp(-gamma * np.linalg.norm(train_X - x, axis=1) ** 2)  # (150,)
        
        projection = np.dot(alphas.T, k_x)  # shape: (3,)

        return np.linalg.norm(projection - x_proj) ** 2

    x0 = np.mean(train_X, axis=0)
    res = minimize(objective, x0, method='L-BFGS-B')
    return res.x

def kernel_pca_reconstruct(X, X_distorted, n_components = 3, gamma = 0.5):
    
    eigenvals, eigenvecs, K = kernel_pca(X, n_components=X.shape[0])
    eigenvals = eigenvals[:n_components]
    eigenvecs = eigenvecs[:, :n_components]

    one_n = np.ones((K.shape[0], K.shape[0])) / K.shape[0]
    K_mean = K.mean(axis=0)

    recon_images = []
    for x in X_distorted:
        k_x = np.exp(-gamma * np.linalg.norm(X - x, axis=1) ** 2)
        k_x_centered = k_x - K_mean
        x_proj = np.dot(eigenvecs.T, k_x_centered)
        x_pre = compute_preimage_part_c(x_proj, eigenvecs, eigenvals, X, gamma)
        recon_images.append(x_pre.reshape(64, 64))
    
    return np.array(recon_images)


def display_reconstructions(original_images, distorted_images, reconstructed_images, label, indices=[20, 80, 140]):

    original_images = original_images.reshape(original_images.shape[0], 64, 64)
    distorted_images = distorted_images.reshape(distorted_images.shape[0], 64, 64)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i, idx in enumerate(indices):
        axes[i, 0].imshow(original_images[idx], cmap='gray')
        axes[i, 0].set_title(f"Original {idx}")
        axes[i, 1].imshow(distorted_images[idx], cmap='gray')
        axes[i, 1].set_title(f"Distorted {idx}")
        axes[i, 2].imshow(reconstructed_images[idx], cmap='gray')
        axes[i, 2].set_title(f"{label} Reconstructed {idx}")

    for ax in axes.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    
    folder_path = r"anatomicalSegmentations"  
    X = load_dataset(folder_path)
    print("Loaded dataset shape:", X.shape) # (150, 4096)

    """
    sample_image = X[97, :]
    sample_image = sample_image.reshape((64, 64))
    plt.imshow(sample_image, cmap='gray')
    """

    #plot_partA(*(apply_pca(X)))
    eigenvals, eigenvecs, K = kernel_pca(X, n_components=X.shape[0])
    print("K = ", K.shape) 
    print("eigenvals = ", eigenvals.shape) 
    print("eigenvecs = ", eigenvecs.shape) 
    #plot_eigen_spectrum(eigenvals)

    #pre_image = get_pre_image(eigenvecs, X, gamma = 1.0)
    #plot_pre_image(pre_image)

    folder_path = r"anatomicalSegmentationsDistorted"  
    X_distorted = load_dataset(folder_path)
    print("Loaded distorted dataset shape:", X_distorted.shape) # (150, 4096)

    #reconstructed_images = pca_reconstruct(X, X_distorted)
    #print("reconstructed_images = ", reconstructed_images.shape)
    #display_reconstructions(X, X_distorted, reconstructed_images, label="PCA")

    start_time = time.time()
    X = X[:25, :]
    X_distorted = X_distorted[:25, :]

    reconstructed_images_kernel = kernel_pca_reconstruct(X, X_distorted)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"For 50 images Time elapsed: {elapsed_time:.2f} minutes")
    print("reconstructed_images_kernel = ", reconstructed_images_kernel.shape)
    display_reconstructions(X, X_distorted, reconstructed_images_kernel, label="Kernel PCA", indices=[10,20,21])
