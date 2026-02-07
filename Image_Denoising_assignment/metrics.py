import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import h5py

def rrmse(noiselss_image: np.array, estimated_image: np.array):

    """
    Computes the relative root mean squared error between two images.
    """

    A = np.abs(noiselss_image)
    B = np.abs(estimated_image)

    numerator = np.sqrt(np.sum((A - B) ** 2))
    denominator = np.sqrt(np.sum(A ** 2))

    epsilon = 1e-7

    return numerator/(denominator + epsilon)

if __name__=="__main__":

    print("\n\nFetching data ...............................")
    with h5py.File(r'data\assignmentImageDenoising_microscopy.mat', 'r') as f:
        
        print(list(f.keys()))

        noiseless_image = f['microscopyImageOrig'][:]
        noisy_image = f['microscopyImageNoisyScale350sigma0point06'][:]
    print("Successfully fetched noisy and noiseless images!!\n")

    C, H, W = noisy_image.shape
    noisy_image = noisy_image.reshape(H, W, C)
    noiseless_image = noiseless_image.reshape(H, W, C)

    noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min())
    noiseless_image = (noiseless_image - noiseless_image.min()) / (noiseless_image.max() - noiseless_image.min())
    
    image_path = rf"results3\noiseless.png"
    plt.imsave(image_path, noiseless_image, cmap="jet")

    image_path = rf"results3\noisy.png"
    plt.imsave(image_path, noisy_image, cmap="jet")

    print("Successfully fetched noisy and noiseless images!!\n")
    print("RRMSE between noisy and noiseless image = ", rrmse(noiseless_image, noisy_image))

