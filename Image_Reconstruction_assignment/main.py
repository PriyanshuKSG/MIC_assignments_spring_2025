from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import cv2

def myXrayIntegration(image: np.array, t, theta_degrees, interp_func, delta_s: float = 1.0):
    
    H, W = image.shape

    theta_rad = np.deg2rad(theta_degrees)
    s_vals = np.arange(-1*max(W, H), max(W, H), delta_s)
    x_vals = t * np.cos(theta_rad) - s_vals * np.sin(theta_rad)
    y_vals = t * np.sin(theta_rad) + s_vals * np.cos(theta_rad)

    sampled_values = interp_func(np.vstack((y_vals, x_vals)).T)
    
    sampled_values = np.array(sampled_values)
    return np.sum(sampled_values) * delta_s

def myXrayCTRadonTransform(image: np.array, delta_s: float = 1.0, delta_t: int = 5, delta_theta: int = 5):

    H, W = image.shape
    
    theta_degrees = np.array([i for i in range(0, 176, delta_theta)])
    t_vals = np.array([i for i in range(-90, 91, delta_t)])
    interp_func = RegularGridInterpolator((np.arange(H), np.arange(W)), image, method="linear", bounds_error=False, fill_value=0)

    radon_transform = np.array([
        [myXrayIntegration(image=image, t=t, theta_degrees=theta, interp_func=interp_func, delta_s=delta_s) for t in t_vals]
        for theta in theta_degrees
    ])
    
    return radon_transform

def plot_radon_comparison(image: np.array, delta_t: int = 5, delta_theta: int = 5):
    delta_s_values = [0.5, 1.0, 3.0]
    theta_values = [0, 90]  

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    for i, delta_s in enumerate(delta_s_values):
        radon_image = myXrayCTRadonTransform(image=image, delta_s=delta_s, delta_t=delta_t, delta_theta=delta_theta)

        ax_img = axes[i, 0]
        ax_img.imshow(radon_image, cmap="gray", aspect="auto")
        ax_img.set_title(f"Radon Transform (∆s={delta_s})")
        ax_img.set_xlabel("t values")
        ax_img.set_ylabel("θ values")

        for j, theta in enumerate(theta_values):
            ax_plot = axes[i, j + 1]
            ax_plot.plot(np.arange(-90, 91, delta_t), radon_image[theta // delta_theta], label=f"θ={theta}°")
            ax_plot.set_title(f"1D Profile (θ={theta}°), ∆s={delta_s}")
            ax_plot.set_xlabel("t values")
            ax_plot.set_ylabel("Rf(t, θ)")
            ax_plot.legend()

    plt.tight_layout()
    plt.show()


def myFilter(sinogram: np.array, freq, filter_type='ram-lak', L=None):

    # you get sinogram after applying radon transform to the image
    
    filter_response = np.abs(freq)  # Ram-Lak filter (|w|)
    
    if filter_type == 'shepp-logan':
        filter_response *= (np.sinc(np.pi* freq / (2 * L)) / (np.pi* freq / (2 * L)))
    elif filter_type == 'cosine':
        filter_response *= np.cos(np.pi * freq / (2 * L))  

  
    # Apply cutoff: Zero out frequencies beyond L
    filter_response[freq > L] = 0
    
    filtered_sinogram = np.fft.rfft(sinogram, axis=1)  
    filtered_sinogram *= filter_response  
    filtered_sinogram = np.fft.irfft(filtered_sinogram, axis=1) 
    
    return filtered_sinogram

def plot_question_2(image, sinogram, reconstructed_image, filter_name):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(sinogram, cmap="gray", aspect="auto")
    plt.title(f"{filter_name} sinogram")

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"{filter_name} Backprojection")

    plt.show()

def add_gaussian_noise(image, kernel_size, sigma0):
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma0)

def rrmse(A, B):
    numerator = np.sqrt(np.sum((A - B) ** 2))
    denominator = np.sqrt(np.sum(A ** 2))
    return numerator / denominator

if __name__=="__main__":

    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (128, 128), mode='reflect', anti_aliasing=True)
    
    """
    # Question 1, part c) and d)
    values = [(1,1), (5,5), (10,10)]
    for delta_t, delta_theta in values:
        print(f"delta_t={delta_t}, delta_theta={delta_theta}")
        plot_radon_comparison(phantom, delta_t=delta_t, delta_theta=delta_theta)
    """
    
    
    # Question 2, part a)
    theta_degrees = np.array([i for i in range(0, 178, 3)])
    sinogram = radon(phantom, theta=theta_degrees, circle=True)

    _, num_projections = sinogram.shape

    freq = np.fft.rfftfreq(num_projections)  # Positive frequencies only
    wmax = np.max(freq)  # Nyquist frequency
    L = [wmax, wmax/2.0]
      
    for l in L:
        unfiltered_reconstruction = iradon(sinogram, theta=theta_degrees, filter_name=None)
        plot_question_2(image=phantom, sinogram=sinogram, reconstructed_image=unfiltered_reconstruction, filter_name="Unfiltered")

        ram_lak_sinogram = myFilter(sinogram, freq, filter_type='ram-lak', L=l)
        ram_lak_reconstruction = iradon(ram_lak_sinogram, theta=theta_degrees, filter_name=None)
        plot_question_2(image=phantom, sinogram=ram_lak_sinogram, reconstructed_image=ram_lak_reconstruction, filter_name="Ram-Lak")

        shepp_logan_sinogram = myFilter(sinogram, freq, filter_type='shepp-logan', L=l)
        shepp_logan_reconstruction = iradon(shepp_logan_sinogram, theta=theta_degrees, filter_name=None)
        plot_question_2(image=phantom, sinogram=shepp_logan_sinogram, reconstructed_image=shepp_logan_reconstruction, filter_name="Shepp-Logan")

        cosine_sinogram = myFilter(sinogram, freq, filter_type='cosine', L=l)
        cosine_reconstruction = iradon(cosine_sinogram, theta=theta_degrees, filter_name=None)
        plot_question_2(image=phantom, sinogram=cosine_sinogram, reconstructed_image=cosine_reconstruction, filter_name="Cosine")
    
    
    S0 = phantom
    S1 = add_gaussian_noise(phantom, kernel_size=11, sigma0=1.0)
    S5 = add_gaussian_noise(phantom, kernel_size=51, sigma0=5.0)

    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(S0, cmap="gray")
    plt.title("S0")

    plt.subplot(1, 3, 2)
    plt.imshow(S1, cmap="gray")
    plt.title("S1")

    plt.subplot(1, 3, 3)
    plt.imshow(S5, cmap="gray")
    plt.title("S5")

    plt.show()
    """

    theta_degrees = np.array([i for i in range(0, 178, 3)])
    S0_sinogram = radon(S0, theta=theta_degrees, circle=True)
    _, num_projections = S0_sinogram.shape
    freq = np.fft.rfftfreq(num_projections)
    wmax = np.max(freq)  # Nyquist frequency

    
    ram_lak_sinogram = myFilter(S0_sinogram, freq, filter_type='ram-lak', L=wmax)
    R0 = iradon(ram_lak_sinogram, theta=theta_degrees, filter_name=None)

    S1_sinogram = radon(S1, theta=theta_degrees, circle=True)
    ram_lak_sinogram = myFilter(S1_sinogram, freq, filter_type='ram-lak', L=wmax)
    R1 = iradon(ram_lak_sinogram, theta=theta_degrees, filter_name=None)

    S5_sinogram = radon(S5, theta=theta_degrees, circle=True)
    ram_lak_sinogram = myFilter(S5_sinogram, freq, filter_type='ram-lak', L=wmax)
    R5 = iradon(ram_lak_sinogram, theta=theta_degrees, filter_name=None)

    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(R0, cmap="gray")
    plt.title("R0")

    plt.subplot(1, 3, 2)
    plt.imshow(R1, cmap="gray")
    plt.title("R1")

    plt.subplot(1, 3, 3)
    plt.imshow(R5, cmap="gray")
    plt.title("R5")

    plt.show()
    """
    
    print(f"RRMSE(S0, R0) = {rrmse(S0, R0):.4f}")
    print(f"RRMSE(S1, R1) = {rrmse(S1, R1):.4f}")
    print(f"RRMSE(S5, R5) = {rrmse(S5, R5):.4f}")
    
    """
    new_wmax = wmax/50.0
    L_vals = [i*new_wmax for i in range(1, 51)]
    rrmse0, rrmse1, rrmse5 = [], [], []

    for l in L_vals:
        ram_lak_sinogram = myFilter(S0_sinogram, freq, filter_type='ram-lak', L=l)
        R0 = iradon(ram_lak_sinogram, theta=theta_degrees, filter_name=None)
        rrmse0.append(rrmse(S0, R0))

        ram_lak_sinogram = myFilter(S1_sinogram, freq, filter_type='ram-lak', L=l)
        R1 = iradon(ram_lak_sinogram, theta=theta_degrees, filter_name=None)
        rrmse1.append(rrmse(S1, R1))

        ram_lak_sinogram = myFilter(S5_sinogram, freq, filter_type='ram-lak', L=l)
        R5 = iradon(ram_lak_sinogram, theta=theta_degrees, filter_name=None)
        rrmse5.append(rrmse(S5, R5))
    
    plt.figure(figsize=(12, 4))
    plt.plot(L_vals, rrmse0, label="S0")
    plt.plot(L_vals, rrmse1, label="S1")
    plt.plot(L_vals, rrmse5, label="S5")
    plt.xlabel("L")
    plt.ylabel("RRMSE")
    plt.legend()
    plt.show()
"""

