from scipy.special import i0, i1
import numpy as np

"""
This script contains all the noise models which are essentially modeled as log likelihood of some 
popular models. In addition to the likelihood, their respective gradient functions can also be
defined. Here only rician log likelihood is present. Others can also be added.
"""

def rician_weighted_log_likelihood(noisy_image: np.array,
                                noiseless_image: np.array,
                                alpha: float,
                                sigma: float = 1.0):
    """
    log likelihood = summation over all pixels (ln(alpha*yi/sigma^2) - alpha*(yi^2 + xi^2)/2*sigma^2 + ln(I0(alpha*xi*yi/sigma^2)))
    When passing arguments during the functional call, we would give (1-alpha) value for the parameter named 'alpha'.
    """
    epsilon = 1e-7
    term1 = np.log(np.maximum(alpha * (noisy_image + epsilon) / (sigma**2), 1e-10))
    term2 = -alpha * (noisy_image**2 + noiseless_image**2) / (2 * sigma**2)
    term3 = np.log(i0(alpha * noisy_image * noiseless_image / sigma**2))
    return np.sum(term1 + term2 + term3)


def rician_weighted_gradient(noisy_image: np.array,
                            noiseless_image: np.array,
                            alpha: float,
                            sigma: float = 1.0):
    """
    If you differentiate the rician weighted likelihood, you'll get the below gradient.
    gradient = -alpha*xi/sigma^2 + (alpha*yi/sigma^2)*(I1(alpha*xi*yi/sigma^2)/I0(alpha*xi*yi/sigma^2))
    When passing arguments during the functional call, we would give (1-alpha) value for the parameter named 'alpha'.
    """
    numerator = i1(np.maximum(alpha * noisy_image * noiseless_image / sigma**2, 1e-10))
    denominator = (i0(np.maximum(alpha * noisy_image * noiseless_image / sigma**2, 1e-10))+1e-7)
    bessel_ratio = numerator / denominator
    gradient = (-alpha * noiseless_image / sigma**2) + (alpha * noisy_image / sigma**2) * bessel_ratio
    return gradient

if __name__=="__main__":
    test_image = np.ones((256,256,3))
    print(rician_weighted_gradient(test_image, test_image, 0.1).shape)