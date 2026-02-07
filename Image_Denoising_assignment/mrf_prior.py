import numpy as np

"""
This script contains the gradients of all the prior models as mentioned in the assignment.
"""

def get_neighbors(noiseless_image: np.array):
    """
    Compute all neighbours (i,j) in the noiseless image.
    """

    try:
        H, W, _ = noiseless_image.shape
        #print("rgb image")
    except:
        H, W = noiseless_image.shape
        #print("gray image")
    
    neighbors = list()

    for row in range(H):
        for col in range(W):
            index = row*H + col

            neighbors.append((index, row * W + (col - 1) % W))  # Left
            neighbors.append((index, row * W + (col + 1) % W))  # Right
            neighbors.append((index, ((row - 1) % H) * W + col))  # Up
            neighbors.append((index, ((row + 1) % H) * W + col))  # Down

    return neighbors

def quadratic_weighted_prior_gradient(alpha: float,
                                    gamma: float,
                                    noiseless_image: np.array):
    """
    Computes the gradient of quadratic prior using the weight = alpha
    """

    H, W = noiseless_image.shape
    x = noiseless_image.flatten()
    gradient = np.zeros_like(x)

    neighbors = get_neighbors(noiseless_image=noiseless_image)

    for (i, j) in neighbors:
        gradient[i] += 2 * alpha * (x[i] - x[j])
        gradient[j] -= 2 * alpha * (x[i] - x[j]) # equal and opposite force 

    return gradient.reshape(H, W)

def quadratic_prior_value(alpha: float, gamma: float, noiseless_image: np.array):
    neighbors = get_neighbors(noiseless_image)
    x = noiseless_image.flatten()
    log_prior = alpha * np.sum([(x[i] - x[j]) ** 2 for (i, j) in neighbors])
    return log_prior

def derivative_of_huber(u: float, gamma: float):
    """
    Computes the derivative of huber adaptive discontinuous function at any u.
    """
    if abs(u) <= gamma:
        return u
    else:
        return gamma*np.sign(u)

def huber_prior_value(alpha: float, gamma: float, noiseless_image: np.array):
    neighbors = get_neighbors(noiseless_image)
    x = noiseless_image.flatten()
    log_prior = 0

    for (i, j) in neighbors:
        u = x[i] - x[j]
        if abs(u) <= gamma:
            log_prior += alpha * 0.5 * u**2
        else:
            log_prior += alpha * (gamma * abs(u) - 0.5 * gamma**2)
    
    return log_prior

def huber_weighted_prior_gradient(alpha: float, gamma: float, noiseless_image: np.array):

    """
    Computes gradients for huber adaptive discontinuous function.
    """

    H, W = noiseless_image.shape
    neighbors = get_neighbors(noiseless_image=noiseless_image)
    x = noiseless_image.flatten()
    gradient = np.zeros_like(x)

    for (i, j) in neighbors:
        gradient[i] += alpha*derivative_of_huber(u=(x[i] - x[j]), gamma=gamma)
        gradient[j] -= alpha*derivative_of_huber(u=(x[i] - x[j]), gamma=gamma) # equal and opposite force
    
    return gradient.reshape((H,W))

def derivative_discontinuous_adaptive(u: float, gamma: float):
    """
    Computes the derivative of third prior given in the assignment.
    """

    return (np.sign(u))*(gamma - gamma/(1 + (abs(u)/gamma)))

def third_prior_gradient(alpha: float, gamma: float, noiseless_image: np.array):
    """
    Computes the gradient of third prior given in the assignment.
    """

    H, W = noiseless_image.shape
    x = noiseless_image.flatten()
    neighbors = get_neighbors(noiseless_image=noiseless_image)
    gradient = np.zeros_like(x)

    for (i, j) in neighbors:
        gradient[i] += alpha*derivative_discontinuous_adaptive(u=(x[i] - x[j]), gamma=gamma)
        gradient[j] -= alpha*derivative_discontinuous_adaptive(u=(x[i] - x[j]), gamma=gamma) # equal and opposite force
    
    return gradient.reshape((H, W))

def third_prior_value(alpha: float, gamma: float, noiseless_image: np.array):
    neighbors = get_neighbors(noiseless_image)
    x = noiseless_image.flatten()

    log_prior = 0

    for(i,j) in neighbors:
        u = x[i] - x[j]
        log_prior += gamma*abs(u) - (gamma**2 * np.log(1 + (abs(u) / gamma)))
    
    return log_prior



def q3A_prior_value(noiseless_image: np.array, alpha: float, gamma: float):
    
    assert noiseless_image.shape[2] == 3, f"Image is not RGB image, found image with shape = {noiseless_image.shape}"
    
    log_prior = 0
    
    H, W, C = noiseless_image.shape
    x = noiseless_image.reshape(-1, C)

    neighbors = get_neighbors(noiseless_image)

    for (i, j) in neighbors:
        diff = x[i%(H*W)] - x[j%(H*W)] # here x[i].shape = (3,)
        log_prior += alpha*np.sum(diff**2)
    
    return log_prior

def q3A_prior_gradient(alpha: float, gamma: float, noiseless_image: np.array):
    
    assert noiseless_image.shape[2] == 3, f"Image is not RGB image, found image with shape = {noiseless_image.shape}"
   
    H, W, C = noiseless_image.shape
    x = noiseless_image.reshape(-1, C)
    neighbors = get_neighbors(noiseless_image)

    gradient = np.zeros_like(x)

    for (i, j) in neighbors:
        diff = x[i%(H*W)] - x[j%(H*W)]
        gradient[i%(H*W)] += alpha*2*diff
        gradient[j%(H*W)] -= alpha*2*diff

    return gradient.reshape(H,W,C)

def q3B_prior_value(noiseless_image: np.array, alpha: float, gamma: float):
    
    assert noiseless_image.shape[2] == 3, f"Image is not RGB image, found image with shape = {noiseless_image.shape}"
    
    log_prior = 0
    
    H, W, C = noiseless_image.shape
    x = noiseless_image.reshape(-1, C)

    neighbors = get_neighbors(noiseless_image)

    for (i, j) in neighbors:
        diff = x[i%(H*W)] - x[j%(H*W)] # here x[i].shape = (3,)
        log_prior += alpha*np.sqrt(np.sum(diff**2))
    
    return log_prior

def q3B_prior_gradient(alpha: float, gamma: float, noiseless_image: np.array):

    assert noiseless_image.shape[2] == 3, f"Image is not RGB image, found image with shape = {noiseless_image.shape}"

    H, W, C = noiseless_image.shape
    x = noiseless_image.reshape(-1, C)
    gradient = np.zeros_like(x)
    neighbors = get_neighbors(noiseless_image)

    for (i, j) in neighbors:
        diff = x[i%(H*W)] - x[j%(H*W)]
        norm = np.linalg.norm(diff, ord=2)

        # avoid dividing by zero
        if norm > 1e-6:
            gradient[i%(H*W)] += alpha*(diff / norm)
            gradient[j%(H*W)] -= alpha*(diff / norm)
        else:
            gradient[i%(H*W)] += np.zeros_like(diff)
            gradient[j%(H*W)] -= np.zeros_like(diff)
    
    return gradient.reshape(H,W,C)

def q3C_prior_value(noiseless_image: np.array, alpha: float, gamma: float):
    
    assert noiseless_image.shape[2] == 3, f"Image is not RGB image, found image with shape = {noiseless_image.shape}"
    
    H, W, C = noiseless_image.shape
    neighbors = get_neighbors(noiseless_image)
    x = noiseless_image.reshape(-1, C)
    log_prior = 0

    for (i, j) in neighbors:
        u = np.sum(x[i%(H*W)] - x[j%(H*W)])
        if abs(u) <= gamma:
            log_prior += alpha * 0.5 * u**2
        else:
            log_prior += alpha * (gamma * abs(u) - 0.5 * gamma**2)
    
    return log_prior

def q3C_prior_gradient(alpha: float, gamma: float, noiseless_image: np.array):
    
    assert noiseless_image.shape[2] == 3, f"Image is not RGB image, found image with shape = {noiseless_image.shape}"
    
    H, W, C = noiseless_image.shape
    neighbors = get_neighbors(noiseless_image=noiseless_image)
    x = noiseless_image.reshape(-1, C)
    gradient = np.zeros_like(x)

    for (i, j) in neighbors:
        gradient[i%(H*W)] += alpha*derivative_of_huber(u=np.sum(x[i%(H*W)] - x[j%(H*W)]), gamma=gamma)
        gradient[j%(H*W)] -= alpha*derivative_of_huber(u=np.sum(x[i%(H*W)] - x[j%(H*W)]), gamma=gamma) # equal and opposite force
    
    return gradient.reshape(H,W,C)

if __name__=="__main__":
    
    test_image = np.ones((528,393,3))
    #print(test_image.shape)
    
    alpha = 1.0 

    #computed_gradient = q3C_prior_gradient(alpha=alpha, gamma=0.1, noiseless_image=test_image)
    value = q3A_prior_value(test_image, alpha, 1.0)

    print(value)
