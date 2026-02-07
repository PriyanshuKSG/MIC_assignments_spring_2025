import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def plot_shapes(shapes, num_shapes=5):
    plt.figure(figsize=(6, 10))
    for i in range(num_shapes):
        shape = shapes[i]  # Select the i-th shape (32 points, 2D)
        plt.subplot(num_shapes, 1, i+1)
        plt.scatter(shape[:, 0], shape[:, 1], marker='o', color='b')
        plt.plot(shape[:, 0], shape[:, 1], linestyle='-', color='gray')  # Connect points
        plt.title(f"Shape {i+1}")
        plt.axis("equal")
    plt.show()

def plot_all_shapes(shapes):
    plt.figure(figsize=(6,6))
    for shape in shapes:
        plt.plot(shape[:, 0], shape[:, 1], linestyle='-', alpha=0.3, color='gray')
    
    plt.title("Initial Shapes")
    plt.axis("equal")
    plt.legend()
    plt.show()

def compute_procrustes_mean(shapes, max_iter=10, tol=1e-6, distace_type = "squared"):
    M, N, D = shapes.shape  

    centered_shapes = shapes - np.mean(shapes, axis=1, keepdims=True)
    print("centered shapes = ", centered_shapes.shape)

    norms = np.linalg.norm(centered_shapes.reshape(M, -1), axis=1, keepdims=True)
    print("norms = ", norms.shape)
    norms_broadcasted = norms[:, np.newaxis]
    print("norms_broadcasted = ", norms_broadcasted.shape)
    normalized_shapes = centered_shapes / norms_broadcasted

    print("Normalized shapes = ", normalized_shapes.shape)

    mean_shape = np.mean(normalized_shapes, axis=0) # initial no weights
    print("Mean shape = ", mean_shape.shape)

    for iteration in range(max_iter):
        aligned_shapes = []
        procrustes_distances = []
        for shape in normalized_shapes:
            
            #print("shape = ", shape.shape)

            U, _, Vt = np.linalg.svd(shape.T @ mean_shape)
            R_opt = U @ Vt  
            new_shape = shape @ R_opt
            aligned_shapes.append(new_shape)
            difference = (new_shape - mean_shape)**2
            if distace_type != "squared":
                difference = np.sqrt(difference)
            procrustes_distances.append(np.sum(difference))

        aligned_shapes = np.array(aligned_shapes)
        print("aligned_shapes = ", aligned_shapes.shape)

        procrustes_distances = np.array(procrustes_distances) # (357,)
        print("procrustes_distances = ", procrustes_distances.shape)

        weights = 1.0 / (1.0 + procrustes_distances)
        weights = np.exp(weights)
        weights = np.exp(weights)
        weights = np.exp(weights)
        print(weights[290:310])
        weights = weights[:, np.newaxis, np.newaxis] # (357, 1, 1)
        print("weights = ", weights.shape)

        new_mean_shape = np.sum(weights * aligned_shapes, axis=0) / np.sum(weights) # (32, 2)

        if np.linalg.norm(new_mean_shape - mean_shape) < tol:
            break

        mean_shape = new_mean_shape  # Update mean shape

    return mean_shape, aligned_shapes


def plot_mean_and_aligned_shapes(mean_shape, aligned_shapes):
    plt.figure(figsize=(6,6))
    
    # Plot all aligned shapes
    for shape in aligned_shapes:
        plt.plot(shape[:, 0], shape[:, 1], linestyle='-', alpha=0.3, color='gray')

    # Plot the estimated mean shape
    plt.scatter(mean_shape[:, 0], mean_shape[:, 1], color='red', marker='o', label="Mean Shape")
    plt.plot(mean_shape[:, 0], mean_shape[:, 1], linestyle='-', color='red')

    plt.title("Estimated Mean Shape and Aligned Shapes")
    plt.axis("equal")
    plt.legend()
    plt.show()

def plot_procrustes_dist(procrustes_distances, label = ""):
    plt.figure(figsize=(10,6))
    plt.plot([i+1 for i in range(procrustes_distances.shape[0])], procrustes_distances, color='blue')
    plt.scatter([i+1 for i in range(procrustes_distances.shape[0])], procrustes_distances, color='red')
    plt.title("Procrustes Distances for all shapes")
    plt.ylabel(f"{label} Squared Procrustes Distance")
    plt.xlabel("Shape number")
    plt.show()

    plt.hist(procrustes_distances, bins=30, color='blue', alpha=0.7)
    plt.xlabel(f"{label} Squared Procrustes Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Procrustes Distances")
    plt.show()

if __name__=="__main__":
    
    filepath = r"data\robustShapeMean2D.mat" 
    data = scipy.io.loadmat(filepath)

    distance_type = "not squared"

    pointsets = data['pointsets']
    #print(type(pointsets))
    #print(pointsets.shape) 

    plot_shapes(pointsets)
    plot_all_shapes(pointsets)

    #pointsets = pointsets[:300, :, :] # remove outliers

    mean_shape, aligned_shapes = compute_procrustes_mean(pointsets, distace_type=distance_type)
    plot_mean_and_aligned_shapes(mean_shape, aligned_shapes)

    print("mean_shape = ", mean_shape.shape)
    print("aligned_shape = ", aligned_shapes.shape)

    if distance_type == "squared":
        procrustes_distances = np.sum((aligned_shapes - mean_shape) ** 2, axis=(1, 2))
    elif distance_type == "not squared":
        procrustes_distances = np.sum(np.linalg.norm(aligned_shapes - mean_shape, axis=2), axis=1)
    
    print("procrustes_distances = ", procrustes_distances.shape)
    print("Min in distance = ", np.min(procrustes_distances))
    print("Max in distance = ", np.max(procrustes_distances))
    print("Mean in distance = ", np.mean(procrustes_distances))
    print("Variability in distance (STD) = ", np.std(procrustes_distances))

    plot_procrustes_dist(procrustes_distances, label="Not")