import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to create a random dataset
def create_dataset(n_samples, n_features, centers):
    data = []
    np.random.seed(42)
    
    for center in centers:
        points = np.random.randn(n_samples, n_features) + center
        data.append(points)
    
    # Use np.concatenate to combine the lists into one array
    return np.concatenate(data, axis=0)  # Concatenate along the first axis (row-wise)

# Function to assign clusters based on nearest centroid
def assign_clusters(data, centroids):
    distances = np.zeros((len(data), len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(data - centroid, axis=1)
    return np.argmin(distances, axis=1)

# Function to update centroids based on the assigned clusters
def update_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = data[labels == i].mean(axis=0)
    return centroids

# Function to run the K-Means algorithm
def kmeans(data, k=3, max_iters=100):
    np.random.seed(42)
    random_idx = np.random.choice(len(data), k, replace=False)
    centroids = data[random_idx]
    
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)  # Assign clusters
        old_centroids = centroids.copy()  # Store old centroids
        centroids = update_centroids(data, labels, k)  # Update centroids
        
        # Check for convergence
        if np.all(old_centroids == centroids):
            break
            
    return labels, centroids

# Function to plot the clusters
def plot_clusters(data, labels, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
    plt.title('K-Means Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

# Main function to execute the K-Means algorithm
def main():
    # Create a dataset with 3 centers
    centers = [[2, 2], [8, 8], [5, 12]]
    data = create_dataset(100, 2, centers)
    
    # Run K-Means
    k = 3  # Number of clusters
    labels, centroids = kmeans(data, k)
    
    # Plot the results
    plot_clusters(data, labels, centroids)

if __name__ == "__main__":
    main()
