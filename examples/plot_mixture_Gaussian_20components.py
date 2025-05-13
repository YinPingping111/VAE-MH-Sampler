import numpy as np
import matplotlib.pyplot as plt

# Load data from file
def load_samples(file_path):
    return np.loadtxt(file_path)
save_path = "D:/desk/VAE-MH/VAE/examples/Mixture_Gaussian_2D_20components"
# Load Random-Walk MH and VAE-MH samples
random_walk_mh_samples = load_samples("D:/desk/VAE-MH/VAE/examples/Mixture_Gaussian_2D_20components/comparison_samples.dat")
vae_mh_samples = load_samples("D:/desk/VAE-MH/VAE/examples/Mixture_Gaussian_2D_20components/samples.dat")

# Ensure the dimensions of the samples match
assert random_walk_mh_samples.shape[1] == vae_mh_samples.shape[1], "Dimensions of samples must match"

# Define burn-in period (number of initial samples to discard)
burn_in = 10000

# Remove burn-in samples
random_walk_mh_samples = random_walk_mh_samples[burn_in:]
vae_mh_samples = vae_mh_samples[(burn_in + 2000):]

# Function to plot the distribution of samples
def plot_sample_distribution(random_walk_mh_samples, vae_mh_samples):
    plt.figure(figsize=(10, 5))

    # Subplot 1: Random-Walk MH samples
    plt.subplot(1, 2, 1)
    plt.scatter(random_walk_mh_samples[:, 0], random_walk_mh_samples[:, 1], s=1, c='black', alpha=0.6, label="Random-Walk MH")
    plt.title("Random-Walk MH Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Subplot 2: VAE-MH samples
    plt.subplot(1, 2, 2)
    plt.scatter(vae_mh_samples[:, 0], vae_mh_samples[:, 1], s=1, c='black', alpha=0.6, label="VAE-MH")
    plt.title("VAE-MH Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    # Save as PDF
    plt.savefig(f"{save_path}/algorithms_comparison.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

# Function to plot the trajectory of samples
def plot_trajectory(random_walk_mh_samples, vae_mh_samples, num_samples=10000):
    plt.figure(figsize=(10, 5))

    # Subplot 1: Trajectory of Random-Walk MH
    plt.subplot(1, 2, 1)
    plt.plot(random_walk_mh_samples[:num_samples, 0], random_walk_mh_samples[:num_samples, 1], color='blue', linewidth=0.5, alpha=0.7, label="Random-Walk MH")
    plt.scatter(random_walk_mh_samples[:num_samples, 0], random_walk_mh_samples[:num_samples, 1], s=1, c='blue', alpha=0.3)
    plt.title("Trajectory of Random-Walk MH (First 10,000 Samples)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Subplot 2: Trajectory of VAE-MH
    plt.subplot(1, 2, 2)
    plt.plot(vae_mh_samples[:num_samples, 0], vae_mh_samples[:num_samples, 1], color='blue', linewidth=0.5, alpha=0.7, label="VAE-MH")
    plt.scatter(vae_mh_samples[:num_samples, 0], vae_mh_samples[:num_samples, 1], s=1, c='blue', alpha=0.3)
    plt.title("Trajectory of VAE-MH (First 10,000 Samples)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(f"{save_path}/Trajectory_comparison.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

# Plot the sample distributions
plot_sample_distribution(random_walk_mh_samples, vae_mh_samples)

# Plot the trajectories
plot_trajectory(random_walk_mh_samples, vae_mh_samples)