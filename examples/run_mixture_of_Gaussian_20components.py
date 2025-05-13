import scipy.stats
import os
import numpy as np
import scipy
import time
import algorithm
import scipy.stats
import scipy.special

def target(x,dim):
    dimension = 2
    sigma = 0.1
    # Generate means for each component, uniformly distributed within the given bounds
    means = np.array([
        [2.18,5.76],
        [8.67,9.59],
        [4.24,8.48],
        [8.41,1.68],
        [3.93,8.82],
        [3.25,3.47],
        [1.70,0.50],
        [4.59,5.60],
        [6.91,5.81],
        [6.87,5.40],
        [5.41,2.65],
        [2.70,7.88],
        [4.98,3.70],
        [1.14,2.39],
        [8.33,9.50],
        [4.93,1.50],
        [1.83,0.09],
        [2.26,0.31],
        [5.54,6.86],
        [1.69,8.11]
    ])
    # Define the covariance matrix (identity scaled by sigma^2)
    covariance = np.eye(dimension) * sigma**2
    # All components have equal weight
    weight = 0.05
    # Calculate the total probability density of the mixture at point x
    total_density = 0
    for mean in means:
        total_density += weight * scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=covariance)
    return total_density
outdir = 'Mixture_Gaussian_2D_20components'
dim = 2
low_bound = -20 * np.ones(dim)
high_bound = 20 * np.ones(dim)
sigma = 0.13
retrains = 100
samples_per_retrain = 1000
vae_prob = 0.2

# # #Generate pure Metropolis-Hastings comparison samples
# start_time = time.time()
# sigma_pureMH = 0.13
# comparison_sample_size = 100000
# pureMH = algorithm.PureMH(target,comparison_sample_size,dim,
# low_bound,high_bound,sigma_pureMH = sigma_pureMH)
# comparison_samples = np.array(pureMH.run())
# isExist = os.path.exists(outdir)
# if not isExist:
#     os.mkdir(outdir)
# np.savetxt(outdir + '/' + 'comparison_samples.dat',comparison_samples)
# end_time = time.time()
# time_taken_MH = end_time - start_time
# print("Time taken MH:", time_taken_MH)
# Pure MH samples = 99999 Pure MH Acceptance efficiency =  0.5072
# Time taken MH: 565.9853012561798
# #Gaussian initial samples, we pretend we don't know the relative weights of the modes
initial_sample_size = 2000
maxima = np.array([
        [2.18,5.76],
        [8.67,9.59],
        [4.24,8.48],
        [8.41,1.68],
        [3.93,8.82],
        [3.25,3.47],
        [1.70,0.50],
        [4.59,5.60],
        [6.91,5.81],
        [6.87,5.40],
        [5.41,2.65],
        [2.70,7.88],
        [4.98,3.70],
        [1.14,2.39],
        [8.33,9.50],
        [4.93,1.50],
        [1.83,0.09],
        [2.26,0.31],
        [5.54,6.86],
        [1.69,8.11]])
num_components = len(maxima)
covs = np.stack([np.eye(2) for _ in range(num_components)])  # Creates an identity matrix for each component
initial_samples = []
for i in range(num_components):
    means = maxima[i]
    cov = covs[i]
    # Generate a number of samples for each component
    samples_from_component = np.random.multivariate_normal(means, cov, initial_sample_size // num_components)
    initial_samples.extend(samples_from_component)

initial_samples = np.array(initial_samples)
input_dim = 2
inter_dim = 48
inter_dim1 = 32
latent_dim = 4
batch_size = 128
epochs = 20
lr = 0.001
desired_sample_size = 10000
bandwidth = 0.5
start_time_vae = time.time()
VAE_MH_sampler = algorithm.MH_VAE(target=target,dim=dim,input_dim=input_dim,inter_dim=inter_dim,inter_dim1=inter_dim1, latent_dim=latent_dim, epochs=epochs, low_bound=low_bound,high_bound=high_bound,
                         initial_samples=initial_samples,retrains=retrains,samples_per_retrain=samples_per_retrain,outdir = outdir,vae_prob = vae_prob,desired_sample_size = desired_sample_size,bandwidth = bandwidth,lr=lr,batch_size=batch_size,sigma = sigma)
samples, vae_samples, MH_samples, vae_rate = VAE_MH_sampler.run()
end_time_vae = time.time()
time_taken_vae = end_time_vae - start_time_vae
print("Time taken vae:", time_taken_vae)
isExist = os.path.exists(outdir + '/' + 'samples.dat')
if isExist:
    os.remove(outdir + '/' + 'samples.dat')
np.savetxt(outdir + '/' + 'samples.dat', samples)

isExist = os.path.exists(outdir + '/' + 'vae_samples.dat')
if isExist:
    os.remove(outdir + '/' + 'vae_samples.dat')
np.savetxt(outdir + '/' + 'vae_samples.dat', vae_samples)

isExist = os.path.exists(outdir + '/' + 'MH_samples.dat')
if isExist:
    os.remove(outdir + '/' + 'MH_samples.dat')
np.savetxt(outdir + '/' + 'MH_samples.dat', MH_samples)

isExist = os.path.exists(outdir + '/' + 'vae_acceptance.dat')
if isExist:
    os.remove(outdir + '/' + 'vae_acceptance.dat')
np.savetxt(outdir + '/' + 'vae_acceptance.dat', vae_rate)
# VAE acceptance efficiency = 0.15789473684210525
# Metropolis acceptance efficiency = 0.4654320987654321
# Previous Retrain Time = 8.0 seconds
# Total Retrain Time = 763.0 seconds
# Time taken vae: 765.3335039615631