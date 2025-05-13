import sys,os
import math
import numpy as np
import scipy
import time
from random import randrange
import algorithm

def target(x,dim):
    # Define the parameters of the two t-distributions
    mean_1 = [10,0]
    mean_2 = [-10,0]
    # Calculate the probability density function of each t-distribution
    gaussian_1 = scipy.stats.multivariate_normal.pdf(x, mean=mean_1, cov=[[1,0],[0,1]])
    gaussian_2 = scipy.stats.multivariate_normal.pdf(x, mean=mean_2, cov=[[1,0],[0,1]])
    # Calculate the likelihood of the mixture distribution
    likelihood =  gaussian_1 + gaussian_2
    # Return the log-likelihood
    return likelihood

outdir = 'Gaussian_2D'
dim = 2
low_bound = -50*np.ones(dim)
high_bound = 50*np.ones(dim)
sigma = 1.2
retrains = 100
samples_per_retrain = 1000
vae_prob = 0.2
plot_initial = False

# #Generate pure Metropolis-Hastings comparison samples
start_time = time.time()
sigma_pureMH = 1.2
comparison_sample_size = 100000
pureMH = algorithm.PureMH(target,comparison_sample_size,dim,
low_bound,high_bound,sigma_pureMH = sigma_pureMH)
comparison_samples = np.array(pureMH.run())
isExist = os.path.exists(outdir)
if not isExist:
    os.mkdir(outdir)
np.savetxt(outdir + '/' + 'comparison_samples.dat',comparison_samples)
end_time = time.time()
time_taken_MH = end_time - start_time
print("Time taken MH:", time_taken_MH)
# Pure MH samples = 99999 Pure MH Acceptance efficiency =  0.48561
# Time taken MH: 113.19518852233887
# #Gaussian initial samples, we pretend we don't know the relative weights of the modes
initial_sample_size = 2000
maxima = np.array([[10,0],[-10,0]])
covs = np.stack([np.eye(2), np.eye(2)], axis=0)
initial_samples = []
for i in range(0,len(maxima)):
    means = maxima[i]
    cov = covs[i]
    initial_samples.extend(np.random.multivariate_normal(means,cov,
    initial_sample_size//len(maxima)))
initial_samples = np.array(initial_samples)
input_dim = 2
inter_dim = 48
inter_dim1 = 32
latent_dim = 4
batch_size = 128
epochs = 20
bins = 20
lr = 0.001
desired_sample_size = 5000
bandwidth = 0.5
start_time_vae = time.time()
VAE_MH_sampler = algorithm.MH_VAE(target=target,dim=dim,input_dim=input_dim,inter_dim=inter_dim,inter_dim1=inter_dim1, latent_dim=latent_dim, epochs=epochs, low_bound=low_bound,high_bound=high_bound,
                         initial_samples=initial_samples,retrains=retrains,samples_per_retrain=samples_per_retrain,outdir = outdir,vae_prob = vae_prob,desired_sample_size = desired_sample_size,bandwidth = bandwidth,lr=lr,batch_size=batch_size,sigma = sigma)
samples,vae_samples,MH_samples,vae_rate = VAE_MH_sampler.run()
end_time_vae = time.time()
time_taken_vae = end_time_vae - start_time_vae
print("Time taken vae:", time_taken_vae)

isExist = os.path.exists(outdir + '/' + 'samples.dat')
if isExist:
    os.remove(outdir + '/' + 'samples.dat')
np.savetxt(outdir + '/' + 'samples.dat',samples)

isExist = os.path.exists(outdir + '/' + 'vae_samples.dat')
if isExist:
    os.remove(outdir + '/' + 'vae_samples.dat')
np.savetxt(outdir + '/' + 'vae_samples.dat',vae_samples)

isExist = os.path.exists(outdir + '/' + 'MH_samples.dat')
if isExist:
    os.remove(outdir + '/' + 'MH_samples.dat')
np.savetxt(outdir + '/' + 'MH_samples.dat',MH_samples)

isExist = os.path.exists(outdir + '/' + 'vae_acceptance.dat')
if isExist:
    os.remove(outdir + '/' + 'vae_acceptance.dat')
np.savetxt(outdir + '/' + 'vae_acceptance.dat',vae_rate)
# VAE acceptance efficiency = 0.7611940298507462
# Metropolis acceptance efficiency = 0.5006257822277848
# Previous Retrain Time = 4.0 seconds
# Total Retrain Time = 512.0 seconds
# Time taken vae: 515.1134822368622