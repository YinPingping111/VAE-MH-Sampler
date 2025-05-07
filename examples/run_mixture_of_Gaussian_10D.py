import sys,os
import math
import numpy as np
import scipy
import time
from random import randrange
import algorithm


def log_likelihood(x,input_dim):
    mean_1 = [8,3,0,0,0,0,0,0,0,0]
    std_1 = 0.6666
    mean_2 = [-2,3,0,0,0,0,0,0,0,0]
    std_2 = 0.3333
    gauss_1 = scipy.stats.multivariate_normal.pdf(x,mean =
    mean_1,cov = [[0.6666,0,0,0,0,0,0,0,0,0],[0,0.6666,0,0,0,0,0,0,0,0],[0,0,0.6666,0,0,0,0,0,0,0],[0,0,0,0.6666,0,0,0,0,0,0],[0,0,0,0,0.6666,0,0,0,0,0],[0,0,0,0,0,0.6666,0,0,0,0],[0,0,0,0,0,0,0.6666,0,0,0],[0,0,0,0,0,0,0,0.6666,0,0],[0,0,0,0,0,0,0,0,0.6666,0],[0,0,0,0,0,0,0,0,0,0.6666]])
    gauss_2 = scipy.stats.multivariate_normal.pdf(x,mean =
    mean_2*np.ones(input_dim),cov = [[0.3333,0,0,0,0,0,0,0,0,0],[0,0.3333,0,0,0,0,0,0,0,0],[0,0,0.3333,0,0,0,0,0,0,0],[0,0,0,0.3333,0,0,0,0,0,0],[0,0,0,0,0.3333,0,0,0,0,0],[0,0,0,0,0,0.3333,0,0,0,0],[0,0,0,0,0,0,0.3333,0,0,0],[0,0,0,0,0,0,0,0.3333,0,0],[0,0,0,0,0,0,0,0,0.3333,0],[0,0,0,0,0,0,0,0,0,0.3333]])
    likelihood = gauss_1 + gauss_2
    return likelihood

outdir = 'Gaussian_10D'
dim = 10
low_bound = -5*np.ones(dim)
high_bound = 10*np.ones(dim)
sigma = 0.32
retrains = 100
samples_per_retrain = 1000
vae_prob = 0.2
plot_initial = False

# # # #Generate pure Metropolis-Hastings comparison samples
# start_time = time.time()
# sigma_pureMH = 0.32
# comparison_sample_size = 100000
# pureMH = algorithm.PureMH(log_likelihood,comparison_sample_size,dim,
# low_bound,high_bound,sigma_pureMH = sigma_pureMH)
# comparison_samples = np.array(pureMH.run())
# isExist = os.path.exists(outdir)
# if not isExist:
#     os.mkdir(outdir)
# np.savetxt(outdir + '/' + 'comparison_samples.dat',comparison_samples)
# end_time = time.time()
# time_taken_MH = end_time - start_time
# print("Time taken MH:", time_taken_MH)
#Gaussian initial samples, we pretend we don't know the relative weights of the modes
initial_sample_size = 2000
maxima = np.array([[8,3,0,0,0,0,0,0,0,0],[-2,3,0,0,0,0,0,0,0,0]])
covs = [[[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],[[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]]
initial_samples = []
for i in range(0,len(maxima)):
    means = maxima[i]
    cov = covs[i]
    initial_samples.extend(np.random.multivariate_normal(means,cov,
    initial_sample_size//len(maxima)))
initial_samples = np.array(initial_samples)
input_dim = 10
inter_dim = 32
inter_dim1 = 48
latent_dim = 8
batch_size = 128
epochs = 30
bins = 20
lr = 0.001
start_time_vae = time.time()
algo1 = algorithm.MH_VAE(log_likelihood=log_likelihood,dim=dim,input_dim=input_dim,inter_dim=inter_dim,inter_dim1=inter_dim1, latent_dim=latent_dim, batch_size=batch_size, epochs=epochs, low_bound=low_bound,high_bound=high_bound,
                         initial_samples=initial_samples,retrains=retrains,samples_per_retrain=samples_per_retrain,outdir = outdir,sigma = sigma,vae_prob = vae_prob,bins =bins,plot_initial = plot_initial,lr=lr)
samples,vae_samples,MH_samples,vae_rate = algo1.run()
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
