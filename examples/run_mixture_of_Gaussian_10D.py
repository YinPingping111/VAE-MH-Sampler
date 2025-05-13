import sys,os
import numpy as np
import time
import algorithm
from scipy.stats import multivariate_normal

def target(x,dim):
    w_A = 2 / 3
    w_B = 1 / 3
    mu_A = np.array([8, 3] + [0] * 8)
    mu_B = np.array([-2, 3] + [0] * 8)
    cov = np.eye(10)  # 协方差矩阵（单位矩阵）
    pdf_A = multivariate_normal.pdf(x, mean=mu_A, cov=cov)
    pdf_B = multivariate_normal.pdf(x, mean=mu_B, cov=cov)
    return w_A * pdf_A + w_B * pdf_B

outdir = 'Gaussian_10D'
dim = 10
low_bound = -10*np.ones(dim)
high_bound = 15*np.ones(dim)
sigma = 0.48
retrains = 100
samples_per_retrain = 1000
vae_prob = 0.2

# # # #Generate pure Metropolis-Hastings comparison samples
# start_time = time.time()
# sigma_pureMH = 0.48
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
# Pure MH samples = 99999 Pure MH Acceptance efficiency =  0.46435
# Time taken MH: 396.3302958011627
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
inter_dim = 64
inter_dim1 = 32
latent_dim = 10
batch_size = 128
epochs = 20
lr = 0.001
desired_sample_size = 10000
bandwidth = 0.75
start_time_VAE_MH = time.time()
VAE_MH_sampler = algorithm.MH_VAE(target=target,dim=dim,input_dim=input_dim,inter_dim=inter_dim,inter_dim1=inter_dim1, latent_dim=latent_dim, epochs=epochs, low_bound=low_bound,high_bound=high_bound,
                         initial_samples=initial_samples,retrains=retrains,samples_per_retrain=samples_per_retrain,outdir = outdir,vae_prob = vae_prob,desired_sample_size = desired_sample_size,bandwidth = bandwidth,lr=lr,batch_size=batch_size,sigma = sigma,)
samples,vae_samples,MH_samples,vae_rate = VAE_MH_sampler.run()
end_time_VAE_MH = time.time()
time_taken_VAE_MH = end_time_VAE_MH - start_time_VAE_MH
print("Time taken VAE_MH:", time_taken_VAE_MH)
# Previous Retrain Time = 11.0 seconds
# Total Retrain Time = 1019.0 seconds
# Time taken VAE_MH: 1022.0337378978729

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
