import os
import numpy as np
import scipy
import time
import algorithm
import scipy.stats
import scipy.special


def target(x,dim):
    # Define the parameters of the two t distributions
    mean_1 = np.array([10,0])
    mean_2 = np.array([-10,0])
    cov_1 = np.array([[1,0],[0,1]])
    cov_2 = np.array([[1,0],[0,1]])
    df1 = 1 # degrees of freedom
    df2 = 4 # degrees of freedom
    # Define a function to calculate the multivariate t pdf
    def multivariate_t_pdf(x, mu, Sigma, v):
        d = len(x)
        c = scipy.special.gamma((v+d)/2) / (scipy.special.gamma(v/2) * (v*np.pi)**(d/2) * np.linalg.det(Sigma)**(1/2))
        e = (1 + (x-mu).T @ np.linalg.inv(Sigma) @ (x-mu) / v)**(-(v+d)/2)
        return c * e
    # Calculate the probability density function of each t distribution
    t_pdf_1 = multivariate_t_pdf(x, mean_1, cov_1, df1)
    t_pdf_2 = multivariate_t_pdf(x, mean_2, cov_2, df2)
    # Calculate the likelihood of the mixture distribution
    likelihood = t_pdf_1 + t_pdf_2
    return likelihood


outdir = 'T2D_14'
dim = 2
low_bound = -20*np.ones(dim)
high_bound = 20*np.ones(dim)
sigma = 1.5
retrains = 100
samples_per_retrain = 1000
vae_prob = 0.2


# # # #Generate pure Metropolis-Hastings comparison samples
# start_time = time.time()
# sigma_pureMH = 1.5
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
# Pure MH samples = 99999 Pure MH Acceptance efficiency =  0.48117
# Time taken MH: 73.5982894897461
# # #Gaussian initial samples, we pretend we don't know the relative weights of the modes
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
# VAE acceptance efficiency = 0.8186274509803921
# Metropolis acceptance efficiency = 0.4472361809045226
# Previous Retrain Time = 5.0 seconds
# Total Retrain Time = 485.0 seconds
# Time taken vae: 486.7203583717346