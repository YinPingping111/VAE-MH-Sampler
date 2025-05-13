import sys
import os
import numpy as np
from scipy import stats
import time
from random import randrange
from vae import VAE_model
from sklearn.neighbors import KernelDensity

def lprint(msg):
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()

class MH_VAE:
    def __init__(self, target, dim, input_dim, inter_dim, inter_dim1, latent_dim,
                 epochs, low_bound, high_bound, initial_samples, retrains, samples_per_retrain,
                 outdir, vae_prob, desired_sample_size, bandwidth,lr, batch_size, sigma):
        self.target = target
        self.dim = dim
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.inter_dim1 = inter_dim1
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.initial_samples = initial_samples
        self.retrains = retrains
        self.samples_per_retrain = samples_per_retrain
        self.outdir = outdir
        self.vae_prob = vae_prob
        self.desired_sample_size = desired_sample_size
        self.bandwidth = bandwidth
        self.lr = lr
        self.batch_size = batch_size
        self.sigma = sigma

    def run(self):
        print('VAE MCMC Chain Started')
        isExist = os.path.exists(self.outdir)
        if not isExist:
            os.mkdir(self.outdir)

        training_samples = self.initial_samples
        desired_sample_size = self.desired_sample_size
        bandwidth = self.bandwidth
        # Train VAE model on initial seeded samples
        model = VAE_model(self.input_dim, self.inter_dim, self.inter_dim1,
                          self.latent_dim, desired_sample_size, training_samples,
                          self.batch_size, self.epochs, lr=self.lr)
        model.train(training_samples)
        vae_samples = model.generate_samples(desired_sample_size)
        samples_final = self.initial_samples
        theta = samples_final[np.random.randint(0, len(samples_final))]
        accepted_vae = []
        accepted_MH = []
        total_time = 0
        vae_rate = []

        for retrain_iter in range(self.retrains):
            start = time.time()
            naccepted_vae = 0
            nattempted_vae = 0
            naccepted = 0
            nattempted = 0

            # Build KDE using VAE-generated samples
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(vae_samples)

            for i in range(self.samples_per_retrain):
                rand = np.random.uniform()

                # VAE proposal with probability vae_prob
                if rand < self.vae_prob:
                    nattempted_vae += 1
                    rand_pick = randrange(len(vae_samples))
                    theta_prime = vae_samples[rand_pick]

                    # Ensure the proposal stays within bounds
                    for j in range(self.dim):
                        while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                            theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()

                    # Evaluate likelihood ratio and KDE-based Q ratio
                    Q = np.exp(kde.score_samples(theta.reshape(1, -1)))
                    Q_prime = np.exp(kde.score_samples(theta_prime.reshape(1, -1)))
                    Q_ratio = Q / Q_prime
                    L_ratio = self.target(theta_prime, self.dim) / self.target(theta, self.dim)
                    prob_accept = L_ratio * Q_ratio
                    a = min(1, prob_accept)
                    u = np.random.uniform()

                    if u < a:
                        naccepted_vae += 1
                        theta = theta_prime
                        accepted_vae.append(theta_prime)

                # Metropolis-Hastings fallback
                else:
                    nattempted += 1
                    theta_prime = np.array([
                        theta[j] + stats.norm(0, self.sigma).rvs()
                        for j in range(self.dim)
                    ])

                    # Ensure the proposal stays within bounds
                    for j in range(self.dim):
                        while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                            theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()

                    L_ratio = self.target(theta_prime, self.dim) / self.target(theta, self.dim)
                    a = min(1, L_ratio)
                    u = np.random.uniform()

                    if u < a:
                        naccepted += 1
                        theta = theta_prime
                        accepted_MH.append(theta_prime)

                samples_final = np.vstack((samples_final, theta))

            # Retrain VAE using recent samples
            training_samples = np.vstack((samples_final[-6000:], self.initial_samples))
            model.train(training_samples)
            vae_samples = model.generate_samples(desired_sample_size)

            end = time.time()
            total_time += end - start

            print(len(training_samples))
            print(f'Number of retrains = {retrain_iter + 1}/{self.retrains}\n'
                  f'VAE acceptance efficiency = {naccepted_vae / nattempted_vae if nattempted_vae > 0 else 0}\n'
                  f'Metropolis acceptance efficiency = {naccepted / nattempted if nattempted > 0 else 0}\n'
                  f'Previous Retrain Time = {np.round(end - start)} seconds\n'
                  f'Total Retrain Time = {np.round(total_time)} seconds')

            if not retrain_iter == self.retrains - 1:
                for _ in range(0, 5):
                    UP = '\033[1A'
                    CLEAR = '\x1b[2K'
                    print(UP, end=CLEAR)

            vae_rate.append(naccepted_vae / nattempted_vae if nattempted_vae > 0 else 0)

        return samples_final, accepted_vae, accepted_MH, vae_rate

class PureMH:

    def __init__(self, target, nsamples, dim, low_bound, high_bound, sigma_pureMH=0.3):
        self.target = target
        self.nsamples = nsamples
        self.dim = dim
        self.sigma = sigma_pureMH
        self.low_bound = low_bound
        self.high_bound = high_bound

    def run(self):
        theta = np.zeros(self.dim)
        theta_prime = np.zeros(self.dim)
        naccepted = 0
        samples = []

        for i in range(self.nsamples):
            lprint('Pure MH samples = ' + str(i))
            theta_prime = np.zeros(self.dim)
            for j in range(self.dim):
                theta_prime[j] = theta[j] + stats.norm(0,self.sigma).rvs()  # proposal sample, taken from gaussian centred on current sample with std = sigma
                while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                    theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
            a = min(1, self.target(theta_prime, self.dim) / self.target(theta, self.dim))
            u = np.random.uniform()
            if u < a:
                naccepted += 1
                theta = theta_prime
            samples.append(theta)
        print(' Pure MH Acceptance efficiency = ', naccepted / self.nsamples)
        return samples
