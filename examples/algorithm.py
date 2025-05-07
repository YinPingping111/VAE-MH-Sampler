import sys,os
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
    def __init__(self,log_likelihood,dim,input_dim,inter_dim,inter_dim1,latent_dim, epochs, low_bound,high_bound,initial_samples,retrains,samples_per_retrain,outdir,
    vae_prob,bins,lr,batch_size,sigma,plot_initial = True ):
        self.log_likelihood = log_likelihood
        self.dim = dim
        self.input_dim = input_dim
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.initial_samples = initial_samples
        self.retrains = retrains
        self.samples_per_retrain = samples_per_retrain
        self.outdir = outdir
        self.sigma = sigma
        self.vae_prob = vae_prob
        self.bins = bins
        self.plot_initial = plot_initial
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.inter_dim = inter_dim
        self.latent_dim = latent_dim
        self.inter_dim1= inter_dim1
    def run(self):
        print('VAE MCMC Chain Started')

        isExist = os.path.exists(self.outdir)
        if not isExist:
            os.mkdir(self.outdir)

        training_samples = self.initial_samples
        initial_sample_size = len(training_samples)
        #desired_sample_size = 2*initial_sample_size # Gaussian2D T14
        #desired_sample_size = 3 * initial_sample_size #for 10D t55 20componets
        desired_sample_size = 5 * initial_sample_size  # for 10D t55 20componets
        # Train vae model on initial seeded samples
        model = VAE_model(self.input_dim, self.inter_dim, self.inter_dim1,self.latent_dim, desired_sample_size, training_samples, self.batch_size, self.epochs, lr=0.001)
        model.train(training_samples)

        vae_samples= model.generate_samples(desired_sample_size)
        samples_final = self.initial_samples
        theta = samples_final[np.random.randint(0,len(samples_final))]
        accepted_vae = []
        accepted_MH = []
        total_time = 0
        vae_rate = []

        for retrain_iter in range(self.retrains):

            start = time.time()
            H = []
            edges = []
            for dim_iter in range(0,self.input_dim):
                H_temp,edges_temp = np.histogram(vae_samples[:,dim_iter],
                bins = self.bins)
                H.append(H_temp/np.sum(H_temp))
                edges.append(edges_temp)

            naccepted_vae = 0
            nattempted_vae = 0
            naccepted = 0
            nattempted = 0
            for i in range(self.samples_per_retrain):
                rand = np.random.uniform()
                # VAE as proposal some of the time
                if rand < self.vae_prob:
                    nattempted_vae +=1
                    rand_pick = randrange(len(vae_samples))
                    theta_prime = vae_samples[rand_pick]

                    # edge_loc = []
                    # for dim_iter in range(0,self.dim):
                    #     for edge_iter in range(0,len(edges[dim_iter])):
                    #         if theta_prime[dim_iter] >= edges[dim_iter][len(edges[dim_iter])-1]:
                    #             edge_loc.append(len(edges[dim_iter])-2)
                    #             break
                    #         elif theta_prime[dim_iter] < edges[dim_iter][edge_iter]:
                    #             edge_loc.append(edge_iter-1)
                    #             break
                    #
                    # Q_prime = 1
                    # for dim_iter in range(0,self.dim):
                    #     Q_prime = Q_prime*H[dim_iter][edge_loc[dim_iter]]
                    # if Q_prime == float(0):
                    #     Q_prime = 0.000001
                    #
                    # edge_loc = []
                    # for dim_iter in range(0,self.dim):
                    #     for edge_iter in range(0,len(edges[dim_iter])):
                    #         if theta[dim_iter] >= edges[dim_iter][len(edges[dim_iter])-1]:
                    #             edge_loc.append(len(edges[dim_iter])-2)
                    #             break
                    #         elif theta[dim_iter] < edges[dim_iter][edge_iter]:
                    #             edge_loc.append(edge_iter-1)
                    #             break
                    #
                    # Q = 1
                    # for dim_iter in range(0,self.dim):
                    #     Q = Q*H[dim_iter][edge_loc[dim_iter]]
                    # if Q == float(0):
                    #     Q = 0.000001
                    #
                    # Q_ratio = Q/Q_prime

                    # 在 retrain 循环内部替换 H 和 edges 的构建部分
                    #kde = KernelDensity(bandwidth=0.6).fit(vae_samples)# Gaussian_2D，Student_T14
                    kde = KernelDensity(bandwidth=0.45).fit(vae_samples)  # Gaussian_10D
                    #kde = KernelDensity(bandwidth=0.2).fit(vae_samples)  # Gaussian_20components
                    #kde = KernelDensity(bandwidth=0.9).fit(vae_samples)# T_2D_15
                    #kde = KernelDensity(bandwidth=0.9).fit(vae_samples)# T_2D_55
                    #kde = KernelDensity(bandwidth=0.9).fit(vae_samples)  # Gaussian_10D

                    Q = np.exp(kde.score_samples(theta.reshape(1, -1)))
                    Q_prime = np.exp(kde.score_samples(theta_prime.reshape(1, -1)))
                    Q_ratio = Q / Q_prime

                    for j in range(self.dim):
                        while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                            theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
                    L_ratio = self.log_likelihood(theta_prime,self.dim)/self.log_likelihood(theta,self.dim)
                    prob_accept = L_ratio*Q_ratio
                    a = min(1, prob_accept)
                    u = np.random.uniform()
                    if u < a:
                        naccepted_vae +=1
                        theta = theta_prime
                        accepted_vae.append(theta_prime)
                # M-H the rest of the time
                else:
                    nattempted +=1
                    theta_prime = np.zeros(self.dim)

                    for j in range(self.dim):

                        theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
                        while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                            theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
                    theta_prime = np.array(theta_prime)
                    a = min(1, self.log_likelihood(theta_prime,self.dim)/self.log_likelihood(theta,self.dim))
                    u = np.random.uniform()
                    if u < a:
                        naccepted +=1
                        theta = theta_prime
                        accepted_MH.append(theta_prime)
                samples_final = np.vstack((samples_final,theta))
            # Retrain VAE
            training_samples = np.vstack((samples_final[-5000:], self.initial_samples))
            #training_samples = samples_final
            #training_samples = self.initial_samples
            model.train(training_samples)
            vae_samples= model.generate_samples(desired_sample_size)
            print("VAE samples",vae_samples)
            end = time.time()
            total_time += end-start
            print(len(training_samples))
            print('Number of retrains = ' + str(retrain_iter+1) + '/' +
            str(self.retrains) + '\n' + 'VAE acceptance efficiency = ' +
            str(naccepted_vae/nattempted_vae) + '\n' +
            'Metropolis acceptance efficiency = ' + str(naccepted/nattempted) +
            '\n' + 'Previous Retrain Time = ' + str(np.round(end-start)) +
            ' seconds ' + '\n' + 'Total Retrain Time = ' +
            str(np.round(total_time)) + ' seconds ')

            if not retrain_iter == self.retrains-1:
                for _ in range(0,5):
                    UP = '\033[1A'
                    CLEAR = '\x1b[2K'
                    print(UP, end=CLEAR)

            vae_rate.append(naccepted_vae/nattempted_vae)
        return samples_final,accepted_vae,accepted_MH,vae_rate


class PureMH:

    def __init__(self,log_likelihood,nsamples,dim,low_bound,high_bound,sigma_pureMH = 0.3):
        self.log_likelihood = log_likelihood
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
                theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs() # proposal sample, taken from gaussian centred on current sample with std = sigma
                while theta_prime[j] < self.low_bound[j] or theta_prime[j] > self.high_bound[j]:
                    theta_prime[j] = theta[j] + stats.norm(0, self.sigma).rvs()
            a = min(1, self.log_likelihood(theta_prime,self.dim)/self.log_likelihood(theta,self.dim))
            u = np.random.uniform()
            if u < a:
                naccepted +=1
                theta = theta_prime
            samples.append(theta)
        print(' Pure MH Acceptance efficiency = ',naccepted/self.nsamples)
        return samples
