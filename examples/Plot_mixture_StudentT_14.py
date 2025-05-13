import numpy as np
import matplotlib
matplotlib.use('Agg')
import pygtc
import os

outdir = 'T2D_14'
dim = 2
low_bound = -20 * np.ones(dim)
high_bound = 20 * np.ones(dim)

samples_total_VAE_MH = 102000
samples_burn_in_VAE_MH = 12000
samples_total_RMH = 100000
samples_burn_in_RMH = 10000

VAE_MH_samples = np.loadtxt("D:/desk/VAE-MH/VAE/examples/T2D_14/samples.dat")
VAE_MH_samples = VAE_MH_samples[samples_burn_in_VAE_MH:samples_total_VAE_MH]
VAE_MH_samples = np.vstack((VAE_MH_samples, low_bound, high_bound))

RMH_samples = np.loadtxt("D:/desk/VAE-MH/VAE/examples/T2D_14/comparison_samples.dat")
RMH_samples = RMH_samples[samples_burn_in_RMH:samples_total_RMH]
RMH_samples = np.vstack((RMH_samples, low_bound, high_bound))
os.makedirs(outdir, exist_ok=True)


Ranges = ((-20, 20), (-20, 20))
names = ['θ1', 'θ2']
truth = (-100, 100)  # 真实值（可选）


GTC_VAE_MH = pygtc.plotGTC(
    chains=[VAE_MH_samples],
    nContourLevels=3,
    paramRanges=Ranges,
    do1dPlots=True,
    nBins=50,
    paramNames=names,
    figureSize=15,
    customLabelFont={'family': 'Arial', 'size': 30},
    customLegendFont={'family': 'Arial', 'size': 20},
    customTickFont={'family': 'Arial', 'size': 40},
    legendMarker='None',
    truths=truth,
    truthLineStyles='-'
)
GTC_RMH = pygtc.plotGTC(
    chains=[RMH_samples],
    nContourLevels=3,
    paramRanges=Ranges,
    do1dPlots=True,
    nBins=50,
    paramNames=names,
    figureSize=15,
    customLabelFont={'family': 'Arial', 'size': 30},
    customLegendFont={'family': 'Arial', 'size': 20},
    customTickFont={'family': 'Arial', 'size': 40},
    legendMarker='None',
    truths=truth,
    truthLineStyles='-'
)
GTC_VAE_MH.savefig(os.path.join(outdir, 'VAE_MH(T_2D_14).pdf'), bbox_inches='tight')
print(f"Figure saved to {outdir}/VAE_MH(T_2D_14).pdf")
GTC_RMH.savefig(os.path.join(outdir, 'RMH(T_2D_14).pdf'), bbox_inches='tight')
print(f"Figure saved to {outdir}/RMH(T_2D_14).pdf")