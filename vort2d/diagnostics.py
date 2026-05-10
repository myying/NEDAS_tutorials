import numpy as np

# domain-averaged RMSE
def rmse(Xens, Xt):
    return np.sqrt(np.mean((np.mean(Xens, axis=1)-Xt)**2, axis=(1,2,3)))

# domain-averaged ensemble spread
def sprd(Xens):
    return np.sqrt(np.mean(np.std(Xens, axis=1)**2, axis=(1,2,3)))

# spectrum of error and ensemble spread
from NEDAS.diag.metrics.spectral import pwrspec2d
from NEDAS.utils.fft_lib import get_wn

def err_spec(Xens, Xt):
    wn, err_pwr = pwrspec2d(np.mean(Xens, axis=0)-Xt)
    return wn, err_pwr

def sprd_spec(Xens):
    nens, nv, nj, ni = Xens.shape
    ki, kj = get_wn(Xens[-2:])
    nup = int(max(ki.max(), kj.max()))
    Xmean = np.mean(Xens, axis=0)
    pwr_ens = np.zeros((nens, nv, nup))
    wn = None
    for m in range(nens):
        wn, pwr_ens[m, ...] = pwrspec2d(Xens[m, ...] - Xmean)
    sprd_pwr = np.sum(pwr_ens, axis=0) / (nens-1)
    return wn, sprd_pwr
