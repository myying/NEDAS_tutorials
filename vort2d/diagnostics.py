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

def grid_to_spec(X):
    nt, nv, nj, ni = X.shape
    wn = None
    pwr = []
    for n in range(nt):
        wn, pwr_n = pwrspec2d(X[n,...])
        pwr.append(pwr_n)
    return wn, np.array(pwr)

def variance_spec(Xens):
    nt, nens, nv, nj, ni = Xens.shape
    ki, kj = get_wn(Xens[-2:])
    nup = int(max(ki.max(), kj.max()))
    wn = None
    sprd_pwr = []
    for n in range(nt):
        Xmean = np.mean(Xens[n,...], axis=0)
        pwr_ens = np.zeros((nens, nv, nup))
        for m in range(nens):
            wn, pwr_ens[m,...] = pwrspec2d(Xens[n,m,...] - Xmean)
        sprd_pwr.append(np.sum(pwr_ens, axis=0) / (nens-1))
    return wn, np.array(sprd_pwr)

def ens_corr(X, Y):
    nens = X.shape[0]
    assert Y.shape[0] == nens

    cov = np.zeros(X.shape[1:]+Y.shape[1:])
    varX = np.zeros(X.shape[1:])
    varY = np.zeros(Y.shape[1:])

    # compute correlation
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    for m in range(nens):
        cov += (X[m,...] - X_mean) * (Y[m,...] - Y_mean)
        varX += (X[m,...] - X_mean)**2
        varY += (Y[m,...] - Y_mean)**2
    cov /= nens-1
    varX /= nens-1
    varY /= nens-1

    varX = np.atleast_1d(varX)
    varX[np.where(varX==0)] = 1e-10
    varY = np.atleast_1d(varY)
    varY[np.where(varY==0)] = 1e-10

    corr = cov / np.sqrt(varX * varY)
    return corr