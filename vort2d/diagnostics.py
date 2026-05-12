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

def get_ens_corr(c, grid):
    # compute correlation
    fld_prior_ens = np.zeros((c.config.nens,)+c.grid.x.shape)
    for m in range(c.config.nens):
        fld_prior_ens[m,...] = c.state.fields_prior[m,0][0,...]
    fld_prior_mean = np.mean(fld_prior_ens, axis=0)

    obs_prior_ens = np.array([c.obs.obs_prior[m,0][0] for m in range(c.config.nens)])
    obs_prior_mean = np.mean(obs_prior_ens, axis=0)

    cov = np.zeros(grid.x.shape)
    fld_prior_var = np.zeros(grid.x.shape)
    obs_prior_var = 0
    for m in range(c.config.nens):
        cov += ((fld_prior_ens[m,...] - fld_prior_mean) * (obs_prior_ens[m] - obs_prior_mean))
        fld_prior_var += (fld_prior_ens[m,...] - fld_prior_mean)**2
        obs_prior_var += (obs_prior_ens[m] - obs_prior_mean)**2
    cov /= c.config.nens-1
    fld_prior_var /= c.config.nens-1
    obs_prior_var /= c.config.nens-1

    fld_prior_var[np.where(fld_prior_var==0)] = 1e-10

    corr = cov / np.sqrt(fld_prior_var * obs_prior_var)
    return corr