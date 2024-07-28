from dataclasses import dataclass
import time
import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.special import logsumexp
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm, rv_continuous


@dataclass
class GaussianMixture(rv_continuous):
    means: np.ndarray
    covs: np.ndarray
    weights: np.ndarray

    def __post_init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sqrt_covs = np.array([np.linalg.cholesky(cov) for cov in self.covs])
        self.weights = self.weights / np.sum(self.weights)

    def logpdf(self, x):
        logps = np.stack([multivariate_normal.logpdf(x, mean=mean, cov=cov) for mean, cov in zip(self.means, self.covs)], -1)
        return logsumexp(logps, axis=-1, b=self.weights)

    def _rvs(self, size=None, random_state=None):
        component = np.random.choice(len(self.weights), size=size, p=self.weights).astype(int)
        samples = norm.rvs(size=(*size, self.means.shape[1]), random_state=random_state)
        samples = np.einsum("...ij,...j->...i", self.sqrt_covs[component], samples) + self.means[component]
        return samples
    





@dataclass
class IsoScaler():
    mean: float = 0.
    std: float = 1.

    def fit(self, x):
        assert x.ndim == 2
        self.mean = np.mean(x, 0)
        self.std = np.std(x)
    def transform(self, x):
        return (x - self.mean) / self.std
    def inverse_transform(self, x):
        return x * self.std + self.mean
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def bw_scott(sources):
    n, d = sources.shape
    return n ** (-1 / (d + 4))

def bw_cvml(sources):
    """
    Bandwidth selection via cross-validation, 
    see e.g. https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    """
    scott = bw_scott(sources)
    kde = KernelDensity(kernel='gaussian', atol=1e-3, rtol=1e-3)

    grid_kde = GridSearchCV(kde, {'bandwidth': np.logspace(0, 1, 3, base=2) * scott}, cv=5, verbose=2)
    grid_kde.fit(sources)
    best_bw = grid_kde.best_params_['bandwidth']
    print(best_bw, scott, np.log2(best_bw / scott))
    return best_bw

def KDE_bw(sources, bandwidth='scott'):
    n, d = sources.shape
    # scaler = StandardScaler()
    scaler = IsoScaler()
    sources = scaler.fit_transform(sources)
    scaler_correction = - d * np.log(scaler.std)
    # scaler = PCA(whiten=True)
    # sources = scaler.fit_transform(sources)
    # scaler_correction = np.log(scaler.n_samples_**(d/2) / scaler.singular_values_.prod())

    if bandwidth == 'scott':
        bw = bw_scott(sources)
    elif bandwidth == 'cvml':
        bw = bw_cvml(sources)
    elif bandwidth == 'kNN':
        bw = bw_kNN(sources)

    print(bw, bw_scott(sources))
    kde = KernelDensity(kernel='gaussian', bandwidth=bw, atol=1e-5, rtol=1e-5)

    kde.fit(sources)
    logp_fn = lambda x: kde.score_samples(scaler.transform(x)) + scaler_correction
    return logp_fn






def k_AMISE(sources):
    n, d = sources.shape
    return np.maximum(np.rint(0.6 * n ** (1 - d / (d + 4))).astype(int), 2)

def gauss(sigma, d):
    def f(u):
        return -((u/sigma)**2)/2 - np.log(sigma**2 * 2*np.pi) * d / 2
    return f

def bw_kNN(sources):
    """
    Propose a bandwidth based on nearest-neighbors.
    Compute the median of distances from sources to their k-th nearest neighbor.
    k is chosen based on heuristic from Orava2011 www.sav.sk/journals/uploads/0127102604orava.pdf
    """
    k_n = k_AMISE(sources)
    neigh = NearestNeighbors(n_neighbors=k_n)
    neigh.fit(sources)

    dists = neigh.kneighbors(sources, k_n, return_distance=True)[0]
    bws = dists[...,-1]
    return np.median(bws)

def KDE_kNN(sources):
    """
    KDE with adaptive bandwidth based on nearest-neighbors, 
    see e.g. Orava2011 www.sav.sk/journals/uploads/0127102604orava.pdf

    Seems to perform poorly on tails. 
    However, can be used to propose bandwidths for fixed-bandwidth KDE.
    """
    n, d = sources.shape
    scaler = IsoScaler()
    sources = scaler.fit_transform(sources)
    scaler_correction = - d * np.log(scaler.std)
    
    k_n = k_AMISE(sources)
    neigh = NearestNeighbors(n_neighbors=k_n)
    neigh.fit(sources)

    
    def logp_fn(x):
        x = scaler.transform(x)
        dists, ids = neigh.kneighbors(x, k_n, return_distance=True)
        bws = dists[...,-1]
        # dists = np.linalg.norm(x - np.moveaxis(sources[ids], 1, 0), axis=-1)
        # return logsumexp(gauss(bws)(dists), 0, 1/k_n) + scaler_correction

        # bws = bw_scott(sources)
        dists = np.linalg.norm(x - sources[:,None], axis=-1)
        return logsumexp(gauss(bws, d)(dists), 0, 1/n) + scaler_correction

    return logp_fn