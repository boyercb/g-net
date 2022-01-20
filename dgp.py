#!/usr/bin/env python3

import numpy as np
import pandas as pd

from numpy.random import default_rng
from scipy.stats import logistic

def expit(x):
    return logistic.cdf(x)

class DGP:
    def __init__(self, params):
        self.C = params['C'] if 'C' in params else None 
        self.L = params['L'] if 'L' in params else None 
        self.A = params['A'] if 'A' in params else None 
        self.D = params['D'] if 'D' in params else None 
        self.Y = params['Y'] if 'Y' in params else None 
        self.K = params['K'] if 'K' in params else None 
        self.P = params['P'] if 'P' in params else None 

        if not 'eof' in self.Y:
            self.Y['eof'] = False
        if not 'family' in self.Y:
            self.Y['family'] = 'binomial'
        if not 'survival' in self.Y:
            self.Y['survival'] = False

    def __generator(self, N=None, d=None, truth=False, long=True, shift={}, seed=None):
        K = self.K
        P = self.P

        if N is None:
             N = np.shape(d)[0] if not long else np.shape(d)[0] // K

        C_coefs = self.C['coefs'] if self.C is not None else None
        L_coefs = self.L['coefs']
        A_coefs = self.A['coefs']
        D_coefs = self.D['coefs'] if self.D is not None else None
        Y_coefs = self.Y['coefs']

        if P == 1:
            L_coefs = np.reshape(L_coefs, (1, K * 3))

        L_sigma = self.L['sigma']
        Y_sigma = self.Y['sigma'] if 'sigma' in self.Y else None

        C = np.zeros((N, K))
        L = np.zeros((N, K * P))
        A = np.zeros((N, K))
        D = np.zeros((N, K))
        Y = np.zeros((N, K))

        X = np.column_stack((np.zeros((N, K)), L, A))

        cnames = ["C"] + ["L" + str(p + 1) for p in range(P)] + ["A", "D", "Y"]

        rng = default_rng(seed) #! should i change this to an object attribute?

        for k in range(K):
            h = list(range(k, -1, -1)) + list(range(k + 1, K))
            h = list(range(K)) + [x * K + j + K for x in range(P) for j in h] + [K * P + j + K for j in h]
            iL = [x * K + k + K for x in range(P)]
            iA = K * P + k + K

            if not truth:
                X[:, k] = np.ones(N)
                
                if self.C is not None:
                    C[:, k] = rng.binomial(n=1, p=expit(np.dot(X[:, h], C_coefs)), size=N)

                for p in range(P):
                    X[:, iL[p]] = rng.normal(loc=np.dot(X[:, h], L_coefs[p, :]), scale=L_sigma[p], size=N)

                X[:, iA] = rng.binomial(n=1, p=expit(np.dot(X[:, h], A_coefs)), size=N)

                if self.D is not None:
                    D[:, k] = rng.binomial(n=1, p=expit(np.dot(X[:, h], D_coefs)), size=N)

                if self.Y['eof'] and k < K - 1:
                    continue
                else: 
                    if self.Y['family'] == 'normal':
                        Y[:, k] = rng.normal(loc=np.dot(X[:, h], Y_coefs), scale=Y_sigma, size=N)
                    elif self.Y['family'] == 'binomial':
                        Y[:, k] = rng.binomial(n=1, p=expit(np.dot(X[:, h], Y_coefs)), size=N)
                
            else: 
                if k == 0:
                    if long:
                         X[:, iL + [iA]] = d.loc[d["time"] == 0, cnames[1:(P+1)] + ["A"]].to_numpy()
                    else:
                         X[:, iL + [iA]] = d[[l + "_0" for l in cnames[1:(P+1)]]].to_numpy()
                
                X[:, k] = np.ones(N)
            
                if 'C' in shift:
                    C[:, k] = shift['C'](X[:, h])
                elif self.C is not None:
                    C[:, k] = expit(np.dot(X[:, h], C_coefs))


                for p in range(P):
                    if 'L' + str(p + 1) in shift:
                        X[:, iL[p]] = shift['L' + str(p + 1)](X[:, h])
                    else: 
                        X[:, iL[p]] = np.dot(X[:, h], L_coefs[p, :])

                if 'A' in shift:
                    X[:, iA] = shift['A'](X[:, h])
                else:
                    X[:, iA] = expit(np.dot(X[:, h], A_coefs))

                if 'D' in shift:
                    D[:, k] = shift['D'](X[:, h])
                elif self.D is not None:
                    D[:, k] = expit(np.dot(X[:, h], D_coefs))

                if 'Y' in shift:
                    Y[:, k] = shift['Y'](X[:, h])
                else:
                    if self.Y['eof'] and k < K - 1:
                        continue
                    else: 
                        if self.Y['family'] == 'normal':
                            Y[:, k] = np.dot(X[:, h], Y_coefs)
                        elif self.Y['family'] == 'binomial':
                            Y[:, k] = expit(np.dot(X[:, h], Y_coefs))

        if not truth:
            for k in range(K):   
                iL = [x * K + k + K for x in range(P)]
                iA = K * P + k + K

                last_C = C[:, k-1] if k > 0 else np.zeros(N)
                last_D = D[:, k-1] if k > 0 else np.zeros(N)
                last_Y = Y[:, k-1] if k > 0 and self.Y['survival'] else np.zeros(N)

                C[:, k] = np.where((last_D == 1) | (last_Y == 1), np.nan, C[:, k])
                C[:, k] = np.where(last_C == 1, 1, C[:, k])

                for p in range(P):
                    X[:, iL[p]] = np.where((C[:, k] == 1) | (last_D == 1) | (last_Y == 1), np.nan, X[:, iL[p]])

                X[:, iA] = np.where((C[:, k] == 1) | (last_D == 1) | (last_Y == 1), np.nan, X[:, iA])
                D[:, k] = np.where((C[:, k] == 1) | (last_Y == 1), np.nan, D[:, k])
                D[:, k] = np.where(last_D == 1, 1, D[:, k])
                Y[:, k] = np.where((C[:, k] == 1) | (D[:, k] == 1), np.nan, Y[:, k])
                Y[:, k] = np.where(last_Y == 1, 1, Y[:, k])

                if self.Y['eof'] and k < K - 1:
                    Y[:, k] = np.nan

        d = pd.DataFrame(np.column_stack((C, X[:, K:], D, Y)))

        d.columns = [n + "_" + str(k) for n in cnames for k in range(K)]
    
        d["id"] = d.index
        d = pd.wide_to_long(d, cnames, sep="_", i="id", j="time") if long else d
        d = d.sort_values(by=["id", "time"]).reset_index()

        return d

    def generate_data(self, N, long=True, shift={}, seed=None):
        return self.__generator(N=N, long=long, shift=shift, seed=seed)

    def generate_truth(self, d, long=True, shift={}, seed=None):
        return self.__generator(d=d, truth=True, long=long, shift=shift, seed=seed)

    def transform_covariates(self, d):
        print('TBD!')
