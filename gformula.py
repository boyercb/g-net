#!/usr/bin/env python3
   
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

class Gformula:
    def __init__(self, df, id, time):
        # initialize
        self.outcome = None
        self.compevent = None
        self.censoring = None
        self.id = id
        self.time = time
        self.df = df.sort_values(by=[id, time]).copy()
        self.models = {}
        self.fits = {}
        self._weights = None
        self.eof = None
        self.survival = None

    def create_lags(self, lags):
        for var in lags:
            for lag in range(lags[var]):
                name = 'lag' + str(lag + 1) + '_' + var
                self.df[name] = self.df.groupby(self.id)[var].shift(lag + 1, fill_value=0)

    def outcome_model(self, formula, family, eof=False, survival=False, restriction=None):
        if eof and survival:
            ValueError("outcome cannot be both end of follow up and survival")

        mod = self.__define_glm(formula, self.df, family, self._weights, restriction)

        self.models.update({'outcome': mod})
        self.outcome = mod.endog_names
        self.eof = eof
        self.survival = survival 

    def censoring_model(self, formula, family, restriction=None):
        mod = self.__define_glm(formula, self.df, family, self._weights, restriction)
        
        self.models.update({'censoring': mod})
        self.censoring = mod.endog_names
    
    def compevent_model(self, formula, family, restriction=None):
        mod = self.__define_glm(formula, self.df, family, self._weights, restriction)
        
        self.models.update({'compevent': mod})
        self.compevent = mod.endog_names

    def covariate_model(self, covariate, formula, family, restriction=None, append=True):
        if not append or not 'covariates' in self.models:
            self.models['covariates'] = {}

        mod = self.__define_glm(formula, self.df, family, self._weights, restriction)
        
        self.models['covariates'].update({covariate: mod})
    
    def covariate_models(self, defns, append=True): 
        if not append or not 'covariates' in self.models:
            self.models['covariates'] = {}

        for cov in defns:
            if not 'restriction' in defns[cov]:
                defns[cov]['restriction'] = None 
            
            self.covariate_model(cov, defns[cov]['formula'], defns[cov]['family'], defns[cov]['restriction'])

    def models():
        print('tbd')

    def print_models():
        print('tbd')

    def fit(self, newdata=None, verbose=False):
        for var in ['outcome', 'censoring', 'compevent']:
            if var in self.models:
                self.fits[var] = self.models[var].fit()

        self.fits['covariates'] = {}

        for cov in self.models['covariates']:
            self.fits['covariates'][cov] = self.models['covariates'][cov].fit()

        if verbose:
            keys = [x for x in self.fits if x != 'covariates']
            for var in keys:
                print(var.capitalize() + ' Model:')
                print('==============================================================================')
                print(self.fits[var].summary())
                print('')

            for cov in self.fits['covariates']: 
                print('Covariate Model: ' + cov)
                print('==============================================================================')
                print(self.fits['covariates'][cov].summary())
                print('')

    # revise to allow for out-of-sample predictions based on fitted g-formula
    def simulate(self, interventions, data=None, lags=None, samples=10000, start=0, stop=None, seed=None):
        self.mc_start = start
        self.mc_stop = stop
        self.mc_samples = samples
        self.seed = seed

        rng = np.random.default_rng(seed)

        if not isinstance(interventions, list):
            interventions = [ interventions ]
        
        # add natural course to intervention list
        nc = {'label': 'natural course (parametric)',
              'variables': [],
              'g': None}
        interventions.insert(0, nc)
        
        self.interventions = interventions
        
        if stop is None:
            stop = self.df[self.time].max()
        
        # restrict to baseline values
        if data is not None:
            df = data.copy()
        else:
            df = self.df.loc[self.df[self.time] == start].copy()

        # draw monte carlo samples
        if samples > df.shape[0]:
            df = df.sample(n=samples, replace=True, random_state=seed)
        elif samples < df.shape[0]:
            df = df.sample(n=samples, random_state=seed)
        
        df.insert(0, '_uid', [x for x in range(samples)])
        sims = []

        # loop over all specified interventions
        for inter in interventions:
            g = inter['g']
            
            d = df.copy()
            dlist = []

            if not isinstance(inter['variables'], list):
                inter['variables'] = [ inter['variables'] ]

            # monte carlo simulation loop
            for k in range(start, stop + 1):
                d[self.time] = k

                # do not simulate covariates in first time step, unless after treatment
                if k == start:
                    sim_covs = False
                else:
                    sim_covs = True

                # simulate or intervene on time-varying covariates
                for cov, fit in self.fits['covariates'].items():
                    if cov in inter['variables']:
                        sim_covs = True         
                        if isinstance(g, dict):
                            d[cov] = g[cov](d, k)
                        else:    
                            d[cov] = g(d, k)
                    else:
                        if sim_covs:
                            d[cov] = self.__predict(
                                newdata=d,
                                rng=rng, 
                                fit=fit, 
                                family=fit.family
                            )
                
                # simulate competing event
                if self.compevent is not None:
                    if self.compevent in inter['variables']:
                        if isinstance(g, dict):
                            d[self.compevent] = g[self.compevent](d, k)
                        else:    
                            d[self.compevent] = g(d, k)
                    else:
                        d['_pD'] = self.fits['compevent'].predict(d)
                        d[self.compevent] = self.__predict(
                            newdata=d, 
                            rng=rng,
                            fit=self.fits['compevent'], 
                            family=self.fits['compevent'].family
                        )

                # simulate outcome 
                if self.eof and k < stop:
                    d['_pY'] = np.nan
                    d[self.outcome] = np.nan
                else: 
                    d['_pY'] = self.fits['outcome'].predict(d)
                    d[self.outcome] = self.__predict(
                        newdata=d, 
                        rng=rng,
                        fit=self.fits['outcome'], 
                        family=self.fits['outcome'].family
                    )

                # simulate censoring (for next time step)
                if self.censoring is not None:
                    if self.censoring in inter['variables']:
                        if isinstance(g, dict):
                            d[self.censoring] = g[self.censoring](d, k)
                        else:    
                            d[self.censoring] = g(d, k)
                    else:
                        d['_pC'] = self.fits['censoring'].predict(d)
                        d[self.censoring] = self.__predict(
                            newdata=d, 
                            rng=rng,
                            fit=self.fits['censoring'], 
                            family=self.fits['censoring'].family
                        )
                
                # update recodes 

                # append to list
                dlist.append(d.copy())

                # update lags
                if lags is not None:
                    for var in lags:
                        # parse number of lags
                        nlags = d.filter(regex='^lag[0-9]+_' + str(var)).shape[1]

                        for l in range(nlags):
                            if l > 0:
                                d['lag' + str(l + 1) + '_' + str(var)] = d['lag' + str(l) + '_' + str(var)]
                            else:
                                d['lag' + str(l + 1) + '_' + str(var)] = d[var]
            
            sims.append(pd.concat(dlist).sort_values(by=['_uid', self.time]).reset_index(drop=True))

        
        return GFormulaResult(self, sims)

    @staticmethod
    def __define_glm(formula, data, family, weights=None, restriction=None):
        if restriction is not None:
            data = data.loc[eval(restriction)].copy()

        if weights is None:  # Unweighted g-formula
            model = smf.glm(formula, data, family=family)

        else:  # Weighted g-formula
            model = smf.glm(formula, data, freq_weights=data[weights], family=family)
        
        return model

    @staticmethod
    def __predict(newdata, rng, fit, family):
        pp = fit.predict(newdata)
        if isinstance(family, sm.families.Binomial):
            pred = rng.binomial(1, pp, size=len(pp))
        elif isinstance(family, sm.families.Gaussian):
            pred = rng.normal(loc=pp, scale=fit.scale, size=len(pp))
        else:
            raise ValueError('That option is not supported')
        return pred
    

class GFormulaResult:
    def __init__(self, gformula, sims):
        # initialize
        self.sims = sims
        self.gformula = gformula
        self.bootstrap = False

    def summary(self):
        tab = pd.DataFrame(columns=['Intervention', 'Mean'])

        if self.bootstrap:
            print('bootstrapped results')
        else:
            for sim, interv in zip(self.sims, self.gformula.interventions):
                if not self.gformula.survival:
                   m = np.mean(sim[sim[self.gformula.time] == self.gformula.mc_stop]['_pY'])
                lab = interv['label']
                #f"{lab}: {m:.3f}"
                tab = tab.append({'Intervention': lab, 'Mean': m}, ignore_index=True)

            print(tab.to_string())
        

    def bootstrap(self, reps):
        # draw bootstrap sample
        boot = self.df.sample(n=reps, replace=True)

        for b in range(reps):
            # re-fit models
            self.gformula.fit(newdata = boot)

            # re-run monte carlo simulation
            self.gformula.simulate(

            )

            # collect results
            
        
        # print summary
        print('tbd')

# class GformulaICE:

