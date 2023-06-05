import pandas as pd
import numpy as np
import ruptures as rpt

from   scipy.stats import norm
from   scipy.special import logsumexp

from bayesian_changepoint_detection.priors import const_prior
import bayesian_changepoint_detection
from functools import partial
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
import bayesian_changepoint_detection.online_likelihoods as online_ll


def bayes_offline(data, truncate=-100):
    '''Offline changepoint analysis for signal processing and finding changepoints.
    implemended with 'bayesian changepoint detection' package (see references below)
    
    Requires a continuous 1D dataset as numpy array:
    
    Also:
    
    Tuning variables should include:

    truncate term: NOT SURE WHAT THIS IS TRUNCATING yet


    references:
    https://github.com/hildensia/bayesian_changepoint_detection

    
    '''
    #data = GR_rollAvg_[:2100]
    prior_function = partial(const_prior, p=1/(len(data) + 100))
    _trunc = truncate
    
    Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(),truncate=_trunc)
    
    return Q, P, Pcp


def bayes_online_CP(data, **kwargs):
    '''
    Online, changepoint analysis for signal processing and finding changepoints.
    implemended with 'bayesian changepoint detection' package (see references below)
    
    Requires a continuous 1D dataset as numpy array:
    
    Also:
    
    Tuning variables should include:
    
    hazard value that corresponds to the maximum likely continuous sequence.
    
    alpha, beta =  SURE WHAT THESE ARE YET but likely two variables to determine the beta distribution prior.
    
    mu = expected mean of dataset 
    
    references:
    https://github.com/hildensia/bayesian_changepoint_detection
    Adams and Mackay 2007 "Bayesian Online Changepoint Detection" 
    
    
    '''
    
    _mu = kwargs.get('mu')
    _hazard = kwargs.get('hazard')
    _alpha = kwargs.get('alpha')
    _beta = kwargs.get('beta')
    _kappa = kwargs.get('kappa')
    
    
    hazard_function = partial(constant_hazard, _hazard)
    
    R, maxes = online_changepoint_detection(data, hazard_function, online_ll.StudentT(alpha=_alpha, beta=_beta , kappa=_kappa, mu=_mu))
    
    return R, maxes





def pelt_bkps(data, pen=120, min_size=250):

    '''PELT search method for offline changepoint search.

    will find an arbitrary number of changepoints

    takes in a continuous sequence of data and
    through the minimization of a cost function or 'model' 
    identifies changepoints:

    Function returns array of dataset relative changepoints my_bkps

    min_size refers to the minimum continuous data length that will be 
    allowed between change points; pen is the threshold for the search 
    method, in that higher numbers find fewer changepoints.
        
        
        
    model options:
        "l2" (least squares)
        "ar" (auto regressive)
        "cosine" (Cosine similarity)
        ... refer to ruptures documentation for others
    '''
        
    _pen = pen
    _min_size = min_size

    model = "l2"
    algo = rpt.Pelt(model=model, min_size=_min_size).fit(data)
    my_bkps = algo.predict(pen=_pen)
    
    return my_bkps

def CP_GRAPH(data, pen=120, min_size=250):
        
    _min_size = min_size
    _pen = pen
    data_work = data
    #data_work = fill_out_data_ends_complete(data)

    CPoints = pelt_bkps(data_work, pen=pen, min_size=min_size)
    print(datalabel,CPoints)
    rpt.show.display(data_work, CPoints, figsize=(10, 3))

    return CPoints

class change_point_tools(object):
    if __name__ == '__main__':
        T      = len(data)   # Number of observations.
        hazard = 1/100  # Constant prior on changepoint probability.
        mean0  = 50      # The prior mean on the mean parameter.
        var0   = 10      # The prior variance for mean parameter.
        varx   = 7      # The known variance of the data.

#       data, cps      = generate_data(varx, mean0, var0, T, hazard) # generated data
        #data, cps      = well_data_bkps(data)
        #model          = GaussianUnknownMean(mean0, var0, varx)
        #R, pmean, pvar = bocd(data, model, hazard)

        #plot_posterior(T, data, cps, R, pmean, pvar)

    #plot_posterior(T, data, R, pmean, pvar)


    






        """============================================================================
    Author: Gregory Gundersen
    Python implementation of Bayesian online changepoint detection for a normal
    model with unknown mean parameter. For algorithm details, see
        Adams & MacKay 2007
        "Bayesian Online Changepoint Detection"
        https://arxiv.org/abs/0710.3742
    For Bayesian inference details about the Gaussian, see:
        Murphy 2007
        "Conjugate Bayesian analysis of the Gaussian distribution"
        https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    This code is associated with the following blog posts:
        http://gregorygundersen.com/blog/2019/08/13/bocd/
        http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
    ============================================================================"""




    # -----------------------------------------------------------------------------

    def bocd(data, model, hazard):
        """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
        """
        # 1. Initialize lower triangular matrix representing the posterior as
        #    function of time. Model parameters are initialized in the model class.
        #    
        #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
        #    visualization.
        #
        T           = len(data)
        log_R       = -np.inf * np.ones((T+1, T+1))
        log_R[0, 0] = 0              # log 0 == 1
        pmean       = np.empty(T)    # Model's predictive mean.
        pvar        = np.empty(T)    # Model's predictive variance. 
        log_message = np.array([0])  # log 0 == 1
        log_H       = np.log(hazard)
        log_1mH     = np.log(1 - hazard)

        for t in range(1, T+1):
            # 2. Observe new datum.
            x = data[t-1]

            # Make model predictions.
            pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
            pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * model.var_params[:t])
            
            # 3. Evaluate predictive probabilities.
            log_pis = model.log_pred_prob(t, x)

            # 4. Calculate growth probabilities.
            log_growth_probs = log_pis + log_message + log_1mH

            # 5. Calculate changepoint probabilities.
            log_cp_prob = logsumexp(log_pis + log_message + log_H)

            # 6. Calculate evidence
            new_log_joint = np.append(log_cp_prob, log_growth_probs)

            # 7. Determine run length distribution.
            log_R[t, :t+1]  = new_log_joint
            log_R[t, :t+1] -= logsumexp(new_log_joint)

            # 8. Update sufficient statistics.
            model.update_params(t, x)

            # Pass message.
            log_message = new_log_joint

        R = np.exp(log_R)
        return R, pmean, pvar


    # -----------------------------------------------------------------------------


    class GaussianUnknownMean:
        
        def __init__(self, mean0, var0, varx):
            """Initialize model.
            
            meanx is unknown; varx is known
            p(meanx) = N(mean0, var0)
            p(x) = N(meanx, varx)
            """
            self.mean0 = mean0
            self.var0  = var0
            self.varx  = varx
            self.mean_params = np.array([mean0])
            self.prec_params = np.array([1/var0])
        
        def log_pred_prob(self, t, x):
            """Compute predictive probabilities \pi, i.e. the posterior predictive
            for each run length hypothesis.
            """
            # Posterior predictive: see eq. 40 in (Murphy 2007).
            post_means = self.mean_params[:t]
            post_stds  = np.sqrt(self.var_params[:t])
            return norm(post_means, post_stds).logpdf(x)
        
        def update_params(self, t, x):
            """Upon observing a new datum x at time t, update all run length 
            hypotheses.
            """
            # See eq. 19 in (Murphy 2007).
            new_prec_params  = self.prec_params + (1/self.varx)
            self.prec_params = np.append([1/self.var0], new_prec_params)
            # See eq. 24 in (Murphy 2007).
            new_mean_params  = (self.mean_params * self.prec_params[:-1] + \
                                (x / self.varx)) / new_prec_params
            self.mean_params = np.append([self.mean0], new_mean_params)

        @property
        def var_params(self):
            """Helper function for computing the posterior variance.
            """
            return 1./self.prec_params + self.varx

    # -----------------------------------------------------------------------------

    def generate_data(varx, mean0, var0, T, cp_prob):
        """Generate partitioned data of T observations according to constant
        changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
        """
        data  = []
        cps   = []
        meanx = mean0
        for t in range(0, T):
            if np.random.random() < cp_prob:
                meanx = np.random.normal(mean0, var0)
                cps.append(t)
            data.append(np.random.normal(meanx, varx))
        return data, cps


    # -----------------------------------------------------------------------------

    def plot_posterior(T, data, cps, R, pmean, pvar):
        fig, axes = plt.subplots(2, 1, figsize=(20,10))

        ax1, ax2 = axes

        ax1.scatter(range(0, T), data, alpha=0.2)
        ax1.plot(range(0, T), data)
        ax1.set_xlim([0, T])
        ax1.margins(0)
        
        # Plot predictions.
        ax1.plot(range(0, T), pmean, c='k')
        _2std = 2 * np.sqrt(pvar)
        ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
        ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

        ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
                norm=LogNorm(vmin=0.0001, vmax=1))
        ax2.set_xlim([0, T])
        ax2.margins(0)

        for cp in cps:
            ax1.axvline(cp, c='red', ls='dotted', alpha=0.5)
            #ax2.axvline(cp, c='red', ls='dotted', alpha=0.5)

        plt.tight_layout()
        plt.show()



    

        