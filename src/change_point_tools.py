import pandas as pd
import numpy as np
import ruptures as rpt
from scipy.signal import find_peaks
from   scipy.stats import norm
from   scipy.special import logsumexp
import sdt.changepoint as sdt_cp
#import numba






'''===========TOOLS  - Split dataset; Peak Finder============='''

def split_dataset(dataset, segment_size):
    '''Computational cost for offline bayesian search method is ????(look this up, I think it's O(n^2)
    
    If you can expect that the probability for change point segments to be less than a certain log distance
    then splitting the log into many segments can greatly improve computation time
    
    This is currently only built with the sdt-python offline Bayesian method that concatenates a continuous
    probabilty curve for the input dataset.

    Inputs:
    Dataset (numpy array, this will handle multi-dimentonal data as well, in a horizonal stack)
    Segment size  ( legth of segments, last segment will be what is a remainder after n complete segments)

    
    '''

    segments = []
    current_segment = []
    segment_num = 0

    for idx, item in enumerate(dataset):
        current_segment.append(item)
        if len(current_segment) == segment_size:
            segment_num += 1
            segments.append([segment_num, idx - segment_size + 1, idx + 1])
            current_segment = []

    if current_segment:
        segment_num += 1
        segments.append([segment_num, len(dataset) - len(current_segment), len(dataset)])

    return segments


def find_prob_peaks(data, height=0.1, return_peak_heights=False):
    '''Wrapper for scipy find peaks function
    
    Taks in 1D numpy data array and outputs array of local maximums
    
    hight function sets minimum threshold, default is 0.01 
    '''
    
    peaks, _peak_heights = find_peaks(data, height=height)
    
    if return_peak_heights == True:
        return peaks, _peak_heights
    else:
        return peaks


def bayes_offline_split(data, segment_length=1000, **kwargs):
    '''For computational efficiency, datasets are split into equal segments for ChangePoint detection
    
    Parameters:
    data: np.ndarray, 
    sequence length: this is division length of the segments, any remainder of incomplete
                    segments will be in the tail (last).  
    

    **** DOES NOT UNIQULY HANDLE CORNER CASE WHERE SEGMENT FALLS ON A CHANGEPOINT**** 
    Dunno what this means yet, if it's a problem at all, will have to be addressed later.  

    '''
    
    #Keyword Args assingment
    _prior = kwargs.get('prior')
    _method = kwargs.get('method')
    _engine = kwargs.get('engine')
    _normalize = kwargs.get('normal')
    _p = kwargs.get('p')
    _k = kwargs.get('k')

    #Argument Defaults
    if _method == None:
        _method = "full_cov"
    
    if _prior == None:
        _prior = "const"
    
    if _p == None:
        _p = 0.1

    if _k == None:
        _k=1
    
    if _engine == None:
        _engine ="numba"
    print('method- ' +_method+ ', prior- ' +_prior + ', engine- '+ _engine)#'Output- '+ _full_output +'Threshhold- '+ _thresh)
    


    full_prob = []
    
    #Go straight to CP search segments lengths that are less than the dataset length
    if len(data) <= segment_length:
        print('Segment length is less than dataset length, no spliting necessary')
        full_prob=bayes_offline_sdt(data.T, method=_method, prior=_prior, engine=_engine, p=_p, k=_k)
    
    #if segment is smaller than dataset, go on to splitting it up
    else:
        split = split_dataset(data, segment_length)
    
        #establishing full probablility obj
        

        #loop through segments and add probability dist. to the full_prob obj
        print('Segmenting into '+ str(segment_length))
        for segment in split:
            #print('Segmenting into '+ str(segment_length))
            seg_prob = bayes_offline_sdt(data[segment[1]:segment[2]].T, 
                                        method=_method, prior=_prior, engine=_engine, normal=_normalize, p=_p, k=_k)
            full_prob = np.concatenate((full_prob, seg_prob))
            print('completed segment ' + str(segment[0]) + ' from ' + str(segment[1]) + ': ' + str(segment[2])+ ' of ' + str(len(data)), end="\r")
    print(end='/n')
    return full_prob

def normalize_array(arr, low_clip, high_clip): 
    '''created through CHATGPT-4
    text input:

    build a python function that normalizes a numpy 
    array to values between zero and one as inputs takes in an 
    array to normalize, and two clipping percentage parameters 
    one for the high end and one for the low end'''

    # Calculate the lower and upper percentile values
    low_value = np.percentile(arr, low_clip)
    high_value = np.percentile(arr, 100 - high_clip)
    
    # Clip the array values based on the percentiles
    clipped_arr = np.clip(arr, low_value, high_value)
    
    # Normalize the clipped array to values between zero and one
    normalized_arr = (clipped_arr - low_value) / (high_value - low_value)
    
    return normalized_arr

''' ==========CHANGEPOINT SEARCH TOOLS ==========='''


def bayes_offline_sdt(data, **kwargs):
    '''Implementation of bayesian offline changepoint methods that is
    described in Fearnhead et al 2006, Chosen for its elegant handling of
    both univariate and multivariate data.
    
    Utilizing the protocols from sdt-python: 
    https://schuetzgroup.github.io/sdt-python/
    
    Parameters: 
    data: np.array - 'm x n' matrix of m datasets and n observations

    method: str - Choices - {"gauss" , "ifm", "full_cov"}
    prior: str - Choices - {"const", "geometric", "neg_binomial"}
        -Geometric prior `P(t) =  p (1 - p)^{t - 1}` requires value for 'p' where (0 < p < 1)  default is 0.1 
        -neg_binomial prior ****THIS PRIOR CLASS IS NOT WORKING YET IN SDT-PYTHON, do not use it :( ****
         `P(t) =  {{t - k}\choose{k - 1}} p^k (1 - p)^{t - k}` require value for 
        'p' where (0 < p < 1)  and 'k' (int) default is p=0.1, k=1

    engine: str - Choices - {'numba' or 'numpy' default is 'numba'}
    past: int - number of datapoints that probability seach will include in analysis


    full_output: boolean - change point list or array len(data) probabilities of a changepoint
                            returns 4 elements, prob, Q, P, Pcp
    
    returns probabilities of changepoints
    
    References:
    Fearnhead, Paul: “Exact and efficient Bayesian inference for multiple changepoint problems”,
    Statistics and computing 16.2 (2006), pp. 203–21
    
    Adams and McKay: “Bayesian Online Changepoint Detection”, arXiv:0710.3742
    
    Killick et al.: “Optimal Detection of Changepoints With a Linear Computational Cost”,
    Journal of the American Statistical Association, Informa UK Limited, 2012, 107, 1590–1598
    '''
    
    #KWARGS getter 
    _method = kwargs.get('method')
    _prior = kwargs.get('prior')
    _engine = kwargs.get('engine')
    _full_output = kwargs.get('full_output')
    _thresh = kwargs.get('threshold')
    _past = kwargs.get('past')
    _normalize = kwargs.get('normal')
    _p = kwargs.get('p')
    _k = kwargs.get('k')


    #Default arguments
    if _method == None:
        _method = "full_cov"
    if _engine == None:
        _engine ="numba"
    if _past == None:
        _past = 20
    if _normalize == True:
        data = normalize_array(data, 1, 1)


    #Prior parameters and defaults
    if _prior == None:
        _prior = "const"
        _prior_params = {}
    
    if _p == None:
        _p = 0.1

    if _k == None:
        _k=1

    if _prior == 'const':
        _prior_params = {}

    if _prior == "geometric":
        _prior_params = {'p': _p}

    if _prior == "neg_binomial":
        _prior_params = {'k': _k, 'p': _p}

    
    #if _full_output = True:
    #    _full_output = False
    #print('method- ' +_method+ ', prior- ' +_prior + ', engine- '+ _engine)#'Output- '+ _full_output +'Threshhold- '+ _thresh)
    
    
    #Offline Bayes Class init
    detOffBay = sdt_cp.BayesOffline(_prior, _method, engine=_engine, prior_params=_prior_params)
    
    #Full output 
    if _full_output==True:
        #detOffBay = sdt_cp.BayesOffline(_prior, _method, engine=_engine)
        prob, q, p, pcp = detOffBay.find_changepoints(data.T, full_output=True)
        return prob, q, p, pcp
    
    #Threshold case output
    elif _thresh!=None:
        return detOffBay.find_changepoints(data.T, prob_threshold=_thresh)
                    
    #Everything else output
    else:
        #detOffBay = sdt_cp.BayesOffline(_prior, _method, engine=_engine)
        return detOffBay.find_changepoints(data.T, truncate=-20)



''' Depreciating  this function because sdt-python is much much faster '''
# def bayes_offline_BCP(data, truncate=-20):
#     '''Offline changepoint analysis for signal processing and finding changepoints.
#     implemended with 'bayesian changepoint detection' package (see references below)
    
#     Requires a continuous 1D dataset as numpy array:
    
#     Also:
    
#     Tuning variables should include:

#     truncate term:  Speed up calculations by truncating a sum if
#      the summands provide negligible contributions. This parameter 
#      is the exponent of the threshold. Set to -inf to turn off. Defaults to -20


#     references:
#     https://github.com/hildensia/bayesian_changepoint_detection

    
#     '''
#     #data = GR_rollAvg_[:2100]
#     prior_function = partial(const_prior, p=1/(len(data) + 100))
#     _trunc = truncate
    
#     Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(),truncate=_trunc)
    
#     return Q, P, Pcp

# depreciating because sdt-python is much faster
# def bayes_online_BCP(data, **kwargs):
#     '''
#     Online, changepoint analysis for signal processing and finding changepoints.
#     implemended with 'bayesian changepoint detection' package (see references below)
    
#     Requires a continuous 1D dataset as numpy array:
    
#     Also:
    
#     Tuning variables should include:
    
#     hazard = value that corresponds to the maximum likely continuous sequence.
    
#     alpha, beta = to determine student T distribution prior.
    
#     mu = expected mean of dataset

#     kappa = ??  
    
#     references:
#     https://github.com/hildensia/bayesian_changepoint_detection
#     Adams and Mackay 2007 "Bayesian Online Changepoint Detection" 
    
    
#     '''
    
#     _mu = kwargs.get('mu')
#     _hazard = kwargs.get('hazard')
#     _alpha = kwargs.get('alpha')
#     _beta = kwargs.get('beta')
#     _kappa = kwargs.get('kappa')
    
    
#     hazard_function = partial(constant_hazard, _hazard)
    
#     R, maxes = online_changepoint_detection(data, hazard_function, online_ll.StudentT(alpha=_alpha, beta=_beta , kappa=_kappa, mu=_mu))
    
#     return R, maxes







def pelt_bkps(data, pen=120, min_size=250):

    '''PELT search method for offline changepoint search. Ruptures.

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
        refer to ruptures documentation for others

    references:
    Killick et al.: “Optimal Detection of Changepoints With a Linear Computational Cost”, 
    Journal of the American Statistical Association, Informa UK Limited, 2012, 107, 1590–1598
    '''
        
    _pen = pen
    _min_size = min_size

    model = "l2"
    algo = rpt.Pelt(model=model, min_size=_min_size).fit(data)
    my_bkps = algo.predict(pen=_pen)
    
    return my_bkps



'''=======================VISUALIZATION =============================='''

def CP_GRAPH(data, pen=120, min_size=250):
        
    '''wrapper for changepoint search and Visualization of changepoints using ruptures PELT method'''

    _min_size = min_size
    _pen = pen
    data_work = data
    #data_work = fill_out_data_ends_complete(data)

    CPoints = pelt_bkps(data_work, pen=pen, min_size=min_size)
    print(datalabel,CPoints)
    rpt.show.display(data_work, CPoints, figsize=(10, 3))

    return CPoints



'''===============Online Bayesian as implemented by G. Gunderson======================='''


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