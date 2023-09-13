import numpy as np
import pandas as pd
import scipy
import ruptures as rpt
import scipy.stats as stats
from scipy.signal import find_peaks
from itertools import product
from scipy.stats import linregress
import sdt.changepoint
import well_tools as wtool
import change_point_tools as cp_Tools

def piecewiseFunc(bkpt, avg):   #constructor for piecewise function returns np.piecewise(x, bkpts_bool,averages)
    '''Requires two lists or arrays of equal length to build into a piecewise function constructor
    '''
    
    horz = avg
    bkpt_bool = []
    zero = 0
    zed = np.array(zero) 
    bkpt = np.append(zed, bkpt)  #zero padding at the head of the trends (means) segments 

    for pts in range(1, len(bkpt)):     # this builds the boolean expressions used as *args in the piecewise builder
        boool = ((x>=bkpt[pts-1]) & (x<+bkpt[pts]))
        bkpt_bool.append(boool,)
    why = np.piecewise(x, bkpt_bool, horz)
    return why #np.piecewise(x, bkpt_bool, horz)


def piecewise_linearRegress(data, pen=3, min_size=10):
    
    _pen = pen
    _min_size = min_size
    
    #data = data / data[data.argmax()]
    
    def data_bkps(data, pen=_pen, min_size=_min_size): 
        '''PELT search method for offline changepoint search analysis.
        This function takes in a continuous sequence of data and
        through the minimization of a cost function identifies changepoints:
        min_size refers to the minimum continuous data length that will be 
        allowed between change points; pen is the threshold for the search 
        method, in that higher numbers find fewer changepoints.

        Function returns array of dataset relative changepoints my_bkps'''


        model = "l2"
        algo = rpt.Pelt(model=model, min_size=min_size).fit(data)
        my_bkps = algo.predict(pen=pen)

        return my_bkps
    
    
    
    def MSELin (data, Upbound, LwBound):   #Linear regression paramater tool for a dataset
        dataSeg = data[Upbound:LwBound]
        slope, intercept = scipy.stats.linregress(dataSeg, y=data[Upbound:LwBound].index.values)
        return slope, intercept
    def mean(data, Lwbound, Upbound):
        #dataSeg = data[Lwbound:Upbound].to_numpy()  # if using pandas DF as input
        dataSeg = data[Lwbound:Upbound]
        return np.nanmean(dataSeg)
    
    bkps = data_bkps(data, pen=pen, min_size=min_size)
    bkps = np.array(bkps)#.reshape(1,len(my_bkps))
    #print(len(bkps), bkps)
    DataMean = np.zeros(len(bkps)+1)
    bkps = np.concatenate((np.zeros(1),bkps))
    print(len(bkps), bkps)
    #bkps = np.concatenate((bkps,len(data))) 
    for i in range(1, len(bkps)): 
        #slp = []
        #interc = []
        #slp[i-1] , interc[i-1] = MSELin(data, my_bkps[i], my_bkps[i-1])
        
        mu = mean(data, int(bkps[i-1]), int(bkps[i]))
        DataMean[i] = mu
        #(bkps[i-1],bkps[i],mu)
    print(DataMean)
    #return slp, interc
    return DataMean, bkps


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




def prob_piecewise_linearRegress(data, prob, height=0.2):
    
    if len(data) != len(prob):
        raise ValueError("Dataset and probability sequence length mismatch. "
                         "Check that these two things are related")

    def MSELin (data, Upbound, LwBound):   #Linear regression paramater tool for a dataset
        dataSeg = data[Upbound:LwBound]
        slope, intercept = scipy.stats.linregress(dataSeg, y=data[Upbound:LwBound].index.values)
        return slope, intercept

    def mean(data, Lwbound, Upbound):
        #dataSeg = data[Lwbound:Upbound].to_numpy()  # if using pandas DF as input
        dataSeg = data[Lwbound:Upbound]
        return np.nanmean(dataSeg)

    bkps = find_prob_peaks(prob, height=height, return_peak_heights=False)
    #bkps = np.array(bkps)#.reshape(1,len(my_bkps))
    #print(len(bkps), bkps)
    DataMean = np.zeros(len(bkps)+1)
    bkps = np.concatenate((np.zeros(1),bkps))
    print(len(bkps), bkps)
    #bkps = np.concatenate((bkps,len(data))) 
    for i in range(1, len(bkps)): 
        #slp = []
        #interc = []
        #slp[i-1] , interc[i-1] = MSELin(data, my_bkps[i], my_bkps[i-1])
        
        mu = mean(data, int(bkps[i-1]), int(bkps[i]))
        DataMean[i] = mu
        #(bkps[i-1],bkps[i],mu)
    print(DataMean)
    #return slp, interc
    return DataMean, bkps    

def changepoint_stat(data, prob, height=0.1, **kwargs):
    
    _stat_margin = kwargs.get('stat_margin')

    if _stat_margin==None:
        _stat_margin = 0

    if len(data) != len(prob):
        raise ValueError("Dataset and probability sequence length mismatch. "
                         "Check that these two things are related")

    bkps = find_prob_peaks(prob, height=height, return_peak_heights=False)

    print(str(len(bkps)) + ' Change points: ' +  str(bkps))

    def linearM(data, **kwargs):
        mb = linregress(np.arange(len(data)), data)
        return mb[0] 
    def linearB(data, **kwargs):
        mb = linregress(np.arange(len(data)), data)
        return mb[1] 

    stat_func = [np.mean, np.std, linearM, linearB] #np.meadian

    stats = sdt.changepoint.segment_stats(data, bkps, stat_func, mask=None, stat_margin=_stat_margin, return_len='segments')
    
    
    bkps = np.concatenate((np.zeros(1),bkps))
    stats = np.hstack((bkps.reshape(len(stats[0]),1), stats[1]))
    
    return stats


def df_add_CP_Prob(df, start, stop, log_dict=[], label='CP_Prob_', nan_method='local_average', window_size=200, segment_length=1000, **kwargs):
    ''' Adds a probability curve for change points to a dataFrame
    
    Parameters:
        df (pandas.Dataframe) : input dataFrame with a dataset mentioned in the log_dictionary 
        log_dict (dict) :  Log dictionary
        label (str)  : Prefix for probability curve that will be appended to the dataFrame
        
        **passthrough argument for replace_nans function**
        nan_method (str) : choices ('global_average', 'local_average', 'linear_interpolation') 
        window_size (int) : window size for calculating local averages.
    
        segment_length (int) : split for offline Bayesian Change Point detection,
                                unsefult to control the computational cost of this function

    Returns: None, this merely adds a curve to the input dataframe

    '''
    
     #Keyword Args assingment
    _prior = kwargs.get('prior') # choices are "const", "geometric", "neg_binomial"
    _method = kwargs.get('method')  # choices are 'gauss', 'ifm' or 'full_cov'
    _engine = kwargs.get('engine')  # choices are "numpy" or "numba"
    _normalize = kwargs.get('normal')

    #Argument Defaults
    if _method == None:
        _method = "full_cov"
    if _prior == None:
        _prior = "const"
    if _engine == None:
        _engine ="numba"
    #if _normalize == False:
    
    
    #Fill ends of incomplete df, replace NaN's with parameter method, and changepoint search with parameter segment splits
    for log in log_dict:
        print(log+' finding changepoints')
        labl = label+log
        df[log][start:stop] = wtool.fill_out_data_ends(df[log][start:stop].values)
        
        df[log][start:stop] = wtool.replace_nans(df[log][start:stop].values, method=nan_method, window_size=window_size)
        
        df[labl][start:stop] = cp_Tools.bayes_offline_split(df[log][start:stop].to_numpy(), segment_length=segment_length, method=_method, prior=_prior, engine=_engine, normal=_normalize)
        #df[labl][start:stop] = cp_Tools.bayes_offline_split(df[log][start:stop].values, segment_length=segment_length, method=_method, prior=_prior, engine=_engine, normal=_normalize)


def cpCorrelation(cpComparitee, cpComparitor, **kwargs):
    
    '''Correlation engine for changepoint associations:
    takes in two vectors (m x n) of changepoints/trends. One compariTEE who is compared to the compariTOR.
    
    
    Parameters:
        cpComparitee (np.ndarray) : Matrix of values associated to the changepoints that is being compared to comparitor
        cpComparitor (np.ndarray) : matrix of equal column dimention to comparitee from values of known data

    Relative changepoint (vector[0]) and corresponding trend values (vector[1:]) 
    
    
    Uses a kwarg to choose the geometric comparitor operation:
    
    operator= 
    'cosine'
    'euclidean'
    'theta'
    'triangle'
    'sector'
    'mag'


    Builds output of change points that correspond to all remaining changepoints after the matching input vector. 
    '''

    #for key, value in kwargs.items():
        #print(f"{key}: {value}")

    tee = cpCompConstructor(cpComparitee)
    tor = cpCompConstructor(cpComparitor)
    
    opFunc_dict = {'cosine':Cosine, 'euclidean': Euclidean, 'theta': Theta, 'triangle': Triangle, 'sector': Sector, 'magnitude': Magnitude_Difference} 
    _operator = kwargs.get('operator')
    
    _threshold = kwargs.get('thresh')




    operator = opFunc_dict[_operator]

    QumeProb = np.array([])
    k = 1
    '''    
    cos_thresh = 0.99999 # match is 1.0 (very tight threshhold, all reasonalble values are very close to 1.0)
    euclidean_thresh = 0.1 # match is near zero
    theta_thresh = 0.1745 # match is .1745
    triangle_thresh = 0.2 # match is near zero
    sector_thresh = 0.1 # match is near zero
    '''
    
    for i in range(1,len(tee)):
        for j in range(1,len(tor)):
            #print('iter')
            #print(tor[j][1],tor[j][2])
            #print(tee[i][1],tee[i][2])
            #if tor[j][1] == tee[i][1] and tor[j][2] == tee[i][2]:
            if operator(tor[j][1:], tee[i][1:]) >= _threshold:
                print('*****MATCH' + str(operator(tor[j][1:], tee[i][1:])))
                probAdd = np.diff(tor.T[0])
                #print(tor[i][1],tor[i][2])
                #print(tee[j][1],tee[j][2])
                for q in range(j, len(probAdd)):
                    #CumulativeProb.append(q+np.sum(probAdd[q:]))
                    #print('added')
                    #print(tor[j][0]+np.sum(probAdd[q:]))
                    #QumeProb = np.append(QumeProb, (tor[j][0]+np.sum(probAdd[q:])))
                    QumeProb = np.append(QumeProb, (tee[i][0]+np.sum(probAdd[q:])))
                    k = k+1
            #print('no match')
    #unique, counts = np.unique(CumulativeProb, return_counts=True)
    
    end = QumeProb[QumeProb.argmax()]+1
    #V = np.array([])
    V = np.zeros(int(end))
    unique, counts = np.unique(QumeProb, return_counts=True)
    #print(QumeProb, unique, V)
    for r in range(1, len(unique)):
        #print(r, unique[r], counts[r], end)
        V[int(unique[r])] = counts[r]/k
    
    return V


def prob_cpCorrelation(cpComparitee, cpComparitor, **kwargs):
    
    '''Correlation engine for changepoint associations:
    takes in two matrix (m  x n) of changepoints/trends. One compariTEE who is compared to the compariTOR.
    
    Relative changepoint (vector[0]) and corresponding trend values (vector[1:]) 
    
    
    Uses a kwarg to choose the geometric comparitor operation:
    
    operator= 
    'cosine'
    'euclidean'
    'theta'
    'triangle'
    'sector'
    'mag'


    Builds output of change points that correspond to all remaining changepoints after the matching input vector. 
    '''

    #for key, value in kwargs.items():
        #print(f"{key}: {value}")

    tee = cpCompConstructor(cpComparitee)
    tor = cpCompConstructor(cpComparitor)
    
    opFunc_dict = {'cosine':Cosine, 'euclidean': Euclidean, 'theta': Theta, 'triangle': Triangle, 'sector': Sector, 'magnitude': Magnitude_Difference} 
    _operator = kwargs.get('operator')
    
    _threshold = kwargs.get('thresh')

    _df = kwargs.get('df')
    _log = kwargs.get('log')



    operator = opFunc_dict[_operator]

    QumeProb = np.array([])
    k = 1
    '''    
    cos_thresh = 0.999999 # match is 1.0 (very tight threshhold, all reasonalble values are very close to 1.0)
    euclidean_thresh = 0.1 # match is near zero
    theta_thresh = 0.1745 # match is .1745
    triangle_thresh = 0.2 # match is near zero
    sector_thresh = 0.1 # match is near zero
    '''
    
    for i in range(1,len(tee)):
        for j in range(1,len(tor)):
            #print('iter')
            #print(tor[j][1],tor[j][2])
            #print(tee[i][1],tee[i][2])
            #if tor[j][1] == tee[i][1] and tor[j][2] == tee[i][2]:
            if operator(tor[j][1:], tee[i][1:]) >= _threshold:
                print('*****MATCH' + str(operator(tor[j][1:], tee[i][1:])))
                probAdd = _df[_log].to_numpy()
                #print(tor[i][1],tor[i][2])
                #print(tee[j][1],tee[j][2])    
                QumeProb = np.append(QumeProb, probAdd)     
    #unique, counts = np.unique(CumulativeProb, return_counts=True)
    
    V = QumeProb.sum()
    
    return V
'''This IS HERE STILL ONLY FOR REFERENCE'''
# def prob_cpCorr(cpComparitee, cpComparitor, **kwargs):
    
#     '''Correlation engine for changepoint associations:
#     takes in two matrix (m  x n) of changepoints/trends. One compariTEE who is compared to the compariTOR.
    
#     Relative changepoint (vector[0]) and corresponding trend values (vector[1:]) 
    
    
#     Uses a kwarg to choose the geometric comparitor operation:
    
#     operator= 
#     'cosine'
#     'euclidean'
#     'theta'
#     'triangle'
#     'sector'
#     'mag'


#     Builds output of change points that correspond to all remaining changepoints after the matching input vector. 
#     '''

#     #for key, value in kwargs.items():
#         #print(f"{key}: {value}")

#     tee = cpComparitee
#     tor = cpComparitor
    
#     opFunc_dict = {'cosine':Cosine, 'euclidean': Euclidean, 'theta': Theta, 'triangle': Triangle, 'sector': Sector, 'magnitude': Magnitude_Difference} 
#     _operator = kwargs.get('operator')
    
#     _threshold = kwargs.get('thresh')

#     _df = kwargs.get('df')
#     _log = kwargs.get('log')


#     operator = opFunc_dict[_operator]

#     QumeProb = np.array(([0],[0]))
#     #k = 1
#     '''    
#     cos_thresh = 0.999999 # match is 1.0 (very tight threshhold, all reasonalble values are very close to 1.0)
#     euclidean_thresh = 0.1 # match is near zero
#     theta_thresh = 0.1745 # match is .1745
#     triangle_thresh = 0.2 # match is near zero
#     sector_thresh = 0.1 # match is near zero
#     '''
    
#     for i in range(1,len(tee)):
#         for j in range(1,len(tor)):
#             idx = int(tor[j][0])
#             teeTee = np.concatenate((tee[i-1][1:],tee[i][1:]))
#             torTor = np.concatenate((tor[j-1][1:],tor[j][1:]))
#             #print('iter')
#             #print(tor[j][1],tor[j][2])
#             #print(tee[i][1],tee[i][2])
#             #if tor[j][1] == tee[i][1] and tor[j][2] == tee[i][2]:
#             if operator(torTor, teeTee) >= _threshold:
#                 jump = int(tee[i][0])
#                 #print(type(jump)) 
#                 print('*MATCH at ' + _log + ' Tor depth '+ str(idx)+ ' to Tee depth '+str(jump)+ ' with ' + _operator + ' value ' + str(operator(torTor, teeTee)))
#                 probAdd = _df[_log][jump:].to_numpy()
#                 #print(tor[i][1],tor[i][2])
#                 #print(tee[j][1],tee[j][2])
#                 QumeProb = combine_vector_and_matrix(probAdd, QumeProb, jump)
#                 print('adding Tor probabilites from '+str(jump))     
#     #unique, counts = np.unique(CumulativeProb, return_counts=True)
    
#     V = QumeProb.sum(axis=0)
    
#     return V
def prob_cpCorr(cpComparitee, cpComparitor, **kwargs):
    
    '''Correlation engine for changepoint associations:
    takes in two matrix (m  x n) of changepoints/trends. One compariTEE who is compared to the compariTOR.
    
    Relative changepoint (vector[0]) and corresponding trend values (vector[1:]) 
    
    
    
    Uses a keyword argument to choose the geometric comparitor operation:
    
    operator= 
        'cosine'
        'euclidean'
        'theta'
        'triangle'
        'sector'
        'mag'

    df = dataFrame that holds the changepoint probabiliy curve

    log = dataFrame Column of keyword argument 'df' that contains the change point probability curve

    Builds output of change points that correspond to all remaining changepoints after the matching input vector. 
    

    
    '''

    #for key, value in kwargs.items():
        #print(f"{key}: {value}")

    tee = cpComparitee
    tor = cpComparitor
    
    opFunc_dict = {'cosine':Cosine, 'euclidean': Euclidean, 'theta': Theta, 'triangle': Triangle, 'sector': Sector, 'magnitude': Magnitude_Difference} 
    _operator = kwargs.get('operator')
    
    _threshold = kwargs.get('thresh')

    _df = kwargs.get('df')
    _log = kwargs.get('log')



    operator = opFunc_dict[_operator]

    QumeProb = np.array(([0],[0]))
    #k = 1

    ''' NOTES ABOUT THRESHOLD FOR GEOMETRIC OPERATORS

    cos_thresh = 0.999999 # match is 1.0 (very tight threshhold, all reasonalble values are very close to 1.0)
    euclidean_thresh = 0.1 # match is near zero (AKA Euclidean distance)
    theta_thresh = 0.1745 # match is .1745
    triangle_thresh = 0.2 # match is near zero
    sector_thresh = 0.1 # match is near zero

    '''
    
    for i in range(1,len(tee)):
        for j in range(1,len(tor)):
            idx = int(tor[j][0])
            teeTee = np.concatenate((tee[i-1][1:],tee[i][1:]))
            torTor = np.concatenate((tor[j-1][1:],tor[j][1:]))
            #print('iter')
            #print(tor[j][1],tor[j][2])
            #print(tee[i][1],tee[i][2])
            #if tor[j][1] == tee[i][1] and tor[j][2] == tee[i][2]:
            if operator(torTor, teeTee) >= _threshold:
                jump = int(tee[i][0])
                #print(type(jump)) 
                print('*MATCH at ' + _log + ' Tor depth '+ str(idx)+ ' to Tee depth '+str(jump)+ ' with ' + _operator + ' value ' + str(operator(torTor, teeTee)))
                probAdd = _df[_log][jump:].to_numpy()
                #print(tor[i][1],tor[i][2])
                #print(tee[j][1],tee[j][2])
                QumeProb = combine_vector_and_matrix(probAdd, QumeProb, jump)
                print('adding Tor probabilites to Tee cumulative matix, from '+str(jump))     
    #unique, counts = np.unique(CumulativeProb, return_counts=True)
    
    V = QumeProb.sum(axis=0)
    
    return V #, QumeProb


def combine_vector_and_matrix(vector, matrix, column):
    # Check if the column value is valid
    #if column < 0 or column > matrix.shape[1]:
    #    raise ValueError("Invalid column value")

    # Calculate the dimensions of the new matrix
    num_rows = matrix.shape[0] + 1
    num_cols = max(matrix.shape[1], column + len(vector))

    # Create a new matrix with the desired dimensions, filled with zeros
    combined_matrix = np.zeros((num_rows, num_cols))

    # Copy the original matrix to the new matrix
    combined_matrix[:-1, :matrix.shape[1]] = matrix

    # Copy the vector to the new matrix
    combined_matrix[-1, column:column + len(vector)] = vector

    return combined_matrix

def cpCompConstructor(cpWell):
    '''Creates a ChangePoint input 'length' of Rows by 3 columns matrix 
    that will be used to compare sequentially to another changepoint matix'''
    
    cpBLNK = cpWell
    
    cpCompWellX = np.empty(3*len(cpBLNK[0])).reshape(len(cpBLNK[0]),3)
    
    #Setting special case Zero offset (Start of well) condition, to initial values
    cpCompWellX[0] = np.array((cpBLNK[0,0],cpBLNK[1,0],cpBLNK[1,0]))
    #for loop follows through with the rest of the ChangePTS
    for i in range(1,len(cpBLNK[0])):

        cpCompWellX[i] = np.array((cpBLNK[0,i],cpBLNK[1,i-1],cpBLNK[1,i]))
    return cpCompWellX #Returns matix


def combine_vectors_to_matrix_(*vectors):
    longest = 0
    
    for i in range(len(vectors)):
        if longest < len(vectors[i]):
            longest = len(vectors[i])
        
    num_rows = longest
    num_cols = len(vectors)
    print('rows ', longest, 'columns', len(vectors)) 
    matrix = [[0] * num_cols for _ in range(num_rows)]
    
    for i, vector in enumerate(vectors):
        for j, value in enumerate(vector):
            #print(j,i)
            matrix[j][i] = value
    
    return matrix


'''Glossary:
    Comparitor_vector = base known data that is the foundational knowledge
    
    Comparitee_vector = New data that is the question you "ask" the comparitor'''


def spearman_offset(comparitee_vector, comparitor_vector, **kwargs):

    '''Takes two vectors as input and calculates comparitee (tee) to the comparitor (tor)
     starting from the head of the comparitor vector until the last full comparitor
     segment that is the same length as the comparitee.
    
     outputs the minimum argument loc which correspond to point where the tee:tor have
    highest stat and pvalue correation''' 
    _window = kwargs.get('window')
    _match_param = kwargs.get('param')
    _normalize = kwargs.get('norm')
    stat = []
    pvalue = []
    #normalize_array(df_f11A['GR'][2600:].to_numpy(), 0, 0)
    if _normalize==True:
        tee = normalize_array(comparitee_vector, 0, 0)
        tor = normalize_array(comparitor_vector, 0, 0)
    else:
        tee = comparitee_vector
        tor = comparitor_vector

    if len(tor)<len(tee[:_window]):
        print("Comparitor is shorter than comparitee, this won't do, try a shorter sequence") 
    else:
        for i in range(len(tor)-len(tee)):
            SigRes = (stats.spearmanr(tor[i:i+_window],tee[:_window]))
            stat.append(SigRes.statistic)
            pvalue.append(SigRes.pvalue)
        statAray = np.array(stat)
        statAray = np.nan_to_num(statAray, nan=0.0)
        
        pvalueAray = np.array(pvalue)
        pvalueAray = np.nan_to_num(pvalueAray, nan=1)
        print('statArgmax= ' + str(statAray.argmax()) + ' val ' + str(statAray[statAray.argmax()]))
        print('pvalArgmin= ' + str(pvalueAray.argmin()) + ' val ' + str(pvalueAray[pvalueAray.argmin()]))
        #print
    if _match_param == 'pval':
        return pvalueAray.argmin()
    elif _match_param == 'sval':
        return statAray.argmax()
    else:
        return pvalueAray.argmin()


def generate_progressive_combinations(*number_sets):
    combinations = list(product(*number_sets))
    progressive_combinations = []
    for combination in combinations:
        is_progressive = True
        for i in range(len(combination) - 1):
            if combination[i+1] <= combination[i]:
                is_progressive = False
                break
        if is_progressive:
            progressive_combinations.append(combination)

    return progressive_combinations

#Cosine Similarity - angle between vectors *** does not take magnitude into account***
def Cosine(comparitee_vector, comparitor_vector):
    dot_product = np.dot(comparitee_vector, comparitor_vector.T)
    denominator = (np.linalg.norm(comparitee_vector) * np.linalg.norm(comparitor_vector))
    return dot_product/denominator

# Euclidean Distance - Multi dimentional pythagorean eq.
def Euclidean(comparitee_vector, comparitor_vector):
    vec1 = comparitee_vector.copy()
    vec2 = comparitor_vector.copy()
    if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
    #vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
    return np.linalg.norm(vec1-vec2)

# Adaptation of Cosine similarity
def Theta(comparitee_vector, comparitor_vector):
    return np.arccos(Cosine(comparitee_vector, comparitor_vector)) + np.radians(10)

#Triangle Area Similarity – Sector Area Similarity.
# 1. Angle
# 2. Euclidean Distance between vectors
# 3. Magnitude of vectors

def Triangle(comparitee_vector, comparitor_vector):
    theta = np.radians(Theta(comparitee_vector, comparitor_vector))
    return ((np.linalg.norm(comparitee_vector) * np.linalg.norm(comparitor_vector)) * np.sin(theta))/2

#Magnitude Difference
def Magnitude_Difference(vec1, vec2):
    return abs((np.linalg.norm(vec1) - np.linalg.norm(vec2)))
    
#Sector Area Similarity    
def Sector(comparitee_vector, comparitor_vector):
    ED = Euclidean(comparitee_vector, comparitor_vector)
    MD = Magnitude_Difference(comparitee_vector, comparitor_vector)
    theta = Theta(comparitee_vector, comparitor_vector)
    return np.pi * (ED + MD)**2 * theta/360


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


    '''
    This is a helper function will groom the ends (by holding values) and remove NaN's of an
     incomplete dataset and perform a Bayesian changepoint search.

     results of the search will be added to another row of the dataframe.

     Really should be building a well 'object' that holds this data along with the
     trends and correlation to other wells, but for now it is a dataframe... added to cp_Comparitor.py

    '''

    def return_df_add_CP_Prob(df, start, stop, log_dict=[], label='CP_Prob_', nan_method='local_average', window_size=200, segment_length=1000, **kwargs):
        ''' Adds a probability curve for change points to a dataFrame
    
        Parameters:
            df (pandas.Dataframe) : input dataFrame with a dataset mentioned in the log_dictionary 
            log_dict (dict) :  Log dictionary
            label (str)  : Prefix for probability curve that will be appended to the dataFrame
        
            **passthrough argument for replace_nans function**
            nan_method (str) : choices ('global_average', 'local_average', 'linear_interpolation') 
            window_size (int) : window size for calculating local averages.
    
            segment_length (int) : split for offline Bayesian Change Point detection,
                                unsefult to control the computational cost of this function

        Returns: None, this merely adds a curve to the input dataframe

        '''
    
     #Keyword Args assingment
    _prior = kwargs.get('prior') # choices are "const", "geometric", "neg_binomial"
    _method = kwargs.get('method')  # choices are 'gauss', 'ifm' or 'full_cov'
    _engine = kwargs.get('engine')  # choices are "numpy" or "numba"
    _normalize = kwargs.get('normal')

    #Argument Defaults
    if _method == None:
        _method = "full_cov"
    if _prior == None:
        _prior = "const"
    if _engine == None:
        _engine ="numba"
    #if _normalize == False:
    
    print(str(df))
    #Fill ends of incomplete df, replace NaN's with parameter method, and changepoint search with parameter segment splits
    for log in log_dict:
        print(' finding changepoints in '+log)
        labl = label+log
        df[log][start:stop] = wtool.fill_out_data_ends(df[log][start:stop].values)

        df[log][start:stop] = wtool.replace_nans(df[log][start:stop].values, method=nan_method, window_size=window_size)
        
        changepoint_prob = cp_Tools.bayes_offline_split(df[log][start:stop].values, segment_length=segment_length, method=_method, prior=_prior, engine=_engine, normal=_normalize)
        
        log_index_array = df[log][start:stop].index
        d = {labl : pd.Series(changepoint_prob, index=log_index_array)}
        dfcp = pd.DataFrame(data=d) 
        df = pd.concat([df, dfcp], axis=1,)
        
    return df
        