import numpy as np
import pandas as pd
import scipy
import ruptures as rpt
import scipy.stats as stats
from scipy.signal import find_peaks


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
    
    peaks, _peak_heights = find_peaks(data, height=0.01)
    
    if return_peak_heights == True:
        return peaks, _peak_heights
    else:
        return peaks




def prob_piecewise_linearRegress(data, prob):
    
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

    bkps = find_prob_peaks(prob, height=0.7, return_peak_heights=False)
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





def cpCorrelation(cpComparitee, cpComparitor, **kwargs):
    
    '''Correlation engine for changepoint associations:
    takes in two vectors (m x n) of changepoints/trends. One compariTEE who is compared to the compariTOR.
    
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
                    k = k+1 # this counter is a placeholder for, not currently used.
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
    takes in two vectors (m x n) of changepoints/trends. One compariTEE who is compared to the compariTOR.
    
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

#Spearman Correlation, calculator
# def spearman_offset(comparitee_vector, comparitor_vector, **kwargs):
#     '''Takes two vectors as input and calculates comparitee (tee) to the comparitor (tor)
#     starting from the head of the comparitor vector until the last full comparitor
#     segment that is the same length as the comparitee.
    
#     outputs the minimum argument loc which correspond to point where the tee:tor have
#      highest stat and pvalue correation''' 

#     _window = kwargs.get('window')
#     stat = []
#     pvalue = []
#     tee = comparitee_vector
#     tor = comparitor_vector

#     if len(tor)<len(tee[:_window]):
#         print("Comparitor is shorter than comparitee, this won't do, try a shorter sequence") 
#     else:
#         for i in range(len(tor)-len(tee)):
#             SigRes = (stats.spearmanr(tor[i:i+_window],tee[:_window]))
#             stat.append(SigRes.statistic)
#             pvalue.append(SigRes.pvalue)
#         statAray = np.array(stat)
#         statAray = np.nan_to_num(statAray, nan=0.0)
        
#         pvalueAray = np.array(pvalue)
#         pvalueAray = np.nan_to_num(pvalueAray, nan=1)
#         print('statArgmax= ' + str(statAray.argmax()) + ' val ' + str(statAray[statAray.argmax()]))
#         print('pvalArgmin= ' + str(pvalueAray.argmin()) + ' val ' + str(pvalueAray[pvalueAray.argmin()]))
#         #print
#     return pvalueAray.argmin()
def spearman_offset(comparitee_vector, comparitor_vector, **kwargs):
    '''Takes two vectors as input and calculates comparitee (tee) to the comparitor (tor)
     starting from the head of the comparitor vector until the last full comparitor
     segment that is the same length as the comparitee.
    
     outputs the minimum argument loc which correspond to point where the tee:tor have
    highest stat and pvalue correation''' 

    _window = kwargs.get('window')
    _match_param = kwargs.get('param')
    stat = []
    pvalue = []
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