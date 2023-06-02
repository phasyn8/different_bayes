import numpy as np
import pandas as pd
import scipy
import ruptures as rpt


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


    

def cpCorrelation(cpComparitee, cpComparitor):
    tee = cpCompConstructor(cpComparitee)
    tor = cpCompConstructor(cpComparitor)
    QumeProb = np.array([])
    k = 1
    
    for i in range(1,len(tee)):
        for j in range(1,len(tor)):
            #print('iter')
            #print(tor[j][1],tor[j][2])
            #print(tee[i][1],tee[i][2])
            #if tor[j][1] == tee[i][1] and tor[j][2] == tee[i][2]:
            if Cosine(tor[j][1:], tee[i][1:]) >= 0.9999:
                print('*****MATCH' + str(Cosine(tor[j][1:], tee[i][1:])))
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

def cpCompConstructor(cpWell): # Creates a ChangePoint length Rows by 3 columns matrix that will be used to compare sequentially 
    
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


#Cosine Similarity *** does not take magnitude into account***
def Cosine(comparitee_vector, comparitor_vector):
    dot_product = np.dot(comparitee_vector, comparitor_vector.T)
    denominator = (np.linalg.norm(comparitee_vector) * np.linalg.norm(comparitor_vector))
    return dot_product/denominator

    
# Euclidean Distance
def Euclidean(comparitee_vector, comparitor_vector):
    vec1 = comparitee_vector.copy()
    vec2 = comparitor_vector.copy()
    if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
    #vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
    return np.linalg.norm(vec1-vec2)


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