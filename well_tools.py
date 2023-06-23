'''functions and well log grooming for different_bayes framework'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#class well_tools(object):


def fill_out_data_ends(data):

    '''replaces NaN values at the ends of an incomplete pandas dataset
    
    (one that does not fill out the ends of the index) with first or last value 
    
    for beginning or end encountered'''

    lmin = np.min(np.where(np.isnan(data)==False))
    lmax = np.max(np.where(np.isnan(data)==False))
    if lmin > 0:
        data[:lmin] = data[lmin]
    if lmax < len(data):
        data[lmax:] = data[lmax]
    return data



def remove_nan(data, replace_with='glob_avg'):

    '''Replace NaN values with average of all input data
    
    will soon include a calculation of the local average, or the first or last real datapoint
    encountered.
    '''
    
    if replace_with == 'glob_avg':
  
        data_with_removed_NaN = data[~np.isnan(data)]
        mu = np.mean(data_with_removed_NaN)
        data[np.isnan(data)] = mu
    
    #   elif replace_with = 'local_avg'
    #       avg = df.rolling(periods=2, window=30)

    return data




def df_rolling_avg(data, datalabel, window=30, periods=3):

    _window = window
    _periods = periods        

    '''takes a data segment and constructs a pandas dataframe that is a rolling average
    of the input'''

    string = 'Roll_Avg_'+ str(datalabel)
    
    #df = pd.DataFrame(data, columns=[datalabel])
    df= pd.DataFrame()
    df[string] = data[str(datalabel)].rolling(_window, min_periods=_periods).mean() 

    return df

def compute_all_rolling_avg(data, window, periods):

    '''compute rolling average, takes in a pandas dataframe, an integer period and number of
    windows. Appends rolling average for all column datas and outputs df, retaining originals'''
    
    _window = window
    _periods = periods 
    string = ""
    df = pd.DataFrame(data)
 
    _datalabels = df.columns.unique()
    for datalabel in _datalabels:
        string = 'Roll_Avg_'+ str(datalabel)
    
        #df = pd.DataFrame(df, columns=[datalabel])
        
        df[string] = df[datalabel].rolling(_window, min_periods=_periods).mean()
        
    return df


def label_generator(df_well, df_tops, column_depth, label_name): #From Yohanes Nuwara
    """
    Generate Formation (or other) Labels to Well Dataframe
    (useful for EDA and machine learning purpose)

    Input:

    df_well is your well dataframe (that originally doesn't have the intended label)
    df_tops is your label dataframe (this dataframe should ONLY have 2 columns)
        1st column is the label name (e.g. formation top names)
        2nd column is the depth of each label name

    column_depth is the name of depth column on your df_well dataframe
    label_name is the name of label that you want to produce (e.g. FM. LABEL)

    Output:

    df_well is your dataframe that now has the labels (e.g. FM. LABEL)
    """


    # generate list of formation depths and top names
    fm_tops = df_tops.iloc[:,0]  
    fm_depths = df_tops.iloc[:,1] 

    # create FM. LABEL column to well dataframe
    # initiate with NaNs
    df_well[label_name] = np.full(len(df_well), np.nan)  

    indexes = []
    topnames = []
    for j in range(len(fm_depths)):
        # search index at which the DEPTH in the well df equals to OR
        # larger than the DEPTH of each pick in the pick df
        if (df_well[column_depth].iloc[-1] > fm_depths[j]):
            index = df_well.index[(df_well[column_depth] >= fm_depths[j])][0]
            top = fm_tops[j]
            indexes.append(index)
            topnames.append(top)

        # replace the NaN in the LABEL column of well df
        # at the assigned TOP NAME indexes
            df_well[label_name].loc[indexes] = topnames

        # Finally, using pandas "ffill" to fill all the rows 
        # with the TOP NAMES
            df_well = df_well.fillna(method='ffill')  

    return df_well