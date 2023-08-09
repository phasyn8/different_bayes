'''functions and well log grooming for different_bayes framework'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#class well_tools(object):


def fill_out_data_ends(data):

    '''replaces NaN values at the head and tail ends of an incomplete pandas dataset
        (one that does not fill out the ends of the index)
    
    Parameters:
         data: (pandas.DataFrame)


    Output:
        pandas.DataFrame with dataset extrapolated from the first and last valid values
        extrapolated to the ends
    '''

    lmin = np.min(np.where(np.isnan(data)==False))
    lmax = np.max(np.where(np.isnan(data)==False))
    if lmin > 0:
        data[:lmin] = data[lmin]
    if lmax < len(data):
        data[lmax:] = data[lmax]
    return data


def replace_nans(array, method='global_average', window_size=3):
    """
    Replace NaN values in an array with specified method.
    
    Parameters:
        array (numpy.ndarray): Input array containing NaN values.
        method (str): Method to replace NaNs. Options are 'global_average',
                      'local_average', or 'linear_interpolation'.
                      Default is 'global_average'.
        window_size (int): Size of the window for local averaging. Applicable
                           only if method is 'local_average'. Default is 3.
    
    Returns:
        numpy.ndarray: Array with NaN values replaced.
    """
    if method == 'global_average':
        # Compute the global average excluding NaN values
        global_avg = np.nanmean(array)
        # Replace NaNs with the global average
        array[np.isnan(array)] = global_avg
    
    elif method == 'local_average':
        # Replace NaNs with local average within a window
        for i in range(len(array)):
            if np.isnan(array[i]):
                # Calculate the indices for the window around the NaN
                lower_bound = max(0, i - window_size)
                upper_bound = min(len(array), i + window_size + 1)
                
                # Compute the local average excluding NaN values
                local_avg = np.nanmean(array[lower_bound:upper_bound])
                # Replace NaN with the local average
                array[i] = local_avg
    
    elif method == 'linear_interpolation':
        # Create a mask of non-NaN values
        mask = ~np.isnan(array)
        # Create a linear interpolation function
        interp_func = interp1d(np.where(mask)[0], array[mask])
        # Replace NaNs with interpolated values
        array[np.isnan(array)] = interp_func(np.where(np.isnan(array))[0])
    
    else:
        raise ValueError("Invalid method. Please choose 'global_average', "
                         "'local_average', or 'linear_interpolation'.")
    
    return array




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

    '''
    compute rolling average,takes in a pandas dataframe, an integer period and number of
    windows. Appends rolling average for all column datas and outputs df, retaining originals
    
    Parameters:
        data: (numpy array or pandas DataFrame)** will perform the operation on all rows**
        window: rolling window for the mean calculation 
        period: not clear what this is, please refer to pandas.DataFrame.rolling()

    Returns:
        pandas dataframe including original input data and

    '''
    
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

    Parameters:

        df_well (pandas.DataFrame) is your well dataframe (that originally doesn't have the intended label)
        df_tops (pandas.DataFrame) is your label dataframe (this dataframe should ONLY have 2 columns)
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