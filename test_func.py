import numpy as np
import pandas as pd



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