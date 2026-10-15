import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sampler import Sampler

np.set_printoptions(legacy='1.25')

#input symbols, start date, end date, and an optional dataframe
#first symbol is always the dependent variable Y
#the rest of the variables are independent variables X
class Multilinear_Regression():
    def __init__(self,
                 start = str(date.today() - timedelta(59)),
                 end = str(date.today() - timedelta(1)),
                 interval = '1d',
                 symbol_list=None,
                 optional_df=None):
        if optional_df is not None:
            self.df = optional_df
        else:
            #Download the data from Yahoo Finance
            df = yf.download(symbol_list, start, end, interval = interval, multi_level_index=False, ignore_tz=True)
            self.df = df
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = [' '.join(col).strip() for col in self.df.columns.values]
            self.symbol_list = symbol_list
            #count the number of symbols in the list
            self.n_symbols = len(self.symbol_list)
            #for every symbol in the list, create a new column in the datafram that its returns, and name the column after the symbol
            for symbol in self.symbol_list:
                self.df[f'{symbol}'] = np.log(self.df[f'Close {symbol}']).diff() # returns for each symbol
        #A number identifier to know what samples are being sampled
        self.df['Number'] = np.arange(1, len(self.df) + 1)
        self.df.dropna(inplace = True)
        #Sample frome the given dataframe
        sampler = Sampler(self.df)
        self.df = sampler.sampled_df
        #store arrays for each variable
        for i in range(self.n_symbols):
            setattr(self, f'{self.symbol_list[i]}_array', self.df[f'{self.symbol_list[i]}'])
        self.interval = interval
        #how many rows is the sampled dataframe not including header
        self.n = len(self.df) - 1

        #Preliminary Calculations
        #Mean of each variable and store the mean in a dictionary name:mean
        self.means = {}
        for i in range(self.n_symbols):
            setattr(self, f'mean_{self.symbol_list[i]}', self.df[f'{self.symbol_list[i]}'].mean()) # Xbar for each variable
            self.means[f'{self.symbol_list[i]}'] = getattr(self, f'mean_{self.symbol_list[i]}') #store in dict
        print(f'Means: {self.means}')
        self.idx = self.df.index.get_loc(self.df.index[0])
        self.last_idx = self.df.index.get_loc(self.df.index[-1])
        def SSxy_calculation(return_array_1, return_array_2, last_idx):
            #return array 1 is predictor and return array 2 is response variable
            SSxy = 0
            for return_idx in range(0, last_idx):
                SSxy += (return_array_1.iloc[return_idx] - self.means[f'{return_array_1.name}']) * (return_array_2.iloc[return_idx] - self.means[f'{return_array_2.name}'])
            SSxy = SSxy / (last_idx - 1)
            return SSxy
        def SSxx_calculation(return_array, mean_value, last_idx):
            #Works as SSyy also
            SSxx = 0
            for return_idx in range(0, last_idx):
                SSxx += ((return_array.iloc[return_idx] - mean_value)**2)
            SSxx = SSxx / (last_idx - 1)
            return SSxx
        #Recall Slope is SSXY / SSXX
        #Calculate Slope Coefficients for each variable
        self.slopes = {}
        #we use 1 because 0 is our dependent variable Y
        for i in range(1, self.n_symbols):
            slope = SSxy_calculation(getattr(self, f'{self.symbol_list[i]}_array'), getattr(self, f'{self.symbol_list[0]}_array'), self.last_idx) / SSxx_calculation(getattr(self, f'{self.symbol_list[i]}_array'), self.means[f'{self.symbol_list[i]}'], self.last_idx)
            self.slopes[f'{self.symbol_list[i]}'] = slope
        print(f'Slope Coefficients: {self.slopes}')
    def hypothesis_test(self, significance_level=.05):
        # Null Hypothesis is Slope = 0
        # Alternative Hypothesis is Slope != 0
        t = self.slope / self.standard_error_of_slope
        critical_value = stats.t.ppf(1-(significance_level / 2), self.n - 2)
        if abs(t) > critical_value:
            print(f'We REJECT the Null Hypothesis')
            print('-' * 50)
            print(f'The Test Statistic {round(t,4)} falls in the rejection region of {round(critical_value,4)}')
            print(f"We can conclude that the slope of the linear regression between {self.symbol_list[0]} and {self.symbol_list[1]} is not 0")
            print("Thus, we can predict the value of Y based on a given X, and conclude that X contributes to Y")
        else:
            print(f"We CANNOT reject the Null Hypothesis at this signficance level")
            print(f'The Test Statistic {round(t,4)} does not fall into the rejection region of {round(critical_value,4)}')
            print(f"More than likely no linear relationship exists between {self.symbol_list[0]} and {self.symbol_list[1]}")

    def slope_confidence_interval(self, confidence_level=.95):
        critical_value = stats.t.ppf(1-((1-confidence_level) / 2), self.n - 2)
        upper_level = self.slope + (critical_value * self.standard_error_of_slope)
        lower_level = self.slope - (critical_value * self.standard_error_of_slope)
        interval = [round(lower_level,4), round(upper_level,4)]
        print(f'{confidence_level * 100}% Confidence Level: Slope lies between {interval}')
        return interval
    
    def find_y(self, x, confidence_level=.95, find_mean=False):
        #y is also called Y hat here
        y = self.y_int + self.slope * x
        critical_value = stats.t.ppf(1-((1-confidence_level) / 2), self.n - 2)
        print(f'Given X: {x} | Expect Y: {round(y,4)}')
        if find_mean is True:
            #Whenever we run the model on X, we'd expect Y to lie within this interval
            # i dont think this math is right
            upper_level = y + critical_value * self.standard_error_of_estimate * math.sqrt((1/self.n) + ((x - self.mean_x)**2 / self.ssxx))
            lower_level = y - critical_value * self.standard_error_of_estimate * math.sqrt((1/self.n) + ((x - self.mean_x)**2 / self.ssxx))
            interval = [round(lower_level * 100,4), round(upper_level * 100,4)]
            print (f'{confidence_level * 100}% Confidence Interval of Y Mean: If {self.x_name} moves {x * 100}%, we expect the mean of {self.y_name} to move {round(y,4) * 100}% between the interval {interval}')
        else:
            # Prediction interval for a single test of Y
            upper_level = y + critical_value * (self.standard_error_of_estimate * math.sqrt(1 + (1/self.n) + ((x - self.mean_x)**2 / self.ssxx)))
            lower_level = y - critical_value * (self.standard_error_of_estimate * math.sqrt(1 + (1/self.n) + ((x - self.mean_x)**2 / self.ssxx)))
            interval = [round(lower_level * 100,4), round(upper_level * 100,4)]
            print (f'{confidence_level * 100}% Prediction Interval: If {self.x_name} moves {x * 100}%, we expect {self.y_name} to move {round(y,4) * 100}% between the interval {interval}')

    def scatterplot_matrix(self):
        #for each symbol in the list, create a scatterplot matrix of the returns
        fig = px.scatter_matrix(self.df, 
                                dimensions=self.symbol_list, 
                                title='Scatterplot Matrix of Returns',
                                )
        fig.update_traces(showupperhalf=False, diagonal_visible=False)
        fig.update_layout(width=800, height=800)
        fig.show()
