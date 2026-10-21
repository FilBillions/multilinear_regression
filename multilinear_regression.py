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
        self.n = len(self.df)
        print(f'Number of Samples: {self.n}')
        #Preliminary Calculations
        #Mean of each variable and store the mean in a dictionary name:mean
        self.means = {}
        for i in range(self.n_symbols):
            setattr(self, f'mean_{self.symbol_list[i]}', self.df[f'{self.symbol_list[i]}'].mean()) # Xbar for each variable
            self.means[f'{self.symbol_list[i]}'] = getattr(self, f'mean_{self.symbol_list[i]}') #store in dict
        self.idx = self.df.index.get_loc(self.df.index[0])
        self.last_idx = len(self.df)
        def SSxy_calculation(return_array_1, return_array_2, last_idx):
        #return array 1 is predictor and return array 2 is response variable
            SSxy = 0
            for return_idx in range(0, last_idx):
                SSxy += (return_array_1.iloc[return_idx] - self.means[f'{return_array_1.name}']) * (return_array_2.iloc[return_idx] - self.means[f'{return_array_2.name}'])
            return SSxy
        def SSxx_calculation(return_array, mean_value, last_idx):
        #Works as SSyy also
            SSxx = 0
            for return_idx in range(0, last_idx):
                SSxx += ((return_array.iloc[return_idx] - mean_value)**2)
            return SSxx
    #Store the SSxy and SSxx for each variable in a dictionary name:ssxy and ssxx
        self.ssxy = {}
        self.ssxx = {}
        for i in range(1, self.n_symbols): #we use 1 because 0 is our dependent variable Y
            ssxy = SSxy_calculation(getattr(self, f'{self.symbol_list[i]}_array'), getattr(self, f'{self.symbol_list[0]}_array'), self.last_idx)
            ssxx = SSxx_calculation(getattr(self, f'{self.symbol_list[i]}_array'), self.means[f'{self.symbol_list[i]}'], self.last_idx)
            setattr(self, f'ssxy_{self.symbol_list[i]}', ssxy)
            setattr(self, f'ssxx_{self.symbol_list[i]}', ssxx)
            self.ssxy[f'{self.symbol_list[i]}'] = ssxy
            self.ssxx[f'{self.symbol_list[i]}'] = ssxx
    #Calculate Slope Coefficients for each variable
        self.slopes = {}
        for i in range(1, self.n_symbols): #we use 1 because 0 is our dependent variable Y
            slope = SSxy_calculation(getattr(self, f'{self.symbol_list[i]}_array'), getattr(self, f'{self.symbol_list[0]}_array'), self.last_idx) / SSxx_calculation(getattr(self, f'{self.symbol_list[i]}_array'), self.means[f'{self.symbol_list[i]}'], self.last_idx)
            self.slopes[f'{self.symbol_list[i]}'] = slope
    #store the variances in a dictionary name:variance
        self.variances = {}
        for i in range(self.n_symbols):
            variance = SSxx_calculation(getattr(self, f'{self.symbol_list[i]}_array'), self.means[f'{self.symbol_list[i]}'], self.last_idx) / ((self.n) - 1)
            setattr(self, f'variance_{self.symbol_list[i]}', variance)
            self.variances[f'{self.symbol_list[i]}'] = variance
    #store the covariances in a dictionary name:covariance
        self.covariances = {}
        for i in range(1, self.n_symbols):
            covariance = SSxy_calculation(getattr(self, f'{self.symbol_list[i]}_array'), getattr(self, f'{self.symbol_list[0]}_array'), self.last_idx) / ((self.n) - 1)
            setattr(self, f'covariance_{self.symbol_list[i]}', covariance)
            self.covariances[f'{self.symbol_list[i]}'] = covariance
    #calculate the Y intercept
        self.y_int = self.means[f'{self.symbol_list[0]}'] - sum([self.slopes[f'{self.symbol_list[i]}'] * self.means[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)])
    #Use Y intercept and Slopes to calculate the predicted Y values for each sample -> Y Hat
        self.df['Predicted Y'] = self.y_int + sum([self.slopes[f'{self.symbol_list[i]}'] * getattr(self, f'{self.symbol_list[i]}_array') for i in range(1, self.n_symbols)])
    #Create a new column for residuals -> Y - Y Hat
        self.df['Residuals'] = self.df[f'{self.symbol_list[0]}'] - self.df['Predicted Y']
    #ANOVA Calculations
    #Were using a different SSR calculation that forces the x variables to be uncorrelated
    #we will change that once we add tests for correlation
        self.regression_df = self.n_symbols - 1 #The number of independent variables
        self.residual_df = self.n - self.regression_df - 1
        self.total_df = self.n - 1
        #SSR is the sum of the slope times ssxy
        self.ssr = sum([self.slopes[f'{self.symbol_list[i]}'] * self.ssxy[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)])
        self.sst = sum([(self.df[f'{self.symbol_list[0]}'].iloc[i] - self.means[f'{self.symbol_list[0]}'])**2 for i in range(self.last_idx)])
        self.sse = self.sst - self.ssr
        self.sst_check = self.ssr + self.sse
        self.msr = self.ssr / self.regression_df
        self.mse = self.sse / self.residual_df
        self.F_statistic = self.msr / self.mse
    #Standard Error Calculations
        self.standard_error_of_estimate = math.sqrt(self.mse)
        self.standard_error_of_slope = {}
        for i in range(1, self.n_symbols):
            standard_error = self.standard_error_of_estimate / (math.sqrt(self.ssxx[f'{self.symbol_list[i]}']))
            setattr(self, f'standard_error_of_slope_{self.symbol_list[i]}', standard_error)
            self.standard_error_of_slope[f'{self.symbol_list[i]}'] = standard_error
        self.standard_error_of_intercept = self.standard_error_of_estimate * math.sqrt((1/self.n) + (sum([self.means[f'{self.symbol_list[i]}']**2 / self.ssxx[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)])))
    #Hypothesis Testing
        self.t_statistic = {}
        self.t_intercept = self.y_int / self.standard_error_of_intercept
        self.t_statistic['Intercept'] = self.t_intercept
        setattr(self, f't_statistic_Intercept', self.t_intercept)
        for i in range(1, self.n_symbols):
            t_stat = self.slopes[f'{self.symbol_list[i]}'] / self.standard_error_of_slope[f'{self.symbol_list[i]}']
            setattr(self, f't_statistic_{self.symbol_list[i]}', t_stat)
            self.t_statistic[f'{self.symbol_list[i]}'] = t_stat
    #P-value Calculations
        self.p_values = {}
        self.p_intercept = 2 * (1 - stats.t.cdf(abs(self.t_intercept), self.residual_df))
        self.p_values['Intercept'] = self.p_intercept
        for i in range(1, self.n_symbols):
            p_value = 2 * (1 - stats.t.cdf(abs(self.t_statistic[f'{self.symbol_list[i]}']), self.residual_df))
            self.p_values[f'{self.symbol_list[i]}'] = p_value
    #Confidence Interval Calculations
        self.confidence_intervals_lower = {}
        self.confidence_intervals_upper = {}
        self.confidence_intervals_lower['Intercept'] = self.y_int - (stats.t.ppf(1-((1-.95) / 2), self.residual_df) * self.standard_error_of_intercept)
        self.confidence_intervals_upper['Intercept'] = self.y_int + (stats.t.ppf(1-((1-.95) / 2), self.residual_df) * self.standard_error_of_intercept)
        for i in range(1, self.n_symbols):
            lower_level = self.slopes[f'{self.symbol_list[i]}'] - (stats.t.ppf(1-((1-.95) / 2), self.residual_df) * self.standard_error_of_slope[f'{self.symbol_list[i]}'])
            upper_level = self.slopes[f'{self.symbol_list[i]}'] + (stats.t.ppf(1-((1-.95) / 2), self.residual_df) * self.standard_error_of_slope[f'{self.symbol_list[i]}'])
            self.confidence_intervals_lower[f'{self.symbol_list[i]}'] = lower_level
            self.confidence_intervals_upper[f'{self.symbol_list[i]}'] = upper_level
    #Create presentation Dataframe
        self.presentation_df = pd.DataFrame(columns=['', 'Coefficent', 'Standard Error', 't Stat', 'P-value', 'Lower 95%', 'Upper 95%'])
        self.presentation_df[''] = ['Intercept'] + [f'{self.symbol_list[i]}' for i in range(1, self.n_symbols)]
        self.presentation_df['Coefficent'] = [self.y_int] + [self.slopes[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)]
        self.presentation_df['Standard Error'] = [self.standard_error_of_intercept] + [self.standard_error_of_slope[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)]
        self.presentation_df['t Stat'] = [self.t_statistic['Intercept']] + [self.t_statistic[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)]
        self.presentation_df['P-value'] = [self.p_values['Intercept']] + [self.p_values[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)]
        self.presentation_df['Lower 95%'] = [self.confidence_intervals_lower['Intercept']] + [self.confidence_intervals_lower[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)]
        self.presentation_df['Upper 95%'] = [self.confidence_intervals_upper['Intercept']] + [self.confidence_intervals_upper[f'{self.symbol_list[i]}'] for i in range(1, self.n_symbols)]
    #Create ANOVA Table
        self.anova_df = pd.DataFrame(columns=['', 'DF', 'SS', 'MS', 'F'])
        self.anova_df[''] = ['Regression', 'Residual', 'Total']
        self.anova_df['DF'] = [self.regression_df, self.residual_df, self.total_df]
        self.anova_df['SS'] = [self.ssr, self.sse, self.sst]
        self.anova_df['MS'] = [self.msr, self.mse, '']
        self.anova_df['F'] = [self.F_statistic, '', '']
    
    def residual_plots(self):
        print("Residuals vs Predicted Values Plot")
        print("Things to look for in the plot:")
        print("If the residuals are randomly scattered around 0, then the model is appropriate for the data")
        print("If the residuals show a pattern, then the model is not appropriate for the data")
        print("Outliers  can be used to check for influential points that may be affecting the model")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df['Predicted Y'], y=self.df['Residuals'], mode='markers', name='Data Points'))
        fig.add_trace(go.Scatter(x=self.df['Predicted Y'], y=[0]*len(self.df), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Residuals vs Predicted Values',
                          xaxis_title='Predicted Values',
                          yaxis_title='Residuals',
                          showlegend=True)
        fig.show()

        print("Residuals vs Independent Variables Plot")
        fig = go.Figure()
    # each independent variable will have its own subplot
        for i in range(1, self.n_symbols):
            fig.add_trace(go.Scatter(x=self.df[f'{self.symbol_list[i]}'], y=self.df['Residuals'], mode='markers', name=f'{self.symbol_list[i]}'))
            fig.update_layout(title='Residuals vs Independent Variables',
                          xaxis_title='Independent Variables',
                          yaxis_title='Residuals',
                          showlegend=True)
            fig.show()

    # if theres a trend, print a note that says theres probably a trend
        if self.df['Predicted Y'].corr(self.df['Residuals']) > 0.5:
            print("There is a strong positive correlation between the predicted values and the actual values")
            print("Thus, we can predict the value of Y based on a given X, and conclude that X contributes to Y")
            print("This model is weak!!")
        else:
            print("There is a weak correlation between the predicted values and the actual values, this is good!")
    
    def return_coefficients(self):
        print("If P-value < 0.05, we can reject the null hypothesis that the coefficient is equal to 0")
        print("If P-value > 0.05, we cannot reject the null hypothesis that the coefficient is equal to 0")
        print(self.presentation_df)
        print('-' * 50)
        print(self.anova_df)

    def scatterplot_matrix(self):
        #for each symbol in the list, create a scatterplot matrix of the returns
        fig = px.scatter_matrix(self.df, 
                                dimensions=self.symbol_list, 
                                title='Scatterplot Matrix of Returns',
                                )
        fig.update_traces(showupperhalf=False, diagonal_visible=False)
        fig.update_layout(width=800, height=800)
        fig.show()
