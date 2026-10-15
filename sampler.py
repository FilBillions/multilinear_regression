import numpy as np
import csv
import os
import sys
import pandas as pd
sys.path.append(".")

np.set_printoptions(legacy='1.25')

#Goals
# Usable in any model
# Take in a dataframe and return a randomized sampled data frame
class Sampler:
    def __init__(self, df, sample_size=30):
        self.df = df
        self.sample_size = sample_size
        self.sampled_df = self.generate_sampled_df()

    def generate_sampled_df(self):
        # if sample.csv exists, use that to sample, or else generate random sample
        if os.path.exists('sample.csv'):
            # check if sample.csv has the same columns as df
            sampled_df = pd.read_csv('sample.csv', index_col=0, parse_dates=True)
            if not sampled_df.columns.equals(self.df.columns):
                print("sample.csv columns do not match the input dataframe columns. Generating a new sample.")
                os.remove('sample.csv')
                return self.generate_sampled_df()
            print('-'*20)
            print(f"Using existing sample.csv with {len(sampled_df)} data points.")
            print('-'*20)
            return sampled_df
        else:
            if self.sample_size >= len(self.df):
                return self.df.copy()
            else:
                #Save the old df as population.csv
                self.df.to_csv('population.csv', index_label='Date')
                #sample the dataframe randomly without replacement
                sampled_indices = np.random.choice(self.df.index, size=self.sample_size, replace=False)
                sampled_df = self.df.loc[sampled_indices].sort_index()
                print('-'*20)
                print(f"Sampled {self.sample_size} data points from original {len(self.df)} data points.")
                print('-'*20)
                # Save the df as the sample
                with open('sample.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Date'] + list(self.df.columns))
                    for idx, row in sampled_df.iterrows():
                        writer.writerow([idx] + list(row))
                return sampled_df