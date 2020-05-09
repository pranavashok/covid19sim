import pandas as pd
import os


class Covid19IndiaStatewiseLoader:
    def __init__(self):
        if os.path.exists("data"):
            self.store_location = 'data/covid19india_state_wise_daily.pickle'
        elif os.path.exists("../data"):
            self.store_location = '../data/covid19india_state_wise_daily.pickle'
        elif os.path.exists("../../data"):
            self.store_location = '../../data/covid19india_state_wise_daily.pickle'
        self.url = 'https://api.covid19india.org/csv/latest/state_wise_daily.csv'

    def load(self):
        try:
            # Read from cache
            print("Reading from cache")
            data = pd.read_pickle(self.store_location)
        except:
            # If not available in cache, then download and store to cache
            print(f"Not available in cache, downloading from {self.url}")
            raw_data = pd.read_csv(self.url, parse_dates=[0], dayfirst=True)
            data = raw_data.pivot(index='Date', columns='Status')
            data.columns = data.columns.set_names(["State", "Status"])
            data.drop("TT", axis=1, inplace=True)
            data.to_pickle(self.store_location)
        return data
