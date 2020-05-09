import json
import urllib.request
import pandas as pd
import os


class Covid19IndiaNationalLoader:
    def __init__(self):
        if os.path.exists("../data"):
            self.store_location = '../data/covid19india_national_daily.pickle'
        elif os.path.exists("../../data"):
            self.store_location = '../../data/covid19india_national_daily.pickle'
        self.url = 'https://api.covid19india.org/data.json'

    def load(self):
        try:
            # Read from cache
            print("Reading from cache")
            data = pd.read_pickle(self.store_location)
        except:
            # If not available in cache, then download and store to cache
            print(f"Not available in cache, downloading from {self.url}")
            with urllib.request.urlopen(self.url) as data_url:
                raw_data = json.loads(data_url.read().decode())
            data = pd.json_normalize(raw_data['cases_time_series'])
            data['date'] = pd.to_datetime(data['date']+"2020")
            data = data.set_index('date')
            assert list(data.columns) == ['dailyconfirmed', 'dailydeceased', 'dailyrecovered', 'totalconfirmed',
                                          'totaldeceased', 'totalrecovered']
            data.columns = pd.MultiIndex.from_product([['India'], ['Confirmed', 'Deceased', 'Recovered', 'TotalConfirmed', 'TotalDeceased', 'TotalRecovered']], names=["Country", "Status"])
            data = data.astype(int)
            data.to_pickle(self.store_location)
        return data
