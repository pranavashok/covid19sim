from src.dataset.covid19india_state_wise_loader import Covid19IndiaStatewiseLoader
from src.dataset.covid19india_national_loader import Covid19IndiaNationalLoader


class Dataset:
    def __init__(self, label):
        self.data = None
        if ":" in label:
            country, state = label.split(":")
            if country == "India":
                self.loader = Covid19IndiaStatewiseLoader(state)
        else:
            if label == "India":
                self.loader = Covid19IndiaNationalLoader()
        self.data = self.loader.load()

    def get_confirmed(self):
        return dict(self.data['Confirmed'])

    def get_recovered(self):
        return dict(self.data['Recovered'])

    def get_deceased(self):
        return dict(self.data['Deceased'])

    def get_active(self):
        return dict(self.data['Confirmed'] - self.data['Recovered'] - self.data['Deceased'])