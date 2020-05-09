from src.dataset.covid19india_state_wise_loader import Covid19IndiaStatewiseLoader
from src.dataset.covid19india_national_loader import Covid19IndiaNationalLoader


class Dataset:
    def __init__(self, label):
        self.data = None
        if ":" in label:
            self.country, self.state_code = label.split(":")
            if self.country == "India":
                self.loader = Covid19IndiaStatewiseLoader()
        else:
            if label == "India":
                self.loader = Covid19IndiaNationalLoader()
                self.country = label
                self.state_code = None
        data = self.loader.load()
        if self.state_code in data.columns:
            self.data = data.xs(self.state_code, axis=1, level=0, drop_level=False)
        else:
            self.data = data

    def get_confirmed(self):
        return self.data.xs("Confirmed", axis=1, level=1)

    def get_recovered(self):
        return self.data.xs("Recovered", axis=1, level=1)

    def get_deceased(self):
        return self.data.xs("Deceased", axis=1, level=1)

    def get_active(self):
        return dict(self.data['Confirmed'] - self.data['Recovered'] - self.data['Deceased'])

    def get_dataframe(self):
        return self.data