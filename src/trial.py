from src.dataset.dataset import Dataset

# Get daily data for a state in India
kl_time_series = Dataset("India:KL")
kl_confirmed = kl_time_series.get_confirmed()
print(kl_confirmed)

# Get daily data for a country
india_time_series = Dataset("India")
india_confirmed = india_time_series.get_confirmed()
print(india_confirmed)

# Get all states dataframe
all_states = Dataset("India:AllStates")
states_df = all_states.get_confirmed()
print(states_df)