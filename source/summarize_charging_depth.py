
"""
Date: 240926
Author: danikam
Purpose: Summarize charging depth data for all charging sessions in the NACFE data
"""

from common_tools import get_top_dir
import pandas as pd
MINUTES_PER_HOUR = 60

top_dir = get_top_dir()

# List of truck names
truck_names = ['pepsi_1', 'pepsi_2', 'pepsi_3']


# Read and combine all dataframes into a single dataframe, adding the truck name as a column
combined_df = pd.concat(
    [pd.read_csv(f"{top_dir}/tables/{truck_name}_charging_time_data.csv").assign(truck_name=truck_name) for truck_name in truck_names],
    ignore_index=True
)

######## Calculate the average charging power and C value based on the size of the battery ########
bat_capac_df = pd.read_csv("tables/pepsi_semi_battery_capacities.csv")

# Extract the row where Value is 'Mean' and drop the 'Value' and 'average' columns
mean_capacities = bat_capac_df[bat_capac_df['Value'] == 'Mean'].drop(['Value', 'average'], axis=1)

# Transpose the dataframe so that truck names are in the index and we have one column for Mean
mean_capacities = mean_capacities.transpose()
mean_capacities.columns = ['mean_battery_capacity']

# Merge this with the combined_df
combined_df = pd.merge(combined_df, mean_capacities, how='left', left_on='truck_name', right_index=True)

# Add a column with the average charging power
combined_df["mean_charging_power"] = combined_df["mean_battery_capacity"] * ( combined_df["DoD"] / 100) / (combined_df["charging_time"] / MINUTES_PER_HOUR)

# Add a column with the average C-value (average charging power divided by battery capacity)
combined_df["C_mean"] = combined_df["mean_charging_power"] / combined_df["mean_battery_capacity"]

# Calculate stats of interest

# Minimum state of charge
soc_min_av = combined_df["min_soc"].mean()
soc_min_std = combined_df["min_soc"].std()
print(f"Minimum state of charge: {soc_min_av} ± {soc_min_std}")

soc_max_av = combined_df["max_soc"].mean()
soc_max_std = combined_df["max_soc"].std()
print(f"Maximum state of charge: {soc_max_av} ± {soc_max_std}")

dod_av = combined_df["DoD"].mean()
dod_std = combined_df["DoD"].std()
print(f"Depth of discharge: {dod_av} ± {dod_std}")

P_av = combined_df["mean_charging_power"].mean()
P_std = combined_df["mean_charging_power"].std()
print(f"Mean charging power: {P_av} ± {P_std}")

C_av = combined_df["C_mean"].mean()
C_std = combined_df["C_mean"].std()
print(f"Mean C-value: {C_av} ± {C_std}")

###################################################################################################
