import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter, date2num
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
from common_tools import get_top_dir

SECONDS_PER_HOUR = 3600.
MINUTES_PER_DAY = 60.*24.
DAYS_PER_YEAR = 365.
DISTANCE_UNCERTAINTY = 2.5*np.sqrt(2)/1600.  # Distance measurement uncertainty (from https://community.geotab.com/s/article/How-does-the-GO-device-evaluate-coordinates?language=en_US)
KM_PER_MILE = 1.60934
DAYS_PER_MONTH = 30.437
METERS_PER_MILE = 1609.34
    
# Function to calculate time difference in minutes
def calculate_time_elapsed(row, start_time):
    time_difference_seconds = (row['timestamp'] - start_time).total_seconds()
    time_difference_minutes = time_difference_seconds / 60.
    return time_difference_minutes
    
top_dir = get_top_dir()

# Create the tables, plots and data directories if they don't already exist
if not os.path.exists(f'{top_dir}/data'):
    os.makedirs(f'{top_dir}/data')

if not os.path.exists(f'{top_dir}/tables'):
    os.makedirs(f'{top_dir}/tables')
    
if not os.path.exists(f'{top_dir}/plots'):
    os.makedirs(f'{top_dir}/plots')


# Collect filenames
files = [f'{top_dir}/data/pepsi_1_spd_dist_soc_cs_is_er.csv', f'{top_dir}/data/pepsi_2_spd_dist_cs_is_er.csv', f'{top_dir}/data/pepsi_3_spd_dist_cs_is_er.csv']

names = []
for file in files:
    name = file.split('/')[-1].split('_spd_dist')[0]
    names.append(name)


for file in files:
    name = file.split('/')[-1].split('_spd_dist')[0]
    names.append(name)
    data_df = pd.read_csv(file, low_memory=False)
    
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    
    # Remove events after the timestamp when the state of charge stops being measured
    data_with_soc = data_df[~data_df['socpercent'].isna()]
    
    max_timestamp_with_soc = data_with_soc['timestamp'].max()
    data_df = data_df[data_df['timestamp'] < max_timestamp_with_soc]
    
    # For Pepsi 1, remove events after the last charging event on Sept 16
    if name=='pepsi_1':
        lt_date = pd.Timestamp('2023-09-17 00:00:00', tz='UTC')
        data_df = data_df[data_df['timestamp'] < lt_date]
        data_charging_df = data_df[~data_df['energytype'].isna()]
        max_charging_timestamp = data_charging_df['timestamp'].max()
        data_df = data_df[data_df['timestamp'] < max_charging_timestamp]
    
    # Calculate accumulated distance
    data_df['accumulated_distance'] = data_df['distance'].cumsum()
    data_df.to_csv(f'{top_dir}/data/{name}_additional_cols.csv', index=False)

"""
################################### Collect and save high-level metadata ###################################
for name in names:
    data_df = pd.read_csv(f'{top_dir}/data/{name}_additional_cols.csv', low_memory=False)
    metadata_dict = {
        'total_time': [],
        'total_miles': [],
        'total_kWh_charged': []
        }
        
        # Convert timestamps to datetime
        data_df['timestamp'] = pd.to_datetime(data_driving_df['timestamp'])
        total_time =
        
    vmt_data_df = pd.DataFrame(vmt_data_dict)

############################################################################################################
"""
            
"""
######################################## Analysis of charging power ########################################
charging_powers = {}
for name in names:
    data_df = pd.read_csv(f'{top_dir}/data/{name}_additional_cols.csv', low_memory=False)
    data_charging_df = data_df[data_df['energytype'] == 'energy_from_dc_charger']
    data_charging_df['timestamp'] = pd.to_datetime(data_charging_df['timestamp'])

    # Calculate differences in accumulated energy and timestamps
    data_charging_df['accumulatedkwh_diffs'] = data_charging_df['accumumlatedkwh'] - data_charging_df['accumumlatedkwh'].shift(1)
    data_charging_df['timestamp_diffs_seconds'] = (data_charging_df['timestamp'] - data_charging_df['timestamp'].shift(1)).dt.total_seconds()
    
    # Remove any entries with timestamp differences greater than 40 seconds (they're typically 30 seconds)
    #print(f'Dataset size before removal of time gaps: {len(data_charging_df)}')
    data_charging_df = data_charging_df[data_charging_df['timestamp_diffs_seconds'] < 40]
    #print(f'Dataset size after removal of time gaps: {len(data_charging_df)}')
    #plt.plot(data_charging_df['timestamp_diffs_seconds'])
    #plt.show()
    #plt.close()
    
    # Convert timestamp differences to hours to align units
    data_charging_df['timestamp_diffs_hours'] = data_charging_df['timestamp_diffs_seconds'] / SECONDS_PER_HOUR
    data_charging_df['charging_power'] = data_charging_df['accumulatedkwh_diffs'] / data_charging_df['timestamp_diffs_hours']
    charging_powers[name] = data_charging_df['charging_power']


    # Plot the charging power as a function of timestamp
    fig, ax = plt.subplots(figsize=(18, 3))
    ax.set_title(name.replace('_', ' ').capitalize(), fontsize=20)
    plt.plot(data_charging_df['timestamp'], data_charging_df['charging_power'], 'o', markersize=1)
    plt.ylabel('Charging power (kW)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(f'{top_dir}/plots/{name}_chargingpower_vs_time.png')
    plt.close()
   
# Calculate and print out charging stat results for each truck
data = [charging_powers['pepsi_1'], charging_powers['pepsi_2'], charging_powers['pepsi_3']]
means = [np.mean(d) for d in data]
maxes = [np.max(d) for d in data]
mins = [np.min(d) for d in data]
upper_quantiles = [np.percentile(d, 75) for d in data]  # 75% quantile
lower_quantiles = [np.percentile(d, 25) for d in data]  # 25% quantile

for i in range(3):
    print(f'\n###### Charging results for {name} ######')
    print(f'Mean: {means[i]} kW')
    print(f'75% quantile: {upper_quantiles[i]} kW')
    print(f'25% quantile: {lower_quantiles[i]} kW')
    print(f'Max: {maxes[i]} kW')
    print(f'Min: {mins[i]} kW')
    print(f'#########################################')

# Plot the charging power stats as a box plot for each truck
fig, ax = plt.subplots(figsize=(10, 6))
plt.boxplot(data)
for i, mean in enumerate(means, start=1):
    plt.scatter(i, mean, color='red', marker='o', label='Mean' if i == 1 else "")
plt.xticks([1, 2, 3], [name.replace('_', ' ').capitalize() for name in names])
plt.ylabel('Charging power (kW)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize=16)
plt.savefig(f'{top_dir}/plots/chargingpower_stats.png')
plt.close()
    
############################################################################################################
"""

"""
#################################### Instantaneous electricity per mile ####################################

# Function to binned data with a step plot (don't like matplotlib's options)
def draw_binned_step(binned_data, linecolor='red', linelabel='', linewidth=2):
    previous_bin_mid = None
    previous_data_value = None
    for i, (bin_range, data) in enumerate(binned_data.items()):
        # Bin range as the x-coordinate (use the middle of the bin)
        bin_mid = (bin_range.right + bin_range.left) / 2
        
        # Plot mean
        plt.plot(bin_mid, data, color=linecolor, markersize=0, zorder=100)
        
        # Connect each mean to the next
        if i==0:
            plt.hlines(data, bin_range.left, bin_range.right, color=linecolor, linewidth=linewidth, label=linelabel, zorder=100)
        else:
            plt.hlines(data, bin_range.left, bin_range.right, color=linecolor, linewidth=linewidth, zorder=100)
        if previous_bin_mid is not None and previous_data_value is not None:
            plt.vlines(bin_range.left, previous_data_value, data, color=linecolor, linewidth=linewidth, zorder=100)

        previous_bin_mid = bin_mid
        previous_data_value = data

binned_e_per_d_driving_dict = {}
binned_e_per_d_driving_and_regen_dict = {}
for name in names:
    data_df = pd.read_csv(f'{top_dir}/data/{name}_additional_cols.csv', low_memory=False)
    
    # Collect data during both driving and regen
    data_driving_df = data_df[(data_df['energytype'] == 'driving_energy') & (data_df['distance'].notna()) & (data_df['distance'] != 0)]
    data_regen_df = data_df[(data_df['energytype'] == 'energy_regen') & (data_df['distance'].notna()) & (data_df['distance'] != 0)]
    
    # Convert string timestamps to python datetime format
    data_driving_df['timestamp'] = pd.to_datetime(data_driving_df['timestamp'])
    data_regen_df['timestamp'] = pd.to_datetime(data_regen_df['timestamp'])

    # Calculate differences in accumulated driving energy and distance (in miles)
    # Driving
    data_driving_df['accumulatedkwh_diffs'] = data_driving_df['accumumlatedkwh'] - data_driving_df['accumumlatedkwh'].shift(1)
    data_driving_df['accumulated_distance_diffs'] = data_driving_df['accumulated_distance'] - data_driving_df['accumulated_distance'].shift(1)
    data_driving_df['accumulated_distance_diffs_unc'] = DISTANCE_UNCERTAINTY
    data_driving_df['timestamp_diffs_seconds'] = (data_driving_df['timestamp'] - data_driving_df['timestamp'].shift(1)).dt.total_seconds()
    
    # Regen: need to take negative of accumulatedkWh as energy is being put back into the battery
    data_regen_df['accumulatedkwh_diffs'] = -1*(data_regen_df['accumumlatedkwh'] - data_regen_df['accumumlatedkwh'].shift(1))
    data_regen_df['accumulated_distance_diffs'] = data_regen_df['accumulated_distance'] - data_regen_df['accumulated_distance'].shift(1)
    data_regen_df['timestamp_diffs_seconds'] = DISTANCE_UNCERTAINTY
    data_regen_df['timestamp_diffs_seconds'] = (data_regen_df['timestamp'] - data_regen_df['timestamp'].shift(1)).dt.total_seconds()
    
    # Optionally, concatenate regen to the driving now that we've evaluated differences in energy and distance for each
    data_driving_with_regen_df = pd.concat([data_driving_df, data_regen_df])
    
    # Remove any entries with timestamp differences greater than 40 seconds (they're typically 30 seconds)
    #print(f'Dataset size before removal of time gaps: {len(data_driving_df)}')
    #data_driving_df = data_driving_df[data_driving_df['timestamp_diffs_seconds'] < 40]
    #print(f'Dataset size after removal of time gaps: {len(data_driving_df)}')
    #plt.plot(data_driving_df['timestamp_diffs_seconds'])
    #plt.show()
    #plt.close()
    
    # Calculate the driving energy per distance
    data_driving_df['driving_energy_per_distance'] = data_driving_df['accumulatedkwh_diffs'] / data_driving_df['accumulated_distance_diffs']
    data_driving_with_regen_df['driving_energy_per_distance'] = data_driving_with_regen_df['accumulatedkwh_diffs'] / data_driving_with_regen_df['accumulated_distance_diffs']
    data_regen_df['driving_energy_per_distance'] = data_regen_df['accumulatedkwh_diffs'] / data_regen_df['accumulated_distance_diffs']
    
    # Calculate the overall driving energy per distance
    e_per_d_driving_total = data_driving_df['accumulatedkwh_diffs'].sum() / data_driving_df['accumulated_distance_diffs'].sum()
    e_per_d_driving_and_regen_total = data_driving_with_regen_df['accumulatedkwh_diffs'].sum() / data_driving_with_regen_df['accumulated_distance_diffs'].sum()
    
#    print(data_driving_df[data_driving_df['driving_energy_per_distance'] < 0])
#    print(data_driving_df['driving_energy_per_distance'].mean())
#    print(data_driving_df['accumulatedkwh_diffs'][data_driving_df['speed'] > 60].sum() / data_driving_df['accumulated_distance_diffs'][data_driving_df['speed'] > 60].sum())

    # Calculate overall energy per distance within each 10 mph speed ban
    bins = np.linspace(0, 70, 8)    # Define the bins
    bin_centers = bins[:-1]+(bins[1]-bins[0])/2.
    data_driving_df['binned'] = pd.cut(data_driving_df['speed'], bins)   # Categorize data into speed bins
    data_driving_with_regen_df['binned'] = pd.cut(data_driving_with_regen_df['speed'], bins)
    data_driving_df.groupby('binned')['driving_energy_per_distance']
    data_driving_with_regen_df.groupby('binned')['driving_energy_per_distance']
    binned_e_per_d_driving = data_driving_df.groupby('binned')['accumulatedkwh_diffs'].sum() / data_driving_df.groupby('binned')['accumulated_distance_diffs'].sum()
    binned_e_per_d_driving_dict[name] = binned_e_per_d_driving
    binned_e_per_d_driving_and_regen = data_driving_with_regen_df.groupby('binned')['accumulatedkwh_diffs'].sum() / data_driving_with_regen_df.groupby('binned')['accumulated_distance_diffs'].sum()
    binned_e_per_d_driving_and_regen_dict[name] = binned_e_per_d_driving_and_regen
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(name.replace('_', ' ').capitalize(), fontsize=20)
    ax.set_xlabel('Speed (miles/hour)', fontsize=16)
    ax.set_ylabel('Driving energy per distance (kWh/mile)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.text(0.15, 0.15, 'Total over all bands (driving only): %.2f\nTotal over all bands (driving and regen): %.2f'%(e_per_d_driving_total, e_per_d_driving_and_regen_total), transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
    plt.hlines(0, linestyle='--', linewidth=1, color='black', xmin=0, xmax=70)
    
    # Plot the raw driving energy per distance in each time interval
    ax.plot(data_driving_df['speed'], data_driving_df['driving_energy_per_distance'], 'o', color='blue', markersize=1, label = 'Driving Energy')
    ax.plot(data_regen_df['speed'], data_regen_df['driving_energy_per_distance'], 'o', color='green', markersize=1, label = 'Regen Energy')

    
    # Plot the means within each 10 mph speed band
    draw_binned_step(binned_e_per_d_driving, linecolor='red', linelabel='Overall per speed band (driving only)')
    draw_binned_step(binned_e_per_d_driving_and_regen, linecolor='green', linelabel='Overall per speed band (driving and regen)')
    
    ax.legend(fontsize=14, loc='upper right')
    
    # Save figure (with original y lims and zoomed to help see the totals within each mph band)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/driving_energy_per_distance_{name}.png')
    plt.savefig(f'{top_dir}/plots/driving_energy_per_distance_{name}.pdf')
    ax.set_ylim(-5,10)
    plt.savefig(f'{top_dir}/plots/driving_energy_per_distance_{name}_zoom.png')
    plt.savefig(f'{top_dir}/plots/driving_energy_per_distance_{name}_zoom.pdf')
    
# Plot the driving energy per distance within speed bands for all trucks together

# Light red to dark red gradient
light_blue = "#CCCCFF"  # Lighter blue
medium_blue = "#6666FF" # Medium blue
dark_blue = "#0000CC"   # Darker blue
light_green = "#CCFFCC"  # Lighter green
medium_green = "#66FF66" # Medium green
dark_green = "#00CC00"   # Darker green

blue_gradient = [light_blue, medium_blue, dark_blue]
green_gradient = [light_green, medium_green, dark_green]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Speed (miles/hour)', fontsize=16)
ax.set_ylabel('Driving energy per distance (kWh/mile)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

i=0
for name in names:
    binned_e_per_d_driving = binned_e_per_d_driving_dict[name]
    draw_binned_step(binned_e_per_d_driving, linecolor=blue_gradient[i], linelabel='%s (driving only)'%name.replace('_', ' ').capitalize(), linewidth=1)
    i+=1
    
i=0
for name in names:
    binned_e_per_d_driving_and_regen = binned_e_per_d_driving_and_regen_dict[name]
    draw_binned_step(binned_e_per_d_driving_and_regen, linecolor=green_gradient[i], linelabel='%s (driving and regen)'%name.replace('_', ' ').capitalize(), linewidth=1)
    i+=1
    
# Save figure (with original y lims and zoomed to help see the totals within each mph band)
ax.legend(fontsize=14, loc='upper right')
plt.tight_layout()
plt.savefig(f'{top_dir}/plots/driving_energy_per_distance_all.png')
plt.savefig(f'{top_dir}/plots/driving_energy_per_distance_all.pdf')

############################################################################################################


############################ Analysis of actual driving range and battery energy ###########################

######### Add activity, driving and charging events to dataframes #########
data_df_dict = {}
for name in names:
    data_df = pd.read_csv(f'{top_dir}/data/{name}_additional_cols.csv', low_memory=False)
    
    # Establish which periods are driving vs. charging
    data_df['activity'] = data_df['energytype'].fillna(method='ffill')
    
    data_df['activity'][data_df['activity'] == 'energy_from_dc_charger'] = 'charging'
    data_df['activity'][data_df['activity'] == 'driving_energy'] = 'driving'
    data_df['activity'][data_df['activity'] == 'energy_regen'] = 'driving'
    data_df = data_df.dropna(subset=['activity'])
    
    data_df_dict[name] = data_df


# Add driving and charging event numbers to the dataframes
for name in names:
    data_df = data_df_dict[name]
    
    # Make a column label to distinguish each driving and charging event
    data_df['charging_event'] = np.nan
    data_df['driving_event'] = np.nan
    
    # Initialize variables to track charging events
    charging_event = 0
    driving_event = 0
    prev_activity = data_df.at[data_df.index[0], 'activity']
    prev_soc = -1.
    current_soc = 100

    # Iterate through the DataFrame and perform any needed row-by-row operations
    for index, row in data_df.iterrows():
    
        # Fill the charging_event column with an index representing how many charging events have occurred
        current_activity = row['activity']
        current_soc = row['socpercent']
        
        # Update the activity to driving if the soc has decreased relative to its previous value
        if current_activity == 'charging' and current_soc <= prev_soc:
            current_activity = np.nan
            data_df.at[index, 'activity'] = np.nan
            data_df.at[index, 'charging_event'] = np.nan
            current_activity = 'driving'
        
        # Increment the charging event number if the activity has changed from driving to charging
        if current_activity == 'charging':
            if prev_activity == 'driving' or not isinstance(prev_activity, str):
                charging_event += 1
                soc_diff = 0
            data_df.at[index, 'charging_event'] = charging_event
        
        # Increment the driving event number if the activity has changed from charging to driving, or if the SOC has changed by more than 5%
        if current_activity == 'driving':
            if prev_activity == 'charging' or prev_activity == np.nan:
                driving_event += 1
                soc_diff = 0
            data_df.at[index, 'driving_event'] = driving_event
            
        prev_activity = current_activity
        prev_soc = current_soc
        
        # In cases where there's one nan row in the accumumlatedkwh with floats above and below, interpolate by replacing the nan with the average of the above and below rows
        if pd.notna(row['accumumlatedkwh']):
            continue  # Skip non-NaN rows
        if index > 0 and index < len(data_df) - 1:
            prev_value = data_df.at[index - 1, 'accumumlatedkwh']
            next_value = data_df.at[index + 1, 'accumumlatedkwh']
            if pd.notna(prev_value) and pd.notna(next_value):
                average = (prev_value + next_value) / 2
                data_df.at[index, 'accumumlatedkwh'] = average
                
    data_df.to_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', index=False)
    
###########################################################################

"""

"""
######################### Energy capacity #########################
for name in names:
    battery_data_dict = {
        'charging_event': [],
        'battery_size': [],
        'battery_size_unc': []
        }
        
    battery_data_linear_df = pd.DataFrame(battery_data_dict)
    battery_data_quad_df = pd.DataFrame(battery_data_dict)
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)
            
    # Iterate through all the charging events and plot them
    n_charging_events = int(data_df['charging_event'].max())
    for charging_event in range(1, n_charging_events):
        # Make a cut to select only rows corresponding to the given charging event
        cChargingEvent = (data_df['charging_event'] == charging_event)
        
        # Only keep rows for which both the socpercent and accumumlatedkwh values are not nan
        data_df_event = data_df[cChargingEvent].dropna(subset=['socpercent', 'accumumlatedkwh'])
        
        # Only plot and analyze charging events with at least 10 datapoints
        if len(data_df_event) < 10:
            continue
            
        # Only consider the charging event for battery capacity estimation if >=50% of SOC has been recharged
        if data_df_event['socpercent'].max() - data_df_event['socpercent'].min() < 50:
            continue
        
        # Plot the raw soc vs. energy data
        fig, ax = plt.subplots(figsize=(10, 6))
        #ax.set_title(f"{name.replace('_', ' ').capitalize()}: Charging Event {charging_event}", fontsize=18)
        ax.set_xlabel('State of charge (%)', fontsize=24)
        ax.set_ylabel('Battery energy (kWh)', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.scatter(data_df_event['socpercent'], data_df_event['accumumlatedkwh'], s=50, color='black')
        
        # Evaluate a best fit line
        x_values = data_df_event['socpercent']
        y_values = data_df_event['accumumlatedkwh']
        
        ###### Linear Fit ######
        coefficients, covariance = np.polyfit(x_values, y_values, 1, cov=True)
        
        # Get the linear coefficients and uncertainty
        slope = coefficients[0]
        b = coefficients[1]
        slope_unc = np.sqrt(covariance[0, 0])
        b_unc = np.sqrt(covariance[1, 1])
        
        # Get the RMSE
        y_pred = np.polyval(coefficients, x_values)
        rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))

        # The extrapolated battery size for linear fit is 100% x slope
        battery_size = slope * 100
        battery_size_unc = slope_unc * 100
        
        new_row = pd.DataFrame({
            'charging_event': [charging_event],
            'battery_size': [battery_size],
            'battery_size_unc': [battery_size_unc]
        })
        
        battery_data_linear_df = pd.concat([battery_data_linear_df, new_row], ignore_index=True)
        
        best_fit_line = f"Best-fit Line \ny = mx + b \nm={slope:.3f}$\pm${slope_unc:.3f}\nRMSE: {rmse:.2f}"
        
        ax.text(0.33, 0.25, f'Extrapolated Battery Size: {battery_size:.1f}$\pm${battery_size_unc:.1f} kWh', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=20)
        
        # Plot the data and best fit line
        x_plot = np.linspace(0, 100, 1000)
        plt.plot(x_plot, slope * x_plot + b, color='red', label=best_fit_line, linewidth=3)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_linearfit.png')
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_linearfit.pdf')
        plt.close()
        ########################
        
        ##### Quadratic Fit ####
        fig, ax = plt.subplots(figsize=(10, 6))
        #ax.set_title(f"{name.replace('_', ' ').capitalize()}: Charging Event {charging_event}", fontsize=18)
        ax.set_xlabel('State of charge (%)', fontsize=24)
        ax.set_ylabel('Battery energy (kWh)', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.scatter(data_df_event['socpercent'], data_df_event['accumumlatedkwh'], s=50, color='black')
        
        coefficients, covariance = np.polyfit(x_values, y_values, 2, cov=True)
        
        # Get the quadratic coefficients and uncertainty
        a = coefficients[0]
        b = coefficients[1]
        c = coefficients[2]
        a_unc = np.sqrt(covariance[0, 0])
        b_unc = np.sqrt(covariance[1, 1])
        c_unc = np.sqrt(covariance[2, 2])
        
                # Get the RMSE
        y_pred = np.polyval(coefficients, x_values)
        rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))
        
        # The extrapolated battery size for quadratic fit is (100%)^2a + 100b
        battery_size = a*100**2 + b*100
        battery_size_unc = np.sqrt((a_unc*100**2)**2 + (b_unc*100)**2)
        
        new_row_quad = pd.DataFrame({
            'charging_event': [charging_event],
            'battery_size': [battery_size],
            'battery_size_unc': [battery_size_unc]
        })

        # Use concat to add the new row to the existing DataFrame
        battery_data_quad_df = pd.concat([battery_data_quad_df, new_row_quad], ignore_index=True)
        
        best_fit_line = f"Best-fit Quadratic \ny = ax$^2$ + bx + c \na={a:.4f}$\pm${a_unc:.4f}\nb: {b:.2f}$\pm${b_unc:.2f}\nRMSE: {rmse:.2f}"
        
        ax.text(0.33, 0.25, f'Extrapolated Battery Size: {battery_size:.1f}$\pm${battery_size_unc:.1f} kWh', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=20)
        
        # Plot the data and best fit line
        x_plot = np.linspace(0, 100, 1000)
        plt.plot(x_plot, a * x_plot**2 + b * x_plot + c, color='red', label=best_fit_line, linewidth=3)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_quadfit.png')
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_quadfit.pdf')
        plt.close()
        
        ########################

    battery_data_linear_df.to_csv(f'{top_dir}/tables/{name}_battery_data_linearfit.csv', index=False)
    battery_data_quad_df.to_csv(f'{top_dir}/tables/{name}_battery_data_quadfit.csv', index=False)


# Calculate the weighted average of all estimates for each truck, along with standard deviation
battery_capacity_save = pd.DataFrame({'Value': ['Mean', 'Standard Deviation']})
battery_capacities = np.zeros(0)
for name in names:
    battery_data_linearfit_df = pd.read_csv(f'{top_dir}/tables/{name}_battery_data_linearfit.csv')
    battery_data_quadfit_df = pd.read_csv(f'{top_dir}/tables/{name}_battery_data_quadfit.csv')
    
    # Calculated the weighted average and std among all battery capacity estimates
    weighted_mean_linearfit = np.average(battery_data_linearfit_df['battery_size'], weights=1./battery_data_linearfit_df['battery_size_unc']**2)
    weighted_mean_quadfit = np.average(battery_data_quadfit_df['battery_size'], weights=1./battery_data_quadfit_df['battery_size_unc']**2)
    
    weighted_std_linearfit = np.sqrt(np.average((battery_data_linearfit_df['battery_size']-weighted_mean_linearfit)**2, weights=1./battery_data_linearfit_df['battery_size_unc']**2))
    weighted_std_quadfit = np.sqrt(np.average((battery_data_quadfit_df['battery_size']-weighted_mean_quadfit)**2, weights=1./battery_data_quadfit_df['battery_size_unc']**2))
    
    # Plot all the battery capacity estimates together, along with the weighted mean and std
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.set_title(f"{name.replace('_', ' ').capitalize()}: Summary of Battery Capacity Estimates", fontsize=18)
    ax.set_xlabel('Charging Event', fontsize=24)
    ax.set_ylabel('Battery Capacity (kWh)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.xticks(range(len(battery_data_linearfit_df['charging_event'])), labels=battery_data_linearfit_df['charging_event'].astype(int))
    
    
   # plt.errorbar(range(len(battery_data_linearfit_df['charging_event'])), battery_data_linearfit_df['battery_size'], yerr = battery_data_linearfit_df['battery_size_unc'], capsize=5, label='Linear Fit', marker='o', linestyle='none', color='green')
    #ax.axhline(weighted_mean_linearfit, color='green', linewidth=2, label=f'Weighted mean (linear): {weighted_mean_linearfit:.1f}$\pm${weighted_std_linearfit:.1f}')
    #ax.fill_between(np.linspace(xmin, xmax, 100), weighted_mean_linearfit-weighted_std_linearfit, weighted_mean_linearfit+weighted_std_linearfit, color='green', alpha=0.2, edgecolor='none')
    
    #plt.errorbar(range(len(battery_data_quadfit_df['charging_event'])), battery_data_quadfit_df['battery_size'], yerr = battery_data_quadfit_df['battery_size_unc'], capsize=5, label='Quadratic Fit', marker='o', linestyle='none', color='blue')
    plt.errorbar(range(len(battery_data_quadfit_df['charging_event'])), battery_data_quadfit_df['battery_size'], yerr = battery_data_quadfit_df['battery_size_unc'], capsize=5, marker='o', linestyle='none', color='black', label = 'Extrapolated capacity')
    xmin, xmax = ax.get_xlim()
    ax.axhline(weighted_mean_quadfit, color='blue', linewidth=2, label=f'Weighted mean: {weighted_mean_quadfit:.1f}$\pm${weighted_std_quadfit:.1f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), weighted_mean_quadfit-weighted_std_quadfit, weighted_mean_quadfit+weighted_std_quadfit, color='blue', alpha=0.2, edgecolor='none')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_battery_capacity_summary.png', dpi=300)
    plt.savefig(f'{top_dir}/plots/{name}_battery_capacity_summary.pdf')
    
    battery_capacities = np.append(battery_capacities, weighted_mean_quadfit)
    battery_capacity_save[name] = [weighted_mean_quadfit, weighted_std_quadfit]
    
battery_capacity_save['average'] = [np.mean(battery_capacities), np.std(battery_capacities)]
battery_capacity_save.to_csv('tables/pepsi_semi_battery_capacities.csv')

###################################################################
"""


################ Charging Time and Depth of Discharge #############
for name in names:
    charging_dict = {
        'charging_event': [],
        'min_soc': [],
        'max_soc': [],
        'DoD': [],
        'charging_time': [],
        }
        
    charging_df = pd.DataFrame(charging_dict)
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)

    n_charging_events = int(data_df['charging_event'].max())
    for charging_event in range(1, n_charging_events):
        # Make a cut to select only rows corresponding to the given charging event
        cChargingEvent = (data_df['charging_event'] == charging_event)
        
        # Only keep rows for which both the socpercent and timestamp values are not nan
        data_df_event = data_df[cChargingEvent].dropna(subset=['socpercent', 'timestamp'])
        
        # Convert timestamp format to python datetime
        data_df_event['timestamp'] = pd.to_datetime(data_df_event['timestamp'])
        
        # Only consider the charging event for battery capacity estimation if >=1% of SOC has been recharged
        min_soc = data_df_event['socpercent'].min()
        max_soc = data_df_event['socpercent'].max()
        dod = max_soc - min_soc
        if dod < 1 or np.isnan(dod):
            continue
        
        # Calculate the total charging time (in minutes)
        start_time = data_df_event['timestamp'].iloc[0]
        charging_time = calculate_time_elapsed(data_df_event.iloc[-1], start_time)
        charging_df = charging_df.append({'charging_event': charging_event, 'min_soc': min_soc, 'max_soc': max_soc, 'DoD': dod, 'charging_time': charging_time}, ignore_index=True)
        
        # Plot the raw soc vs. time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Charging Event {charging_event}", fontsize=18)
        ax.set_ylabel('State of Charge (%)', fontsize=18)
        ax.set_xlabel('Charging time elapsed (minutes)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.text(0.35, 0.15, f'Charging Time: {charging_time:.1f} minutes\nDepth of Discharge: {dod:.1f}%\nMinimum Battery Charge: {min_soc:.1f}%', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        start_time = data_df_event['timestamp'].iloc[0]
        charging_time_elapsed = data_df_event.apply(calculate_time_elapsed, axis=1, args=(start_time,))
        ax.scatter(charging_time_elapsed, data_df_event['socpercent'])
        plt.savefig(f'{top_dir}/plots/{name}_time_vs_battery_soc_event_{charging_event}.png')
        plt.savefig(f'{top_dir}/plots/{name}_time_vs_battery_soc_event_{charging_event}.pdf')
        plt.close()
        
    charging_df.to_csv(f'{top_dir}/tables/{name}_charging_time_data.csv', index=False)
    
# Calculate the average and standard deviation among all charging times and depth of discharges
for name in names:
    charging_df = pd.read_csv(f'{top_dir}/tables/{name}_charging_time_data.csv')
    
    # Calculated the average and std among all charging times and depths of discharge
    mean_charging_time = np.average(charging_df['charging_time'])
    min_charging_time = np.min(charging_df['charging_time'])
    max_charging_time = np.max(charging_df['charging_time'])
    std_charging_time = np.std(charging_df['charging_time'])
    
    mean_dod = np.average(charging_df['DoD'])
    min_dod = np.min(charging_df['DoD'])
    max_dod = np.max(charging_df['DoD'])
    std_dod = np.std(charging_df['DoD'])
    
    # Plot the DoDs and charging times together
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title(f"{name.replace('_', ' ').capitalize()}: Summary of Charging Parameters", fontsize=18)
    ax.set_xlabel('Minimum Depth of Discharge (%)', fontsize=18)
    ax.set_ylabel('Charging Time (minutes)', fontsize=18)
    ax.tick_params(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.scatter(charging_df['DoD'], charging_df['charging_time'], color='green')
    
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax-xmin))
    xmin, xmax = ax.get_xlim()
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
    ymin, ymax = ax.get_ylim()
    
    ax.axhline(mean_charging_time, color='green', linewidth=2, label=f'Mean charge time: {mean_charging_time:.1f}$\pm${std_charging_time:.1f} mins\nMin: {min_charging_time:.1f} minutes\nMax: {max_charging_time:.1f} minutes\n')
    ax.fill_between(np.linspace(xmin, xmax, 100), mean_charging_time-std_charging_time, mean_charging_time+std_charging_time, color='green', alpha=0.2, edgecolor='none')

    ax.axvline(mean_dod, color='blue', linewidth=2, label=f'Mean DoD: {mean_dod:.1f}%$\pm${std_dod:.1f}%\nMin: {min_dod:.1f}%\nMax: {max_dod:.1f}%')
    ax.fill_betweenx(np.linspace(ymin, ymax, 100), mean_dod-std_dod, mean_dod+std_dod, color='blue', alpha=0.2, edgecolor='none')
    
    ax.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_charging_summary.png')
    plt.savefig(f'{top_dir}/plots/{name}_charging_summary.pdf')
        
###################################################################


################ Drivecycle and extrapolated range ################

"""
# Read in saved csv file with battery capacities
battery_capacity_df = pd.read_csv('tables/pepsi_semi_battery_capacities.csv')
for name in names:
    drivecycle_data_dict = {
        'Driving event': [],
        'Initial battery charge (%)': [],
        'Final battery charge (%)': [],
        'Depth of Discharge (%)': [],
        'Range (miles)': [],
        'Range unc (miles)': [],
        'Fuel economy (kWh/mile)': [],
        'Fuel economy unc (kWh/mile)': [],
        'RMSE': [],
        }
        
    drivecycle_data_df = pd.DataFrame(drivecycle_data_dict)
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)
    
    # Read in the battery capacity for the given truck
    battery_capacity = battery_capacity_df[name].iloc[0]
    battery_capacity_unc = battery_capacity_df[name].iloc[1]
    
    # Iterate through all the driving events and plot them
    n_driving_events = int(data_df['driving_event'].max())
    for driving_event in range(1, n_driving_events):
        # Make a cut to select only rows corresponding to the given driving event
        cDrivingEvent = (data_df['driving_event'] == driving_event)
        
        # Only keep rows for which both the socpercent and accumumlatedkwh values are not nan
        data_df_event = data_df[cDrivingEvent].dropna(subset=['socpercent', 'accumulated_distance'])
        
        # Only plot and analyze driving events with at least 10 datapoints
        if len(data_df_event) < 10:
            continue
            
        # Only consider the driving event for driving range if >=50% of SOC has been discharged
        if data_df_event['socpercent'].max() - data_df_event['socpercent'].min() < 50:
            continue
            
#        if name=='pepsi_1' and driving_event==16:
#            for index, row in data_df_event.iterrows():
#                print(row)
            
        # Plot the raw soc vs. distance data
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
        #axs[0].set_title(f"{name.replace('_', ' ').capitalize()}: Driving Event {driving_event}", fontsize=18)
        axs[1].set_xlabel('State of charge (%)', fontsize=24)
        axs[0].set_ylabel('Distance Traveled (miles)', fontsize=24)
        axs[1].set_ylabel('Speed (mph)', fontsize=24)
        axs[0].tick_params(axis='both', which='major', labelsize=20)
        axs[1].tick_params(axis='both', which='major', labelsize=20)
        
        # Add major/minor ticks and gridlines
        axs[0].xaxis.set_major_locator(MultipleLocator(20))
        axs[0].xaxis.set_minor_locator(MultipleLocator(5))
        axs[0].grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
        axs[0].grid(which='major', linestyle='-', linewidth=0.5, color='black')
        axs[1].xaxis.set_major_locator(MultipleLocator(20))
        axs[1].xaxis.set_minor_locator(MultipleLocator(5))
        axs[1].grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
        axs[1].grid(which='major', linestyle='-', linewidth=0.5, color='black')
        
        axs[0].scatter(data_df_event['socpercent'], data_df_event['accumulated_distance'], color='black', s=50)
        axs[1].plot(data_df_event['socpercent'], data_df_event['speed'])
            
        # Evaluate a best fit line
        x_values = data_df_event['socpercent']
        y_values = data_df_event['accumulated_distance']
        
        coefficients, covariance = np.polyfit(x_values, y_values, 1, cov=True)
        
        # Get the linear coefficients and uncertainty
        slope = coefficients[0]
        b = coefficients[1]
        slope_unc = np.sqrt(covariance[0, 0])
        b_unc = np.sqrt(covariance[1, 1])
        
        # Get the RMSE
        y_pred = np.polyval(coefficients, x_values)
        rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))
        
        # The extrapolated range for linear fit is 100% x slope
        truck_range = -1 * slope * 100
        truck_range_unc = slope_unc * 100
        
        # Evaluate the depth of discharge
        battery_charge_init = data_df_event['socpercent'].max()
        battery_charge_final = data_df_event['socpercent'].min()
        dod = battery_charge_init - battery_charge_final
    
        # Evaluate the change in battery energy over the drivecycle
        delta_battery_energy = dod * battery_capacity / 100.
        delta_battery_energy_unc = dod * battery_capacity_unc / 100.
        
        # Evaluate the total distance traveled over the drivecycle
        distance_traveled = data_df_event['accumulated_distance'].max() - data_df_event['accumulated_distance'].min()
        
        # Evaluate the extrapolated fuel economy (kWh/mile)
        fuel_economy = delta_battery_energy / distance_traveled
        fuel_economy_unc = delta_battery_energy_unc / distance_traveled
        
        new_row = {
            'Driving event': int(driving_event),
            'Initial battery charge (%)': battery_charge_init,
            'Final battery charge (%)': battery_charge_final,
            'Depth of Discharge (%)': dod,
            'Range (miles)': truck_range,
            'Range unc (miles)': truck_range_unc,
            'Fuel economy (kWh/mile)': fuel_economy,
            'Fuel economy unc (kWh/mile)': fuel_economy_unc,
            'RMSE': rmse
        }
        
        new_row_df = pd.DataFrame([new_row])
        
        drivecycle_data_df = pd.concat([drivecycle_data_df, new_row_df], ignore_index=True)
                
        best_fit_line = f"Best-fit Line \ny = mx + b \nm={slope:.3f}$\pm${slope_unc:.3f}\nRMSE: {rmse:.2f}"

        
        # Plot the data and best fit line
        x_plot = np.linspace(0, 100, 1000)
        axs[0].plot(x_plot, slope * x_plot + b, color='red', label=best_fit_line, linewidth=3)
        axs[0].legend(fontsize=20)
        
        #axs[0].text(0.33, 0.25, f'Extrapolated Range: {truck_range:.1f}$\pm${truck_range_unc:.1f} kWh\nExtrapolated fuel economy: {fuel_economy:.2f}$\pm${fuel_economy_unc:.2f}', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=20)
        fig.text(0.15, 0.45, f'Range: {truck_range:.1f}$\pm${truck_range_unc:.1f} kWh\nEnergy Economy: {fuel_economy:.2f}$\pm${fuel_economy_unc:.2f} kWh/mile', fontsize=20, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7))
        
        xmin, xmax = axs[0].get_xlim()
        axs[1].set_xlim(xmin, xmax)
        
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.png', dpi=300)
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.pdf')
        plt.close()

    drivecycle_data_df['Driving event'] = drivecycle_data_df['Driving event'].astype('int')
    drivecycle_data_df.to_csv(f'tables/{name}_drivecycle_data.csv', index=False)
"""

# Plot the distribution of RMSE
RMSE_cutoff = 10
all_rmse = np.zeros(0)
for name in names:
    drivecycle_data_df = pd.read_csv(f'tables/{name}_drivecycle_data.csv')
    all_rmse = np.append(all_rmse, np.array(drivecycle_data_df['RMSE']))
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xlabel('RMSE', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.hist(all_rmse, bins=50)
ax.axvline(RMSE_cutoff, linestyle='--', linewidth=3, color='red', label='RMSE Cutoff')
ax.legend(fontsize=20)
plt.tight_layout()
plt.savefig(f'{top_dir}/plots/all_RMSE.png', dpi=300)
plt.savefig(f'{top_dir}/plots/all_RMSE.pdf')

"""
# Plot range estimate summaries
for name in names:
    drivecycle_data_df = pd.read_csv(f'tables/{name}_drivecycle_data.csv')
    
    # Separate the linear from non-linear drivecycles according to RMSE
    drivecycle_data_linear = drivecycle_data_df[drivecycle_data_df['RMSE'] < RMSE_cutoff]
    drivecycle_data_nonlinear = drivecycle_data_df[drivecycle_data_df['RMSE'] > RMSE_cutoff]
    
    # Calculate the average and std among all range estimates
    mean_linear = np.average(drivecycle_data_linear['Range (miles)'])
    mean_nonlinear = np.average(drivecycle_data_nonlinear['Range (miles)'])
    
    std_linear = np.std(drivecycle_data_linear['Range (miles)'])
    std_nonlinear = np.std(drivecycle_data_nonlinear['Range (miles)'])
    
    # Plot all the range estimates together, along with their means and stds
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.set_title(f"{name.replace('_', ' ').capitalize()}: Summary of Range Estimates", fontsize=24)
    #ax.set_xlabel('Driving Event', fontsize=18)
    ax.set_ylabel('Extrapolated Range (miles)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks([])
    
    # Plot the ranges evaluated for individual drivecycles, separated between linear and non-linear
    plt.errorbar(range(len(drivecycle_data_linear['Driving event'])), drivecycle_data_linear['Range (miles)'], yerr = drivecycle_data_linear['Range unc (miles)'], capsize=5, marker='o', linestyle='none', color='blue', label='Extrapolated range (linear)')
    
    plt.errorbar(range(len(drivecycle_data_linear['Driving event']), len(drivecycle_data_linear['Driving event'])+len(drivecycle_data_nonlinear['Driving event'])), drivecycle_data_nonlinear['Range (miles)'], yerr = drivecycle_data_nonlinear['Range unc (miles)'], capsize=5, marker='o', linestyle='none', color='green', label='Extrapolated range (nonlinear)')
    xmin, xmax = ax.get_xlim()
    
    # Plot the mean and standard deviation for both linear and non-linear drivecycles
    ax.axhline(mean_linear, color='blue', linewidth=2, label=f'Mean (linear driveycles): {mean_linear:.1f}$\pm${std_linear:.1f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), mean_linear-std_linear, mean_linear+std_linear, color='blue', alpha=0.2, edgecolor='none')
    
    ax.axhline(mean_nonlinear, color='green', linewidth=2, label=f'Mean (nonlinear driveycles): {mean_nonlinear:.1f}$\pm${std_nonlinear:.1f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), mean_nonlinear-std_nonlinear, mean_nonlinear+std_nonlinear, color='green', alpha=0.2, edgecolor='none')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_range_summary.png')
    plt.savefig(f'{top_dir}/plots/{name}_range_summary.pdf')
"""
# Plot fuel economy estimate summaries
for name in names:
    drivecycle_data_df = pd.read_csv(f'tables/{name}_drivecycle_data.csv')
    
    # Separate the linear from non-linear drivecycles according to RMSE
    drivecycle_data_linear = drivecycle_data_df[drivecycle_data_df['RMSE'] < RMSE_cutoff]
    drivecycle_data_nonlinear = drivecycle_data_df[drivecycle_data_df['RMSE'] > RMSE_cutoff]
    
    # Calculate the average and std among all range estimates
    mean_linear = np.average(drivecycle_data_linear['Fuel economy (kWh/mile)'])
    mean_nonlinear = np.average(drivecycle_data_nonlinear['Fuel economy (kWh/mile)'])
    
    std_linear = np.std(drivecycle_data_linear['Fuel economy (kWh/mile)'])
    std_nonlinear = np.std(drivecycle_data_nonlinear['Fuel economy (kWh/mile)'])
    
    # Plot all the range estimates together, along with their means and stds
    fig, ax = plt.subplots(figsize=(10, 6))
    #ax.set_xlabel('Driving Event', fontsize=18)
    ax.set_ylabel('Extrapolated Energy Economy (kWh/mile)', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xticks([])
    
    # Plot the ranges evaluated for individual drivecycles, separated between linear and non-linear
    plt.errorbar(range(len(drivecycle_data_linear['Driving event'])), drivecycle_data_linear['Fuel economy (kWh/mile)'], yerr = drivecycle_data_linear['Fuel economy unc (kWh/mile)'], capsize=5, marker='o', linestyle='none', color='blue', label='Extrapolated energy economy (linear)')
    
    plt.errorbar(range(len(drivecycle_data_linear['Driving event']), len(drivecycle_data_linear['Driving event'])+len(drivecycle_data_nonlinear['Driving event'])), drivecycle_data_nonlinear['Fuel economy (kWh/mile)'], yerr = drivecycle_data_nonlinear['Fuel economy unc (kWh/mile)'], capsize=5, marker='o', linestyle='none', color='green', label='Extrapolated energy economy (nonlinear)')
    xmin, xmax = ax.get_xlim()
    
    # Plot the mean and standard deviation for both linear and non-linear drivecycles
    ax.axhline(mean_linear, color='blue', linewidth=2, label=f'Mean (linear driveycles): {mean_linear:.2f}$\pm${std_linear:.2f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), mean_linear-std_linear, mean_linear+std_linear, color='blue', alpha=0.2, edgecolor='none')
    
    ax.axhline(mean_nonlinear, color='green', linewidth=2, label=f'Mean (nonlinear driveycles): {mean_nonlinear:.2f}$\pm${std_nonlinear:.2f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), mean_nonlinear-std_nonlinear, mean_nonlinear+std_nonlinear, color='green', alpha=0.2, edgecolor='none')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_fuel_economy_summary.png')
    plt.savefig(f'{top_dir}/plots/{name}_fuel_economy_summary.pdf')
    
###################################################################


"""
########################## Drive Cycle ##########################
for name in names:
        
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)
        
    # Iterate through all the driving events and plot them
    n_driving_events = int(data_df['driving_event'].max())
    for driving_event in range(1, n_driving_events):
        # Make a cut to select only rows corresponding to the given driving event
        cDrivingEvent = (data_df['driving_event'] == driving_event)
        
        # Only keep rows for which both the socpercent and accumumlatedkwh values are not nan
        data_df_event = data_df[cDrivingEvent].dropna(subset=['timestamp', 'speed'])
        
        # Only plot and analyze charging events with at least 10 datapoints
        if len(data_df_event) < 10:
            continue
        
        # Convert timestamp format to python datetime
        data_df_event['timestamp'] = pd.to_datetime(data_df_event['timestamp'])
        
        # Add a column with the time elapsed (in minutes)
        start_time = data_df_event['timestamp'].iloc[0]
        data_df_event['time_elapsed'] = data_df_event.apply(calculate_time_elapsed, axis=1, args=(start_time,))
            
        # Only consider the driving event where the drive time is >5 minutes
        total_drive_time = data_df_event['time_elapsed'].max()
        if data_df_event['time_elapsed'].max() < 5:
            continue
            
        # Evaluate the depth of discharge
        battery_charge_init = data_df_event['socpercent'].max()
        battery_charge_final = data_df_event['socpercent'].min()
        dod = battery_charge_init - battery_charge_final
        
        # Plot the speed vs. driving time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Driving Event {driving_event}", fontsize=18)
        ax.set_ylabel('Speed (mph)', fontsize=18)
        ax.set_xlabel('Driving time (minutes)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.plot(data_df_event['time_elapsed'], data_df_event['speed'])
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.6)
        ymin, ymax = ax.get_ylim()
        total_drive_time_hours = total_drive_time / 60.
        
        plt.text(0.45, 0.65, f'Total drive time: {total_drive_time_hours:.1f} hours\nInitial Battery Charge: {battery_charge_init:.1f}%\nFinal Battery Charge: {battery_charge_final:.1f}%\nDepth of Discharge: {dod:.1f}%', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_{driving_event}.png')
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_{driving_event}.pdf')
        plt.close()
        
        # Plot the speed vs. driving time in a way that's comparable with the long-haul drivecycle
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Driving Event {driving_event}", fontsize=18)

        # Add major/minor ticks and gridlines
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax.grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
        ax.grid(which='major', linestyle='-', linewidth=0.5, color='black')

        ax.set_ylabel('Speed (mph)', fontsize=18)
        ax.set_xlabel('Driving time (minutes)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.plot(data_df_event['time_elapsed'], data_df_event['speed'])
        
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_{driving_event}_paper.png')
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_{driving_event}_paper.pdf')
        plt.close()
        
        # Also plot the speed vs. state of charge to validate with best fit events
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Driving Event {driving_event}", fontsize=18)
        ax.set_ylabel('Speed (mph)', fontsize=18)
        ax.set_xlabel('State of charge (%)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        data_df_event_plot = data_df_event.dropna(subset=['socpercent'])
        ax.plot(data_df_event_plot['socpercent'], data_df_event_plot['speed'])
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.6)
        ymin, ymax = ax.get_ylim()
        
        battery_charge_init = data_df_event['socpercent'].max()
        battery_charge_final = data_df_event['socpercent'].min()
        dod = battery_charge_init - battery_charge_final
        
        plt.text(0.45, 0.65, f'Total drive time: {total_drive_time_hours:.1f} hours\nInitial Battery Charge: {battery_charge_init:.1f}%\nFinal Battery Charge: {battery_charge_final:.1f}%\nDepth of Discharge: {dod:.1f}%', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_soc_{driving_event}.png')
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_soc_{driving_event}.pdf')
        plt.close()
        
        # Extract the drive cycle (time and speed columns)
        drive_cycle_df = data_df_event.filter(['time_elapsed','speed'], axis=1)
        
        # Convert the time elapsed from  minutes to seconds
        drive_cycle_df['time_elapsed'] = drive_cycle_df['time_elapsed'] * 60
        
        # Convert the speed to km/h
        drive_cycle_df['speed'] = drive_cycle_df['speed'] * KM_PER_MILE
        
        # Add a column of zeros as a stand-in for the road grade (which unfortunately wasn't included in the NACFE data)
        drive_cycle_df['road_grade'] = drive_cycle_df['speed'] * 0
        
        # Save to a csv file
        drive_cycle_df.to_csv(f'{top_dir}/tables/{name}_drive_cycle_{driving_event}.csv', header=['Time (s)', 'Vehicle speed (km/h)', 'Road grade (%)'], index=False)
        
        # Add in the accumulated distance since the start of the drivecycle
        drive_cycle_df['accumulated_distance'] = (data_df_event['accumulated_distance'] - data_df_event['accumulated_distance'].iloc[0]) * METERS_PER_MILE
        
        # Save to a csv file
        drive_cycle_df.to_csv(f'{top_dir}/tables/{name}_drive_cycle_{driving_event}_with_distance.csv', header=['Time (s)', 'Vehicle speed (km/h)', 'Road grade (%)', 'Cumulative distance (m)'], index=False)
        
#################################################################
"""
"""
######################## Extrapolated VMT #######################
for name in names:
    vmt_data_dict = {
        'miles_driven': [],
        'total_time': [],
        'extrapolated_vmt': []
        }
        
    vmt_data_df = pd.DataFrame(vmt_data_dict)
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)

    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])  # Convert to datetime
    
    # Evaluate the timestamp in hours since start
    min_datetime = data_df['timestamp'].min()
    data_df['timestamp_hours'] = data_df['timestamp'].apply(
        lambda x: float((x - min_datetime).total_seconds() / SECONDS_PER_HOUR) if pd.notna(x) else np.nan
    )
    
    # Drop any hours timestamps with nans
    data_df = data_df[data_df['timestamp_hours'].notna()]
    data_df = data_df[~np.isinf(data_df['timestamp_hours'])]
    
    data_df_reduced = data_df.iloc[::100]
        
    # Plot the charging power as a function of timestamp
    fig, axs = plt.subplots(2, 1, figsize=(18, 7), gridspec_kw={'height_ratios': [1, 1]})  # 2 rows, 1 column
    #axs[0].set_title(name.replace('_', ' ').capitalize(), fontsize=20)
    axs[0].plot(data_df_reduced.dropna(subset=['speed'])['timestamp_hours'], data_df_reduced.dropna(subset=['speed'])['speed'])
    axs[1].plot(data_df.dropna(subset=['socpercent', 'timestamp_hours'])['timestamp_hours'], data_df.dropna(subset=['socpercent', 'timestamp_hours'])['socpercent'])
    
    cCharging = (data_df['activity'] == 'charging')
    
    data_df['charging_timestamp'] = data_df['timestamp_hours']
    data_df.loc[~cCharging, 'charging_timestamp'] = np.nan
    
    ymin, ymax = axs[0].get_ylim()
    axs[0].fill_between(data_df['charging_timestamp'], ymin, ymax, color='green', alpha=0.2, edgecolor='none')
    axs[0].set_ylim(ymin, ymax)
    
    ymin, ymax = axs[1].get_ylim()
    axs[1].fill_between(data_df['charging_timestamp'], ymin, ymax, color='green', alpha=0.2, edgecolor='none', label='Charging')
    axs[1].set_ylim(ymin, ymax)
    
    cDriving = (data_df['activity'] == 'driving')
    
    data_df['driving_timestamp'] = data_df['timestamp_hours']
    data_df.loc[~cDriving, 'driving_timestamp'] = np.nan
    
    ymin, ymax = axs[0].get_ylim()
    axs[0].fill_between(data_df['driving_timestamp'], ymin, ymax, color='purple', alpha=0.2, edgecolor='none')
    axs[0].set_ylim(ymin, ymax)
    
    ymin, ymax = axs[1].get_ylim()
    axs[1].fill_between(data_df['driving_timestamp'], ymin, ymax, color='purple', alpha=0.2, edgecolor='none', label='Driving')
    axs[1].set_ylim(ymin, ymax)
    
    axs[0].set_ylabel('Speed (mph)', fontsize=22)
    axs[1].set_ylabel('State of Charge (%)', fontsize=22)
    axs[1].set_xlabel('Time Elapsed (h)', fontsize=22)
    
    axs[0].set_xlim(data_df['timestamp_hours'].min(), data_df['timestamp_hours'].max())
    axs[1].set_xlim(data_df['timestamp_hours'].min(), data_df['timestamp_hours'].max())
    
    axs[0].xaxis.set_tick_params(labelbottom=False)
    axs[0].tick_params(axis='both', which='major', labelsize=20)
    axs[0].xaxis.set_major_locator(MaxNLocator(10))
    
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    axs[1].xaxis.set_major_locator(MaxNLocator(10))
    
    axs[1].legend(fontsize=22, bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_speed_vs_time.png', dpi=300)
    plt.close()
    
    # Plot accumulated miles traveled over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{name.replace('_', ' ').capitalize()}: Extrapolation of Annual VMT", fontsize=18)
    ax.set_ylabel('Accumulated distance (miles)', fontsize=18)
    ax.set_xlabel('Accumulated data collection time (days)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    start_time = data_df['timestamp'].iloc[0]
    collection_time_elapsed = data_df.apply(calculate_time_elapsed, axis=1, args=(start_time,)) / MINUTES_PER_DAY
    total_distance = data_df['accumulated_distance'].max()
    total_data_collection_time = collection_time_elapsed.max()
    extrapolated_vmt = (total_distance / total_data_collection_time) * DAYS_PER_YEAR
    vmt_data_df['miles_driven'] = total_distance
    vmt_data_df['total_time'] = total_data_collection_time
    vmt_data_df['extrapolated_vmt'] = extrapolated_vmt
    
    ax.plot(collection_time_elapsed, data_df['accumulated_distance'], 'o', markersize=1)
    plt.text(0.35, 0.15, f'Total Distance: {total_distance:.1f} miles\nTotal Data Collection Time: {total_data_collection_time:.2f} days\nExtrapolated VMT: {extrapolated_vmt:.0f} miles/year', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_distance_vs_time.png')
    plt.close()
    
    # Save the vmt data to a csv file
    vmt_data_df.to_csv(f'{top_dir}/tables/{name}_vmt_data.csv', header=['Distance traveled (miles)', 'Total time (days)', 'Extrapolated Annual VMT (miles/year)'], index=False)
        
#################################################################
"""

############################################################################################################
"""
################################### Extrapolated Energy Delivered per Month ################################
for name in names:
    energy_data_dict = {
        'energy_delivered': [],
        'total_time': [],
        'extrapolated_energy_per_month': []
        }
        
    energy_data_df = pd.DataFrame(energy_data_dict)
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df_charging = data_df[data_df['energytype'] == 'energy_from_dc_charger']
        
    # Plot accumulated charging energy delivered over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{name.replace('_', ' ').capitalize()}: Extrapolation of Energy Delivered by Chargers per Month", fontsize=18)
    ax.set_ylabel('Accumulated electricity delivered (MWh)', fontsize=18)
    ax.set_xlabel('Accumulated data collection time (days)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    start_time = data_df['timestamp'].iloc[0]
    collection_time_elapsed = data_df.apply(calculate_time_elapsed, axis=1, args=(start_time,)) / MINUTES_PER_DAY
    total_energy_delivered = data_df_charging['accumumlatedkwh'].max() / 1e3        # Convert from kWh to MWh
    total_data_collection_time = collection_time_elapsed.max()
    extrapolated_energy_per_month = (total_energy_delivered / total_data_collection_time) * DAYS_PER_MONTH
    energy_data_df['energy_delivered'] = total_energy_delivered
    energy_data_df['total_time'] = total_data_collection_time
    energy_data_df['extrapolated_energy_per_month'] = extrapolated_energy_per_month
    
    ax.plot(collection_time_elapsed[data_df['energytype'] == 'energy_from_dc_charger'], data_df_charging['accumumlatedkwh'] / 1e3, 'o', markersize=1)
    plt.text(0.35, 0.15, f'Total Distance: {total_distance:.1f} miles\nTotal Data Collection Time: {total_data_collection_time:.2f} days\nExtrapolated Energy/Month: {extrapolated_energy_per_month:.1f} MWh/month', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_charging_energy_vs_time.png')
    plt.close()
    
    # Save the vmt data to a csv file
    energy_data_df.to_csv(f'{top_dir}/tables/{name}_energy_per_month_data.csv', header=['Energy Delivered (kWh)', 'Total time (days)', 'Extrapolated Energy Delivered/Month (kWh/month)'], index=False)

############################################################################################################
"""
