import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
from pathlib import Path

SECONDS_PER_HOUR = 3600
DISTANCE_UNCERTAINTY = 2.5*np.sqrt(2)/1600  # Distance measurement uncertainty (from https://community.geotab.com/s/article/How-does-the-GO-device-evaluate-coordinates?language=en_US)
KM_PER_MILE = 1.60934

def get_top_dir():
    '''
    Gets the path to the top level of the git repo (one level up from the source directory)
        
    Parameters
    ----------
    None

    Returns
    -------
    top_dir (string): Path to the top level of the git repo
    '''
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    top_dir = os.path.dirname(source_dir)
    return top_dir
    
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

# Read CSV files into Pandas DataFrames
files = [f'{top_dir}/data/pepsi_1_spd_dist_soc_cs_is_er.csv', f'{top_dir}/data/pepsi_2_spd_dist_cs_is_er.csv', f'{top_dir}/data/pepsi_3_spd_dist_cs_is_er.csv']
data_df_dict = {}
names = []
for file in files:
    name = file.split('/')[-1].split('_spd_dist')[0]
    data_df_dict[name] = pd.read_csv(file, low_memory=False)
    names.append(name)
    
    # Calculate accumulated distance
    data_df_dict[name]['accumulated_distance'] = data_df_dict[name]['distance'].cumsum()
    data_df_dict[name].to_csv(f'{top_dir}/data/{name}_additional_cols.csv', index=False)
            

######################################## Analysis of charging power ########################################
charging_powers = {}
for name in names:
    data_df = data_df_dict[name]
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


    # Plot the charging power as a function of timestamp, both overall and on a daily basis
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
##################################### Analysis of electricity per mile #####################################

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
    data_df = data_df_dict[name]
    
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
        if current_soc < prev_soc:
            current_activity = 'driving'
            row['activity'] = 'driving'
            data_df.at[index, 'charging_event'] = np.nan
        
        # Increment the charging event number if the activity has changed from driving to charging
        if current_activity == 'charging':
            if prev_activity == 'driving':
                charging_event += 1
                soc_diff = 0
            data_df.at[index, 'charging_event'] = charging_event
        
        # Increment the driving event number if the activity has changed from charging to driving, or if the SOC has changed by more than 5%
        if current_activity == 'driving':
            if prev_activity == 'charging':
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
                
    data_df_dict[name] = data_df
    data_df_dict[name].to_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', index=False)
    
###########################################################################



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
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Charging Event {charging_event}", fontsize=18)
        ax.set_xlabel('State of charge (%)', fontsize=16)
        ax.set_ylabel('Battery energy (kWh)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.scatter(data_df_event['socpercent'], data_df_event['accumumlatedkwh'])
        
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

        # The extrapolated battery size for linear fit is 100% x slope
        battery_size = slope * 100
        battery_size_unc = slope_unc * 100
        
        battery_data_linear_df = battery_data_linear_df.append({'charging_event': charging_event, 'battery_size': battery_size, 'battery_size_unc': battery_size_unc}, ignore_index=True)
        
        best_fit_line = f"Best-fit Line \ny = mx + b \nm={slope:.3f}$\pm${slope_unc:.3f}\nExtrapolated Battery Size: {battery_size:.1f}$\pm${battery_size_unc:.1f} kWh"
        
        # Plot the data and best fit line
        x_plot = np.linspace(0, 100, 1000)
        plt.plot(x_plot, slope * x_plot + b, color='red', label=best_fit_line)
        plt.xlabel('State of Charge (%)')
        plt.ylabel('Accumulated Battery Energy (kWh)')
        plt.legend(fontsize=14)
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_linearfit.png')
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_linearfit.pdf')
        plt.close()
        ########################
        
        ##### Quadratic Fit ####
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Charging Event {charging_event}", fontsize=18)
        ax.set_xlabel('State of charge (%)', fontsize=16)
        ax.set_ylabel('Battery energy (kWh)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.scatter(data_df_event['socpercent'], data_df_event['accumumlatedkwh'])
        
        coefficients, covariance = np.polyfit(x_values, y_values, 2, cov=True)
        
        # Get the quadratic coefficients and uncertainty
        a = coefficients[0]
        b = coefficients[1]
        c = coefficients[2]
        a_unc = np.sqrt(covariance[0, 0])
        b_unc = np.sqrt(covariance[1, 1])
        c_unc = np.sqrt(covariance[2, 2])
        
        # The extrapolated battery size for quadratic fit is (100%)^2a + 100b
        battery_size = a*100**2 + b*100
        battery_size_unc = np.sqrt((a_unc*100**2)**2 + (b_unc*100)**2)
        
        battery_data_quad_df = battery_data_quad_df.append({'charging_event': charging_event, 'battery_size': battery_size, 'battery_size_unc': battery_size_unc}, ignore_index=True)
        
        best_fit_line = f"Best-fit Quadratic \ny = ax$^2$ + bx + c \na={a:.4f}$\pm${a_unc:.4f}\nb: {b:.2f}$\pm${b_unc:.2f}\nExtrapolated Battery Capacity: {battery_size:.1f}$\pm${battery_size_unc:.1f} kWh"
        
        # Plot the data and best fit line
        x_plot = np.linspace(0, 100, 1000)
        plt.plot(x_plot, a * x_plot**2 + b * x_plot + c, color='red', label=best_fit_line)
        plt.xlabel('State of Charge (%)')
        plt.ylabel('Accumulated Battery Energy (kWh)')
        plt.legend(fontsize=14)
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_quadfit.png')
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_energy_event_{charging_event}_quadfit.pdf')
        plt.close()
        
        ########################

    battery_data_linear_df.to_csv(f'{top_dir}/tables/{name}_battery_data_linearfit.csv', index=False)
    battery_data_quad_df.to_csv(f'{top_dir}/tables/{name}_battery_data_quadfit.csv', index=False)


# Calculate the weighted average of all estimates for each truck, along with standard deviation
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
    ax.set_title(f"{name.replace('_', ' ').capitalize()}: Summary of Battery Capacity Estimates", fontsize=18)
    ax.set_xlabel('Charging Event', fontsize=18)
    ax.set_ylabel('Extrapolated Battery Capacity (kWh)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(range(len(battery_data_linearfit_df['charging_event'])), labels=battery_data_linearfit_df['charging_event'].astype(int))
    
    
    plt.errorbar(range(len(battery_data_linearfit_df['charging_event'])), battery_data_linearfit_df['battery_size'], yerr = battery_data_linearfit_df['battery_size_unc'], capsize=5, label='Linear Fit', marker='o', linestyle='none', color='green')
    xmin, xmax = ax.get_xlim()
    ax.axhline(weighted_mean_linearfit, color='green', linewidth=2, label=f'Weighted mean (linear): {weighted_mean_linearfit:.1f}$\pm${weighted_std_linearfit:.1f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), weighted_mean_linearfit-weighted_std_linearfit, weighted_mean_linearfit+weighted_std_linearfit, color='green', alpha=0.2, edgecolor='none')
    
    plt.errorbar(range(len(battery_data_quadfit_df['charging_event'])), battery_data_quadfit_df['battery_size'], yerr = battery_data_quadfit_df['battery_size_unc'], capsize=5, label='Quadratic Fit', marker='o', linestyle='none', color='blue')
    ax.axhline(weighted_mean_quadfit, color='blue', linewidth=2, label=f'Weighted mean (quad): {weighted_mean_quadfit:.1f}$\pm${weighted_std_quadfit:.1f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), weighted_mean_quadfit-weighted_std_quadfit, weighted_mean_quadfit+weighted_std_quadfit, color='blue', alpha=0.2, edgecolor='none')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_battery_capacity_summary.png')
    plt.savefig(f'{top_dir}/plots/{name}_battery_capacity_summary.pdf')
###################################################################



################ Charging Time and Depth of Discharge #############
for name in names:
    charging_dict = {
        'charging_event': [],
        'min_soc': [],
        'max_soc': [],
        'delta_soc': [],
        'charging_time': [],
        }
        
    charging_df = pd.DataFrame(charging_dict)
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)

    n_charging_events = int(data_df['charging_event'].max())
    for charging_event in range(1, n_charging_events):
        # Make a cut to select only rows corresponding to the given charging event
        cChargingEvent = (data_df['charging_event'] == charging_event)
        
        # Only keep rows for which both the socpercent and accumumlatedkwh values are not nan
        data_df_event = data_df[cChargingEvent].dropna(subset=['socpercent', 'timestamp'])
        
        # Convert timestamp format to python datetime
        data_df_event['timestamp'] = pd.to_datetime(data_df_event['timestamp'])
        
        # Only consider the charging event for battery capacity estimation if >=1% of SOC has been recharged
        min_soc = data_df_event['socpercent'].min()
        max_soc = data_df_event['socpercent'].max()
        delta_soc = max_soc - min_soc
        if delta_soc < 1 or np.isnan(delta_soc):
            continue
        
        # Calculate the total charging time (in minutes)
        start_time = data_df_event['timestamp'].iloc[0]
        charging_time = calculate_time_elapsed(data_df_event.iloc[-1], start_time)
        charging_df = charging_df.append({'charging_event': charging_event, 'min_soc': min_soc, 'max_soc': max_soc, 'delta_soc': delta_soc, 'charging_time': charging_time}, ignore_index=True)
        
        # Plot the raw soc vs. time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Charging Event {charging_event}", fontsize=18)
        ax.set_ylabel('State of Charge (%)', fontsize=18)
        ax.set_xlabel('Charging time elapsed (minutes)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.text(0.35, 0.15, f'Charging Time: {charging_time:.1f} minutes\nChange in Battery Charge: {delta_soc:.1f}%\nMinimum depth of discharge: {min_soc:.1f}%', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
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
    
    mean_dod = np.average(charging_df['min_soc'])
    min_dod = np.min(charging_df['min_soc'])
    max_dod = np.max(charging_df['min_soc'])
    std_dod = np.std(charging_df['min_soc'])
    
    # Plot the DoDs and charging times together
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title(f"{name.replace('_', ' ').capitalize()}: Summary of Charging Parameters", fontsize=18)
    ax.set_xlabel('Minimum Depth of Discharge (%)', fontsize=18)
    ax.set_ylabel('Charging Time (minutes)', fontsize=18)
    ax.tick_params(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.scatter(charging_df['min_soc'], charging_df['charging_time'], color='green')
    
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax-xmin)*0.5)
    xmin, xmax = ax.get_xlim()
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
    ymin, ymax = ax.get_ylim()
    
    ax.axhline(mean_charging_time, color='green', linewidth=2, label=f'Mean charging time: {mean_charging_time:.1f}$\pm${std_charging_time:.1f} minutes\nMin: {min_charging_time:.1f} minutes\nMax: {max_charging_time:.1f} minutes\n')
    ax.fill_between(np.linspace(xmin, xmax, 100), mean_charging_time-std_charging_time, mean_charging_time+std_charging_time, color='green', alpha=0.2, edgecolor='none')

    ax.axvline(mean_dod, color='blue', linewidth=2, label=f'Mean depth of discharge: {mean_dod:.1f}%$\pm${std_dod:.1f}%\nMin: {min_dod:.1f}%\nMax: {max_dod:.1f}%')
    ax.fill_betweenx(np.linspace(ymin, ymax, 100), mean_dod-std_dod, mean_dod+std_dod, color='blue', alpha=0.2, edgecolor='none')
    
    ax.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_charging_summary.png')
    plt.savefig(f'{top_dir}/plots/{name}_charging_summary.pdf')
        
###################################################################



########################## Driving Range ##########################

for name in names:
    range_data_dict = {
        'driving_event': [],
        'range': [],
        'range_unc': []
        }
        
    range_data_df = pd.DataFrame(range_data_dict)
    data_df = pd.read_csv(f'{top_dir}/data/{name}_with_driving_charging.csv', low_memory=False)
        
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
            
        # Only consider the driving event for battery capacity estimation if >=50% of SOC has been recharged
        if data_df_event['socpercent'].max() - data_df_event['socpercent'].min() < 50:
            continue
            
#        if name=='pepsi_1' and driving_event==16:
#            for index, row in data_df_event.iterrows():
#                print(row)
            
        # Plot the raw soc vs. distance data
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})  # 2 rows, 1 column
        axs[0].set_title(f"{name.replace('_', ' ').capitalize()}: Driving Event {driving_event}", fontsize=18)
        axs[1].set_xlabel('State of charge (%)', fontsize=16)
        axs[0].set_ylabel('Distance Traveled (miles)', fontsize=16)
        axs[1].set_ylabel('Speed (miles/hour)', fontsize=16)
        axs[0].tick_params(axis='both', which='major', labelsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        
        # Add major/minor ticks and gridlines
        axs[0].xaxis.set_major_locator(MultipleLocator(20))
        axs[0].xaxis.set_minor_locator(MultipleLocator(5))
        axs[0].grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
        axs[0].grid(which='major', linestyle='-', linewidth=0.5, color='black')
        axs[1].xaxis.set_major_locator(MultipleLocator(20))
        axs[1].xaxis.set_minor_locator(MultipleLocator(5))
        axs[1].grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
        axs[1].grid(which='major', linestyle='-', linewidth=0.5, color='black')
        
        axs[0].scatter(data_df_event['socpercent'], data_df_event['accumulated_distance'])
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
        
        # The extrapolated range for linear fit is 100% x slope
        truck_range = -1 * slope * 100
        truck_range_unc = slope_unc * 100
        
        range_data_df = range_data_df.append({'driving_event': driving_event, 'range': truck_range, 'range_unc': truck_range_unc}, ignore_index=True)
        
        best_fit_line = f"Best-fit Line \ny = mx + b \nm={slope:.3f}$\pm${slope_unc:.3f}\nExtrapolated Range: {truck_range:.1f}$\pm${truck_range_unc:.1f} miles"
        
        # Plot the data and best fit line
        x_plot = np.linspace(0, 100, 1000)
        axs[0].plot(x_plot, slope * x_plot + b, color='red', label=best_fit_line)
        axs[0].legend(fontsize=14)
        
        xmin, xmax = axs[0].get_xlim()
        axs[1].set_xlim(xmin, xmax)
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.png')
        plt.savefig(f'{top_dir}/plots/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.pdf')
        plt.close()

    range_data_df.to_csv(f'tables/{name}_range_data_linearfit.csv', index=False)

# Calculate the average of all estimates for each truck, along with standard deviation
for name in names:
    range_data_linearfit_df = pd.read_csv(f'tables/{name}_range_data_linearfit.csv')
    
    # Calculated the average and std among all range estimates
    mean_linearfit = np.average(range_data_linearfit_df['range'])
    
    std_linearfit = np.std(range_data_linearfit_df['range'])
    
    # Plot all the range estimates together, along with the mean and std
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{name.replace('_', ' ').capitalize()}: Summary of Range Estimates", fontsize=18)
    ax.set_xlabel('Driving Event', fontsize=18)
    ax.set_ylabel('Extrapolated Range (miles)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(range(len(range_data_linearfit_df['driving_event'])), labels=range_data_linearfit_df['driving_event'].astype(int))
    
    
    plt.errorbar(range(len(range_data_linearfit_df['driving_event'])), range_data_linearfit_df['range'], yerr = range_data_linearfit_df['range_unc'], capsize=5, marker='o', linestyle='none', color='green')
    xmin, xmax = ax.get_xlim()
    ax.axhline(mean_linearfit, color='green', linewidth=2, label=f'Mean: {mean_linearfit:.1f}$\pm${std_linearfit:.1f}')
    ax.fill_between(np.linspace(xmin, xmax, 100), mean_linearfit-std_linearfit, mean_linearfit+std_linearfit, color='green', alpha=0.2, edgecolor='none')
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
    ax.set_xlim(xmin, xmax)
    ax.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{top_dir}/plots/{name}_range_summary.png')
    plt.savefig(f'{top_dir}/plots/{name}_range_summary.pdf')
    
###################################################################


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
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Driving Event {driving_event}", fontsize=18)
        ax.set_ylabel('Speed (mph)', fontsize=18)
        ax.set_xlabel('Driving time (minutes)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.plot(data_df_event['time_elapsed'], data_df_event['speed'])
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
        ymin, ymax = ax.get_ylim()
        total_drive_time_hours = total_drive_time / 60.
        plt.text(0.45, 0.75, f'Total drive time: {total_drive_time_hours:.1f} hours', transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_{driving_event}.png')
        plt.savefig(f'{top_dir}/plots/{name}_drive_cycle_{driving_event}.pdf')
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
        drive_cycle_df.to_csv(f'{top_dir}/tables/{name}_drive_cycle_{driving_event}.csv', header=['Time (s)', 'Vehicle speed (km/h)', 'Road Grade (%)'], index=False)

#################################################################

############################################################################################################



