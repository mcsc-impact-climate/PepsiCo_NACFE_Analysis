"""
PepsiCo NACFE Data Analysis Script
Refactored with modular functions for easier development and iteration.

DATASET_TYPE: Choose 'pepsi' or 'messy_middle' to analyze either dataset
  - 'pepsi': Analyzes Pepsi 1/2/3 trucks (data/)
  - 'messy_middle': Analyzes joyride/4gen/nevoya (data_messy_middle/)

Toggle each analysis section by commenting/uncommenting in the main() function.
Set FAST_MODE = True for quick iteration (PNG only), False for PNG + PDF output.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
from matplotlib.dates import DateFormatter, date2num
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
import numpy as np
import os
from common_tools import get_top_dir

# ============================================================================
# CONFIGURATION
# ============================================================================
FAST_MODE = True  # Set to False to generate PDFs and zoom plots (slower)
DATASET_TYPE = 'messy_middle'  # 'pepsi' or 'messy_middle'

SECONDS_PER_HOUR = 3600.
MINUTES_PER_DAY = 60.*24.
DAYS_PER_YEAR = 365.
DISTANCE_UNCERTAINTY = 2.5*np.sqrt(2)/1600.
KM_PER_MILE = 1.60934
KM_TO_MILES = 0.621371
DAYS_PER_MONTH = 30.437
METERS_PER_MILE = 1609.34
RMSE_cutoff = 10

# Data caching to avoid re-reading CSV files
_csv_cache = {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def read_csv_cached(filepath, low_memory=False):
    """Read CSV file with caching to avoid redundant reads."""
    if filepath not in _csv_cache:
        _csv_cache[filepath] = pd.read_csv(filepath, low_memory=low_memory)
    return _csv_cache[filepath].copy()

def calculate_time_elapsed(row, start_time):
    """Calculate time difference in minutes from a start time."""
    time_difference_seconds = (row['timestamp'] - start_time).total_seconds()
    return time_difference_seconds / 60.

def setup_directories(top_dir):
    """Create necessary directories for data, tables, and plots."""
    if DATASET_TYPE == 'pepsi':
        directories = ['data', 'tables', 'plots']
    else:  # messy_middle
        directories = ['data_messy_middle', 'tables_messy', 'plots_messy']
    
    for directory in directories:
        dirpath = f'{top_dir}/{directory}'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            print(f"Created directory: {dirpath}")

def get_file_list(top_dir):
    """Get list of input CSV files and truck names based on dataset type."""
    if DATASET_TYPE == 'pepsi':
        files = [
            f'{top_dir}/data/pepsi_1_spd_dist_soc_cs_is_er.csv',
            f'{top_dir}/data/pepsi_2_spd_dist_cs_is_er.csv',
            f'{top_dir}/data/pepsi_3_spd_dist_cs_is_er.csv'
        ]
        names = [file.split('/')[-1].split('_spd_dist')[0] for file in files]
    else:  # messy_middle
        files = [
            f'{top_dir}/data_messy_middle/joyride.csv',
            f'{top_dir}/data_messy_middle/4gen.csv',
            f'{top_dir}/data_messy_middle/nevoya.csv'
        ]
        names = ['joyride', '4gen', 'nevoya']
    
    return files, names

def get_output_dir(top_dir, subdir):
    """Get output directory based on dataset type."""
    if DATASET_TYPE == 'pepsi':
        return f'{top_dir}/{subdir}'
    else:  # messy_middle
        if subdir == 'tables':
            return f'{top_dir}/tables_messy'
        elif subdir == 'plots':
            return f'{top_dir}/plots_messy'
        else:
            return f'{top_dir}/data_messy_middle'

def draw_binned_step(binned_data, linecolor='red', linelabel='', linewidth=2):
    """Helper function to plot binned data with a step plot."""
    previous_bin_mid = None
    previous_data_value = None
    for i, (bin_range, data) in enumerate(binned_data.items()):
        bin_mid = (bin_range.right + bin_range.left) / 2
        plt.plot(bin_mid, data, color=linecolor, markersize=0, zorder=100)
        
        if i==0:
            plt.hlines(data, bin_range.left, bin_range.right, color=linecolor, 
                      linewidth=linewidth, label=linelabel, zorder=100)
        else:
            plt.hlines(data, bin_range.left, bin_range.right, color=linecolor, 
                      linewidth=linewidth, zorder=100)
        
        if previous_bin_mid is not None and previous_data_value is not None:
            plt.vlines(bin_range.left, previous_data_value, data, color=linecolor, 
                      linewidth=linewidth, zorder=100)

        previous_bin_mid = bin_mid
        previous_data_value = data

def get_column_name(df, possible_names):
    """Find the first available column from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def normalize_data(df):
    """Normalize data from either Pepsi or messy middle format to a common structure."""
    df = df.copy()
    
    if DATASET_TYPE == 'messy_middle':
        # Convert distance from KM to miles
        distance_col = get_column_name(df, ['distance_km', 'distance'])
        if distance_col:
            df['distance'] = df[distance_col] * KM_TO_MILES
        
        # Map battery percentage column
        soc_col = get_column_name(df, ['battery_percent', 'socpercent', 'soc'])
        if soc_col and soc_col != 'socpercent':
            df['socpercent'] = df[soc_col]
        
        # For messy middle, energy_from_charge_kwh is already accumulated within each charging event
        # Just rename it - don't cumsum it again
        if 'accumumlatedkwh' not in df.columns:
            if 'energy_from_charge_kwh' in df.columns:
                df['accumumlatedkwh'] = df['energy_from_charge_kwh'].fillna(0)
            else:
                df['accumumlatedkwh'] = 0.0
        
        # Map activity to energytype-like structure
        if 'truck_activity' in df.columns:
            df['energytype'] = df['truck_activity']
        
        # Normalize speed (convert from kmh to mph)
        speed_col = get_column_name(df, ['speed_kmh', 'speed_mph', 'speed', 'avg_speed'])
        if speed_col:
            if 'kmh' in speed_col.lower() or 'km' in speed_col.lower():
                df['speed'] = df[speed_col] * KM_TO_MILES
            else:
                df['speed'] = df[speed_col]
    
    return df

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def preprocess_data(top_dir, files, names):
    """Preprocess raw CSV files: convert timestamps, filter data, calculate accumulated distance."""
    print("\n" + "="*70)
    print("PREPROCESSING DATA")
    print("="*70)
    
    for file, name in zip(files, names):
        print(f"Processing {name}...")
        data_df = read_csv_cached(file, low_memory=False)
        data_df = normalize_data(data_df)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        
        if DATASET_TYPE == 'pepsi':
            # Remove events after SOC measurement stops
            data_with_soc = data_df[~data_df['socpercent'].isna()]
            max_timestamp_with_soc = data_with_soc['timestamp'].max()
            data_df = data_df[data_df['timestamp'] < max_timestamp_with_soc]
            
            # For Pepsi 1, remove events after last charging on Sept 16
            if name == 'pepsi_1':
                lt_date = pd.Timestamp('2023-09-17 00:00:00', tz='UTC')
                data_df = data_df[data_df['timestamp'] < lt_date]
                data_charging_df = data_df[~data_df['energytype'].isna()]
                max_charging_timestamp = data_charging_df['timestamp'].max()
                data_df = data_df[data_df['timestamp'] < max_charging_timestamp]
        else:  # messy_middle
            # Remove rows with no SOC data
            data_with_soc = data_df[~data_df['socpercent'].isna()]
            if len(data_with_soc) > 0:
                max_timestamp_with_soc = data_with_soc['timestamp'].max()
                data_df = data_df[data_df['timestamp'] <= max_timestamp_with_soc]
        
        # Calculate accumulated distance
        data_df['accumulated_distance'] = data_df['distance'].fillna(0).cumsum()
        
        output_dir = get_output_dir(top_dir, 'data')
        data_df.to_csv(f'{output_dir}/{name}_additional_cols.csv', index=False)
        print(f"  ✓ Saved {name}_additional_cols.csv")

def analyze_charging_power(top_dir, names):
    """Analyze charging power statistics and create visualizations."""
    print("\n" + "="*70)
    print("ANALYZING CHARGING POWER")
    print("="*70)
    
    charging_powers = {}
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    
    for name in names:
        print(f"Processing {name}...")
        data_df = read_csv_cached(f'{output_dir}/{name}_additional_cols.csv', low_memory=False)
        data_df = normalize_data(data_df)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        
        if DATASET_TYPE == 'pepsi':
            data_charging_df = data_df[data_df['energytype'] == 'energy_from_dc_charger'].copy()
        else:  # messy_middle
            data_charging_df = data_df[data_df['truck_activity'] == 'charging'].copy()

        # Calculate power differences
        data_charging_df['accumulatedkwh_diffs'] = data_charging_df['accumumlatedkwh'] - data_charging_df['accumumlatedkwh'].shift(1)
        data_charging_df['timestamp_diffs_seconds'] = (data_charging_df['timestamp'] - data_charging_df['timestamp'].shift(1)).dt.total_seconds()
        data_charging_df = data_charging_df[data_charging_df['timestamp_diffs_seconds'] < 40]
        
        data_charging_df['timestamp_diffs_hours'] = data_charging_df['timestamp_diffs_seconds'] / SECONDS_PER_HOUR
        data_charging_df['charging_power'] = data_charging_df['accumulatedkwh_diffs'] / data_charging_df['timestamp_diffs_hours']
        charging_powers[name] = data_charging_df['charging_power']

        # Plot charging power vs time
        fig, ax = plt.subplots(figsize=(18, 3))
        ax.set_title(name.replace('_', ' ').capitalize(), fontsize=20)
        plt.plot(data_charging_df['timestamp'], data_charging_df['charging_power'], 'o', markersize=1)
        plt.ylabel('Charging power (kW)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig(f'{plot_dir}/{name}_chargingpower_vs_time.png')
        plt.close()
        print(f"  ✓ Saved charging power plot for {name}")
   
    # Calculate statistics
    data = [charging_powers[name] for name in names]
    means = [np.mean(d) for d in data]
    maxes = [np.max(d) for d in data]
    mins = [np.min(d) for d in data]
    upper_quantiles = [np.percentile(d, 75) for d in data]
    lower_quantiles = [np.percentile(d, 25) for d in data]

    print("\nCharging Power Statistics:")
    for i, name in enumerate(names):
        print(f"\n{name}:")
        print(f"  Mean: {means[i]:.1f} kW")
        print(f"  75% quantile: {upper_quantiles[i]:.1f} kW")
        print(f"  25% quantile: {lower_quantiles[i]:.1f} kW")

    # Plot box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.boxplot(data)
    for i, mean in enumerate(means, start=1):
        plt.scatter(i, mean, color='red', marker='o', label='Mean' if i == 1 else "")
    plt.xticks([1, 2, 3], [name.replace('_', ' ').capitalize() for name in names])
    plt.ylabel('Charging power (kW)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f'{plot_dir}/chargingpower_stats.png')
    plt.close()
    print("\n  ✓ Saved charging power statistics plot")

def analyze_instantaneous_energy(top_dir, names):
    """Analyze instantaneous energy consumption per mile."""
    print("\n" + "="*70)
    print("ANALYZING INSTANTANEOUS ENERGY PER MILE")
    print("="*70)
    
    binned_e_per_d_driving_dict = {}
    binned_e_per_d_driving_and_regen_dict = {}
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    
    for name in names:
        print(f"Processing {name}...")
        data_df = read_csv_cached(f'{output_dir}/{name}_additional_cols.csv', low_memory=False)
        data_df = normalize_data(data_df)
        
        # Separate driving and regen data
        data_driving_df = data_df[(data_df['energytype'] == 'driving_energy') & 
                                  (data_df['distance'].notna()) & 
                                  (data_df['distance'] != 0)].copy()
        data_regen_df = data_df[(data_df['energytype'] == 'energy_regen') & 
                               (data_df['distance'].notna()) & 
                               (data_df['distance'] != 0)].copy()
        
        data_driving_df['timestamp'] = pd.to_datetime(data_driving_df['timestamp'])
        data_regen_df['timestamp'] = pd.to_datetime(data_regen_df['timestamp'])

        # Calculate differences
        data_driving_df['accumulatedkwh_diffs'] = data_driving_df['accumumlatedkwh'] - data_driving_df['accumumlatedkwh'].shift(1)
        data_driving_df['accumulated_distance_diffs'] = data_driving_df['accumulated_distance'] - data_driving_df['accumulated_distance'].shift(1)
        data_driving_df['accumulated_distance_diffs_unc'] = DISTANCE_UNCERTAINTY
        data_driving_df['timestamp_diffs_seconds'] = (data_driving_df['timestamp'] - data_driving_df['timestamp'].shift(1)).dt.total_seconds()
        
        data_regen_df['accumulatedkwh_diffs'] = -1*(data_regen_df['accumumlatedkwh'] - data_regen_df['accumumlatedkwh'].shift(1))
        data_regen_df['accumulated_distance_diffs'] = data_regen_df['accumulated_distance'] - data_regen_df['accumulated_distance'].shift(1)
        data_regen_df['timestamp_diffs_seconds'] = (data_regen_df['timestamp'] - data_regen_df['timestamp'].shift(1)).dt.total_seconds()
        
        data_driving_with_regen_df = pd.concat([data_driving_df, data_regen_df])
        
        # Calculate energy per distance
        data_driving_df['driving_energy_per_distance'] = data_driving_df['accumulatedkwh_diffs'] / data_driving_df['accumulated_distance_diffs']
        data_driving_with_regen_df['driving_energy_per_distance'] = data_driving_with_regen_df['accumulatedkwh_diffs'] / data_driving_with_regen_df['accumulated_distance_diffs']
        data_regen_df['driving_energy_per_distance'] = data_regen_df['accumulatedkwh_diffs'] / data_regen_df['accumulated_distance_diffs']
        
        e_per_d_driving_total = data_driving_df['accumulatedkwh_diffs'].sum() / data_driving_df['accumulated_distance_diffs'].sum()
        e_per_d_driving_and_regen_total = data_driving_with_regen_df['accumulatedkwh_diffs'].sum() / data_driving_with_regen_df['accumulated_distance_diffs'].sum()
        
        # Bin by speed
        bins = np.linspace(0, 70, 8)
        data_driving_df['binned'] = pd.cut(data_driving_df['speed'], bins)
        data_driving_with_regen_df['binned'] = pd.cut(data_driving_with_regen_df['speed'], bins)
        
        binned_e_per_d_driving = data_driving_df.groupby('binned', observed=False)['accumulatedkwh_diffs'].sum() / data_driving_df.groupby('binned', observed=False)['accumulated_distance_diffs'].sum()
        binned_e_per_d_driving_dict[name] = binned_e_per_d_driving
        binned_e_per_d_driving_and_regen = data_driving_with_regen_df.groupby('binned', observed=False)['accumulatedkwh_diffs'].sum() / data_driving_with_regen_df.groupby('binned', observed=False)['accumulated_distance_diffs'].sum()
        binned_e_per_d_driving_and_regen_dict[name] = binned_e_per_d_driving_and_regen
        
        # Plot per-truck results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(name.replace('_', ' ').capitalize(), fontsize=20)
        ax.set_xlabel('Speed (miles/hour)', fontsize=16)
        ax.set_ylabel('Driving energy per distance (kWh/mile)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.text(0.15, 0.15, f'Total (driving only): {e_per_d_driving_total:.2f}\nTotal (with regen): {e_per_d_driving_and_regen_total:.2f}',
                transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        plt.hlines(0, linestyle='--', linewidth=1, color='black', xmin=0, xmax=70)
        
        ax.plot(data_driving_df['speed'], data_driving_df['driving_energy_per_distance'], 'o', color='blue', markersize=1, label='Driving Energy')
        ax.plot(data_regen_df['speed'], data_regen_df['driving_energy_per_distance'], 'o', color='green', markersize=1, label='Regen Energy')
        
        draw_binned_step(binned_e_per_d_driving, linecolor='red', linelabel='Overall per speed band (driving only)')
        draw_binned_step(binned_e_per_d_driving_and_regen, linecolor='green', linelabel='Overall per speed band (driving and regen)')
        
        ax.legend(fontsize=14, loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/energy_by_speed_{name}.png')
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/energy_by_speed_{name}.pdf')
        ax.set_ylim(-5,10)
        plt.savefig(f'{plot_dir}/energy_by_speed_{name}_zoom.png')
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/energy_by_speed_{name}_zoom.pdf')
        plt.close()
        print(f"  ✓ Saved energy per distance plots for {name}")
    
    # Plot all trucks together
    light_blue, medium_blue, dark_blue = "#CCCCFF", "#6666FF", "#0000CC"
    light_green, medium_green, dark_green = "#CCFFCC", "#66FF66", "#00CC00"
    blue_gradient = [light_blue, medium_blue, dark_blue]
    green_gradient = [light_green, medium_green, dark_green]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Speed (miles/hour)', fontsize=16)
    ax.set_ylabel('Driving energy per distance (kWh/mile)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    for i, name in enumerate(names):
        draw_binned_step(binned_e_per_d_driving_dict[name], linecolor=blue_gradient[i], 
                        linelabel=f'{name.replace("_", " ").capitalize()} (driving only)', linewidth=1)
    
    for i, name in enumerate(names):
        draw_binned_step(binned_e_per_d_driving_and_regen_dict[name], linecolor=green_gradient[i], 
                        linelabel=f'{name.replace("_", " ").capitalize()} (driving and regen)', linewidth=1)
    
    ax.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/energy_by_speed_all.png')
    if not FAST_MODE:
        plt.savefig(f'{plot_dir}/energy_by_speed_all.pdf')
    plt.close()
    print("  ✓ Saved combined energy per distance plot")

def prepare_driving_charging_data(top_dir, names):
    """Prepare data with driving and charging event labels."""
    print("\n" + "="*70)
    print("PREPARING DRIVING/CHARGING EVENT DATA")
    print("="*70)
    
    output_dir = get_output_dir(top_dir, 'data')
    
    for name in names:
        print(f"Processing {name}...")
        data_df = read_csv_cached(f'{output_dir}/{name}_additional_cols.csv', low_memory=False)
        data_df = normalize_data(data_df)
        
        # Label activities
        if DATASET_TYPE == 'pepsi':
            data_df['activity'] = data_df['energytype'].ffill()
            data_df.loc[data_df['activity'] == 'energy_from_dc_charger', 'activity'] = 'charging'
            data_df.loc[data_df['activity'] == 'driving_energy', 'activity'] = 'driving'
            data_df.loc[data_df['activity'] == 'energy_regen', 'activity'] = 'driving'
        else:  # messy_middle
            data_df['activity'] = data_df['truck_activity'].ffill()
        data_df = data_df.dropna(subset=['activity'])
        
        # Add event numbers
        data_df['charging_event'] = np.nan
        data_df['driving_event'] = np.nan
        
        charging_event = 0
        driving_event = 0
        prev_activity = data_df.at[data_df.index[0], 'activity']
        prev_soc = -1.

        for index, row in data_df.iterrows():
            current_activity = row['activity']
            current_soc = row['socpercent']
            
            if current_activity == 'charging' and current_soc <= prev_soc:
                current_activity = np.nan
                data_df.at[index, 'activity'] = np.nan
                current_activity = 'driving'
            
            if current_activity == 'charging':
                if prev_activity == 'driving' or not isinstance(prev_activity, str):
                    charging_event += 1
                data_df.at[index, 'charging_event'] = charging_event
            
            if current_activity == 'driving':
                if prev_activity == 'charging' or prev_activity == np.nan:
                    driving_event += 1
                data_df.at[index, 'driving_event'] = driving_event
                
            prev_activity = current_activity
            prev_soc = current_soc
            
            # Interpolate single NaN rows in accumumlatedkwh
            if pd.notna(row['accumumlatedkwh']):
                continue
            if index > 0 and index < len(data_df) - 1:
                prev_value = data_df.at[index - 1, 'accumumlatedkwh']
                next_value = data_df.at[index + 1, 'accumumlatedkwh']
                if pd.notna(prev_value) and pd.notna(next_value):
                    data_df.at[index, 'accumumlatedkwh'] = (prev_value + next_value) / 2
                    
        data_df.to_csv(f'{output_dir}/{name}_with_driving_charging.csv', index=False)
        print(f"  ✓ Saved {name}_with_driving_charging.csv")

def analyze_battery_capacity(top_dir, names):
    """Analyze battery capacity from charging events using linear and quadratic fits."""
    print("\n" + "="*70)
    print("ANALYZING BATTERY CAPACITY")
    print("="*70)
    
    battery_capacities = []
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    table_dir = get_output_dir(top_dir, 'tables')
    
    # Determine correct filenames based on dataset type
    if DATASET_TYPE == 'pepsi':
        battery_quadfit_suffix = '_battery_data_quadfit.csv'
    else:  # messy_middle
        battery_quadfit_suffix = '_battery_data_quadfit.csv'
    
    for name in names:
        print(f"Processing {name}...")
        battery_data_dict = {
            'charging_event': [],
            'battery_size': [],
            'battery_size_unc': []
        }
        
        battery_data_linear_df = pd.DataFrame(battery_data_dict)
        battery_data_quad_df = pd.DataFrame(battery_data_dict)
        data_df = read_csv_cached(f'{output_dir}/{name}_with_driving_charging.csv', low_memory=False)
        
        n_charging_events = int(data_df['charging_event'].max())
        for charging_event in range(1, n_charging_events):
            cChargingEvent = (data_df['charging_event'] == charging_event)
            data_df_event = data_df[cChargingEvent].dropna(subset=['socpercent', 'accumumlatedkwh'])
            
            if len(data_df_event) < 10 or (data_df_event['socpercent'].max() - data_df_event['socpercent'].min()) < 50:
                continue
            
            x_values = data_df_event['socpercent']
            y_values = data_df_event['accumumlatedkwh']
            
            # Linear Fit
            coefficients, covariance = np.polyfit(x_values, y_values, 1, cov=True)
            slope, b = coefficients[0], coefficients[1]
            slope_unc, b_unc = np.sqrt(covariance[0, 0]), np.sqrt(covariance[1, 1])
            y_pred = np.polyval(coefficients, x_values)
            rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))
            
            battery_size = slope * 100
            battery_size_unc = slope_unc * 100
            
            new_row = pd.DataFrame({
                'charging_event': [charging_event],
                'battery_size': [battery_size],
                'battery_size_unc': [battery_size_unc]
            })
            battery_data_linear_df = pd.concat([battery_data_linear_df, new_row], ignore_index=True)
            
            # Plot linear fit
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel('State of charge (%)', fontsize=24)
            ax.set_ylabel('Battery energy (kWh)', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.scatter(data_df_event['socpercent'], data_df_event['accumumlatedkwh'], s=50, color='black')
            
            best_fit_line = rf"Best-fit Line \ny = mx + b \nm={slope:.3f}$\pm${slope_unc:.3f}\nRMSE: {rmse:.2f}"
            ax.text(0.33, 0.25, rf'Extrapolated Battery Size: {battery_size:.1f}$\pm${battery_size_unc:.1f} kWh', 
                   transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=20)
            
            x_plot = np.linspace(0, 100, 1000)
            plt.plot(x_plot, slope * x_plot + b, color='red', label=best_fit_line, linewidth=3)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_energy_event_{charging_event}_linearfit.png')
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_energy_event_{charging_event}_linearfit.pdf')
            plt.close()
            
            # Quadratic Fit
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel('State of charge (%)', fontsize=24)
            ax.set_ylabel('Battery energy (kWh)', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.scatter(data_df_event['socpercent'], data_df_event['accumumlatedkwh'], s=50, color='black')
            
            coefficients, covariance = np.polyfit(x_values, y_values, 2, cov=True)
            a, b, c = coefficients[0], coefficients[1], coefficients[2]
            a_unc, b_unc = np.sqrt(covariance[0, 0]), np.sqrt(covariance[1, 1])
            
            y_pred = np.polyval(coefficients, x_values)
            rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))
            
            battery_size = a*100**2 + b*100
            battery_size_unc = np.sqrt((a_unc*100**2)**2 + (b_unc*100)**2)
            
            new_row_quad = pd.DataFrame({
                'charging_event': [charging_event],
                'battery_size': [battery_size],
                'battery_size_unc': [battery_size_unc]
            })
            battery_data_quad_df = pd.concat([battery_data_quad_df, new_row_quad], ignore_index=True)
            
            best_fit_line = rf"Best-fit Quadratic \ny = ax$^2$ + bx + c \na={a:.4f}$\pm${a_unc:.4f}\nb: {b:.2f}$\pm${b_unc:.2f}\nRMSE: {rmse:.2f}"
            ax.text(0.33, 0.25, rf'Extrapolated Battery Size: {battery_size:.1f}$\pm${battery_size_unc:.1f} kWh', 
                   transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=20)
            
            x_plot = np.linspace(0, 100, 1000)
            plt.plot(x_plot, a * x_plot**2 + b * x_plot + c, color='red', label=best_fit_line, linewidth=3)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_energy_event_{charging_event}_quadfit.png')
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_energy_event_{charging_event}_quadfit.pdf')
            plt.close()
        
        battery_data_linear_df.to_csv(f'{table_dir}/{name}_battery_data_linearfit.csv', index=False)
        battery_data_quad_df.to_csv(f'{table_dir}/{name}_battery_data_quadfit.csv', index=False)
        
        # Calculate weighted averages
        battery_data_quadfit_df = pd.read_csv(f'{table_dir}/{name}_battery_data_quadfit.csv')
        weighted_mean_quadfit = np.average(battery_data_quadfit_df['battery_size'], 
                                          weights=1./battery_data_quadfit_df['battery_size_unc']**2)
        weighted_std_quadfit = np.sqrt(np.average((battery_data_quadfit_df['battery_size']-weighted_mean_quadfit)**2, 
                                                  weights=1./battery_data_quadfit_df['battery_size_unc']**2))
        
        # Plot battery capacity summary
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('Charging Event', fontsize=24)
        ax.set_ylabel('Battery Capacity (kWh)', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.xticks(range(len(battery_data_quadfit_df['charging_event'])), 
                  labels=battery_data_quadfit_df['charging_event'].astype(int))
        
        plt.errorbar(range(len(battery_data_quadfit_df['charging_event'])), battery_data_quadfit_df['battery_size'], 
                    yerr=battery_data_quadfit_df['battery_size_unc'], capsize=5, marker='o', linestyle='none', 
                    color='black', label='Extrapolated capacity')
        xmin, xmax = ax.get_xlim()
        ax.axhline(weighted_mean_quadfit, color='blue', linewidth=2, 
                  label=rf'Weighted mean: {weighted_mean_quadfit:.1f}$\pm${weighted_std_quadfit:.1f}')
        ax.fill_between(np.linspace(xmin, xmax, 100), weighted_mean_quadfit-weighted_std_quadfit, 
                       weighted_mean_quadfit+weighted_std_quadfit, color='blue', alpha=0.2, edgecolor='none')
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{top_dir}/plots/{name}_battery_capacity_summary.png', dpi=300)
        if not FAST_MODE:
            plt.savefig(f'{top_dir}/plots/{name}_battery_capacity_summary.pdf')
        plt.close()
        
        battery_capacities.append(weighted_mean_quadfit)
        print(f"  ✓ Saved battery capacity analysis for {name}")
    
    # Save battery capacities
    battery_capacity_save = pd.DataFrame({'Value': ['Mean', 'Standard Deviation']})
    for i, name in enumerate(names):
        battery_data_quadfit_df = pd.read_csv(f'{table_dir}/{name}_battery_data_quadfit.csv')
        weighted_mean = np.average(battery_data_quadfit_df['battery_size'], 
                                  weights=1./battery_data_quadfit_df['battery_size_unc']**2)
        weighted_std = np.sqrt(np.average((battery_data_quadfit_df['battery_size']-weighted_mean)**2, 
                                         weights=1./battery_data_quadfit_df['battery_size_unc']**2))
        battery_capacity_save[name] = [weighted_mean, weighted_std]
    
    battery_capacity_save['average'] = [np.mean(battery_capacities), np.std(battery_capacities)]
    battery_capacity_save.to_csv(f'{table_dir}/battery_capacities.csv')
    print("  ✓ Saved battery capacities summary")

def analyze_charging_time_dod(top_dir, names):
    """Analyze charging time and depth of discharge."""
    print("\n" + "="*70)
    print("ANALYZING CHARGING TIME AND DEPTH OF DISCHARGE")
    print("="*70)
    
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    table_dir = get_output_dir(top_dir, 'tables')
    
    for name in names:
        print(f"Processing {name}...")
        charging_dict = {
            'charging_event': [],
            'min_soc': [],
            'max_soc': [],
            'DoD': [],
            'charging_time': [],
        }
        
        charging_df = pd.DataFrame(charging_dict)
        data_df = read_csv_cached(f'{output_dir}/{name}_with_driving_charging.csv', low_memory=False)

        n_charging_events = int(data_df['charging_event'].max())
        for charging_event in range(1, n_charging_events):
            cChargingEvent = (data_df['charging_event'] == charging_event)
            data_df_event = data_df[cChargingEvent].dropna(subset=['socpercent', 'timestamp'])
            data_df_event['timestamp'] = pd.to_datetime(data_df_event['timestamp'])
            
            min_soc = data_df_event['socpercent'].min()
            max_soc = data_df_event['socpercent'].max()
            dod = max_soc - min_soc
            if dod < 1 or np.isnan(dod):
                continue
            
            start_time = data_df_event['timestamp'].iloc[0]
            charging_time = calculate_time_elapsed(data_df_event.iloc[-1], start_time)
            
            new_row = pd.DataFrame([{
                'charging_event': charging_event,
                'min_soc': min_soc,
                'max_soc': max_soc,
                'DoD': dod,
                'charging_time': charging_time
            }])
            charging_df = pd.concat([charging_df, new_row], ignore_index=True)
            
            # Plot charging profile
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{name.replace('_', ' ').capitalize()}: Charging Event {charging_event}", fontsize=18)
            ax.set_ylabel('State of Charge (%)', fontsize=18)
            ax.set_xlabel('Charging time elapsed (minutes)', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            plt.text(0.35, 0.15, f'Charging Time: {charging_time:.1f} minutes\nDepth of Discharge: {dod:.1f}%\nMin Battery: {min_soc:.1f}%', 
                    transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
            
            charging_time_elapsed = data_df_event.apply(calculate_time_elapsed, axis=1, args=(start_time,))
            ax.scatter(charging_time_elapsed, data_df_event['socpercent'])
            plt.savefig(f'{plot_dir}/{name}_time_vs_battery_soc_event_{charging_event}.png')
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_time_vs_battery_soc_event_{charging_event}.pdf')
            plt.close()
        
        charging_df.to_csv(f'{table_dir}/{name}_charging_time_data.csv', index=False)
        
        # Plot summary statistics
        mean_charging_time = np.average(charging_df['charging_time'])
        min_charging_time = np.min(charging_df['charging_time'])
        max_charging_time = np.max(charging_df['charging_time'])
        std_charging_time = np.std(charging_df['charging_time'])
        
        mean_dod = np.average(charging_df['DoD'])
        min_dod = np.min(charging_df['DoD'])
        max_dod = np.max(charging_df['DoD'])
        std_dod = np.std(charging_df['DoD'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Summary of Charging Parameters", fontsize=18)
        ax.set_xlabel('Minimum Depth of Discharge (%)', fontsize=18)
        ax.set_ylabel('Charging Time (minutes)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.scatter(charging_df['DoD'], charging_df['charging_time'], color='green')
        
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax + (xmax-xmin))
        xmin, xmax = ax.get_xlim()
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
        ymin, ymax = ax.get_ylim()
        
        ax.axhline(mean_charging_time, color='green', linewidth=2, 
                  label=rf'Mean charge time: {mean_charging_time:.1f}$\pm${std_charging_time:.1f} mins\nMin: {min_charging_time:.1f}\nMax: {max_charging_time:.1f}')
        ax.fill_between(np.linspace(xmin, xmax, 100), mean_charging_time-std_charging_time, 
                       mean_charging_time+std_charging_time, color='green', alpha=0.2, edgecolor='none')

        ax.axvline(mean_dod, color='blue', linewidth=2, 
                  label=rf'Mean DoD: {mean_dod:.1f}%$\pm${std_dod:.1f}%\nMin: {min_dod:.1f}%\nMax: {max_dod:.1f}%')
        ax.fill_betweenx(np.linspace(ymin, ymax, 100), mean_dod-std_dod, mean_dod+std_dod, 
                        color='blue', alpha=0.2, edgecolor='none')
        
        ax.legend(loc='upper right', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_charging_summary.png')
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/{name}_charging_summary.pdf')
        plt.close()
        print(f"  ✓ Saved charging time analysis for {name}")

def analyze_drive_cycles(top_dir, names):
    """Analyze individual drive cycles and extrapolated range/fuel economy."""
    print("\n" + "="*70)
    print("ANALYZING DRIVE CYCLES")
    print("="*70)
    
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    table_dir = get_output_dir(top_dir, 'tables')
    
    # Read battery capacities
    battery_capacity_df = pd.read_csv(f'{table_dir}/battery_capacities.csv')
    
    for name in names:
        print(f"Processing {name}...")
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
        data_df = read_csv_cached(f'{output_dir}/{name}_with_driving_charging.csv', low_memory=False)
        
        battery_capacity = battery_capacity_df[name].iloc[0]
        battery_capacity_unc = battery_capacity_df[name].iloc[1]
        
        n_driving_events = int(data_df['driving_event'].max())
        for driving_event in range(1, n_driving_events):
            cDrivingEvent = (data_df['driving_event'] == driving_event)
            data_df_event = data_df[cDrivingEvent].dropna(subset=['socpercent', 'accumulated_distance'])
            
            if len(data_df_event) < 10 or (data_df_event['socpercent'].max() - data_df_event['socpercent'].min()) < 50:
                continue
            
            # Plot driving profile
            fig, axs = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1]})
            axs[1].set_xlabel('State of charge (%)', fontsize=24)
            axs[0].set_ylabel('Distance Traveled (miles)', fontsize=24)
            axs[1].set_ylabel('Speed (mph)', fontsize=24)
            axs[0].tick_params(axis='both', which='major', labelsize=20)
            axs[1].tick_params(axis='both', which='major', labelsize=20)
            
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
            
            x_values = data_df_event['socpercent']
            y_values = data_df_event['accumulated_distance']
            
            coefficients, covariance = np.polyfit(x_values, y_values, 1, cov=True)
            slope, b = coefficients[0], coefficients[1]
            slope_unc = np.sqrt(covariance[0, 0])
            
            y_pred = np.polyval(coefficients, x_values)
            rmse = np.sqrt(np.mean((y_values - y_pred) ** 2))
            
            truck_range = -1 * slope * 100
            truck_range_unc = slope_unc * 100
            
            battery_charge_init = data_df_event['socpercent'].max()
            battery_charge_final = data_df_event['socpercent'].min()
            dod = battery_charge_init - battery_charge_final
            
            delta_battery_energy = dod * battery_capacity / 100.
            delta_battery_energy_unc = dod * battery_capacity_unc / 100.
            distance_traveled = data_df_event['accumulated_distance'].max() - data_df_event['accumulated_distance'].min()
            
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
            
            best_fit_line = rf"Best-fit Line \ny = mx + b \nm={slope:.3f}$\pm${slope_unc:.3f}\nRMSE: {rmse:.2f}"
            
            x_plot = np.linspace(0, 100, 1000)
            axs[0].plot(x_plot, slope * x_plot + b, color='red', label=best_fit_line, linewidth=3)
            axs[0].legend(fontsize=20)
            
            fig.text(0.15, 0.45, rf'Range: {truck_range:.1f}$\pm${truck_range_unc:.1f} kWh\nEnergy Economy: {fuel_economy:.2f}$\pm${fuel_economy_unc:.2f} kWh/mile', 
                    fontsize=20, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7))
            
            xmin, xmax = axs[0].get_xlim()
            axs[1].set_xlim(xmin, xmax)
            
            plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.png', dpi=300)
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.pdf')
            plt.close()

        drivecycle_data_df['Driving event'] = drivecycle_data_df['Driving event'].astype('int')
        drivecycle_data_df.to_csv(f'{table_dir}/{name}_drivecycle_data.csv', index=False)
        print(f"  ✓ Saved drive cycle analysis for {name}")
    
    # Plot RMSE distribution
    all_rmse = np.zeros(0)
    for name in names:
        drivecycle_data_df = pd.read_csv(f'{table_dir}/{name}_drivecycle_data.csv')
        all_rmse = np.append(all_rmse, np.array(drivecycle_data_df['RMSE']))
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel('RMSE', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.hist(all_rmse, bins=50)
    ax.axvline(RMSE_cutoff, linestyle='--', linewidth=3, color='red', label='RMSE Cutoff')
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/all_RMSE.png', dpi=300)
    if not FAST_MODE:
        plt.savefig(f'{plot_dir}/all_RMSE.pdf')
    plt.close()
    
    # Plot range and fuel economy summaries
    for name in names:
        drivecycle_data_df = pd.read_csv(f'{table_dir}/{name}_drivecycle_data.csv')
        
        drivecycle_data_linear = drivecycle_data_df[drivecycle_data_df['RMSE'] < RMSE_cutoff]
        drivecycle_data_nonlinear = drivecycle_data_df[drivecycle_data_df['RMSE'] > RMSE_cutoff]
        
        # Range summary
        mean_linear = np.average(drivecycle_data_linear['Range (miles)'])
        mean_nonlinear = np.average(drivecycle_data_nonlinear['Range (miles)'])
        std_linear = np.std(drivecycle_data_linear['Range (miles)'])
        std_nonlinear = np.std(drivecycle_data_nonlinear['Range (miles)'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel('Extrapolated Range (miles)', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.xticks([])
        
        plt.errorbar(range(len(drivecycle_data_linear['Driving event'])), drivecycle_data_linear['Range (miles)'], 
                    yerr=drivecycle_data_linear['Range unc (miles)'], capsize=5, marker='o', linestyle='none', 
                    color='blue', label='Extrapolated range (linear)')
        
        plt.errorbar(range(len(drivecycle_data_linear['Driving event']), 
                          len(drivecycle_data_linear['Driving event'])+len(drivecycle_data_nonlinear['Driving event'])), 
                    drivecycle_data_nonlinear['Range (miles)'], yerr=drivecycle_data_nonlinear['Range unc (miles)'], 
                    capsize=5, marker='o', linestyle='none', color='green', label='Extrapolated range (nonlinear)')
        xmin, xmax = ax.get_xlim()
        
        ax.axhline(mean_linear, color='blue', linewidth=2, label=rf'Mean (linear): {mean_linear:.1f}$\pm${std_linear:.1f}')
        ax.fill_between(np.linspace(xmin, xmax, 100), mean_linear-std_linear, mean_linear+std_linear, 
                       color='blue', alpha=0.2, edgecolor='none')
        
        ax.axhline(mean_nonlinear, color='green', linewidth=2, label=rf'Mean (nonlinear): {mean_nonlinear:.1f}$\pm${std_nonlinear:.1f}')
        ax.fill_between(np.linspace(xmin, xmax, 100), mean_nonlinear-std_nonlinear, mean_nonlinear+std_nonlinear, 
                       color='green', alpha=0.2, edgecolor='none')
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_range_summary.png')
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/{name}_range_summary.pdf')
        plt.close()
        
        # Fuel economy summary
        mean_linear = np.average(drivecycle_data_linear['Fuel economy (kWh/mile)'])
        mean_nonlinear = np.average(drivecycle_data_nonlinear['Fuel economy (kWh/mile)'])
        std_linear = np.std(drivecycle_data_linear['Fuel economy (kWh/mile)'])
        std_nonlinear = np.std(drivecycle_data_nonlinear['Fuel economy (kWh/mile)'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel('Extrapolated Energy Economy (kWh/mile)', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.xticks([])
        
        plt.errorbar(range(len(drivecycle_data_linear['Driving event'])), drivecycle_data_linear['Fuel economy (kWh/mile)'], 
                    yerr=drivecycle_data_linear['Fuel economy unc (kWh/mile)'], capsize=5, marker='o', linestyle='none', 
                    color='blue', label='Extrapolated energy economy (linear)')
        
        plt.errorbar(range(len(drivecycle_data_linear['Driving event']), 
                          len(drivecycle_data_linear['Driving event'])+len(drivecycle_data_nonlinear['Driving event'])), 
                    drivecycle_data_nonlinear['Fuel economy (kWh/mile)'], 
                    yerr=drivecycle_data_nonlinear['Fuel economy unc (kWh/mile)'], capsize=5, marker='o', linestyle='none', 
                    color='green', label='Extrapolated energy economy (nonlinear)')
        xmin, xmax = ax.get_xlim()
        
        ax.axhline(mean_linear, color='blue', linewidth=2, label=rf'Mean (linear): {mean_linear:.2f}$\pm${std_linear:.2f}')
        ax.fill_between(np.linspace(xmin, xmax, 100), mean_linear-std_linear, mean_linear+std_linear, 
                       color='blue', alpha=0.2, edgecolor='none')
        
        ax.axhline(mean_nonlinear, color='green', linewidth=2, label=rf'Mean (nonlinear): {mean_nonlinear:.2f}$\pm${std_nonlinear:.2f}')
        ax.fill_between(np.linspace(xmin, xmax, 100), mean_nonlinear-std_nonlinear, mean_nonlinear+std_nonlinear, 
                       color='green', alpha=0.2, edgecolor='none')
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.4)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_fuel_economy_summary.png')
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/{name}_fuel_economy_summary.pdf')
        plt.close()

def analyze_vmt(top_dir, names):
    """Analyze and extrapolate vehicle miles traveled."""
    print("\n" + "="*70)
    print("ANALYZING VEHICLE MILES TRAVELED (VMT)")
    print("="*70)
    
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    table_dir = get_output_dir(top_dir, 'tables')
    
    for name in names:
        print(f"Processing {name}...")
        vmt_data_dict = {
            'miles_driven': [],
            'total_time': [],
            'extrapolated_vmt': []
        }
        
        vmt_data_df = pd.DataFrame(vmt_data_dict)
        data_df = read_csv_cached(f'{output_dir}/{name}_with_driving_charging.csv', low_memory=False)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        
        min_datetime = data_df['timestamp'].min()
        data_df['timestamp_hours'] = data_df['timestamp'].apply(
            lambda x: float((x - min_datetime).total_seconds() / SECONDS_PER_HOUR) if pd.notna(x) else np.nan
        )
        
        data_df = data_df[data_df['timestamp_hours'].notna()]
        data_df = data_df[~np.isinf(data_df['timestamp_hours'])]
        data_df_reduced = data_df.iloc[::100]
        
        # Plot speed and SOC vs time
        fig, axs = plt.subplots(2, 1, figsize=(18, 7), gridspec_kw={'height_ratios': [1, 1]})
        axs[0].plot(data_df_reduced.dropna(subset=['speed'])['timestamp_hours'], 
                   data_df_reduced.dropna(subset=['speed'])['speed'])
        axs[1].plot(data_df.dropna(subset=['socpercent', 'timestamp_hours'])['timestamp_hours'], 
                   data_df.dropna(subset=['socpercent', 'timestamp_hours'])['socpercent'])
        
        cCharging = (data_df['activity'] == 'charging')
        data_df['charging_timestamp'] = data_df['timestamp_hours'].where(cCharging, np.nan)
        
        ymin, ymax = axs[0].get_ylim()
        axs[0].fill_between(data_df['charging_timestamp'], ymin, ymax, color='green', alpha=0.2, edgecolor='none')
        axs[0].set_ylim(ymin, ymax)
        
        ymin, ymax = axs[1].get_ylim()
        axs[1].fill_between(data_df['charging_timestamp'], ymin, ymax, color='green', alpha=0.2, edgecolor='none', label='Charging')
        axs[1].set_ylim(ymin, ymax)
        
        cDriving = (data_df['activity'] == 'driving')
        data_df['driving_timestamp'] = data_df['timestamp_hours'].where(cDriving, np.nan)
        
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
        plt.savefig(f'{plot_dir}/{name}_speed_vs_time.png', dpi=300)
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/{name}_speed_vs_time.pdf')
        plt.close()
        
        # Plot accumulated distance vs time
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
        plt.text(0.35, 0.15, f'Total Distance: {total_distance:.1f} miles\nTotal Time: {total_data_collection_time:.2f} days\nExtrapolated VMT: {extrapolated_vmt:.0f} miles/year', 
                transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{top_dir}/plots/{name}_distance_vs_time.png')
        if not FAST_MODE:
            plt.savefig(f'{top_dir}/plots/{name}_distance_vs_time.pdf')
        plt.close()
        
        vmt_data_df.to_csv(f'{table_dir}/{name}_vmt_data.csv', 
                          header=['Distance traveled (miles)', 'Total time (days)', 'Extrapolated Annual VMT (miles/year)'], index=False)
        print(f"  ✓ Saved VMT analysis for {name}")

def analyze_energy_delivered(top_dir, names):
    """Analyze and extrapolate energy delivered per month."""
    print("\n" + "="*70)
    print("ANALYZING ENERGY DELIVERED PER MONTH")
    print("="*70)
    
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    table_dir = get_output_dir(top_dir, 'tables')
    
    for name in names:
        print(f"Processing {name}...")
        energy_data_dict = {
            'energy_delivered': [],
            'total_time': [],
            'extrapolated_energy_per_month': []
        }
        
        energy_data_df = pd.DataFrame(energy_data_dict)
        data_df = read_csv_cached(f'{output_dir}/{name}_with_driving_charging.csv', low_memory=False)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df_charging = data_df[data_df['energytype'] == 'energy_from_dc_charger']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{name.replace('_', ' ').capitalize()}: Extrapolation of Energy Delivered/Month", fontsize=18)
        ax.set_ylabel('Accumulated electricity delivered (MWh)', fontsize=18)
        ax.set_xlabel('Accumulated data collection time (days)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        start_time = data_df['timestamp'].iloc[0]
        collection_time_elapsed = data_df.apply(calculate_time_elapsed, axis=1, args=(start_time,)) / MINUTES_PER_DAY
        total_energy_delivered = data_df_charging['accumumlatedkwh'].max() / 1e3
        total_data_collection_time = collection_time_elapsed.max()
        extrapolated_energy_per_month = (total_energy_delivered / total_data_collection_time) * DAYS_PER_MONTH
        
        energy_data_df['energy_delivered'] = total_energy_delivered
        energy_data_df['total_time'] = total_data_collection_time
        energy_data_df['extrapolated_energy_per_month'] = extrapolated_energy_per_month
        
        ax.plot(collection_time_elapsed[data_df['energytype'] == 'energy_from_dc_charger'], 
               data_df_charging['accumumlatedkwh'] / 1e3, 'o', markersize=1)
        plt.text(0.35, 0.15, f'Total Energy: {total_energy_delivered:.1f} MWh\nTotal Time: {total_data_collection_time:.2f} days\nExtrapolated: {extrapolated_energy_per_month:.1f} MWh/month', 
                transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_charging_energy_vs_time.png')
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/{name}_charging_energy_vs_time.pdf')
        plt.close()
        
        energy_data_df.to_csv(f'{table_dir}/{name}_energy_per_month_data.csv', 
                             header=['Energy Delivered (kWh)', 'Total time (days)', 'Extrapolated Energy/Month (kWh/month)'], index=False)
        print(f"  ✓ Saved energy delivered analysis for {name}")

# ============================================================================
# MAIN ROUTINE
# ============================================================================

def main():
    """Main analysis routine. Comment/uncomment sections as needed."""
    
    top_dir = get_top_dir()
    setup_directories(top_dir)
    files, names = get_file_list(top_dir)
    
    print("\n" + "="*70)
    print(f"ANALYZING {DATASET_TYPE.upper()} DATA")
    print(f"FAST_MODE: {FAST_MODE}")
    print("="*70)
    
    # ======== UNCOMMENT/COMMENT SECTIONS TO RUN ========
    
    # Stage 1: Data Preprocessing
    preprocess_data(top_dir, files, names)
    
    # Stage 2: Charging Analysis
    analyze_charging_power(top_dir, names)
    
    # Stage 3: Energy Analysis
    analyze_instantaneous_energy(top_dir, names)
    
    # Stage 4: Prepare Driving/Charging Events
    prepare_driving_charging_data(top_dir, names)
    
    # Stage 5: Battery Capacity Analysis
    analyze_battery_capacity(top_dir, names)
    
    # Stage 6: Charging Time & DoD
    analyze_charging_time_dod(top_dir, names)
    
    # Stage 7: Drive Cycles
    analyze_drive_cycles(top_dir, names)
    
    # Stage 8: VMT Analysis
    analyze_vmt(top_dir, names)
    
    # Stage 9: Energy Delivered
    analyze_energy_delivered(top_dir, names)
    
    # ====================================================
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
