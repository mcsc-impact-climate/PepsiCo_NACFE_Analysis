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
NO_PLOT_MODE = False  # Set to True to skip plot generation (fastest - data only)
# Debug flag: when True, skip grade smoothing/clipping/interpolation to inspect raw behavior
GRADE_DEBUG_MODE = False
DATASET_TYPE = 'messy_middle'  # 'pepsi' or 'messy_middle'

SECONDS_PER_HOUR = 3600.
MINUTES_PER_DAY = 60.*24.
DAYS_PER_YEAR = 365.
DISTANCE_UNCERTAINTY = 0.02  # miles (~32 meters) - raised to suppress spikes from tiny distance steps
KM_PER_MILE = 1.60934
KM_TO_MILES = 0.621371
DAYS_PER_MONTH = 30.437
METERS_PER_MILE = 1609.34
RMSE_cutoff = 10

# Drive cycle filtering thresholds
MIN_DRIVING_EVENT_POINTS = 50       # Minimum data points per driving event
MIN_DRIVING_DOD = 15                # Minimum depth of discharge (%) for a driving event
MIN_DRIVING_DISTANCE = 10            # Minimum distance traveled (miles) for a driving event

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

def save_plot(filepath, pdf_path=None):
    """Save a plot, respecting NO_PLOT_MODE and FAST_MODE settings."""
    if NO_PLOT_MODE:
        return
    plt.savefig(filepath, dpi=300)
    if not FAST_MODE and pdf_path:
        plt.savefig(pdf_path)


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
            f'{top_dir}/data_messy_middle/nevoya_with_weight.csv',
            f'{top_dir}/data_messy_middle/4gen.csv',
            f'{top_dir}/data_messy_middle/joyride.csv',
            f'{top_dir}/data_messy_middle/saia1_with_elevation.csv',
            f'{top_dir}/data_messy_middle/saia2_with_elevation.csv'
        ]
        # Derive names from filenames (strip extension and optional suffixes)
        import os
        names = []
        for f in files:
            base = os.path.basename(f)
            name = base.split('.')[0]
            name = name.replace('_with_elevation', '').replace('_with_driving_charging', '')
            names.append(name)
    
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
        
        # Calculate road grade from elevation data
        elevation_col = get_column_name(data_df, ['elevation_final_m', 'elevation_meters'])
        if elevation_col:
            from scipy.signal import savgol_filter
            
            # Smooth elevation with Savitzky-Golay (better edge handling than rolling mean)
            elevation_valid = data_df[elevation_col].notna()
            if elevation_valid.sum() > 11:  # Need at least 12 points for window=9
                elev_array = data_df.loc[elevation_valid, elevation_col].values
                elev_smooth_array = savgol_filter(elev_array, window_length=9, polyorder=2)
                data_df.loc[elevation_valid, 'elevation_smooth'] = elev_smooth_array
                data_df['elevation_smooth'] = data_df['elevation_smooth'].ffill().bfill()
            else:
                data_df['elevation_smooth'] = data_df[elevation_col].rolling(window=5, center=True).mean()
            
            # Use accumulated distance deltas; if quantized/zero while speed > 0, fallback to speed*dt
            dist_step_miles = data_df['accumulated_distance'].diff().fillna(0)
            data_df['distance_change_meters'] = dist_step_miles * METERS_PER_MILE
            data_df['elevation_change'] = data_df['elevation_smooth'].diff()

            # Estimate distance from speed and time gaps to rescue quantized distance (Saia)
            dt_hours = data_df['timestamp'].diff().dt.total_seconds().fillna(0) / SECONDS_PER_HOUR
            speed_for_dist = data_df['speed_smooth'] if 'speed_smooth' in data_df.columns else data_df['speed']
            est_dist_miles = speed_for_dist * dt_hours
            est_dist_meters = est_dist_miles * METERS_PER_MILE

            # For Saia, rely primarily on synthetic distance from smoothed speed to avoid quantized distance flats
            if 'saia' in name:
                data_df['grade_distance_meters'] = est_dist_meters
            else:
                use_est_mask = (
                    (data_df['distance_change_meters'].abs() <= 5.0) | data_df['distance_change_meters'].isna()
                ) & (speed_for_dist.fillna(0) > 2.0) & (dt_hours > 0)
                data_df['grade_distance_meters'] = data_df['distance_change_meters']
                data_df.loc[use_est_mask, 'grade_distance_meters'] = est_dist_meters.loc[use_est_mask]
            
            # Pre-smooth speed aggressively to handle quantized/discrete speed data (Saia is especially noisy)
            speed_window = 81 if 'saia' in name else 15
            data_df['speed_smooth'] = data_df['speed'].rolling(window=speed_window, center=True, min_periods=1).mean()
            
            # Calculate speed change rate using smoothed speed to ignore quantization artifacts
            data_df['speed_change'] = data_df['speed_smooth'].diff().abs()
            
            # Thresholds to suppress spikes from tiny movements / near-zero speed
            min_distance_threshold = 0.1 if 'saia' in name else 0.01  # Saia: allow very small steps from synthetic distance
            min_speed_threshold = 5.0      # mph - raised to reduce GPS noise during deceleration/stops
            max_speed_change = 1.0 if 'saia' in name else 2.0  # tighter for Saia quantized data
            speed_ok = data_df['speed_smooth'].fillna(0) > min_speed_threshold
            dist_ok = data_df['grade_distance_meters'].abs() > min_distance_threshold
            speed_stable = data_df['speed_change'].fillna(0) < max_speed_change
            valid_grade = speed_ok & dist_ok & speed_stable & data_df['elevation_change'].notna()

            # Build a strictly-positive distance step for gradient to avoid divide-by-zero from quantized distance
            grade_step = data_df['grade_distance_meters'].copy()
            grade_step.loc[grade_step.abs() < 0.1] = 0.1
            grade_dist_accum = grade_step.cumsum()

            data_df['road_grade_percent'] = np.nan
            # Keep an unsmoothed/raw grade column for debugging overlays
            data_df['road_grade_percent_raw'] = np.nan

            if 'saia' in name:
                # Verify we're using smoothed profiles, not raw data
                print(f"    Saia {name} grade calculation setup:")
                print(f"      Using speed_smooth: {data_df['speed_smooth'].notna().sum()} non-NaN samples")
                print(f"      Using elevation_smooth: {data_df['elevation_smooth'].notna().sum()} non-NaN samples")
                print(f"      speed_smooth range: [{data_df['speed_smooth'].min():.2f}, {data_df['speed_smooth'].max():.2f}] mph")
                print(f"      elevation_smooth range: [{data_df['elevation_smooth'].min():.1f}, {data_df['elevation_smooth'].max():.1f}] m")
                
                # Time-derivative grade for Saia using ONLY smoothed profiles
                dt_seconds = data_df['timestamp'].diff().dt.total_seconds().fillna(0)
                # ONLY use speed_smooth, NOT raw speed
                speed_mps = data_df['speed_smooth'] * 0.44704
                # ONLY use elevation_smooth, NOT raw elevation
                elev_change = data_df['elevation_smooth'].diff()
                
                # Check for NaNs in smoothed columns
                speed_smooth_nans = data_df['speed_smooth'].isna().sum()
                elev_smooth_nans = data_df['elevation_smooth'].isna().sum()
                print(f"      speed_smooth NaNs: {speed_smooth_nans} samples")
                print(f"      elevation_smooth NaNs: {elev_smooth_nans} samples")
                
                # Only compute elev_dt where dt > 0 to avoid NaN from divide-by-zero
                elev_dt = pd.Series(np.nan, index=data_df.index)
                valid_dt = dt_seconds > 0
                elev_dt.loc[valid_dt] = (elev_change.loc[valid_dt] / dt_seconds.loc[valid_dt]).values
                
                # Check why grade is still NaN: trace through computation
                elev_dt_nans_after = elev_dt.isna().sum()
                elev_dt_zeros = (elev_dt == 0).sum()
                elev_dt_nonzero = (elev_dt != 0).sum()
                print(f"      elev_dt after divide: {elev_dt_nans_after} NaNs, {elev_dt_zeros} zeros, {elev_dt_nonzero} non-zero")
                
                # Compute grade where speed_smooth > 0 (no divide-by-zero)
                grade_time = pd.Series(np.nan, index=data_df.index)
                
                # For driving samples (speed > 0.1 m/s), use standard grade
                driving_mask = speed_mps > 0.1
                valid_driving = valid_dt & driving_mask & elev_dt.notna()
                grade_time.loc[valid_driving] = (elev_dt.loc[valid_driving] / speed_mps.loc[valid_driving] * 100.0).values
                
                # For near-idle samples (0 < speed <= 0.1 m/s), use grace speed
                idle_mask = (speed_mps > 0) & (speed_mps <= 0.1)
                valid_idle = valid_dt & idle_mask & elev_dt.notna()
                grade_time.loc[valid_idle] = (elev_dt.loc[valid_idle] / 0.05 * 100.0).values
                
                # For stopped samples (speed <= 0 or NaN), interpolation will bridge later
                
                # Gate: dt > 0, grade is finite
                time_mask = (dt_seconds > 0) & grade_time.notna() & np.isfinite(grade_time)
                # Store raw (pre-smoothing) values and the time-based mask for inspection
                data_df.loc[grade_time.notna() & np.isfinite(grade_time), 'road_grade_percent_raw'] = grade_time.loc[grade_time.notna() & np.isfinite(grade_time)]
                data_df['road_grade_time_mask'] = time_mask
                data_df.loc[time_mask, 'road_grade_percent'] = grade_time.loc[time_mask]
                
                # Diagnostic
                valid_driving_count = valid_driving.sum()
                valid_idle_count = valid_idle.sum()
                grade_finite_count = (grade_time.notna() & np.isfinite(grade_time)).sum()
                grade_nans = grade_time.isna().sum()
                grade_nofinite = (~np.isfinite(grade_time)).sum()
                total = len(data_df)
                print(f"      Grade computation breakdown:")
                print(f"        Driving (speed_smooth > 0.1): {valid_driving_count} samples")
                print(f"        Idle grace (0 < speed_smooth <= 0.1): {valid_idle_count} samples")
                print(f"        grade_time NaN: {grade_nans} | non-finite: {grade_nofinite}")
                print(f"        grade_time computed: {grade_finite_count} samples")
                print(f"        Final road_grade_percent assigned: {data_df['road_grade_percent'].notna().sum()} samples")
            else:
                # Distance-based grade for non-Saia datasets
                try:
                    grad_vals = np.gradient(data_df['elevation_smooth'], grade_dist_accum) * 100.0
                    # Raw gradient before any smoothing
                    data_df.loc[:, 'road_grade_percent_raw'] = pd.Series(grad_vals, index=data_df.index)
                    data_df.loc[valid_grade, 'road_grade_percent'] = pd.Series(grad_vals, index=data_df.index).loc[valid_grade]
                except Exception:
                    # Fallback raw values
                    raw_vals = (
                        data_df.loc[valid_grade, 'elevation_change'] /
                        grade_step.loc[valid_grade]
                    ) * 100.0
                    data_df.loc[valid_grade, 'road_grade_percent'] = raw_vals
                    data_df.loc[valid_grade, 'road_grade_percent_raw'] = raw_vals
            
            if not GRADE_DEBUG_MODE:
                # Median filter to remove single-sample spikes (wider window for more aggressive filtering)
                data_df['road_grade_percent'] = (
                    data_df['road_grade_percent']
                    .rolling(window=11, center=True, min_periods=1)
                    .median()
                )

                # Smooth the grade values to reduce residual noise (wider Savitzky-Golay window)
                grade_valid = data_df['road_grade_percent'].notna()
                if grade_valid.sum() > 15:
                    grade_array = data_df.loc[grade_valid, 'road_grade_percent'].values
                    try:
                        grade_smooth_array = savgol_filter(grade_array, window_length=15, polyorder=2)
                        data_df.loc[grade_valid, 'road_grade_percent'] = grade_smooth_array
                    except ValueError:
                        # Fallback to rolling mean if not enough points
                        data_df.loc[grade_valid, 'road_grade_percent'] = (
                            data_df.loc[grade_valid, 'road_grade_percent']
                            .rolling(window=7, center=True, min_periods=1)
                            .mean()
                        )

                # Clip extreme values (grades >25% are unrealistic for highways)
                data_df['road_grade_percent'] = data_df['road_grade_percent'].clip(-25, 25)
                # Interpolate gaps, but use a time-based limit to avoid bridging unrelated segments
                # For Saia idle periods (speed=0), we can bridge gaps up to ~5 minutes (often 300-500 samples at 1 Hz)
                # Use a conservative limit of 400 samples for Saia to handle idling without over-smoothing
                interp_limit = 400 if 'saia' in name else 25
                data_df['road_grade_percent'] = data_df['road_grade_percent'].interpolate(limit=interp_limit, limit_direction='both')

            # Diagnostics: quantify where final grade is ~0 while raw grade is non-zero
            if 'road_grade_percent_raw' in data_df.columns:
                eps = 1e-6
                raw_nonzero = data_df['road_grade_percent_raw'].abs() > eps
                final_zero = data_df['road_grade_percent'].abs() <= eps
                final_nan = data_df['road_grade_percent'].isna()
                suppressed = final_zero & raw_nonzero
                total = len(data_df)
                if total > 0:
                    suppressed_count = int(suppressed.sum())
                    suppressed_pct = (suppressed.sum() / total) * 100.0
                    raw_zero_pct = ((~raw_nonzero).sum() / total) * 100.0
                    final_nan_pct = (final_nan.sum() / total) * 100.0
                    valid_final = total - final_nan.sum()
                    final_zero_nonnan_pct = (final_zero.sum() / valid_final * 100.0) if valid_final > 0 else 0
                    print(f"    Grade column stats:")
                    print(f"      Raw grade ≈0: {raw_zero_pct:.1f}%")
                    print(f"      Final grade NaN: {final_nan_pct:.1f}%")
                    print(f"      Final grade ≈0 (excluding NaN): {final_zero_nonnan_pct:.1f}%")
                    print(f"      Suppressed (final≈0 while raw≠0): {suppressed_count} samples ({suppressed_pct:.1f}%)")
        
        output_dir = get_output_dir(top_dir, 'data')
        data_df.to_csv(f'{output_dir}/{name}_additional_cols.csv', index=False)
        print(f"  ✓ Saved {name}_additional_cols.csv")

def analyze_elevation_grade(top_dir, names):
    """Analyze and visualize elevation, road grade, and speed data."""
    print("\n" + "="*70)
    print("ANALYZING ELEVATION AND ROAD GRADE")
    print("="*70)
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    
    for name in names:
        print(f"Processing {name}...")
        data_df = read_csv_cached(f'{output_dir}/{name}_additional_cols.csv', low_memory=False)
        data_df = normalize_data(data_df)
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        
        # Check if elevation data exists
        elevation_col = get_column_name(data_df, ['elevation_final_m', 'elevation_meters'])
        if not elevation_col or 'road_grade_percent' not in data_df.columns:
            print(f"  ⚠ No elevation data available for {name}, skipping analysis")
            continue
        
        # Reduce data for plotting (every 100th point to avoid overcrowding)
        min_datetime = data_df['timestamp'].min()
        data_df['time_hours'] = data_df['timestamp'].apply(
            lambda x: float((x - min_datetime).total_seconds() / SECONDS_PER_HOUR) if pd.notna(x) else np.nan
        )
        data_df_plot = data_df.iloc[::100]
        
        # Create four-panel figure: speed vs time, elevation vs time, grade vs time, SOC vs time
        fig, axs = plt.subplots(4, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        
        # Panel 1: Speed vs time
        data_speed = data_df_plot.dropna(subset=['speed', 'time_hours'])
        axs[0].plot(data_speed['time_hours'], data_speed['speed'], linewidth=1.5, color='steelblue')
        axs[0].set_ylabel('Speed (mph)', fontsize=16)
        axs[0].set_xlim(data_df['time_hours'].min(), data_df['time_hours'].max())
        axs[0].grid(True, alpha=0.3)
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[0].xaxis.set_tick_params(labelbottom=False)
        
        # Panel 2: Elevation vs time
        elevation_col = get_column_name(data_df, ['elevation_final_m', 'elevation_meters'])
        data_elev = data_df_plot.dropna(subset=[elevation_col, 'time_hours'])
        axs[1].plot(data_elev['time_hours'], data_elev[elevation_col], linewidth=1.5, color='darkorange')
        axs[1].set_ylabel('Elevation (m)', fontsize=16)
        axs[1].set_xlim(data_df['time_hours'].min(), data_df['time_hours'].max())
        axs[1].grid(True, alpha=0.3)
        axs[1].tick_params(axis='both', which='major', labelsize=12)
        axs[1].xaxis.set_tick_params(labelbottom=False)
        
        # Panel 3: Road grade vs time
        data_grade = data_df_plot.dropna(subset=['road_grade_percent', 'time_hours'])
        axs[2].plot(data_grade['time_hours'], data_grade['road_grade_percent'], linewidth=1.5, color='darkgreen')
        axs[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axs[2].set_ylabel('Road Grade (%)', fontsize=16)
        axs[2].set_xlim(data_df['time_hours'].min(), data_df['time_hours'].max())
        axs[2].grid(True, alpha=0.3)
        axs[2].tick_params(axis='both', which='major', labelsize=12)
        axs[2].xaxis.set_tick_params(labelbottom=False)
        
        # Panel 4: State of charge vs time (delta SOC - point-to-point differences)
        soc_col = get_column_name(data_df, ['socpercent', 'battery_percent'])
        if soc_col:
            data_soc = data_df_plot.dropna(subset=[soc_col, 'time_hours'])
            delta_soc = data_soc[soc_col].diff()
            axs[3].plot(data_soc['time_hours'], delta_soc, linewidth=1.5, color='purple')
            axs[3].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            axs[3].set_ylabel('Δ SOC (%)', fontsize=16)
            axs[3].set_xlabel('Time Elapsed (hours)', fontsize=16)
            axs[3].set_xlim(data_df['time_hours'].min(), data_df['time_hours'].max())
            axs[3].grid(True, alpha=0.3)
            axs[3].tick_params(axis='both', which='major', labelsize=12)
        else:
            axs[3].text(0.5, 0.5, 'SOC data not available', ha='center', va='center', fontsize=14, transform=axs[3].transAxes)
            axs[3].set_ylabel('SOC (%)', fontsize=16)
            axs[3].set_xlabel('Time Elapsed (hours)', fontsize=16)
        
        # Add title
        fig.suptitle(f"{name.replace('_', ' ').capitalize()}: Speed, Elevation, Road Grade, and SOC Over Time", 
                    fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{name}_elevation_grade_analysis.png', dpi=300)
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/{name}_elevation_grade_analysis.pdf')
        plt.close()
        
        # Print statistics
        grade_mean = data_df['road_grade_percent'].mean()
        grade_std = data_df['road_grade_percent'].std()
        grade_min = data_df['road_grade_percent'].min()
        grade_max = data_df['road_grade_percent'].max()
        
        print(f"  ✓ Saved elevation and grade analysis for {name}")
        print(f"    Road Grade Statistics:")
        print(f"      Mean: {grade_mean:.2f}% ± {grade_std:.2f}%")
        print(f"      Range: {grade_min:.2f}% to {grade_max:.2f}%")

def analyze_charging_power(top_dir, names):
    """Analyze charging power statistics and create visualizations."""
    print("\n" + "="*70)
    print("ANALYZING CHARGING POWER")
    print("="*70)
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
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
    plt.xticks(range(1, len(names) + 1), [name.replace('_', ' ').capitalize() for name in names])
    plt.ylabel('Charging power (kW)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f'{plot_dir}/chargingpower_stats.png')
    plt.close()
    print(f"\n  ✓ Saved {DATASET_TYPE} charging power statistics plot")

def analyze_instantaneous_energy(top_dir, names):
    """Analyze instantaneous energy consumption per mile."""
    print("\n" + "="*70)
    print("ANALYZING INSTANTANEOUS ENERGY PER MILE")
    print("="*70)
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
    binned_e_per_d_driving_dict = {}
    binned_e_per_d_driving_and_regen_dict = {}
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    
    for name in names:
        print(f"Processing {name}...")
        data_df = read_csv_cached(f'{output_dir}/{name}_additional_cols.csv', low_memory=False)
        data_df = normalize_data(data_df)
        
        # Separate driving and regen data (messy_middle uses truck_activity instead of energytype)
        if DATASET_TYPE == 'messy_middle':
            driving_mask = data_df['truck_activity'] == 'driving'
            regen_energy_col = get_column_name(data_df, ['energy_regen_kwh', 'accumumlatedkwh'])
            driving_energy_col = get_column_name(data_df, ['driving_energy_kwh', 'accumumlatedkwh'])
        else:
            driving_mask = data_df['energytype'] == 'driving_energy'
            regen_energy_col = 'accumumlatedkwh'
            driving_energy_col = 'accumumlatedkwh'

        driving_mask &= data_df['accumulated_distance'].notna()
        data_driving_df = data_df[driving_mask].copy()

        if data_driving_df.empty:
            print(f"  ⚠ No driving samples found for {name}; skipping plot")
            continue

        data_driving_df['timestamp'] = pd.to_datetime(data_driving_df['timestamp'])

        # Driving deltas
        data_driving_df['accumulatedkwh_diffs'] = data_driving_df[driving_energy_col].diff()
        data_driving_df['accumulated_distance_diffs'] = data_driving_df['accumulated_distance'].diff()
        data_driving_df['accumulated_distance_diffs_unc'] = DISTANCE_UNCERTAINTY
        data_driving_df['timestamp_diffs_seconds'] = (data_driving_df['timestamp'] - data_driving_df['timestamp'].shift(1)).dt.total_seconds()

        # Regen deltas (aligned to the same driving rows to avoid double-counting distance)
        regen_diff_raw = data_df.loc[driving_mask, regen_energy_col].diff()
        regen_diff = regen_diff_raw.copy()
        regen_diff.loc[regen_diff_raw < 0] = np.nan  # drop resets

        # Drop bad steps and tiny distance steps
        data_driving_df.loc[data_driving_df['accumulatedkwh_diffs'] <= 0, 'accumulatedkwh_diffs'] = np.nan
        data_driving_df = data_driving_df[data_driving_df['accumulated_distance_diffs'] > DISTANCE_UNCERTAINTY]

        # Energy per distance (driving only)
        data_driving_df['driving_energy_per_distance'] = data_driving_df['accumulatedkwh_diffs'] / data_driving_df['accumulated_distance_diffs']

        # Regen energy per distance (negative values to visualize recovery)
        data_driving_df['regen_energy_per_distance'] = -regen_diff / data_driving_df['accumulated_distance_diffs']

        # Net driving+regen per distance using the SAME distance once
        data_driving_df['net_energy_per_distance'] = (
            data_driving_df['accumulatedkwh_diffs'] - regen_diff
        ) / data_driving_df['accumulated_distance_diffs']

        # Build combined for binned stats
        data_driving_with_regen_df = data_driving_df.copy()

        driving_distance_sum = data_driving_df['accumulated_distance_diffs'].sum()
        e_per_d_driving_total = np.nan if driving_distance_sum == 0 else data_driving_df['accumulatedkwh_diffs'].sum() / driving_distance_sum

        net_energy_sum = (data_driving_df['accumulatedkwh_diffs'] - regen_diff).sum()
        e_per_d_driving_and_regen_total = np.nan if driving_distance_sum == 0 else net_energy_sum / driving_distance_sum
        
        # Bin by speed
        bins = np.linspace(0, 70, 8)
        data_driving_df['binned'] = pd.cut(data_driving_df['speed'], bins)
        data_driving_with_regen_df['binned'] = pd.cut(data_driving_with_regen_df['speed'], bins)
        
        binned_dist = data_driving_df.groupby('binned', observed=False)['accumulated_distance_diffs'].sum()
        binned_drive_energy = data_driving_df.groupby('binned', observed=False)['accumulatedkwh_diffs'].sum()
        binned_regen_energy = regen_diff.groupby(data_driving_df['binned']).sum()

        binned_e_per_d_driving = binned_drive_energy / binned_dist
        binned_e_per_d_driving_dict[name] = binned_e_per_d_driving

        binned_net_energy = binned_drive_energy - binned_regen_energy
        binned_e_per_d_driving_and_regen = binned_net_energy / binned_dist
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
        ax.plot(data_driving_df['speed'], data_driving_df['regen_energy_per_distance'], 'o', color='green', markersize=1, label='Regen Energy')
        
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
    # Generate color gradients dynamically based on number of datasets
    n_datasets = len(names)
    blue_gradient = [plt.cm.Blues(0.4 + 0.6 * i / max(1, n_datasets - 1)) for i in range(n_datasets)]
    green_gradient = [plt.cm.Greens(0.4 + 0.6 * i / max(1, n_datasets - 1)) for i in range(n_datasets)]

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
    print(f"  ✓ Saved {DATASET_TYPE} combined energy per distance plot")

def prepare_driving_charging_data(top_dir, names):
    """Prepare data with driving and charging event labels."""
    print("\n" + "="*70)
    print("PREPARING DRIVING/CHARGING EVENT DATA")
    print("="*70)
    
    output_dir = get_output_dir(top_dir, 'data')
    
    # Define time gap thresholds (in minutes) to split events
    # If there's a gap larger than this, it starts a new event
    TIME_GAP_THRESHOLD_CHARGING = 30  # 30 minutes for charging
    TIME_GAP_THRESHOLD_DRIVING = 60   # 60 minutes for driving (increased from 15 min)
    
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
            # For messy_middle, map activities intelligently:
            # - 'charging' stays 'charging'
            # - 'driving' stays 'driving'
            # - 'idling' and 'inactive' are treated as missing for activity assignment
            # but we'll propagate the previous charging/driving state through these periods
            data_df['activity'] = data_df['truck_activity'].copy()
            data_df.loc[data_df['activity'] == 'charging', 'activity'] = 'charging'
            data_df.loc[data_df['activity'] == 'driving', 'activity'] = 'driving'
            # Forward fill to carry charging/driving through idling/inactive periods
            data_df['activity'] = data_df['activity'].ffill()
        
        data_df = data_df.dropna(subset=['activity'])
        
        # Convert timestamp to datetime for time gap calculation
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        
        # Add event numbers
        data_df['charging_event'] = np.nan
        data_df['driving_event'] = np.nan
        
        charging_event = 0
        driving_event = 0
        prev_activity = data_df.at[data_df.index[0], 'activity']
        prev_soc = -1.
        prev_timestamp = data_df.at[data_df.index[0], 'timestamp']
        force_new_charging_event = False
        force_new_driving_event = False

        for index, row in data_df.iterrows():
            current_activity = row['activity']
            current_soc = row['socpercent']
            current_timestamp = row['timestamp']
            
            # Calculate time gap in minutes from previous row
            time_gap_minutes = (current_timestamp - prev_timestamp).total_seconds() / 60.0
            
            # Check for large time gaps or SOC drops (indicates pause or charger off)
            large_gap_charging = time_gap_minutes > TIME_GAP_THRESHOLD_CHARGING
            large_gap_driving = time_gap_minutes > TIME_GAP_THRESHOLD_DRIVING
            soc_decreased_significantly = current_soc < (prev_soc - 2)  # 2% tolerance
            
            # If we were charging and now have a large gap or SOC drop, mark for new event
            if prev_activity == 'charging' and (large_gap_charging or soc_decreased_significantly):
                force_new_charging_event = True
            
            # If we were driving and now have a large gap, mark for new event
            if prev_activity == 'driving' and large_gap_driving:
                force_new_driving_event = True
            
            # Check for invalid charging (SOC decreases during charging)
            if current_activity == 'charging' and current_soc <= prev_soc:
                current_activity = np.nan
                data_df.at[index, 'activity'] = np.nan
                current_activity = 'driving'
            
            # Assign charging events with gap detection
            if current_activity == 'charging':
                # Start new charging event if:
                # 1. Coming from driving/idling/inactive activity, OR
                # 2. Force flag is set (large gap or SOC drop occurred)
                if (prev_activity not in ['charging'] or 
                    force_new_charging_event):
                    charging_event += 1
                    force_new_charging_event = False
                data_df.at[index, 'charging_event'] = charging_event
            
            # Assign driving events with gap detection
            if current_activity == 'driving':
                # Start new driving event if:
                # 1. Coming from charging/idling/inactive activity, OR
                # 2. Force flag is set (large gap occurred)
                if (prev_activity not in ['driving'] or
                    force_new_driving_event):
                    driving_event += 1
                    force_new_driving_event = False
                data_df.at[index, 'driving_event'] = driving_event
                
            prev_activity = current_activity
            prev_soc = current_soc
            prev_timestamp = current_timestamp
            
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
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
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
            
            # Skip events with insufficient data or SOC change
            if len(data_df_event) < 10 or (data_df_event['socpercent'].max() - data_df_event['socpercent'].min()) < 50:
                continue
            
            # Skip events where energy data is invalid (all zeros/negatives, or no energy increase)
            energy_min = data_df_event['accumumlatedkwh'].min()
            energy_max = data_df_event['accumumlatedkwh'].max()
            if energy_max <= 0 or energy_max <= energy_min:
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
        
        # Skip if no valid charging events
        if len(battery_data_quad_df) == 0:
            print(f"  ⚠ No valid charging events found for {name}, skipping battery capacity analysis")
            continue
        
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
        plt.savefig(f'{plot_dir}/{name}_battery_capacity_summary.png', dpi=300)
        if not FAST_MODE:
            plt.savefig(f'{plot_dir}/{name}_battery_capacity_summary.pdf')
        plt.close()
        
        battery_capacities.append(weighted_mean_quadfit)
        print(f"  ✓ Saved battery capacity analysis for {name}")
    
    # Save battery capacities
    battery_capacity_save = pd.DataFrame({'Value': ['Mean', 'Standard Deviation']})
    for i, name in enumerate(names):
        battery_data_quadfit_df = pd.read_csv(f'{table_dir}/{name}_battery_data_quadfit.csv')
        
        # Skip trucks with no valid charging events
        if len(battery_data_quadfit_df) == 0:
            battery_capacity_save[name] = [np.nan, np.nan]
            continue
        
        weighted_mean = np.average(battery_data_quadfit_df['battery_size'], 
                                  weights=1./battery_data_quadfit_df['battery_size_unc']**2)
        weighted_std = np.sqrt(np.average((battery_data_quadfit_df['battery_size']-weighted_mean)**2, 
                                         weights=1./battery_data_quadfit_df['battery_size_unc']**2))
        battery_capacity_save[name] = [weighted_mean, weighted_std]
    
    battery_capacity_save['average'] = [np.mean(battery_capacities), np.std(battery_capacities)]
    battery_capacity_save.to_csv(f'{table_dir}/battery_capacities.csv')
    print(f"  ✓ Saved {DATASET_TYPE} battery capacities summary (quadratic fits)")
    
    # Save battery capacities from linear fits
    battery_capacity_save_linear = pd.DataFrame({'Value': ['Mean', 'Standard Deviation']})
    battery_capacities_linear = []
    for i, name in enumerate(names):
        battery_data_linearfit_df = pd.read_csv(f'{table_dir}/{name}_battery_data_linearfit.csv')
        
        # Skip trucks with no valid charging events
        if len(battery_data_linearfit_df) == 0:
            battery_capacity_save_linear[name] = [np.nan, np.nan]
            continue
        
        weighted_mean = np.average(battery_data_linearfit_df['battery_size'], 
                                  weights=1./battery_data_linearfit_df['battery_size_unc']**2)
        weighted_std = np.sqrt(np.average((battery_data_linearfit_df['battery_size']-weighted_mean)**2, 
                                         weights=1./battery_data_linearfit_df['battery_size_unc']**2))
        battery_capacity_save_linear[name] = [weighted_mean, weighted_std]
        battery_capacities_linear.append(weighted_mean)
    
    battery_capacity_save_linear['average'] = [np.mean(battery_capacities_linear), np.std(battery_capacities_linear)]
    battery_capacity_save_linear.to_csv(f'{table_dir}/battery_capacities_linear.csv')
    print(f"  ✓ Saved {DATASET_TYPE} battery capacities summary (linear fits)")
    
    # Also save to messy_middle_results directory
    if DATASET_TYPE == 'messy_middle':
        messy_results_dir = f'{top_dir}/messy_middle_results'
        os.makedirs(messy_results_dir, exist_ok=True)
        battery_capacity_save_linear.to_csv(f'{messy_results_dir}/battery_capacities_linear_summary.csv')
        print(f"  ✓ Saved battery capacities (linear fits) to messy_middle_results/")

def analyze_charging_time_dod(top_dir, names):
    """Analyze charging time and depth of discharge."""
    print("\n" + "="*70)
    print("ANALYZING CHARGING TIME AND DEPTH OF DISCHARGE")
    print("="*70)
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
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
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
    output_dir = get_output_dir(top_dir, 'data')
    plot_dir = get_output_dir(top_dir, 'plots')
    table_dir = get_output_dir(top_dir, 'tables')
    
    # Read battery capacities
    battery_capacity_df = pd.read_csv(f'{table_dir}/battery_capacities.csv')
    battery_capacity_linear_df = pd.read_csv(f'{table_dir}/battery_capacities_linear.csv')
    
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
        }
        
        drivecycle_data_df = pd.DataFrame(drivecycle_data_dict)
        # Collect per-event energy total deltas for CSV output
        energy_totals_rows = []
        # Collect DoD summary for messy_middle_results
        dod_summary_rows = []
        # Collect detailed drive cycle data for messy_middle_results
        detailed_drivecycle_dict = {}
        data_df = read_csv_cached(f'{output_dir}/{name}_with_driving_charging.csv', low_memory=False)
        
        # Also load the additional_cols file to get elevation/road_grade data
        data_df_additional = read_csv_cached(f'{output_dir}/{name}_additional_cols.csv', low_memory=False)
        
        battery_capacity = battery_capacity_df[name].iloc[0]
        battery_capacity_unc = battery_capacity_df[name].iloc[1]
        # Linear-fit capacity for battery energy change reporting
        battery_capacity_linear = battery_capacity_linear_df[name].iloc[0]
        
        n_driving_events = int(data_df['driving_event'].max())
        for driving_event in range(1, n_driving_events):
            cDrivingEvent = (data_df['driving_event'] == driving_event)
            data_df_event = data_df[cDrivingEvent].dropna(subset=['socpercent', 'accumulated_distance'])
            data_df_event_full = data_df_event.copy()

            # Apply DoD/distance/points filters on the FULL event before any elevation trimming
            dod_full = data_df_event['socpercent'].max() - data_df_event['socpercent'].min()
            dist_full = data_df_event['accumulated_distance'].max() - data_df_event['accumulated_distance'].min()
            if (
                len(data_df_event) < MIN_DRIVING_EVENT_POINTS
                or dod_full < MIN_DRIVING_DOD
                or dist_full < MIN_DRIVING_DISTANCE
            ):
                continue
            
            # Check for multiple gvw values - skip if gvw varies during event
            if 'weight_kg' in data_df_event.columns:
                unique_gvws = data_df_event['weight_kg'].dropna().unique()
                if len(unique_gvws) > 1:
                    print(f"  Skipping event {driving_event}: Multiple gvw values detected ({len(unique_gvws)} unique values)")
                    continue
            
            # Trim to the largest continuous block with elevation data (all datasets)
            elevation_valid_mask = pd.Series(True, index=data_df_event.index)  # default: all valid
            elevation_col_trim = get_column_name(data_df_event, ['elevation_final_m', 'elevation_meters'])
            if elevation_col_trim and elevation_col_trim in data_df_event.columns:
                has_elevation = data_df_event[elevation_col_trim].notna()
                if has_elevation.sum() > 0:
                    elev_groups = (has_elevation != has_elevation.shift()).cumsum()
                    largest_block = has_elevation.groupby(elev_groups).sum().idxmax()
                    block_mask = (elev_groups == largest_block)
                    elevation_valid_mask = block_mask.copy()
                    data_df_event = data_df_event[block_mask].copy()

                    # Within the elevation block, interpolate any occasional NaNs
                    data_df_event[elevation_col_trim] = data_df_event[elevation_col_trim].interpolate(
                        method='linear', limit_direction='both'
                    )
                    if 'elevation_smooth' in data_df_event.columns:
                        data_df_event['elevation_smooth'] = data_df_event['elevation_smooth'].interpolate(
                            method='linear', limit_direction='both'
                        )

            # If trimming removed everything, skip this cycle
            if data_df_event.empty:
                continue
            
            # Convert timestamp to datetime and calculate time_elapsed
            data_df_event = data_df_event.copy()
            data_df_event_full = data_df_event_full.copy()
            data_df_event['timestamp'] = pd.to_datetime(data_df_event['timestamp'])
            data_df_event_full['timestamp'] = pd.to_datetime(data_df_event_full['timestamp'])
            # Use the full-event start as the common zero so backgrounds and trimmed overlays align
            start_time_global = data_df_event_full['timestamp'].iloc[0]
            data_df_event['time_elapsed'] = data_df_event.apply(calculate_time_elapsed, axis=1, args=(start_time_global,))
            data_df_event_full['time_elapsed'] = data_df_event_full.apply(
                calculate_time_elapsed, axis=1, args=(start_time_global,)
            )
            
            # Get initial and final values
            battery_charge_init = data_df_event['socpercent'].iloc[0]
            battery_charge_final = data_df_event['socpercent'].iloc[-1]
            distance_init = data_df_event['accumulated_distance'].iloc[0]
            distance_final = data_df_event['accumulated_distance'].iloc[-1]
            
            dod = battery_charge_init - battery_charge_final
            distance_traveled = distance_final - distance_init
            
            # Extrapolate to full 100% DoD
            if dod > 0:
                truck_range = (distance_traveled / dod) * 100
            else:
                truck_range = 0
            
            # Calculate energy economy
            delta_battery_energy = dod * battery_capacity / 100.
            
            if distance_traveled > 0:
                fuel_economy = delta_battery_energy / distance_traveled
            else:
                fuel_economy = 0
            
            new_row = {
                'Driving event': int(driving_event),
                'Initial battery charge (%)': battery_charge_init,
                'Final battery charge (%)': battery_charge_final,
                'Depth of Discharge (%)': dod,
                'Range (miles)': truck_range,
                'Fuel economy (kWh/mile)': fuel_economy,
            }
            
            new_row_df = pd.DataFrame([new_row])
            drivecycle_data_df = pd.concat([drivecycle_data_df, new_row_df], ignore_index=True)
            
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
            
            fig.text(0.15, 0.45, f'Range: {truck_range:.1f} miles\nEnergy Economy: {fuel_economy:.2f} kWh/mile', 
                    fontsize=20, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7))
            
            axs[0].set_xlim(0, 100)
            axs[1].set_xlim(0, 100)
            
            plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.png', dpi=300)
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_battery_soc_vs_distance_event_{driving_event}_linearfit.pdf')
            plt.close()
            
            # Plot 1b: Basic speed vs. driving time plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title(f"{name.replace('_', ' ').capitalize()}: Driving Event {driving_event}", fontsize=18)
            ax.set_ylabel('Speed (mph)', fontsize=18)
            ax.set_xlabel('Drive time (minutes)', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            ax.plot(data_df_event['time_elapsed'], data_df_event['speed'])
            
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + (ymax-ymin)*0.6)
            total_drive_time = (data_df_event['time_elapsed'].max() - data_df_event['time_elapsed'].min())
            total_drive_time_hours = total_drive_time / 60.
            
            plt.text(0.45, 0.65, f'Total drive time: {total_drive_time_hours:.1f} hours\nInitial Battery Charge: {battery_charge_init:.1f}%\nFinal Battery Charge: {battery_charge_final:.1f}%\nDepth of Discharge: {dod:.1f}%', 
                    transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
            
            plt.savefig(f'{plot_dir}/{name}_drive_cycle_{driving_event}.png')
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_drive_cycle_{driving_event}.pdf')
            plt.close()
            
            # Plot 2: Speed vs. driving time (paper-format drive cycle)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(20))
            ax.grid(which='minor', linestyle='--', linewidth=0.5, color='gray')
            ax.grid(which='major', linestyle='-', linewidth=0.5, color='black')
            ax.set_ylabel('Speed (mph)', fontsize=24)
            ax.set_xlabel('Driving time (minutes)', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            ax.plot(data_df_event['time_elapsed'], data_df_event['speed'], linewidth=2)
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{name}_drive_cycle_{driving_event}_paper.png')
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_drive_cycle_{driving_event}_paper.pdf')
            plt.close()
            
            # Plot 3: Speed vs. state of charge
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_ylabel('Speed (mph)', fontsize=18)
            ax.set_xlabel('State of charge (%)', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            data_df_event_plot = data_df_event.dropna(subset=['socpercent'])
            ax.plot(data_df_event_plot['socpercent'], data_df_event_plot['speed'])
            
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + (ymax-ymin)*0.6)
            
            total_drive_time = (data_df_event['time_elapsed'].max() - data_df_event['time_elapsed'].min())
            total_drive_time_hours = total_drive_time / 60.
            
            plt.text(0.45, 0.65, f'Total drive time: {total_drive_time_hours:.1f} hours\nInitial Battery Charge: {battery_charge_init:.1f}%\nFinal Battery Charge: {battery_charge_final:.1f}%\nDepth of Discharge: {dod:.1f}%', 
                    transform=plt.gcf().transFigure, bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7), fontsize=16)
            
            plt.savefig(f'{plot_dir}/{name}_drive_cycle_soc_{driving_event}.png')
            if not FAST_MODE:
                plt.savefig(f'{plot_dir}/{name}_drive_cycle_soc_{driving_event}.pdf')
            plt.close()
            
            # Plot 4: Three-panel elevation/grade/speed plot
            # Get corresponding data from additional_cols (which has elevation/road_grade)
            # Use .loc instead of .iloc to match by index labels, not positions
            try:
                data_df_event_with_elevation = data_df_additional.loc[data_df_event.index]
                # Also grab the full event (pre-elevation-trim) for semi-opaque background traces
                data_df_event_full_with_elevation = data_df_additional.loc[data_df_event_full.index]
            except Exception as e:
                print(f"    WARNING Drive cycle {driving_event}: Failed to index additional data: {e}")
                continue
            
            elevation_col = get_column_name(data_df_event_with_elevation, ['elevation_final_m', 'elevation_meters'])
            if elevation_col and 'road_grade_percent' in data_df_event_with_elevation.columns:
                # Check if we have valid elevation and grade data
                elevation_valid = data_df_event_with_elevation[elevation_col].notna().sum() > 0
                grade_valid = data_df_event_with_elevation['road_grade_percent'].notna().sum() > 0
                
                if 'saia' in name.lower():
                    print(f"    Drive cycle {driving_event}: elevation_col={elevation_col}, "
                          f"elevation_valid={elevation_valid} (n={data_df_event_with_elevation[elevation_col].notna().sum()}), "
                          f"grade_valid={grade_valid}")
                else:
                    print(f"    Drive cycle {driving_event}: elevation_valid={elevation_valid}, grade_valid={grade_valid}")
                
                if elevation_valid and grade_valid:
                    fig, axs = plt.subplots(6, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1]})
                    
                    # For grade panel: exclude edges to eliminate boundary artifacts
                    # Use max of 10 points or 5% of cycle length (whichever larger), capped at 25% of cycle
                    edge_buffer = max(10, int(len(data_df_event) * 0.05))
                    edge_buffer = min(edge_buffer, int(len(data_df_event) * 0.25))
                    valid_idx = list(range(edge_buffer, len(data_df_event) - edge_buffer)) if len(data_df_event) > edge_buffer * 2 else list(range(len(data_df_event)))

                    # Apply the same edge trimming to all panels so time axes stay aligned
                    data_df_event_core = data_df_event.iloc[valid_idx].copy()
                    data_df_event_with_elevation_core = data_df_event_with_elevation.iloc[valid_idx].copy()
                    
                    # Optional background overlays: full event (pre elevation trim), semi-opaque (all datasets)
                    background_available = False
                    try:
                        bg_time = data_df_event_full['time_elapsed']
                        bg_speed = data_df_event_full['speed']
                        bg_elev_raw = data_df_event_full_with_elevation[elevation_col] if elevation_col in data_df_event_full_with_elevation.columns else None
                        background_available = True
                    except Exception:
                        background_available = False

                    # Mask for periods WITH elevation data (for opacity control)
                    has_elevation_mask = data_df_event_with_elevation_core[elevation_col].notna()
                    
                    # Panel 1: Speed vs time (overlay raw and smoothed)
                    data_speed_plot = data_df_event_core.dropna(subset=['speed'])
                    # Raw speed background across full event
                    if background_available:
                        axs[0].plot(
                            bg_time,
                            bg_speed,
                            linewidth=1.5,
                            color='steelblue',
                            alpha=0.35,
                        )
                    else:
                        axs[0].plot(
                            data_speed_plot['time_elapsed'],
                            data_speed_plot['speed'],
                            linewidth=1.5,
                            color='steelblue',
                            alpha=0.35,
                        )
                    # Smoothed speed: opaque where elevation exists, semi-opaque where missing
                    if 'speed_smooth' in data_df_event_with_elevation_core.columns:
                        speed_smooth_series = data_df_event_with_elevation_core['speed_smooth']
                        speed_smooth_with_elev = speed_smooth_series[has_elevation_mask]
                        axs[0].plot(
                            data_df_event_core['time_elapsed'][has_elevation_mask],
                            speed_smooth_with_elev,
                            linewidth=2,
                            color='royalblue',
                        )
                        no_elev_mask = ~has_elevation_mask
                        if no_elev_mask.sum() > 0:
                            axs[0].plot(
                                data_df_event_core['time_elapsed'][no_elev_mask],
                                speed_smooth_series[no_elev_mask],
                                linewidth=2,
                                color='royalblue',
                                alpha=0.2,
                            )
                    axs[0].set_ylabel('Speed (mph)', fontsize=16)
                    axs[0].grid(True, alpha=0.3)
                    axs[0].tick_params(axis='both', which='major', labelsize=12)
                    if background_available:
                        axs[0].set_xlim(bg_time.min(), bg_time.max())
                    
                    # Panel 2: Elevation vs time
                    # Plot elevation_smooth for ALL samples to accurately represent what's used in grade calculation
                    elev_smooth_available = 'elevation_smooth' in data_df_event_with_elevation_core.columns and \
                        data_df_event_with_elevation_core['elevation_smooth'].notna().sum() > 0

                    # Plot smoothed elevation (dark orange) with opacity tied to elevation availability
                    if elev_smooth_available:
                        elev_smooth_vals = data_df_event_with_elevation_core['elevation_smooth']
                        elev_smooth_with_data = elev_smooth_vals[has_elevation_mask]
                        axs[1].plot(
                            data_df_event_core['time_elapsed'][has_elevation_mask],
                            elev_smooth_with_data,
                            linewidth=2,
                            color='darkorange',
                            label='elevation_smooth (regions with elevation data)'
                        )
                        no_elev_mask = ~has_elevation_mask
                        if no_elev_mask.sum() > 0:
                            axs[1].plot(
                                data_df_event_core['time_elapsed'][no_elev_mask],
                                elev_smooth_vals[no_elev_mask],
                                linewidth=2,
                                color='darkorange',
                                alpha=0.2,
                                label='elevation_smooth (extrapolated, no data)'
                            )
                        
                        n_zeros_smooth = (elev_smooth_vals == 0).sum()
                        n_const = (elev_smooth_vals == elev_smooth_vals.iloc[0]).sum() if len(elev_smooth_vals) > 0 else 0
                        print(f"    DEBUG Drive cycle {driving_event}: Plotting elevation_smooth")
                        print(f"      elevation_smooth samples: {len(elev_smooth_vals)}")
                        print(f"      elevation_smooth range: [{elev_smooth_vals.min():.2f}, {elev_smooth_vals.max():.2f}] m")
                        print(f"      elevation_smooth zeros: {n_zeros_smooth}, constant values: {n_const}")
                        if 'saia' in name.lower():
                            print(f"      Samples with elevation data: {has_elevation_mask.sum()} / {len(has_elevation_mask)}")
                    
                    # Also overlay raw elevation data if available (sparse DEM points)
                    elev_mask = data_df_event_with_elevation_core[elevation_col].notna()
                    if elev_mask.sum() > 0:
                        time_for_raw = data_df_event_core.loc[elev_mask, 'time_elapsed']
                        raw_elev_values = data_df_event_with_elevation_core.loc[elev_mask, elevation_col]
                        axs[1].plot(
                            time_for_raw,
                            raw_elev_values,
                            linewidth=1.5,
                            color='orange',
                            alpha=0.35,
                            label=f'{elevation_col} (sparse DEM points, n={elev_mask.sum()})'
                        )
                    axs[1].set_ylabel('Elevation (meters)', fontsize=16)
                    axs[1].grid(True, alpha=0.3)
                    axs[1].tick_params(axis='both', which='major', labelsize=12)
                    if background_available:
                        axs[1].set_xlim(bg_time.min(), bg_time.max())
                    
                    # Panel 3: Road grade vs time (edges removed for consistency across panels)
                    grade_mask = data_df_event_with_elevation_core['road_grade_percent'].notna()
                    if grade_mask.sum() > 0:
                        time_for_grade = data_df_event_core.loc[grade_mask, 'time_elapsed'].values
                        grade_values = data_df_event_with_elevation_core.loc[grade_mask, 'road_grade_percent'].values
                        # Compute a "raw" grade overlay from unsmoothed elevation where possible
                        try:
                            elev_change_raw = data_df_event_with_elevation_core[elevation_col].diff()
                            dist_meters = data_df_event_with_elevation_core['distance_change_meters']
                            grade_raw = (elev_change_raw / dist_meters) * 100.0
                            raw_mask = grade_raw.notna()
                            axs[2].plot(
                                data_df_event_core.loc[raw_mask, 'time_elapsed'],
                                grade_raw.loc[raw_mask],
                                linewidth=1.0,
                                color='seagreen',
                                alpha=0.3,
                            )
                        except Exception:
                            pass
                        # Overlay the pre-smoothing grade if available
                        if 'road_grade_percent_raw' in data_df_event_with_elevation_core.columns:
                            raw_grade_mask = data_df_event_with_elevation_core['road_grade_percent_raw'].notna()
                            if raw_grade_mask.sum() > 0:
                                # For Saia: show only opaque where we have elevation data
                                if 'saia' in name.lower():
                                    raw_grade_with_elev = raw_grade_mask & has_elevation_mask
                                    if raw_grade_with_elev.sum() > 0:
                                        axs[2].plot(
                                            data_df_event_core.loc[raw_grade_with_elev, 'time_elapsed'],
                                            data_df_event_with_elevation_core.loc[raw_grade_with_elev, 'road_grade_percent_raw'],
                                            linewidth=1.2,
                                            color='mediumseagreen',
                                            alpha=0.35,
                                        )
                                else:
                                    axs[2].plot(
                                        data_df_event_core.loc[raw_grade_mask, 'time_elapsed'],
                                        data_df_event_with_elevation_core.loc[raw_grade_mask, 'road_grade_percent_raw'],
                                        linewidth=1.2,
                                        color='mediumseagreen',
                                        alpha=0.35,
                                    )
                        
                        # For Saia: plot final grade only in regions with elevation data
                        grade_with_elev = grade_mask & has_elevation_mask
                        if grade_with_elev.sum() > 0:
                            axs[2].plot(
                                data_df_event_core.loc[grade_with_elev, 'time_elapsed'],
                                data_df_event_with_elevation_core.loc[grade_with_elev, 'road_grade_percent'],
                                linewidth=2,
                                color='darkgreen',
                                label='Road grade (regions with elevation data)'
                            )
                        grade_no_elev = grade_mask & (~has_elevation_mask)
                        if grade_no_elev.sum() > 0:
                            axs[2].plot(
                                data_df_event_core.loc[grade_no_elev, 'time_elapsed'],
                                data_df_event_with_elevation_core.loc[grade_no_elev, 'road_grade_percent'],
                                linewidth=2,
                                color='darkgreen',
                                alpha=0.2,
                                label='Road grade (extrapolated, no data)'
                            )

                    axs[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                    axs[2].set_ylabel('Road Grade (%)', fontsize=16)
                    axs[2].grid(True, alpha=0.3)
                    axs[2].tick_params(axis='both', which='major', labelsize=12)
                    axs[2].set_ylim(-10, 10)
                    if background_available:
                        axs[2].set_xlim(bg_time.min(), bg_time.max())

                    # Panel 4: GVW (weight_kg) vs time
                    if 'saia' in name.lower():
                        data_df_event_with_elevation_core['weight_kg'] = 31000.0
                        data_df_event_core['weight_kg'] = 31000.0
                    gvw_mask = data_df_event_with_elevation_core['weight_kg'].notna()
                    if gvw_mask.sum() > 0:
                        time_for_gvw = data_df_event_core.loc[gvw_mask, 'time_elapsed']
                        gvw_values = data_df_event_with_elevation_core.loc[gvw_mask, 'weight_kg']
                        axs[3].plot(time_for_gvw, gvw_values, linewidth=2, color='firebrick')
                    axs[3].set_ylabel('GVW (kg)', fontsize=16)
                    axs[3].grid(True, alpha=0.3)
                    axs[3].tick_params(axis='both', which='major', labelsize=12)
                    if background_available:
                        axs[3].set_xlim(bg_time.min(), bg_time.max())

                    # Panel 5: Battery energy delta vs time (delta SOC converted to kWh, with outlier detection)
                    battery_capacity_linear = battery_capacity_linear_df[name].iloc[0]
                    soc_col = get_column_name(data_df_event_with_elevation_core, ['socpercent', 'battery_percent'])
                    
                    # Compute raw delta SOC and apply 5-std outlier filtering
                    raw_battery_energy_delta = None
                    battery_energy_delta_clean = None
                    soc_upper_thr = None
                    soc_lower_thr = None
                    soc_mean_delta = None
                    if soc_col:
                        soc_mask = data_df_event_with_elevation_core[soc_col].notna()
                        if soc_mask.sum() > 0:
                            soc_values = data_df_event_with_elevation_core.loc[soc_mask, soc_col]
                            delta_soc_values = soc_values.diff()
                            # Convert delta SOC (%) to battery energy delta (kWh)
                            raw_battery_energy_delta = delta_soc_values * battery_capacity_linear / 100.0
                            
                            # Apply 5-std outlier filtering to battery energy deltas
                            battery_energy_delta_clean = raw_battery_energy_delta.copy()
                            outlier_mask_soc = pd.Series(False, index=battery_energy_delta_clean.index)
                            valid_deltas = battery_energy_delta_clean.dropna()
                            if len(valid_deltas) > 0:
                                soc_mean_delta = valid_deltas.mean()
                                soc_std_delta = valid_deltas.std()
                                if soc_std_delta > 0:
                                    soc_upper_thr = soc_mean_delta + 5 * soc_std_delta
                                    soc_lower_thr = soc_mean_delta - 5 * soc_std_delta
                                    # Identify outliers: >5 std from mean
                                    outlier_mask_soc = (np.abs(battery_energy_delta_clean - soc_mean_delta) > 5 * soc_std_delta) & battery_energy_delta_clean.notna()
                                    # Replace outliers with NaN for interpolation
                                    battery_energy_delta_clean[outlier_mask_soc] = np.nan
                                    # Interpolate over outliers
                                    battery_energy_delta_clean = battery_energy_delta_clean.interpolate(method='linear', limit_direction='both')
                            
                            # Plot raw battery energy deltas across full period (semi-opaque, for context)
                            if raw_battery_energy_delta.notna().sum() > 0:
                                axs[4].plot(
                                    data_df_event_core.loc[soc_mask, 'time_elapsed'],
                                    raw_battery_energy_delta.loc[soc_mask],
                                    linewidth=1.2,
                                    color='plum',
                                    alpha=0.35,
                                )
                                # Visually mark outlier regions as symmetric full-width horizontal bands beyond ±5σ
                                if soc_upper_thr is not None and soc_lower_thr is not None and soc_mean_delta is not None:
                                    max_abs = max(
                                        abs(raw_battery_energy_delta.max(skipna=True) - soc_mean_delta),
                                        abs(raw_battery_energy_delta.min(skipna=True) - soc_mean_delta),
                                        soc_upper_thr - soc_mean_delta,
                                        soc_mean_delta - soc_lower_thr
                                    )
                                    pad = 0.1 * max_abs if max_abs > 0 else 0.1
                                    y_max_plot = soc_mean_delta + max_abs + pad
                                    y_min_plot = soc_mean_delta - max_abs - pad
                                    axs[4].axhspan(soc_upper_thr, y_max_plot, color='red', alpha=0.06, zorder=0)
                                    axs[4].axhspan(y_min_plot, soc_lower_thr, color='red', alpha=0.06, zorder=0)
                            
                            # Plot cleaned battery energy deltas (opaque where elevation data exists)
                            if battery_energy_delta_clean is not None:
                                elev_with_data = soc_mask & has_elevation_mask
                                if elev_with_data.sum() > 0:
                                    axs[4].plot(
                                        data_df_event_core.loc[elev_with_data, 'time_elapsed'],
                                        battery_energy_delta_clean.loc[elev_with_data],
                                        linewidth=2,
                                        color='darkorchid',
                                    )
                                no_elev = soc_mask & (~has_elevation_mask)
                                if no_elev.sum() > 0:
                                    axs[4].plot(
                                        data_df_event_core.loc[no_elev, 'time_elapsed'],
                                        battery_energy_delta_clean.loc[no_elev],
                                        linewidth=2,
                                        color='darkorchid',
                                        alpha=0.2,
                                    )
                            
                            # Set y-limits to symmetric range around mean using ±5σ envelope
                            if soc_upper_thr is not None and soc_lower_thr is not None and soc_mean_delta is not None:
                                max_abs = max(
                                    abs(raw_battery_energy_delta.max(skipna=True) - soc_mean_delta),
                                    abs(raw_battery_energy_delta.min(skipna=True) - soc_mean_delta),
                                    soc_upper_thr - soc_mean_delta,
                                    soc_mean_delta - soc_lower_thr
                                )
                                pad = 0.1 * max_abs if max_abs > 0 else 0.1
                                axs[4].set_ylim(soc_mean_delta - max_abs - pad, soc_mean_delta + max_abs + pad)
                    
                    axs[4].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                    axs[4].set_ylabel('Δ Battery Energy (kWh)', fontsize=16)
                    axs[4].grid(True, alpha=0.3)
                    axs[4].tick_params(axis='both', which='major', labelsize=12)
                    if background_available:
                        axs[4].set_xlim(bg_time.min(), bg_time.max())

                    # Panel 6: Instantaneous energy economy vs time (using negative battery energy delta)
                    # First, compute energy for FULL event (for raw data across full period)
                    # Use data_df_event_full (original full event before elevation trimming)
                    energy_df_full_orig = data_df_event_full.copy()
                    driving_energy_col_full = get_column_name(energy_df_full_orig, ['driving_energy_kwh', 'accumumlatedkwh'])
                    
                    # Raw driving energy deltas (no smoothing) for full event
                    raw_driving_diffs_full = energy_df_full_orig[driving_energy_col_full].diff()
                    raw_driving_diffs_full.loc[raw_driving_diffs_full <= 0] = np.nan
                    energy_df_full_orig['distance_diffs_full'] = energy_df_full_orig['accumulated_distance'].diff()
                    energy_mask_drive_raw_full = (
                        energy_df_full_orig['distance_diffs_full'].abs() > DISTANCE_UNCERTAINTY
                    ) & raw_driving_diffs_full.notna()
                    
                    # Apply 5-std outlier filtering to driving energy deltas (using full event)
                    driving_diffs_clean_full = raw_driving_diffs_full.copy()
                    outlier_mask_full = pd.Series(False, index=energy_df_full_orig.index)
                    upper_thr_full = None
                    lower_thr_full = None
                    mean_diff = None
                    if energy_mask_drive_raw_full.sum() > 0:
                        valid_diffs = raw_driving_diffs_full[energy_mask_drive_raw_full]
                        mean_diff = valid_diffs.mean()
                        std_diff = valid_diffs.std()
                        if std_diff > 0:
                            upper_thr_full = mean_diff + 5 * std_diff
                            lower_thr_full = mean_diff - 5 * std_diff
                            # Identify outliers: >5 std from mean
                            outlier_mask_full = energy_mask_drive_raw_full & (np.abs(raw_driving_diffs_full - mean_diff) > 5 * std_diff)
                            # Replace outliers with NaN for interpolation
                            driving_diffs_clean_full[outlier_mask_full] = np.nan
                            # Interpolate over outliers
                            driving_diffs_clean_full = driving_diffs_clean_full.interpolate(method='linear', limit_direction='both')

                    # Totals across the full driving event
                    total_driving_energy_delta_full_kwh = driving_diffs_clean_full.loc[energy_df_full_orig.index][energy_mask_drive_raw_full].sum(skipna=True) if energy_mask_drive_raw_full.sum() > 0 else np.nan
                    total_battery_energy_change_linear_kwh = (dod_full * battery_capacity_linear) / 100.0
                    
                    # Now restrict to elevation-valid region for analysis
                    # Use cleaned driving energy data (with outliers removed and interpolated)
                    energy_df = data_df_event_core.copy()
                    # For Saia: restrict energy calculations and plotting strictly to elevation-valid region
                    if 'saia' in name.lower():
                        energy_df = energy_df.loc[has_elevation_mask].copy()
                    driving_energy_col = get_column_name(energy_df, ['driving_energy_kwh', 'accumumlatedkwh'])
                    regen_energy_col = get_column_name(energy_df, ['energy_regen_kwh', 'accumumlatedkwh'])
                    if regen_energy_col is None:
                        regen_energy_col = driving_energy_col

                    # Use cleaned driving energy deltas (outliers removed and interpolated from full event)
                    # Map cleaned data from full event to current subset
                    energy_df['driving_energy_diffs'] = driving_diffs_clean_full.loc[energy_df.index]

                    # Compute distance deltas if not present
                    if 'accumulated_distance' in energy_df.columns:
                        energy_df['distance_diffs'] = energy_df['accumulated_distance'].diff()
                    else:
                        energy_df['distance_diffs'] = np.nan

                    # Ensure speed_smooth exists for smoothed-speed distance calculation
                    if 'speed_smooth' not in energy_df.columns:
                        speed_window_local = 81 if 'saia' in name.lower() else 15
                        energy_df['speed_smooth'] = energy_df['speed'].rolling(
                            window=speed_window_local, center=True, min_periods=1
                        ).mean()

                    # Distance deltas for instantaneous energy using smoothed speed
                    distance_diffs_inst = energy_df['distance_diffs'].copy()
                    if 'timestamp' in energy_df.columns:
                        dt_seconds = energy_df['timestamp'].diff().dt.total_seconds()
                        distance_diffs_inst = energy_df['speed_smooth'] * dt_seconds / 3600.0  # miles
                    energy_df['distance_diffs_inst'] = distance_diffs_inst

                    # Instantaneous energy = -delta_battery_energy / distance (negative because discharging is positive energy)
                    # Use cleaned battery energy deltas (with outliers removed and interpolated)
                    if battery_energy_delta_clean is not None:
                        energy_df['inst_energy_per_mile'] = (
                            -battery_energy_delta_clean / distance_diffs_inst
                        )
                    else:
                        energy_df['inst_energy_per_mile'] = np.nan

                    energy_mask = (
                        energy_df['distance_diffs_inst'].abs() > DISTANCE_UNCERTAINTY
                    ) & energy_df['inst_energy_per_mile'].notna()
                    
                    # Diagnostic: identify periods where inst energy is undefined
                    missing_inst_energy = ~energy_mask & (battery_energy_delta_clean.notna() if battery_energy_delta_clean is not None else False)
                    if missing_inst_energy.sum() > 0:
                        try:
                            pct_missing = (missing_inst_energy.sum() / len(energy_df)) * 100
                            print(f"    Drive cycle {driving_event}: {missing_inst_energy.sum()} samples ({pct_missing:.1f}%) excluded from inst energy")
                            # Show time ranges where inst energy is missing
                            missing_times = energy_df.loc[missing_inst_energy, 'time_elapsed']
                            if len(missing_times) > 0:
                                print(f"      Missing inst energy at times: {missing_times.min():.1f}-{missing_times.max():.1f} min")
                        except Exception as e:
                            print(f"      ERROR in diagnostics: {e}")
                    
                    # Restrict ALL visualizations to periods where inst energy is well-defined
                    energy_df_valid = energy_df.loc[energy_mask].copy()
                    
                    if energy_mask.sum() > 0:
                        final_vals = energy_df_valid['inst_energy_per_mile']
                        print(f"    Drive cycle {driving_event} - Final inst energy (well-defined region):")
                        print(f"      Distance diffs (smoothed speed) - min: {energy_df_valid['distance_diffs_inst'].min():.6f}, max: {energy_df_valid['distance_diffs_inst'].max():.6f}, mean: {energy_df_valid['distance_diffs_inst'].mean():.6f}")
                        print(f"      Final inst energy values - min: {final_vals.min(skipna=True):.4f}, max: {final_vals.max(skipna=True):.4f}, mean: {final_vals.mean(skipna=True):.4f}, count: {final_vals.count()}")
                        
                        axs[5].plot(
                            energy_df_valid['time_elapsed'],
                            energy_df_valid['inst_energy_per_mile'],
                            linewidth=2,
                            color='purple',
                            label='Instantaneous energy'
                        )
                        # Set y-limits for readability
                        y_min_inst = energy_df_valid['inst_energy_per_mile'].min(skipna=True)
                        y_max_inst = energy_df_valid['inst_energy_per_mile'].max(skipna=True)
                        pad_inst = 0.1 * (y_max_inst - y_min_inst) if (y_max_inst - y_min_inst) > 0 else 0.1
                        axs[5].set_ylim(y_min_inst - pad_inst, y_max_inst + pad_inst)
                    
                    axs[5].set_ylabel('Inst. energy (kWh/mile)', fontsize=16)
                    axs[5].set_xlabel('Drive time (minutes)', fontsize=16)
                    axs[5].grid(True, alpha=0.3)
                    axs[5].tick_params(axis='both', which='major', labelsize=12)
                    if background_available:
                        axs[5].set_xlim(bg_time.min(), bg_time.max())
                    
                    # Add title
                    fig.suptitle(f"{name.replace('_', ' ').capitalize()}: Drive Cycle {driving_event} - Speed, Elevation, and Grade", 
                                fontsize=16, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(f'{plot_dir}/{name}_drive_cycle_{driving_event}_elevation_grade.png', dpi=300)
                    if not FAST_MODE:
                        plt.savefig(f'{plot_dir}/{name}_drive_cycle_{driving_event}_elevation_grade.pdf')
                    plt.close()
                    
                    # Collect DoD summary for messy_middle_results
                    dod_summary_rows.append({
                        'Driving event': int(driving_event),
                        'Initial SoC (%)': battery_charge_init,
                        'Final SoC (%)': battery_charge_final,
                        'Depth of Discharge (%)': dod,
                    })
                    
                    # Collect detailed drive cycle data for messy_middle_results
                    if DATASET_TYPE == 'messy_middle':
                        # Build detailed CSV with final selected time periods
                        # Need to get road_grade from the elevation dataframe that corresponds to energy_df_valid indices
                        try:
                            grade_values = data_df_event_with_elevation_core.loc[energy_df_valid.index, 'road_grade_percent'].values  # Using smoothed grade
                        except:
                            grade_values = np.full(len(energy_df_valid), np.nan)
                        
                        try:
                            gvw_values = data_df_event_with_elevation_core.loc[energy_df_valid.index, 'weight_kg'].values
                        except:
                            gvw_values = np.full(len(energy_df_valid), np.nan)
                        
                        try:
                            soc_values = energy_df_valid[soc_col].values if soc_col and soc_col in energy_df_valid.columns else np.full(len(energy_df_valid), np.nan)
                        except:
                            soc_values = np.full(len(energy_df_valid), np.nan)
                        
                        try:
                            delta_battery_values = battery_energy_delta_clean.loc[energy_df_valid.index].values if battery_energy_delta_clean is not None else np.full(len(energy_df_valid), np.nan)
                        except:
                            delta_battery_values = np.full(len(energy_df_valid), np.nan)
                        
                        # Use smoothed elevation if available, otherwise fall back to raw
                        try:
                            if 'elevation_smooth' in data_df_event_with_elevation_core.columns:
                                elevation_values = data_df_event_with_elevation_core.loc[energy_df_valid.index, 'elevation_smooth'].values
                            elif elevation_col and elevation_col in data_df_event_with_elevation_core.columns:
                                elevation_values = data_df_event_with_elevation_core.loc[energy_df_valid.index, elevation_col].values
                            else:
                                elevation_values = np.full(len(energy_df_valid), np.nan)
                        except:
                            elevation_values = np.full(len(energy_df_valid), np.nan)
                        
                        # Use smoothed speed if available, otherwise fall back to raw
                        speed_col_to_use = 'speed_smooth' if 'speed_smooth' in energy_df_valid.columns else 'speed'
                        
                        detailed_data = pd.DataFrame({
                            'Time (s)': (energy_df_valid['time_elapsed'] * 60).values,  # Convert minutes to seconds
                            'Speed (km/h)': (energy_df_valid[speed_col_to_use] * 1.60934).values,  # Convert mph to km/h (using smoothed)
                            'Elevation (m)': elevation_values,  # Using smoothed elevation
                            'Road Grade (%)': grade_values,  # Using smoothed grade
                            'GVW (kg)': gvw_values,
                            'State of Charge (%)': soc_values,
                            'Delta Battery Energy (kWh)': delta_battery_values,  # Using cleaned/filtered values
                            'Instantaneous Energy (kWh/mile)': energy_df_valid['inst_energy_per_mile'].values,  # Calculated from cleaned battery energy and smoothed speed; filtered for sufficient distance traveled
                        })
                        detailed_drivecycle_dict[f'event_{driving_event}'] = detailed_data

        drivecycle_data_df['Driving event'] = drivecycle_data_df['Driving event'].astype('int')
        drivecycle_data_df.to_csv(f'{table_dir}/{name}_drivecycle_data.csv', index=False)
        # Write per-event totals CSV for this dataset
        try:
            if len(energy_totals_rows) > 0:
                energy_totals_df = pd.DataFrame(energy_totals_rows)
                energy_totals_df.to_csv(f'{table_dir}/{name}_drivecycle_energy_totals.csv', index=False)
                print(f"  ✓ Saved per-event energy totals for {name}")
            else:
                print(f"  ⚠ No per-event energy totals to save for {name}")
        except Exception as e:
            print(f"  WARNING: Failed to write energy totals CSV for {name}: {e}")
        
        # Save messy_middle_results outputs
        if DATASET_TYPE == 'messy_middle':
            messy_results_dir = f'{top_dir}/messy_middle_results'
            os.makedirs(messy_results_dir, exist_ok=True)
            
            # Save DoD summary
            if len(dod_summary_rows) > 0:
                dod_summary_df = pd.DataFrame(dod_summary_rows)
                dod_summary_df.to_csv(f'{messy_results_dir}/{name}_dod_summary.csv', index=False)
                print(f"  ✓ Saved DoD summary to messy_middle_results/{name}_dod_summary.csv")
            
            # Save detailed drive cycle CSVs
            for event_key, event_data in detailed_drivecycle_dict.items():
                event_num = event_key.split('_')[1]
                event_data.to_csv(f'{messy_results_dir}/{name}_drivecycle_{event_num}_detailed.csv', index=False)
            if len(detailed_drivecycle_dict) > 0:
                print(f"  ✓ Saved {len(detailed_drivecycle_dict)} detailed drive cycle CSVs to messy_middle_results/")
            
            # Save drivecycle_data summary (using final selected periods)
            drivecycle_data_df.to_csv(f'{messy_results_dir}/{name}_drivecycle_data.csv', index=False)
            print(f"  ✓ Saved drivecycle data summary to messy_middle_results/{name}_drivecycle_data.csv")
        
        print(f"  ✓ Saved drive cycle analysis for {name}")
    
    # Plot range and fuel economy summaries
    for name in names:
        drivecycle_data_df = pd.read_csv(f'{table_dir}/{name}_drivecycle_data.csv')
        
        # Range summary
        mean_range = np.average(drivecycle_data_df['Range (miles)'])
        std_range = np.std(drivecycle_data_df['Range (miles)'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel('Extrapolated Range (miles)', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.xticks([])
        
        plt.scatter(range(len(drivecycle_data_df['Driving event'])), drivecycle_data_df['Range (miles)'], 
                    marker='o', s=100, color='blue', label='Extrapolated range')
        xmin, xmax = ax.get_xlim()
        
        ax.axhline(mean_range, color='blue', linewidth=2, label=rf'Mean: {mean_range:.1f}$\pm${std_range:.1f} miles')
        ax.fill_between(np.linspace(xmin, xmax, 100), mean_range-std_range, mean_range+std_range, 
                       color='blue', alpha=0.2, edgecolor='none')
        
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
        mean_fuel = np.average(drivecycle_data_df['Fuel economy (kWh/mile)'])
        std_fuel = np.std(drivecycle_data_df['Fuel economy (kWh/mile)'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylabel('Extrapolated Energy Economy (kWh/mile)', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.xticks([])
        
        plt.scatter(range(len(drivecycle_data_df['Driving event'])), drivecycle_data_df['Fuel economy (kWh/mile)'], 
                    marker='o', s=100, color='blue', label='Extrapolated energy economy')
        xmin, xmax = ax.get_xlim()
        
        ax.axhline(mean_fuel, color='blue', linewidth=2, label=rf'Mean: {mean_fuel:.2f}$\pm${std_fuel:.2f} kWh/mile')
        ax.fill_between(np.linspace(xmin, xmax, 100), mean_fuel-std_fuel, mean_fuel+std_fuel, 
                       color='blue', alpha=0.2, edgecolor='none')
        
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
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
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
        
        # Load selected charging and driving events
        charging_selected = set()
        try:
            charging_df = pd.read_csv(f'{table_dir}/{name}_charging_time_data.csv')
            charging_selected = set(charging_df['charging_event'].astype(int).values)
        except FileNotFoundError:
            pass
        
        driving_selected = set()
        try:
            driving_df = pd.read_csv(f'{table_dir}/{name}_drivecycle_data.csv')
            driving_selected = set(driving_df['Driving event'].astype(int).values)
        except FileNotFoundError:
            pass
        
        min_datetime = data_df['timestamp'].min()
        data_df['timestamp_hours'] = data_df['timestamp'].apply(
            lambda x: float((x - min_datetime).total_seconds() / SECONDS_PER_HOUR) if pd.notna(x) else np.nan
        )
        
        data_df = data_df[data_df['timestamp_hours'].notna()]
        data_df = data_df[~np.isinf(data_df['timestamp_hours'])]
        data_df_reduced = data_df.iloc[::100]
        
        # Create event selection masks
        data_df['charging_selected'] = data_df['charging_event'].isin(charging_selected) & (data_df['activity'] == 'charging')
        data_df['driving_selected'] = data_df['driving_event'].isin(driving_selected) & (data_df['activity'] == 'driving')
        
        # Plot speed and SOC vs time
        fig, axs = plt.subplots(2, 1, figsize=(18, 7), gridspec_kw={'height_ratios': [1, 1]})
        axs[0].plot(data_df_reduced.dropna(subset=['speed'])['timestamp_hours'], 
                   data_df_reduced.dropna(subset=['speed'])['speed'])
        axs[1].plot(data_df.dropna(subset=['socpercent', 'timestamp_hours'])['timestamp_hours'], 
                   data_df.dropna(subset=['socpercent', 'timestamp_hours'])['socpercent'])
        
        # Highlight unselected charging events (faded)
        cChargingUnselected = (data_df['activity'] == 'charging') & ~data_df['charging_selected']
        data_df['charging_unselected_timestamp'] = data_df['timestamp_hours'].where(cChargingUnselected, np.nan)
        
        ymin, ymax = axs[0].get_ylim()
        axs[0].fill_between(data_df['charging_unselected_timestamp'], ymin, ymax, color='green', alpha=0.1, edgecolor='none')
        axs[0].set_ylim(ymin, ymax)
        
        ymin, ymax = axs[1].get_ylim()
        axs[1].fill_between(data_df['charging_unselected_timestamp'], ymin, ymax, color='green', alpha=0.1, edgecolor='none')
        axs[1].set_ylim(ymin, ymax)
        
        # Highlight selected charging events (bright)
        cChargingSelected = data_df['charging_selected']
        data_df['charging_selected_timestamp'] = data_df['timestamp_hours'].where(cChargingSelected, np.nan)
        
        ymin, ymax = axs[0].get_ylim()
        axs[0].fill_between(data_df['charging_selected_timestamp'], ymin, ymax, color='darkgreen', alpha=0.4, edgecolor='none')
        axs[0].set_ylim(ymin, ymax)
        
        ymin, ymax = axs[1].get_ylim()
        axs[1].fill_between(data_df['charging_selected_timestamp'], ymin, ymax, color='darkgreen', alpha=0.4, edgecolor='none', label='Charging (selected)')
        axs[1].set_ylim(ymin, ymax)
        
        # Highlight unselected driving events (faded)
        cDrivingUnselected = (data_df['activity'] == 'driving') & ~data_df['driving_selected']
        data_df['driving_unselected_timestamp'] = data_df['timestamp_hours'].where(cDrivingUnselected, np.nan)
        
        ymin, ymax = axs[0].get_ylim()
        axs[0].fill_between(data_df['driving_unselected_timestamp'], ymin, ymax, color='purple', alpha=0.1, edgecolor='none')
        axs[0].set_ylim(ymin, ymax)
        
        ymin, ymax = axs[1].get_ylim()
        axs[1].fill_between(data_df['driving_unselected_timestamp'], ymin, ymax, color='purple', alpha=0.1, edgecolor='none')
        axs[1].set_ylim(ymin, ymax)
        
        # Highlight selected driving events (bright)
        cDrivingSelected = data_df['driving_selected']
        data_df['driving_selected_timestamp'] = data_df['timestamp_hours'].where(cDrivingSelected, np.nan)
        
        ymin, ymax = axs[0].get_ylim()
        axs[0].fill_between(data_df['driving_selected_timestamp'], ymin, ymax, color='indigo', alpha=0.4, edgecolor='none')
        axs[0].set_ylim(ymin, ymax)
        
        ymin, ymax = axs[1].get_ylim()
        axs[1].fill_between(data_df['driving_selected_timestamp'], ymin, ymax, color='indigo', alpha=0.4, edgecolor='none', label='Driving (selected)')
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
        
        # Add info box showing event counts
        n_charging_total = data_df[data_df['activity'] == 'charging']['charging_event'].nunique()
        n_charging_selected = len(charging_selected)
        n_driving_total = data_df[data_df['activity'] == 'driving']['driving_event'].nunique()
        n_driving_selected = len(driving_selected)
        
        info_text = f'Charging Events: {n_charging_selected}/{n_charging_total} selected\n'
        info_text += f'Driving Events: {n_driving_selected}/{n_driving_total} selected'
        
        axs[0].text(0.02, 0.98, info_text, transform=axs[0].transAxes, fontsize=14,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
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
    
    if NO_PLOT_MODE:
        print("(Skipping analysis - NO_PLOT_MODE is enabled)")
        return
    
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
    
    # # Stage 1: Data Preprocessing
    # preprocess_data(top_dir, files, names)
    
    # # Stage 1.5: Elevation and Road Grade Analysis
    # analyze_elevation_grade(top_dir, names)
    
    # # Stage 2: Charging Analysis
    # analyze_charging_power(top_dir, names)
    
    # # Stage 3: Energy Analysis
    # analyze_instantaneous_energy(top_dir, names)
    
    # # Stage 4: Prepare Driving/Charging Events
    # prepare_driving_charging_data(top_dir, names)
    
    # Stage 5: Battery Capacity Analysis
    # analyze_battery_capacity(top_dir, names)
    
    # # Stage 6: Charging Time & DoD
    # analyze_charging_time_dod(top_dir, names)
    
    # Stage 7: Drive Cycles
    analyze_drive_cycles(top_dir, names)
    
    # # Stage 8: VMT Analysis
    # analyze_vmt(top_dir, names)
    
    # # Stage 9: Energy Delivered
    # analyze_energy_delivered(top_dir, names)
    
    # ====================================================
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
