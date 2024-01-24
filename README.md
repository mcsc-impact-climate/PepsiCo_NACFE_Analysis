# Analysis of PepsiCo Data from the NACFE Trucking Pilot

This repo contains some code to extract the following operational parameters for PepsiCo's Tesla Semi pilot in California using data published [here](https://runonless.com/run-on-less-electric-depot-reports/) by NACFE from their 2023 Run On Less pilot:
* Charging power
* Extrapolated battery capacity
* Extrapolated truck range
* Energy consumed per mile
* Charging time
* Depth of discharge
* Drive cycles

# Download the NACFE Run on Less Data

```bash
mkdir tmp
wget https://runonless.com/wp-content/uploads/ROL23-Web-data.zip -O tmp/RoL_Data.zip
unzip tmp/RoL_Data.zip -d tmp/RoL_Data
mv tmp/RoL_Data/"ROL23 Web data"/*.csv data
rm -r tmp
```

# Run the analysis

## Download needed python packages
```bash
pip install -r requirements.txt 
```

## Execute the analysis code
```bash
python source/AnalyzeData.py
```

Plots and tables produced by the analysis code should now be available in the `tables` and `plots` directories.
