# Can We Accurately Model Building Energy Usage? (Being Edited)
<div align="center">
        <img src="images/meter-1240897.jpg" width="" height="">
    </div>
<br>

## Background and Motivation
I have spent almost half of my career in the energy efficiency industry. It's been a passion of mine since late in high school. What drove me to energy efficiency was an appreciation for the natural environment instilled in me as a child, a realization of detrimental impacts global climate change, and the huge impact reducing energy consumption can have.

Reducing energy consumption, and thus CO2 emissions, is key to lessoning the worst impacts of climate change. According to recent data from the U.S. Energy Information Administration building energy usage between residential, commercial, and industrial sectors accounts for 71% of all energy usage. Most utility companies offer financial incentives for their customers to reduce energy consumption in the form of rebates for certain energy efficiency upgrades. In the case of commercial and industrial this can be in the form of a cash rebate tied directly to how much energy is saved. In order to accurately calculate how much energy was save with any particular upgrade you need to know how much energy would have been used had that upgrade not been installed, a baseline usage. For this project I used past energy usage data for over 1,400 commercial and industrial buildings and weather data from 16 different sites to determine if I could model this baseline energy usage.

## Data
The data I used for this project was from a Kaggle competition hosted by the American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE). You can find a link to the competition site <a href="https://www.kaggle.com/c/ashrae-energy-prediction/overview">here</a>. The dataset consisted of five different csv files as well as a template for submitting predictions to Kaggle for scoring. 

### Metadata
The building metadata file had one row for each of the 1448 buildings with the below information:

| Feature Name  | Data Type  |  Description |
|---|---|---|
|  site_id | int  | Primary key for each of 16 sites that each building was located at  |
|  building_id | int  | Primary key for each building  |
|  primary_use | object(string)  | Indicator of the primary category of activities for the building based on EnergyStar property type definitions  |
|  square_feet | int  | Gross floor area of the building  |
| year_built | float  | Year building was opened  |
|  floor_count  | float  | Number of floors  |

Primary uses included Education, Entertainment/public assembly, Food sales and service, Healthcare, Lodging/residential, Manufacturing/industrial, Office, Other, Parking, Public services, Retail, Services, Technology/science, Utility, and Warehouse/storage.

### Training Data
The training data file consisted of over 20 million rows of hour-by-hour energy consumption readings for each building and each of its metered energy types for 2016.

| Feature Name  | Data Type  |  Description |
|---|---|---|
|  building_id | int  | Foreign key for the metadata.  |
|  meter | int  | The meter id code. Read as {0: electricity, 1: chilled water, 2: steam, 3: hot water}. Not every building has all meter types.  |
|  timestamp | object(string)  | When the reading was taken  |
|  meter_reading | float  | The target variable. Energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error. UPDATE: as discussed here, the site 0 electric meter readings are in kBTU.

### Weather Data
The weather data file consisted of 140,000 rows of hour-by-hour energy consumption readings for each of the 16 sites for 2016.

| Feature Name  | Data Type  |  Description |
|---|---|---|
|  site_id | int  | Foreign key for the metadata.  |
|  timestamp | object(string)  | When the reading was taken  |
|  air_temperature | float  | Degrees Celsius  |
|  cloud_coverage | float  | Portion of the sky covered in clouds, in oktas  |
|  dew_temperature  |  float  |  Degrees Celsius  |
|  precip_depth_1_hr  |  float  |  Millimeters  |
|  sea_level_pressure  |  float  |  Millibar/hectopascals  |
|  wind_direction  |  float  |  Compass direction (0-360)  |
|  wind_speed  |  float  |  Meters per second  |

### Test Meter Reading and Weather Data
The remaining two files mirrored the training and weather data files accept the data spanned 2017 and half of 2018 and it was over 40 million rows of data. It was also missing the target value, meter reading.

## Data Cleaning and EDA
Due to the very large size of both the training and test datasets, and my limited time, I choose to use just the training data initially. This would require me to alter the time stamp column in such a way that would keep its cyclical properties while allowing me to randomly split the data into a train and holdout set for validation.


After merging all three training data frames together I first took a look how many missing or null values I had.

<div align="center">
        <img src="images/percent_missing_values_plot.png" width="" height="">
    </div>
<br>

Of the 16 features I had floor count, year built, cloud coverage, precipitation depth each hour, wind direction, and sea level pressure were all missing a large number of values. The most disappointing being floor count because the height of a building, as well as the difference in temperature inside and outside, directly impacts heated air loss and thus energy usage. Wind speed, dew temperature, and air temperature where also missing values though not nearly as many. At least initially I choose to drop all columns with missing values with the exception air temperature, which I believed to be one of the most important features. 

In the case of air temperature, I choose to impute the missing values. In order to do so I first created a lookup table, from the weather data file, with the mean temperatures for each day at each of the 16 sites. I then created a function to iterate through my merged data frame and when it found missing air temperature values it used the site id and date to search the lookup table and input the mean value for that day.

 Next, I converted the timestamps to datetime format and then split out the date and time into their own columns. Afterword’s I used masking to replace each date with 1st Quarter, 2nd Quarter,3rd Quarter, or 4th Quarter. I then did the same thing with time, but replaced them with early morning, morning, afternoon, and evening. Lastly, I used one-hot encoding to convert these categorical values to numerical. This would allow me to split the data into a train and holdout set for validation.

After reformatting the timestamp, I used one-hot encoding on the primary usage and meter columns. I also took a look at the number of buildings with each primary use type and percentage of meter readings by metered energy type.

<div align="center">
        <img src="images/primary_use_bar_graph.png" width="" height="">
    </div>
<br>

With respect to primary use types, interestingly Education made up the majority of the buildings in this dataset.

<div align="center">
        <img src="images/meter_type_distribution.png" width="" height="">
    </div>
<br>

With respect to metered energy types, electricity made up more up more than 60 percent of the metered energy readings.

Lastly, I created a function to convert the meter readings in site 0 to kWh for usage in the model and then back to kBtu for submittal to Kaggle.

## Modeling Using Random Forest
As a baseline model I choose the random forest regressor because of its simplicity and accuracy with minimal hyperparameter tuning. I also chose to use Root Mean Squared Log Error (RMSLE) as my er tuning. I also chose to use Root Mean Squared Log Error (RMSLE) as my error metric because it doesn't allow large differences in the target values to skew the error value making for easier comparison. While doing EDA I noted that usage values for hot water were significantly higher than those for the other three metered energy types.

I initially decided to model each of the metered energy types individually thinking that may make for more accurate modeling. After modeling them this way though I realized there was a stark difference between the most important features for electricity and the other three metered energy types.

<div align="center">
        <img src="images/Electricity_feature_importance_bar_graph.png" width="" height="">
    </div>
<br>

For electricity square footage was the most important feature over all other features.

<div align="center">
        <img src="images/Water_feature_importance_bar_graph.png" width="" height="">
    </div>
<br>

For the other three however air temperature was also very important.

After remodeling these two subsets, trying multiple hyperparameter combinations, and then recombining the scores I got an RMSLE of **0.855**. This is in comparison to the winning RMSLE in the Kaggle competition of **0.931**.

## AWS Optimization
One of the major challenges of this project was running my code on the more than 20 million rows of data. As part of the additional work I did on this project I spent time learning how to create my own Dockerfile and image to be deployed on an AWS EC2 instance. After much trial and error, I was able to put my cleaning script on an EC2 instance. When run on my local machine this script took almost 7 hours to run, but by using an EC2 instance I was able to cut that down to just over 2 hours. This significant decrease in processing time will allow me to more effectively run new code in the future.

## Conclusions
With the minimal amount of time I had available for this project I was able to create model that predicts pretty well with an RMSLSE of 0.855 as compared to the winning RMSLE in the Kaggle competition of **0.931**. However, my model was only used on the training data whereas to be an accurate comparison I will need to run my model on the test data. That's something I'd like to do in the future.

By utilizing AWS EC2 instance I was able to decrease processing time of my cleaning script by over 50%. With the test data set being over 40 million rows of data utilizing an EC2 instance will be crucial.

When modeling electricity usage square footage was the most important feature though with the three water types, hot water, chilled water, and steam, air temperature was also very important. It is possible though that some of the features I removed could be important as well. With my cleaning script now running much quicker it may now be possible to import more of the weather features I removed.

## Future Direction
* Run my cleaning script and model on the test dataset using an AWS EC2 instance and then submit the predictions to Kaggle
* Depending on how this model predicts on the test data it could be useful to use a Grid Search to try other models
* Look into imputing the missing values from one or more of the dropped columns in the weather data

## Sources
* Data - https://www.kaggle.com/c/ashrae-energy-prediction/leaderboard
* Energy stats - https://www.eia.gov/
* Biden admins climate goals - https://www.nytimes.com/live/2021/04/22/us/biden-earth-day-climate-summit


