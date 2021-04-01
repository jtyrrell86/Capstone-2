# Can We Accurately Predict Energy Usage?
<div align="center">
        <img src="images/meter-1240897.jpg" width="" height="">
    </div>
<br>
## Background and Motivation
Reducing energy consumption, and thus CO2 emissions, is key to lessoning the worst impacts of climate change. According to recent data from the U.S. Energy Information Administration building energy usage between residential, commercial, and industrial sectors accounts for 71% of all energy
usage. In Many utility companies offer incentives to spur their customers to adopt energy efficiency upgrades. What incentives are offered depends on the predicted energy saving achieved and without an accurate baseline usage it’s impossible to accurately predict savings. I am using usage data from over 1,400 commercial and industrial buildings to create a model to predict future usage.

## Data Cleaning
The data I used for this project was from a Kaggle compitition, but it was actually provided by the American Society of Heating, Refrigerating and Air-Conditioning Engineers (ASHRAE) and their partners. The dataset consisted of six different CSV files. The building metadata csv contained a building_id, primary_use, square_feet, year_built, and floor count for 1,448 commercial and industrial buildings. Each building also had one of sixteen site_id's that tied it to the weather csv's.

<div align="center">
        <img src="images/building_meta_table" width="" height="">
    </div>
<br>

<div align="center">
        <img src="images/train_table" width="" height="">
    </div>
<br>

<div align="center">
        <img src="images/weather_table" width="" height="">
    </div>
<br>

<div align="center">
        <img src="images/train_table" width="" height="">
    </div>
<br>

The weather train csv contained a years worth of hour by hour air temperatives, cloud cover

## EDA
## Modeling Using Random Forest
<div align="center">
        <img src="images/Modeling_scores" width="" height="">
    </div>
<br>

<div align="center">
        <img src="images/feature_importance_plot.png" width="" height="">
    </div>
<br>

## Conclusions
## Goals for Capstone 3
* 