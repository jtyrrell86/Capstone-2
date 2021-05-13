FROM python:3.7-stretch
MAINTAINER Justin Tyrrell <Justin.d.tyrrell@gmail.com> 

# install build utilities
RUN apt-get update && apt-get install -y gcc make apt-transport-https ca-certificates build-essential 

# check our python environment
RUN python3 --version
RUN pip3 --version 

# set the working directory for containers
WORKDIR /Users/justintyrrell/Documents/Galvanize_DSI/Capstone-2

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/cleaning.py src/cleaning.py
COPY data/ashrae-energy-prediction/building_metadata.csv data/building_metadata.csv
COPY data/ashrae-energy-prediction/train.csv data/train.csv
COPY data/ashrae-energy-prediction/weather_train.csv data/weather_train.csv