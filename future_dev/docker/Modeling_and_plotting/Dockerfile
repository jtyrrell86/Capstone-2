FROM python:3.7-stretch
MAINTAINER Justin Tyrrell <Justin.d.tyrrell@gmail.com> 

# install build utilities
RUN apt-get update && apt-get install -y gcc make apt-transport-https ca-certificates build-essential 

# check our python environment
RUN python3 --version
RUN pip3 --version 

# set the working directory for containers
WORKDIR /Users/justintyrrell/Documents/Galvanize_DSI/Capstone-2/docker/Modeling_and_plotting

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in needed files
COPY ../../src/plotting_and_models.py src/
COPY ../../data/chilledwater_subset.csv data/
COPY ../../data/electricity_subset.csv data/
COPY ../../data/steam_subset.csv data/
#COPY ../../data/combined_df.csv