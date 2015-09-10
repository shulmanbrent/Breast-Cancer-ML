# ML-webapp
Uses a neural network, decision tree, and logisitc regression to predict breast cancer from patient data

## Running the application
The best way to run the web application is using [Docker](https://www.docker.com/). If you do not have docker installed or dont know what it is, see below for instructions.

Once you have your [Docker](https://www.docker.com/) environment up and running, navigate to your local folder containing this repository

## Docker
From there you need to do the following three steps:
1. Run the ```./1-build.sh``` command
  - This will build the container as specified in the Docker file
  - It may take a litle bit initially, so be patient!
  - Once the container is built your will not need to build it again
2. Run the ``./2-run.sh``` command
  - This will drop you into a Command Line interface for the container with proper settings and port sharing enabled
3. Run ```cd /data && python application.py```.
  - This will start the applications server running in your browser!

## Without Docker
- #TODO
