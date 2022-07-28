# Big Data Course (5294BIDA6Y): PySpark-DuckDB DBLP Project

This repository contains code corresponding to the analyses presented during the Final Project Symposium, in partial fulfillment for the degree of MSc. in Data Science at the University of Amsterdam.

## Executive Summary

Real-world data is often dirty, erroneous, and disaggregated over different files. This group project sought to create a scalable Machine Learning Pipeline to clean and store data from the DBLP computer science bibliography, and perform binary classification to accurately predict duplicate bibliographic entries. The proposed pipeline utilized PySpark to clean the text in parallelization and then stored results using DuckDB. Next, PySpark was used to compute Jaccard Similarity scores between listed article authors, keys, and titles, while a parallelized Random Forest Classifier was implemented for final classification. Results found that storing unique characters, as opposed to cleaned text, reduced SQL storage costs by 25% while maintaining an acceptable prediciton accuracy (79.58%). The presented poster can be found within the repo [link]((https://raw.github.com/jah377/PySpark_DBLP/main/poster.pdf))


## Setup

Installing python dependencies in a virtual environment

```bash
# Creation of the virtual environment
python -m venv ./bigd
```

And then you can mount this environment with: 
```bash
# when using fish
source ./bigd/bin/activate.fish
# or other (multiple activation scripts are in this folder)
source ./bigd/bin/activate
```

```bash
# Installation of dependencies
pip install -r requirements.txt
```

To start the spark cluster, use [docker-compose](https://docs.docker.com/compose/install/) (in Mac this comes out of the box with Docker Desktop, for others refer to the [Install Instructions](https://docs.docker.com/compose/install/))

```bash
# verify whether docker-compose is installed already, else install it
docker-compose -v
```


```bash
# to start all containers
docker-compose up 

# or to just start a dedicated container by name
docker-compose up spark
```
---
## Spark Overview
https://surfdrive.surf.nl/files/index.php/s/0gfxnfn1jnngoi6
