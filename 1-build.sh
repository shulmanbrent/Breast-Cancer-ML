#!/bin/bash

# image name
__image=shulmanbrent/ml_flask

# build image
docker build -t $__image .