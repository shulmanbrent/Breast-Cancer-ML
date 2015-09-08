#!/bin/bash

# image name
__image=shulmanbrent/ml_flask
__volume_host=/$(pwd)
__volume_cntr=/data

# run image
docker run -it \
	--volume=$__volume_host:$__volume_cntr \
	--publish=5000:5000 \
	--publish=8888:8888 \
	$__image