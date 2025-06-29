#!/usr/bin/env bash

cd "$(dirname "$0")"

# Define year/month for test
YEAR=2023
MONTH=1
BUCKET=s3://nyc-duration

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then 
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="stream-model-duration:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    docker build -t ${LOCAL_IMAGE_NAME} ..
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

docker-compose up -d

# Create bucket
echo "Creating Bucket: ${BUCKET}"
aws --endpoint-url=http://localhost:4566 s3 mb $BUCKET

# Wait a little bit
sleep 1

# Create test data set
echo "Creating Dataset for ${YEAR}-0${MONTH}"
python integration_test.py $YEAR $MONTH

# Run batch.py inside container
echo "Running Prediction with batch.py"
docker-compose run backend $YEAR $MONTH

# Stop container
docker-compose down