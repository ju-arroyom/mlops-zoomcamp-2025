#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import pandas as pd


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    # Change bucket name to match Q4
    default_output_pattern = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename):
    endpoint_url =  os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    options = {
    'client_kwargs': {
        'endpoint_url': endpoint_url
    }
    }
    return pd.read_parquet(filename, storage_options=options)

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def create_predictions(df):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = lr.predict(X)
    return y_pred

def create_results_df(y_pred, year, month):
    df = pd.DataFrame()
    df['predictions'] = y_pred
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

def save_data(df_input, output_path):
    # Set options to write to s3 localstack
    endpoint_url =  os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")
    options = {
    'client_kwargs': {
        'endpoint_url': endpoint_url
        }
        }
    df_input.to_parquet(
        output_path,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def main(year, month):
    # Global vars
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    categorical = ['PULocationID', 'DOLocationID']
    # Read data
    print(f"Reading Data from {input_file}")
    df_raw = read_data(input_file)
    # Prepare data
    df = prepare_data(df_raw, categorical)
    # Generate predictions
    y_pred = create_predictions(df)
    print('predicted mean duration:', y_pred.mean())
    print('sum of predicted durations:', y_pred.sum())
     # Create results df
    df_result = create_results_df(y_pred, year, month)
    # Save Results
    print(f"Saving Data to {output_file}")
    save_data(df_result, output_file)

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)