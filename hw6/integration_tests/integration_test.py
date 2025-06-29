import os
import sys
import pandas as pd
from datetime import datetime

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def prepare_input_data():
    data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)
    return df_input

def save_data(df_input, input_path):
    # Set options to write to s3 localstack
    options = {
    'client_kwargs': {
        'endpoint_url': "http://localhost:4566"
        }
        }
    print(f"Saving Data to {input_path}")
    df_input.to_parquet(
        input_path,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

if __name__ == "__main__":
    print("Uploading input data to S3")
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    input_file = get_input_path(year, month)
    df_input = prepare_input_data()
    save_data(df_input, input_file)

