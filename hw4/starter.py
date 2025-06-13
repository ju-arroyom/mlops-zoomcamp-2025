import logging
import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    # Fill na categorical vars
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def create_predictions(year, month):
    # Define categorical vars
    categorical = ['PULocationID', 'DOLocationID']
    # Read raw data
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:4d}-{month:02d}.parquet'
    logger.info(f"Creating prediction for {filename}")
    df = read_data(filename, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    # Load model
    logger.info("Loading Model")
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    # Prepare features
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    # Create results df
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred
    logger.info(f"Mean predicted duration: {y_pred.mean()}")
    # Write results to output folder
    ouput_file = f"./output/yellow_taxi_{year:4d}_{month:02d}.parquet"
    logger.info(f"Writing output to {ouput_file}")
    df_result.to_parquet(ouput_file,
                         engine='pyarrow',
                         compression=None,
                         index=False)
