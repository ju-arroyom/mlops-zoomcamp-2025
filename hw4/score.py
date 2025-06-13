import logging
import pickle
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

def read_data(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    filename = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:4d}-{month:02d}.parquet'
    logger.info(f"Creating prediction for {filename}")
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    # Fill na categorical vars
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def create_predictions(df):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    return y_pred

def create_results_df(y_pred, year, month):
    df = pd.DataFrame()
    df['predictions'] = y_pred
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df
    
def save_predictions(df_result, year, month):
    ouput_file = f"./output/yellow_taxi_{year:4d}_{month:02d}.parquet"
    logger.info(f"Writing output to {ouput_file}")
    df_result.to_parquet(ouput_file,
                         engine='pyarrow',
                         compression=None,
                         index=False)


def generate_predictions(year, month):
    # Read raw data
    df = read_data(year, month)
    # Load model
    logger.info("Feature Engineering & Model Prediction")
    y_pred = create_predictions(df)
    # Create results df
    df_result = create_results_df(y_pred, year, month)
    mean_preds = df_result['predictions'].mean()
    logger.info(f"Mean predicted duration: {mean_preds:.2f}")
    # Write results to output folder
    save_predictions(df_result, year, month)

