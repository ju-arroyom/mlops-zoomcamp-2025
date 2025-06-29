import batch
import pandas as pd
from pandas import Timestamp
from datetime import datetime

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
    df = pd.DataFrame(data, columns=columns)
    return df

def test_prepare_data():
    input_data = prepare_input_data()
    expected_output = {'PULocationID': {0: '-1', 1: '1'},
                        'DOLocationID': {0: '-1', 1: '1'},
                        'tpep_pickup_datetime': {0: Timestamp('2023-01-01 01:01:00'),
                        1: Timestamp('2023-01-01 01:02:00')},
                        'tpep_dropoff_datetime': {0: Timestamp('2023-01-01 01:10:00'),
                        1: Timestamp('2023-01-01 01:10:00')},
                        'duration': {0: 9.0, 1: 8.0}}
    actual_output = batch.prepare_data(input_data, 
                                       categorical=['PULocationID', 'DOLocationID']).to_dict()
    assert expected_output==actual_output