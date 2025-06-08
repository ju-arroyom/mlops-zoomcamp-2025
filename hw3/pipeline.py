
import argparse
from ingest import read_dataframe
from features import create_X
from train import train_model


def main(year, month):
    
    df = read_dataframe(year, month)
    X, dv = create_X(df)
    # Define target
    target = 'duration'
    y = df[target].values 
    # Train model
    run_id = train_model(X, y, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()
    run_id = main(args.year, args.month)