import click 
from starter import create_predictions


@click.command()
@click.option('--year', required=True, type=int, help='Year to generate predictions for')
@click.option('--month', required=True, type=int, help='Month to generate predictions for')
def main(year, month):
    """CLI for generating predictions."""
    create_predictions(year, month)

if __name__ == '__main__':
    main()