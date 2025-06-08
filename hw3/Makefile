# Default values
YEAR=2023
MONTH=3

# Build all Docker images
build:
	docker-compose build

# Start only the MLflow server
mlflow-up:
	docker-compose up -d mlflow  # Start mlflow in detached mode

# Run the trainer pipeline (which will trigger MLflow tracking)
train:
	docker-compose run --rm trainer --year $(YEAR) --month $(MONTH)

# Bring up the full stack: mlflow (daemonized), then trainer (once)
up: mlflow-up train

# Tear down all services
down:
	docker-compose down

# Clean everything
clean:
	docker-compose down --volumes --remove-orphans

