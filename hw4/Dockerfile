FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Ensure latest pip is installed
RUN pip install -U pip

# Install pipenv
RUN pip install pipenv

# Create working directory
WORKDIR /app

# Copy Pipfile, lock file
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies to the system
RUN pipenv install --system --deploy

# Copy scripts
COPY ["score.py", "run_predictions.py", "./"]

# Create the output directory
RUN mkdir -p /app/output

# Command to run predictions (optional args will be passed via command line)
ENTRYPOINT ["python", "run_predictions.py"]