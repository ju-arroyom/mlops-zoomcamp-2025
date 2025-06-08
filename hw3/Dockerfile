FROM python:3.10-slim


WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "pipeline.py", "--year", "2023", "--month", "3"]
