# CPU environment for v10 (N = 20, 25 arms, cross-validation). Build: docker build -t v10 .
FROM python:3.13-slim
WORKDIR /v10
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY . .
CMD ["python3", "crossval.py"]
