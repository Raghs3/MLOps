# Dockerfile
# 1. Use an official lightweight Python base image
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy dependency file first (leverages Docker layer caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code and trained model into the container
COPY app.py .
COPY iris_model.pkl .

# 6. Expose the port Flask runs on
EXPOSE 5000

# 7. Command to run the application when the container starts
CMD ["python", "app.py"]
