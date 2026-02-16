FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose backend port (Flask usually 5000)
EXPOSE 5000

# Run the backend app
CMD ["python", "main.py"]
