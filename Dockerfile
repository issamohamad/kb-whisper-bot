FROM python:3.9-slim

WORKDIR /app

# Install ffmpeg explicitly
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Print directory contents for debugging
RUN ls -la

# Command to run the application
CMD ["python", "main.py"]
