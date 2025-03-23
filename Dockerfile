FROM python:3.9-slim

WORKDIR /app

# Copy application code
COPY app.py .

# Print directory contents for debugging
RUN ls -la

# Command to run the application
CMD ["python", "app.py"]
