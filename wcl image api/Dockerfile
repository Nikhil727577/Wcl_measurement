# Use an official Python runtime as a base image
FROM python:3.11-slim
 
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on
 
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    libjpeg-dev \
    zlib1g-dev \
&& rm -rf /var/lib/apt/lists/*
 
# Set the working directory in the container
WORKDIR /app
 
# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
 
# Copy the rest of your application code into the container
COPY . /app/
 
# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]


# # Use an official Python runtime as a parent image
# FROM python:3.11

# # Set environment variables (optional)
# #ENV PYTHONUNBUFFERED=1 \
#  #   PYTHONDONTWRITEBYTECODE=1

# # Set the working directory in the container
# WORKDIR /images

# # Copy the requirements file into the container at /app
# COPY requirements.txt /images/

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code into the container
# COPY . /images/

# # Expose the port your application runs on (if needed)
# EXPOSE 8002

# # Command to run the application
# CMD ["python", "object_measurement.py"]
