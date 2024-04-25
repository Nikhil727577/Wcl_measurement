# Use an official Python runtime as a parent image
FROM python:3.11

# Set environment variables (optional)
#ENV PYTHONUNBUFFERED=1 \
 #   PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /images

# Copy the requirements file into the container at /app
COPY requirements.txt /images/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /images/

# Expose the port your application runs on (if needed)
EXPOSE 8002

# Command to run the application
CMD ["python", "object_measurement.py"]
