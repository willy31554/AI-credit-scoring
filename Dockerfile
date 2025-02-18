# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Ensure pip is up-to-date
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create the models directory
RUN mkdir -p /app/models

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PORT 80

# Run uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
