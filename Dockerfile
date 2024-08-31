# Use the official Python 3.11.7 image as a base image
FROM python:3.11.7

# Set the working directory in the container
WORKDIR /app

# Copy all the files from your local machine to the container
COPY . /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the Streamlit port (default is 8501)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
