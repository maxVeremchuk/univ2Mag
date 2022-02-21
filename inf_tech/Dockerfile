FROM python:3.8-slim

RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools

RUN mkdir /app

# Copy the current directory contents into the container at /app
ADD ./bot.py /app/app.py

COPY ./requirements.txt /app/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

RUN pip install numpy
RUN pip install opencv-python

EXPOSE $PORT
CMD ["python", "app.py"]