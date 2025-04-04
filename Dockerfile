FROM python:3.10.1-buster

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER aballas@hua.gr

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt