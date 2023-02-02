# pull the base image with python 3.8 as a runtime the lambda function
FROM public.ecr.aws/lambda/python:3.8

# copy the requirements file to the container
COPY requirements.txt ./

# install the requirements
RUN python3.8 -m pip install -r requirements.txt

RUN python3.8 -m pip install -U protobuf==3.9.2

# copy the main file
COPY app.py ./

# set the CMD to the lambda_handler() function
CMD ["app.lambda_handler"]