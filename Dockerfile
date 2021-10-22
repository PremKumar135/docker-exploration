#install ubuntu
FROM python:3.7.5-slim

#INSTALL python
RUN apt-get update -y \
    && apt-get install -y python3-pip python3-dev \
    && apt-get install -y build-essential cmake \
    && apt-get install -y libopenblas-dev liblapack-dev \
    && apt-get install -y libx11-dev libgtk-3-dev

#create a source directory and make it as a working directory
RUN mkdir src/
WORKDIR src/

#copy the file
COPY vec.pkl best_model_LR.pkl app.py requirements.txt src/

#install the libraries
# SHELL ["/bin/bash", "-c"]
RUN pip3 install -r src/requirements.txt \
    && python -m nltk.downloader all 
    
#execute while running the docker
ENTRYPOINT ["python3"]
CMD ["src/app.py"]



