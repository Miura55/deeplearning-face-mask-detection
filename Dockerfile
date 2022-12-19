FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /work

COPY . /work
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

EXPOSE 8888
CMD [ "bash" ]
