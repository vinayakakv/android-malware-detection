FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
COPY requirements.txt /mnt/
RUN apt-get update && apt-get install -y git graphviz graphviz-dev && rm -rf /var/lib/apt/lists/*
RUN pip install -r /mnt/requirements.txt
RUN git clone https://github.com/androguard/androguard.git && cd androguard && python setup.py install
VOLUME /model
WORKDIR /model