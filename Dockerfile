FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime AS base
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir -p /tmp/hpe-swarmcli-pkg
COPY swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl /tmp/hpe-swarmcli-pkg/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl
RUN pip install /tmp/hpe-swarmcli-pkg/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl
