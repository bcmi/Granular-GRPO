#!/bin/bash

# install torch
pip install torch==2.5.0 torchvision

# install FA2 and diffusers
pip install packaging ninja 

pip install flash-attn

pip install -r requirements-lint.txt

# install fastvideo
pip install -e .

pip install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 huggingface_hub==0.24.0 protobuf==3.20.0 accelerate

pip install opencv-python-headless, open-clip-torch