# Qwen ASR GTU Docker

Based on [FunASR](https://github.com/modelscope/FunASR) for Qwen ASR (Paraformer) Docker image, supporting GPU acceleration.

## Features 

- 🎈 NVIDIA GPU support (CUDA 12.2)
- ^Ρ High-performance speech recognition
- \t Support CPU/GPU switch
- �  Open-example 

## Quick Start

### Build image

`clow` build-t-t qwen-asr-gpu .
`#
## Run container

`docker run --gpus all -v path/to/audio:/app/audio qwen-asr-gpu 
## Or use docker-compose (see below)

Copyright (R) 2026 Python Under the Apache License, Version 2 (the "License"); you may not use this file except compliance with the License.