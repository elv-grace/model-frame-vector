#!/bin/bash

set -e

git submodule update --init --recursive

# Every checkpoint (SigLIP 2 from the HuggingFace hub, the detectors from ultralytics' assets
# and HF) is pulled at runtime into a mounted cache, so there are no baked-in weights to sync
# before building.
exec buildscripts/build_container.bash -t "model-frame-vector:${IMAGE_TAG:-latest}" . -f Containerfile
