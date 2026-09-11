FROM continuumio/miniconda3:latest
WORKDIR /elv

# Layers are ordered heaviest and least-frequently-changing first: system packages, conda env,
# ssh known_hosts, package scaffold, pip deps, then source. Anything below an edited layer is
# rebuilt, so nothing that changes per-commit sits above the pip install.

# ffmpeg provides the ffprobe/ffmpeg CLIs used by common_ml.video_processing (and the
# codecs PyAV decodes video with); build-essential for any source builds.
RUN apt-get update && apt-get install -y build-essential ffmpeg && rm -rf /var/lib/apt/lists/*

# transformers>=5 / torch 2.8 require Python 3.11+
RUN conda create -n mlpod python=3.11 -y

# Create the SSH directory with correct permissions and add GitHub to known_hosts to bypass
# host verification (common-ml is a git dependency).
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh \
    && ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

# The torch 2.8 pip wheels bundle their own CUDA runtime, so no host CUDA toolkit is
# needed; the container just needs the NVIDIA driver + container toolkit at run time
# (--device nvidia.com/gpu=...).

# setup.py declares packages=["general_detection"], so the dir must exist at install time.
# Kept above `COPY setup.py` so editing setup.py doesn't rebuild this layer.
RUN mkdir -p general_detection

# Install dependencies before copying source so the heavy dependency layer is cached.
# `pip install .` installs the dependencies (incl. common-ml from git); the tagger code
# itself runs from the source copied below (WORKDIR is on sys.path).
COPY setup.py .
# The cache mount keeps pip's wheel/HTTP cache outside the image, so editing setup.py still
# reinstalls but re-downloads nothing (torch + CUDA wheels are several GB). Nothing is committed
# to the image. Needs a BuildKit-capable builder, which buildscripts/build_container.bash
# already assumes via --secret/--ssh.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/conda/envs/mlpod/bin/pip install .

# No weights are baked into this image. Downloads happen on first load, and only for what the
# request actually needs:
#   - SigLIP 2 from the HuggingFace hub, into HF_HOME. Always, it is the embedder.
#   - the detector checkpoint (Grounding DINO or YOLOE) and, for YOLOE, the MobileCLIP text
#     encoder that get_text_pe() needs, into storage.cache_path (config.yml), and ONLY when a
#     request sets `detect_target`; ultralytics resolves those relative to the CWD, which
#     general_detection/detector.py handles by chdir-ing into the cache during load.
# All live under /root/.cache, so ONE mounted volume there covers them. Without it they
# land in the container's ephemeral writable layer and are re-fetched on every run.
# Kept below the pip install: editing these then costs only the source COPYs, not a reinstall.
ENV HF_HOME=/root/.cache
# Keep ultralytics' settings/config out of the ephemeral layer too.
ENV YOLO_CONFIG_DIR=/root/.cache/Ultralytics

# Source last (changes most often).
COPY general_detection ./general_detection
COPY config.yml run.py config.py ./

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "-u", "run.py"]
