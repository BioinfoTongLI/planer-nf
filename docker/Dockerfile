FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      procps git python3 python3-dev python3-pip python-is-python3 libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
     numpy==1.21.2 \
     tifffile==2021.11.2 \
     dask-image==2021.12.0 \
     "dask[distributed]"==2021.12.0 \
     cupy-cuda114 ome-types fire \
     planer planer-zoo onnx onnxruntime \
     --no-cache-dir
