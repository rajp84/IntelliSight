# Development

## Start Milvus + deps

Required Milvus, MiniIO, and etcd.  Also starts attu for web gui on http://localhost:5530

```
docker compose up -d
```

## Backend

``` 
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements/requirements.txt
```

run backend with hot reload

```
uvicorn app.main:app --reload --port 3001
```


### Setup TensorRT 8.6.1.6
Download and extract to ~/TensorRT8.6.1.6

then in your venv:

```
pip install ~/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
```

## Frontend
```
cd frontend
npm install
ng serve -o
```


# Build Docker Image

```
docker build -t intellisight:latest .
docker run -d --name intellisight-app --network host --gpus all intellisight:latest
```
















# Setup to run training

Need Cuda 12.1 and TRT 8.6.16 for installing pycuda.

## Installing Cuda 12.1

```
# Example for CUDA 12.1 runtime + cuDNN (adjust version to what you prefer)
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# (If you use the NVIDIA repo, you can also `sudo apt-get install libcudnn8 libcudnn8-dev` for CUDA 12)
```

## TensorRT 8.6.1.6
Download and extract to ~/TensorRT8.6.1.6

then in your venv:

```
pip install ~/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
```
## Setup paths
```
export TRT_HOME="~/TensorRT-8.6.1.6"
export CUDA_HOME="/usr/local/cuda-12.1"   # or cuda-12.0/12.2 depending on what you installed
export LD_LIBRARY_PATH="$TRT_HOME/lib:$CUDA_HOME/lib64:$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v 'cuda-11' | paste -sd:)"
```

## Check

```
ldconfig -p | grep -E 'libcublasLt\.so|libcublas\.so'   # should show *.so.12
python - <<'PY'
import ctypes
ctypes.CDLL("libcublasLt.so.12")
ctypes.CDLL("libcublas.so.12")
print("CUDA12 cuBLASLt visible")
PY
```