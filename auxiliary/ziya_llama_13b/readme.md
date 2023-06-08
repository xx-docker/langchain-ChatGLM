# 自用姜子牙大模型 (CPU运行)
- https://zhuanlan.zhihu.com/p/539369065
- https://github.com/ggerganov/llama.cpp/tree/master/examples/main

## ubuntu 下安装 openBLAS 

```bash

sed -i 's/security.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list
sed -i 's/archive.ubuntu.com/mirrors.tencent.com/g' /etc/apt/sources.list


apt-get update && \
# common utils for download sources tarball/zipball
apt-get install -y --no-install-recommends curl wget ca-certificates gnupg2 && \
# openblas deps
apt-get install -y --no-install-recommends g++ gcc gfortran git make && \
# cleanup
apt-get remove --purge -y


# 如果是 centos 需要安装  gcc-gfortran
# https://zhuanlan.zhihu.com/p/539369065
# https://github.com/ggerganov/llama.cpp

OPENBLAS_VERSION=0.3.23 && \
wget "https://github.com/xianyi/OpenBLAS/archive/v${OPENBLAS_VERSION}.tar.gz" && \
tar zxf v${OPENBLAS_VERSION}.tar.gz && cd OpenBLAS-${OPENBLAS_VERSION} && \
make TARGET=CORE2 DYNAMIC_ARCH=1 DYNAMIC_OLDER=1 USE_THREAD=0 USE_OPENMP=0 FC=gfortran CC=gcc COMMON_OPT="-O3 -g -fPIC" FCOMMON_OPT="-O3 -g -fPIC -frecursive" NMAX="NUM_THREADS=128" LIBPREFIX="libopenblas" LAPACKE="NO_LAPACKE=1" INTERFACE64=0 NO_STATIC=1

make -j8 PREFIX=/usr NO_STATIC=1 install 

```

<details> <summary>测试blas</summary>

```c 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cblas.h"
int main(void) {
  int i;
  double A[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
  double B[6] = {1.0, 2.0, 1.0, -3.0, 4.0, -1.0};
  double C[9] = {.5, .5, .5, .5, .5, .5, .5, .5, .5};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
              3, 3, 2, 1, A, 3, B, 3, 2, C, 3);
  for (i = 0; i < 9; i++)
    printf("%lf ", C[i]);
  printf("\\n");
  if (fabs(C[0]-11) > 1.e-5) abort();
  if (fabs(C[4]-21) > 1.e-5) abort();
  return 0;
}

// cc -static -o test-openblas test-openblas.c -I /usr/include/ -L/usr/lib -lopenblas -lpthread -lgfortran
// ./test-openblas; test $? -eq 0 && echo "a happy ending"
```
</details>

## Install LLama.cpp 

```bash
make LLAMA_OPENBLAS=1
# 
## 或者使用 cmake 
mkdir build
cd build
cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
cmake --build . --config Release
```

## 命令行进行调用 
```bash

# 
./main -m /root/ggml-vic13b-q8_0.bin --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.1
./main -m /root/ggml-vic13b-q8_0.bin --color -p "how to study chinese, please anwser my question in chinese" -n 512 -c 10 -c 2048

```

## 使用 convert 进行格式转换 bin/pth
- 参考 `chinse-lamma-apaca`, `llama.cpp`


## 使用 llama-cpp-python
```bash

# LLAMA_OPENBLAS=1 /usr/local/miniconda3/bin/pip install llama-cpp-python --verbose
# llama-cpp-python-0.1.57
# https://github.com/abetlen/llama-cpp-python/issues/303
```
- Dockerfile 已经有了。
  - https://github.com/abetlen/llama-cpp-python/tree/main/docker



## 测试 llama-ggml
```bash
# echo 1 > /proc/sys/vm/drop_caches
# docker run -itd --privileged --name=u1 -v /data/:/data -p 38080:8000 ub2204-imange bash

ulimit -l unlimited
LLAMA_OPENBLAS=1 HOST=0.0.0.0 PORT=38080 python3 -B -m llama_cpp.server --model /data/workdir/models/Ziya-LLaMA-13B-v1-ggml-q5_1.bin --n_threads 10 --n_ctx 2048 --last_n_tokens_size 1024 
```


## 
## 
## 
