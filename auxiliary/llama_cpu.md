# llama cpu 


```bash 
docker run -v /data/workdir/Chinese-LLaMA-Alpaca/:/models ghcr.io/ggerganov/llama.cpp:full --all-in-one "/models/" 7B
docker run -v /data/workdir/Chinese-LLaMA-Alpaca/:/models ghcr.io/ggerganov/llama.cpp:full \
    --run -m /models/7B/ggml-model-q4_0.bin \
    -p "Building a website can be done in 10 simple steps:" \
    -n 512

```

## INstall openblas 
- https://zhuanlan.zhihu.com/p/539369065
