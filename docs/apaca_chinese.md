# 合并`alpaca chinese`的权重
- `https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/手动模型合并与转换`


```bash

git lfs clone https://huggingface.co/decapoda-research/llama-7b-hf
git lfs clone https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b
git lfs clone https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b
```

```bash
# protobuf==3.20.1 peft==0.3.0

git clone --depth 1 https://github.com/ymcui/Chinese-LLaMA-Alpaca/
cd Chinese-LLaMA-Alpaca/

# 如果是在 googleddrive下进行的话 
ln -sf /content/drive/MyDrive/ColabWorkSpace/models /data/workdir/models

python scripts/merge_llama_with_chinese_lora.py \
    --base_model /data/workdir/models/llama-7b-hf/ \
    --lora_model /data/workdir/models/chinese-llama-plus-lora-7b/,/data/workdir/models/chinese-alpaca-plus-lora-7b/ \
    --output_type huggingface \
    --output_dir /data/workdir/models/alpaca-chinese-7b-hf/

# https://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:x-oauth-basic@
git remote add origin  git@hf.co:redauzhang/alpaca-chinese-7b-hf

# 正常使用查看。
python ./scripts/inference_hf.py \
    --base_model /data/workdir/models/alpaca-chinese-7b-hf/ \
    --with_prompt \
    --interactive
```
