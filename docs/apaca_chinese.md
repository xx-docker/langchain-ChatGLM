# 合并`alpaca chinese`的权重
- `https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/手动模型合并与转换`


```bash

python scripts/merge_llama_with_chinese_lora.py \
    --base_model /data/workdir/models/llama-7b-hf/ \
    --lora_model /data/workdir/models/chinese-llama-plus-lora-7b/,/data/workdir/models/chinese-alpaca-plus-lora-7b/ \
    --output_type huggingface \
    --output_dir /data/workdir/models/alpaca-chinese-7b-hf/

# https://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:x-oauth-basic@
git remote add origin  git@hf.co:redauzhang/alpaca-chinese-7b-hf

```
