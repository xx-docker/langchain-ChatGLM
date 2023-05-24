# !/bin/bash ``
MODEL_DIR="/data/workdir/models"
mkdir -p  $MODEL_DIR || echo "File exsit"

cd $MODEL_DIR
git lfs install --depth 1 https://huggingface.co/GanymedeNil/text2vec-large-chinese
# git lfs install --depth 1 https://huggingface.co/THUDM/chatglm-6b
# git lfs install --depth 1 https://huggingface.co/fnlp/moss-moon-003-sft-int8
git lfs install --depth 1 https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
# git lfs clone https://huggingface.co/fnlp/moss-moon-003-sft-int4
