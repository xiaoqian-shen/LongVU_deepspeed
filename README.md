# LongVU

> **LongVU**
>
> Authors
>
> <a href=''><img src='https://img.shields.io/badge/arXiv-paper-red'></a>  <a href='https://huggingface.co/spaces/Vision-CAIR/LongVU'><img src='https://img.shields.io/badge/demo-space-blue'></a> <a href=''><img src='https://img.shields.io/badge/model-checkpoints-green'></a>

## Quick Start

```
conda create -n longvu python=3.10
conda activate longvu
pip install -r requirements.txt
```

Download our checkpoints from [here](), then put it under `./checkpoints`

See quick inference code in [inference.md](https://github.com/xiaoqian-shen/LongVU/blob/main/docs/inference.md)

Try our model in [Huggingface Space]()

## Training

### Dataset

+ image-text stage: [LLaVA-OneVision-Single](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)
+ video-text stage: [VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)

### Scripts

Stay tuned ...

## Acknowledgement

+ The model architecture of LongVU follows [LLaVA](https://github.com/haotian-liu/LLaVA) and [Cambrian](https://github.com/cambrian-mllm/cambrian)
+ We base [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and [Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) as the language backbone
+ We use [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [DINOv2](https://huggingface.co/facebook/dinov2-giant) as the vision encoder
