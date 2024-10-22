# LongVU

> **LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding**
>
> <a href=''><img src='https://img.shields.io/badge/arXiv-paper-red'></a> <a href='https://vision-cair.github.io/LongVU'><img src='https://img.shields.io/badge/project-LongVU-blue'></a> <a href='https://huggingface.co/spaces/Vision-CAIR/LongVU'><img src='https://img.shields.io/badge/demo-space-blue'></a> <a href='https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a> 

<div align="center">
    <a href='https://vision-cair.github.io/LongVU'><img src="https://longvu.s3.amazonaws.com/assets/demo.gif" alt="Demo GIF" style="width: 100%; max-width: 650px;"></a>
</div>

## :rocket: Quick Start

Try our model on [🤗 Space](https://huggingface.co/spaces/Vision-CAIR/LongVU)

Or demploy from local

```
conda create -n longvu python=3.10
conda activate longvu
pip install -r requirements.txt
```

Download our checkpoints and put it under `./checkpoints`

| Modality | LongVU_Qwen2_7B | LongVU_Llama3_2_3B |
:--------------------------:| :--------------------------:|:--------------------------:
| Image | [Download]() | [Download]() |
| Video | [Download](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B) | [Download](https://huggingface.co/Vision-CAIR/LongVU_Llama3_2_3B) |

See quick inference code in [inference.md](https://github.com/xiaoqian-shen/LongVU/blob/main/docs/inference.md)

## Training

### Dataset

+ image-text stage: [LLaVA-OneVision-Single](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)
+ video-text stage: [VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)

### Scripts

Experiments are run on 64 H100-96G

```
# download data to specific path, then run
sh scripts/train_video_qwen.sh
sh scripts/train_video_llama3_2.sh
```

## Evaluation

See detailed evaluation code in [eval.md](https://github.com/xiaoqian-shen/LongVU/blob/main/docs/eval.md)

## Acknowledgement

+ The model architecture of LongVU follows [LLaVA](https://github.com/haotian-liu/LLaVA) and [Cambrian](https://github.com/cambrian-mllm/cambrian)
+ We base [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) and [Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) as the language backbone
+ We use [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [DINOv2](https://huggingface.co/facebook/dinov2-giant) as the vision encoder

## Citation

```
@article{shen2024longvu,
  author    = {},
  title     = {LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding},
  journal   = {},
  year      = {2024},
}
```
