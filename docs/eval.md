# Evaluation

## MVBench

Download [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) in  ./data/MVBench

```
python -m torch.distributed.launch --nproc-per-node=8 eval/eval_mvbench.py --data_path ./data/MVBench --version qwen --model_path ./checkpoints/longvu_qwen
```
