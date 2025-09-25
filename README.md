This is a test example for LongVU DeepSpeed.

Please download [LongVU_Qwen2_7B](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B) to ./checkpoints/cambrian_qwen_7b
```
git clone https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B
```

Download example data to ./mlvu for debuging 
(Please use `huggingface-cli login` if encountered `Authentication Failed`)

```
from huggingface_hub import list_repo_files, hf_hub_download
import json
repo_id = "MLVU/MVLU"
subfolder = "MLVU/video/1_plotQA"
```

```
conda env create -f environment.yml
sh run.sh
```

