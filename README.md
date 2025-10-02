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
local_dir = f"./mlvu/{subfolder}"
os.makedirs(local_dir, exist_ok=True)

files = list_repo_files(repo_id=repo_id, repo_type="dataset")
subfolder_files = [f for f in files if f.startswith(subfolder)]

for file_path in subfolder_files:
    local_file_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        local_dir="./data",
        local_dir_use_symlinks=False
    )
```

```
conda env create -f environment.yml
sh run.sh
```

