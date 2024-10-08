import numpy as np

import torch

from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

version = "qwen"
model_name = "cambrian_qwen"
input_model_local_path = "./checkpoints/longvu_qwen"

device = "cuda:7"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    input_model_local_path, None, model_name, device=device
)
model.get_model().config.tokenizer_model_max_length = 8192
model.get_model().config.inference_max_length = 128
model.config.use_cache = True
print(model.device)

model.eval()

video_path = "./examples/video1.mp4"
qs = "Describe this video in detail"

vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
fps = float(vr.get_avg_fps())
frame_indices = np.array(
    [
        i
        for i in range(
            0,
            len(vr),
            round(fps),
        )
    ]
)
video = []
for frame_index in frame_indices:
    img = vr[frame_index].asnumpy()
    video.append(img)
video = np.stack(video)
image_sizes = [video[0].shape[:2]]
video = process_images(video, image_processor, model.config)
video = [item.unsqueeze(0) for item in video]

qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

conv = conv_templates[version].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .to(model.device)
)

if "llama3" in version:
    input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=video,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0.2,
        max_new_tokens=128,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
    )

if isinstance(output_ids, tuple):
    output_ids = output_ids[0]

pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

print("pred: ", pred, flush=True)
