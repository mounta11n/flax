import json
import os, shutil
import random


from PIL import Image
import jax
from transformers import FlaxVisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from huggingface_hub import hf_hub_download


# create target model directory
model_dir = './models/'
os.makedirs(model_dir, exist_ok=True)

files_to_download = [
    "config.json",
    "flax_model.msgpack",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "preprocessor_config.json",
]

# copy files from checkpoint hub:
for fn in files_to_download:
    file_path = hf_hub_download("ydshieh/vit-gpt2-coco-en-ckpts", f"ckpt_epoch_3_step_6900/{fn}")
    shutil.copyfile(file_path, os.path.join(model_dir, fn))

model = FlaxVisionEncoderDecoderModel.from_pretrained(model_dir)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


@jax.jit
def generate(pixel_values):
    output_ids = model.generate(pixel_values, **gen_kwargs).sequences
    return output_ids


def predict(image):

    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    pixel_values = feature_extractor(images=image, return_tensors="np").pixel_values

    output_ids = generate(pixel_values)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds[0]


def _compile():

    image_path = 'samples/val_000000039769.jpg'
    image = Image.open(image_path)
    predict(image)
    image.close()


_compile()


sample_dir = './samples/'
sample_image_ids = tuple(["None"] + [int(f.replace('COCO_val2017_', '').replace('.jpg', '')) for f in os.listdir(sample_dir) if f.startswith('COCO_val2017_')])

with open(os.path.join(sample_dir, "coco-val2017-img-ids.json"), "r", encoding="UTF-8") as fp:
    coco_2017_val_image_ids = json.load(fp)


def get_random_image_id():

    image_id = random.sample(coco_2017_val_image_ids, k=1)[0]
    return image_id
