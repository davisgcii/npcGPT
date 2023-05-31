import os
import json
import torch
import pickle
from tqdm import tqdm
import numpy as np
import tiktoken
from PIL import Image
from datasets import load_dataset # huggingface datasets
from transformers import ViTFeatureExtractor, ViTForImageClassification

# load the datasets
images_path = '/opt/conda/envs/pytorch/lib/python3.9/site-packages/data/yfcc_images'
image_paths = [os.path.join(images_path, image_path) for image_path in os.listdir(images_path)]
data_path = '/opt/conda/envs/pytorch/lib/python3.9/site-packages/data/image_chat/'
print(f'Found {len(image_paths)} images in the YFCC dataset.')

# load the pretrained vision transformer models
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")

# # tokenize the dataset
# tokenized = split_dataset.map(
#     process,
#     remove_columns=['text'],
#     desc="tokenizing the splits",
#     num_proc=num_proc,
# )

# a helper function to process each example
def process(example):
	dialog, image_name = example["dialog"], f'{example["image_hash"]}.jpg'
	
	prompt_ids = enc.encode_ordinary("Given")
	prompt_ids.append(0)	# null token
	
	prompt = ", "
	for i, d in enumerate(dialog):
		if i == len(dialog) - 1:
			break
		prompt += d[0] + ": " + d[1] + ". "
	
	prompt += dialog[-1][0] + ": "
	answer = dialog[-1][1]
	
	# get the word embeddings for the (Q, A) pairs
	prompt_ids.extend(enc.encode_ordinary(prompt)) # encode_ordinary ignores any special tokens
	answer_ids = enc.encode_ordinary(answer) # encode_ordinary ignores any special tokens
	
	prompt_ids.append(enc.eot_token) # add the end of question token, e.g. 50256 for gpt2 bpe
	answer_ids.append(enc.eot_token) # add the end of question token, e.g. 50256 for gpt2 bpe
	
	# convert to numpy arrays
	prompt_ids = np.array(prompt_ids)
	answer_ids = np.array(answer_ids)
	
	# get the image features from the vision transformer model
	image_path = os.path.join(images_path, image_name)
	if image_path not in image_paths:
		print(f'\tImage not found for hash {image_name}')
		image_features = 0
	else:
		image = Image.open(image_path)
		normalized_pixels = feature_extractor(images=image, return_tensors="pt")
		outputs = vit_model(**normalized_pixels)
		image_features = outputs.logits

	return (prompt_ids, image_features, answer_ids)

# # concatenate all the ids in each dataset into one large file we can use for training
# for split, dset in tokenized.items():
#     arr_len = np.sum(dset['len'])
#     filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
#     dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
#     arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
#     total_batches = 1024
	
#     idx = 0
#     for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
#         # Batch together samples for faster write
#         batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
#         arr_batch = np.concatenate(batch['ids'])
#         # Write into mmap
#         arr[idx : idx + len(arr_batch)] = arr_batch
#         idx += len(arr_batch)
#     arr.flush()

# read the raw training data
with open(os.path.join(data_path, 'train.json')) as f:
	raw_train = json.load(f)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# memory maps used to save processed data
seq_len, features_len = 160, 1000
queries_memmap = np.memmap('queries.bin', dtype=np.uint16, mode='w+', shape=(len(raw_train), seq_len))
features_memmap = np.memmap('features.bin', dtype=np.uint16, mode='w+', shape=(len(raw_train), features_len))
answers_memmap = np.memmap('answers.bin', dtype=np.uint16, mode='w+', shape=(len(raw_train), seq_len))

for idx, example in enumerate(raw_train):
	queries, features, answers = process(example)
	
	# save the data into the memory map
	queries_memmap[idx] = np.pad(queries, (0, seq_len - queries.shape[0]), 'constant')
	answers_memmap[idx] = np.pad(answers, (0, seq_len - answers.shape[0]), 'constant')
	
	if isinstance(features, int):
		print(f'Ran into an error for example {idx + 1}.')
	else:
		features_memmap[idx] = features.detach().numpy()
	
	# log output every 10-iterations and also flush output into the memory map
	if idx % 10 == 0:
		print(f'Processed {idx + 1} out of {len(raw_train)} examples.')
		features_memmap.flush()

features_memmap.flush()
