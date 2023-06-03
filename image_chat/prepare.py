import os
import json
import torch
import pickle
from tqdm import tqdm
import numpy as np
import tiktoken
from PIL import Image
# from torchinfo import summary
from datasets import load_dataset # huggingface datasets
from transformers import ViTFeatureExtractor, ViTForImageClassification

# load the datasets
images_path = '/opt/conda/envs/pytorch/lib/python3.9/site-packages/data/yfcc_images'
image_paths = [os.path.join(images_path, image_path) for image_path in os.listdir(images_path) if image_path.endswith('.jpg')]
data_path = '/opt/conda/envs/pytorch/lib/python3.9/site-packages/data/image_chat/'
print(f'Found {len(image_paths)} images in the YFCC dataset.')

# load the pretrained vision transformer models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
vit_model.to(device)
# summary(vit_model, input_size=(3, 224, 224))

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")

# define the maximum sequence length and the features length
seq_len, features_len = 256, 1000

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
	
	answer = dialog[-1][1]
	prompt += dialog[-1][0] + ": " + answer
	
	# get the word embeddings for the (Q, A) pairs
	prompt_ids.extend(enc.encode_ordinary(prompt)) # encode_ordinary ignores any special tokens
	answer_ids = enc.encode_ordinary(answer) # encode_ordinary ignores any special tokens
	
	# trim the size to be at most sequence length - 1, then we append the end token
	if len(prompt_ids) >= seq_len:
		prompt_ids = prompt_ids[:seq_len - 1]
	if len(answer_ids) >= seq_len:
		answer_ids = answer_ids[:seq_len - 1]
	
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
		
		# make sure we are using the right image mode
		rgb_mode = 'RGB'
		if image.mode != rgb_mode:
			print(f'Found image mode {image.mode}, converting to mode {rgb_mode}.')
			image = image.convert(rgb_mode)
		
		normalized_pixels = feature_extractor(images=image, return_tensors="pt").to(device)
		outputs = vit_model(**normalized_pixels)
		image_features = outputs.logits

	return (prompt_ids, image_features, answer_ids)

def prepare(split, truncate=False):
	print(f'Processing the data split {split} with truncate set to {truncate}...')
	
	# read the raw training data
	with open(os.path.join(data_path, f'{split}.json')) as f:
		raw_train = json.load(f)

	base_path = f"{split}"
	
	# truncate data to a 100 values if Flag is True
	if truncate:
		raw_train = raw_train[:100]
		base_path += "-truncated"
	
	# create an empty folder to store the data for the given split
	os.makedirs(base_path, exist_ok=True)

	# memory maps used to save processed data
	queries_memmap = np.memmap(f'{base_path}/queries.bin', dtype=np.uint16, mode='w+', shape=(len(raw_train), seq_len))
	answers_memmap = np.memmap(f'{base_path}/answers.bin', dtype=np.uint16, mode='w+', shape=(len(raw_train), seq_len))
	features_memmap = np.memmap(f'{base_path}/features.bin', dtype=np.float64, mode='w+', shape=(len(raw_train), features_len))

	for idx, example in enumerate(raw_train):
		query, features, answer = process(example)
		
		# save the data into the memory map
		queries_memmap[idx] = np.pad(query, (0, seq_len - query.shape[0]), 'constant')
		answers_memmap[idx] = np.pad(answer, (0, seq_len - answer.shape[0]), 'constant')
		
		if isinstance(features, int):
			print(f'Ran into an error for example {idx + 1}.')
		else:
			features_memmap[idx] = features.cpu().detach().numpy()
		
		# log output every 100-iterations and also flush output into the memory map
		if idx % 100 == 0:
			print(f'Processed {idx + 1} out of {len(raw_train)} examples.')
			queries_memmap.flush()
			answers_memmap.flush()
			features_memmap.flush()
	
	# flush any output that is still in the buffer
	queries_memmap.flush()
	answers_memmap.flush()
	features_memmap.flush()
	
	print(f'Finished processing the data split {split}.')
	print()

if __name__ == "__main__":
	# initial logs for GPU/CPU information
	print('Using device:', device)
	print()

	# additional info when using cuda
	if device.type == 'cuda':
		print('Device name: ', torch.cuda.get_device_name(0))
		print('Memory Usage:')
		print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
		print('Cached:   ', round(torch.cuda.memory_reserved(0)  / 1024**3, 1), 'GB')
		print()
	
	# print("Model summary for the pretrained model 'google/vit-base-patch16-224'")
	# print(vit_model)
	
	# process the three datasets
	splits = ["valid", "train", "test"]
	for split in splits:
		prepare(split, True)
	for split in splits:
		prepare(split)
