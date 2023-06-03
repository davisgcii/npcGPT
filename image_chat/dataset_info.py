import os
import json
from PIL import Image
from prepare import process, image_paths

data_path = '/opt/conda/envs/pytorch/lib/python3.9/site-packages/data/image_chat/'

def dataset_info(split):
	print(f'Processing the data split {split}...')
	
	# load the data for the given split
	with open(os.path.join(data_path, f'{split}.json')) as f:
		data = json.load(f)
	
	# variables to store the length for query, answer, and the entire conversation (query + answer)
	query_lengths = []
	answer_lengths = []
	conversation_lengths = []
	
	# gather information for each example
	for idx, example in enumerate(data):
		query, features, answer = process(example)
		
		query_lengths.append([idx, len(query)])
		answer_lengths.append([idx, len(answer)])
		conversation_lengths.append([idx, len(query) + len(answer)])
		
		# log output every 10-iterations and also flush output into the memory map
		if idx % 100 == 0:
			print(f'Processed {idx + 1} out of {len(data)} examples.')
	
	# the top n values
	top_n = 5
	query_sorted = sorted(query_lengths, key=lambda x: x[1], reverse=True)
	answer_sorted = sorted(answer_lengths, key=lambda x: x[1], reverse=True)
	conversation_sorted = sorted(conversation_lengths, key=lambda x: x[1], reverse=True)
	
	print(f'Top {top_n} query lengths:', query_sorted[:top_n])
	print(f'Top {top_n} answer lengths:', answer_sorted[:top_n])
	print(f'Top {top_n} conversation lengths:', conversation_sorted[:top_n])
	
	print(f'Finished processing the data split {split}.')

if __name__ == "__main__":
	# get information on the images
	print(f'Getting information on image sizes...')
	
	image_sizes = set()
	for idx, image_path in enumerate(image_paths):
		image = Image.open(image_path)
		width, height = image.size
		image_sizes.add((width, height))
		
		# log output every 1000-iterations and also flush output into the memory map
		if idx % 1000 == 0:
			print(f'Processed {idx + 1} out of {len(image_paths)} images.')
	
	# print the set of unique image sizes
	for image_size in sorted(image_sizes):
		print(image_size)
	
	print('Finished gathering information on images.')
	
	splits = ["valid", "train", "test"]
	for split in splits:
		dataset_info(split)
