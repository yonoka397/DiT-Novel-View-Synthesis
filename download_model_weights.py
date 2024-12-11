import json
import os
import requests
from tqdm import tqdm


def download_Objaverse_model_weights():
	download_url = "https://drive.usercontent.google.com/download?id=192R_AH5Cex-pr1o37xoNilfC6lR9LP2d&export=download&authuser=0&confirm=t&uuid=58530a24-e314-4e48-a0a0-c0da376aa35e&at=APvzH3qqYnVLgZOHLj93Vj3SrCXE:1733503158385"
	save_path = "./model_weights_Objaverse.pth"
	download_file(download_url=download_url, save_path=save_path)
	return save_path


def download_MVC_model_weights():
	download_url = "https://drive.usercontent.google.com/download?id=1D78jGRJT9m6B1Apeb2orlKJ_E6AwKEta&export=download&authuser=1&confirm=t&uuid=9e62177b-c911-4f5a-84b5-7b8db598db7e&at=AENtkXYNZGHYeWuWKYb3UJbrKTd9%3A1733146544355"
	save_path = "./model_weights_MVC.pth"
	download_file(download_url=download_url, save_path=save_path)
	return save_path


def download_file(download_url, save_path):
	with tqdm(total=100, desc="Downloading", unit="%") as pbar:
		for progress in download_file_progressive(download_url, save_path):
			data = json.loads(progress.split("data:")[-1].strip())
			percent = data['percent']
			pbar.update(percent - pbar.n)


#  model weights download with front-end progress bar
def download_file_progressive_test(download_url, save_path, progress_callback=None):

	with requests.get(download_url, stream=True) as response:
		total_size = int(response.headers.get('content-length', 0))
		downloaded_size = 0

		with open(save_path, 'wb') as file:
			for chunk in response.iter_content(chunk_size=1024):
				if chunk:
					file.write(chunk)
					downloaded_size += len(chunk)
					# update current progress to callback
					if progress_callback:
						progress_callback(downloaded_size, total_size)


#  normal model weights download with progress
def download_file_progressive(download_url, save_path):
	# if exist then stop download
	if os.path.exists(save_path):
		yield f"data:{json.dumps({'percent': 100})}\n\n"
		return

	with requests.get(download_url, stream=True) as response:
		total_size = int(response.headers.get('content-length', 0))
		downloaded_size = 0

		with open(save_path, 'wb') as file:
			for chunk in response.iter_content(chunk_size=1024):
				if chunk:
					file.write(chunk)
					downloaded_size += len(chunk)
					percent = int(downloaded_size / total_size * 100)
					yield f"data:{json.dumps({'percent': percent})}\n\n"
