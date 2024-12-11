import argparse
from flask import Flask, Response, render_template, request, jsonify, send_from_directory
import json
import os
import requests
import tempfile
from threading import Thread
from time import sleep
from werkzeug.utils import secure_filename

from download_model_weights import download_file_progressive_test
from sample import main

app = Flask(__name__)

output_image_path = None

download_complete = False

progress = 0

#  generate progress to front-end
def generate_progress():
	global progress
	while progress < 100:
		sleep(1)  # wait before sending next progress update
		yield f"data: {progress}\n\n"

def progress_callback(current_step, total_steps):
	global progress
	progress = (current_step / total_steps) * 100
	
@app.route('/progress')
def progress_route():
	return Response(generate_progress(), content_type='text/event-stream')


#  fron-end route
@app.route('/')
def index():
	return render_template('/index.html')  #  front-end


#  start download model weights
@app.route('/download_model_weights', methods=['POST'])
def progress_download():
	global progress
	progress = 0

	data = request.get_json()
	dataset_train = data.get('dataset_train')
	
	download_url = "https://drive.usercontent.google.com/download?id=192R_AH5Cex-pr1o37xoNilfC6lR9LP2d&export=download&authuser=0&confirm=t&uuid=58530a24-e314-4e48-a0a0-c0da376aa35e&at=APvzH3qqYnVLgZOHLj93Vj3SrCXE:1733503158385" if dataset_train == "Objaverse" else "https://drive.usercontent.google.com/download?id=1D78jGRJT9m6B1Apeb2orlKJ_E6AwKEta&export=download&authuser=1&confirm=t&uuid=9e62177b-c911-4f5a-84b5-7b8db598db7e&at=AENtkXYNZGHYeWuWKYb3UJbrKTd9%3A1733146544355"
	
	save_path = "./model_weights_Objaverse.pth" if dataset_train == "Objaverse" else "./model_weights_MVC.pth"
	
	download_complete = False
	def run_downloading():
		global progress, download_complete
		download_file_progressive_test(download_url, save_path, progress_callback=progress_callback)
		progress = 100
		download_complete = True
	Thread(target=run_downloading).start()

	return jsonify({'message': 'downloading started'})


#  check model weights download result
@app.route('/download_result', methods=['GET'])
def download_result():
	return jsonify({'download_complete': download_complete})


#  check file existence
@app.route('/check_file_exists', methods=['GET'])
def check_file_exists():
	file_path = request.args.get('file_path')
	
	if not file_path:
		return jsonify({'error': 'file_path is required'}), 400
	
	if os.path.exists(file_path):
		return jsonify({'exists': True})
	else:
		return jsonify({'exists': False})


#  upload file to server
@app.route('/upload', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		return jsonify({'error': 'No file part'}), 400
	file = request.files['file']
	if file.filename == '':
		return jsonify({'error': 'No selected file'}), 400

	dir_name = "upload_images"
	dir_path = os.path.join("./", dir_name)
	if not os.path.isdir(dir_path):
		os.makedirs(dir_path)

	user_temp_dir = tempfile.mkdtemp(dir=dir_path)

	filename = secure_filename(file.filename)

	file_path = os.path.join(user_temp_dir, filename)
	file.save(file_path)
	
	return jsonify({'uploaded_image_path': file_path})


#  start sample image
@app.route('/sample', methods=['POST'])
def sample_route():
	global progress
	progress = 0

	#  get the img_path from the request JSON payload
	data = request.get_json()
	img_path = data.get('img_path')
	dataset_train = data.get('dataset_train')

	if not img_path:
		return jsonify({'error': 'img_path is required'}), 400

	args = argparse.Namespace(
		dataset_train=dataset_train,
		image_load_path=img_path,
		input_size=64,
		patch_size=4,
		hidden_size=512,
		in_channel=3,
		num_heads=8,
		base=10000,
		depth=8,
		progress_callback=progress_callback,
	)

	output_image_path = None
	#  run the sampling function
	def run_sampling():
		global progress, output_image_path
		output_image_path = main(args)
		progress = 100
	
	Thread(target=run_sampling).start()

	return jsonify({'message': 'Sampling started'})


#  check image sampled
@app.route('/sample_result', methods=['GET'])
def sample_result():
	return jsonify({'output_image_path': output_image_path})


#  path to uploaded image
@app.route('/upload_images/<path:subpath>')
def serve_uploaded_file(subpath):
	directory = os.path.join(os.getcwd(), 'upload_images')
	full_path = os.path.join(directory, subpath)
	return send_from_directory(directory, subpath)


if __name__ == '__main__':
	app.run(debug=True, threaded=False)
