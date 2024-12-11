/* main of javascript including global content and functionals */
import { globalState } from './global_var.js';
import { openModal } from './modal_image.js';
import {
	addStatusItem,
	adjustExpandBtnBottom,
	wrapTextInSpans,
	updateStatusTime,
} from './status_list.js';

const upload_container = document.getElementById("upload-container");
const preview_container = document.getElementById("preview-container");
const button_image_upload = document.getElementById("button-image-upload");


let error_occured = false;

let upload_image_url;
let uploaded_image_path;
let model_path;


/* main button to start sampling image */
button_image_upload.addEventListener("click", async () => {
	/* reject resample while previous process isn't complete */
	if (globalState.disableButton) {
		alert('Cannot upload image when running progress.');
		return;
	}
	hideUploadContainer();
	globalState.disableButton = true;
	
	/* upload image to the server(default pah: ./uploads/*.jpeg) */
	error_occured = false;
	try {
		await uploadFile(upload_image_url);
	}catch(error){
		error_occured = true;
	}
	
	/* return when error occur in upload image */
	if (error_occured) {
		globalState.disableButton = false;
		return;
	}
	
	/* check the model weights existence */
	model_path = "./model_weights_" + globalState.modelSelected + ".pth";
	try {
		const exists = await isExist(model_path);
		
		if (exists) {
			updateStatus("model weights existed.", "#333");
		} else {
			/* if model weights not found then download model weights */
			updateStatus("model weights not found.", "#ABAB49");
			updateStatus("downloading model weights...", "#5630DC");
			
			showProgressBar();
			
			const result = await downloadModelWeights();
			if (result) {
				updateStatus("model weights downloaded!", "#45B08A");
			} else {
				updateStatus("model weights download failed.", "#DC584F");
				error_occured = true;
			}
			closeProgressBar();
		}
	} catch (error) {
		updateStatus("error downloading weights.", "#DC584F");
		error_occured = true;
	}
	
	
	/* return when error occur in upload image */
	if (error_occured) {
		globalState.disableButton = false;
		return;
	}
	
	/* sample image */
	updateStatus("sampling image...", "#5630DC");
	showProgressBar();
	try {
		const result = await sampleImage(uploaded_image_path);
		if (result) {
			updateStatus("image sampled!", "#45B08A");
			const img = document.getElementById("sampled-image");
			img.src = result;
			img.onclick = () => openModal(img.src);
			
			const img_container = document.getElementById("result-container");
			img_container.style.display = "block"
			
		} else {
			updateStatus("image sample failed.", "#DC584F");
			error_occured = true;
		}
	} catch (error) {
		updateStatus("error sampling image.", "#DC584F");
		error_occured = true;
	}
	closeProgressBar();
	
	/* scroll window to show the result */
	window.scrollTo({
		top: document.body.offsetHeight,
		behavior: "smooth"
	});
	globalState.disableButton = false;
});

/* process file loading */
function handleFiles(files) {
	let validFileFound = false;
	[...files].forEach(file => {
		if (file.type.startsWith("image/")) {
			validFileFound = true;
			
			previewImg(file);
			
		} else {
			alert("Unsupport image format");
		}
	});
	
	if (validFileFound) {
		hideUploadContainer();
		
		preview_container.style.display = "block";
	}
}

/* add loaded image to preview zone */
function previewImg(file){
	const reader = new FileReader();
		reader.onload = () => {
			const img = document.getElementById("original-image");
			img.src = reader.result;
			img.onclick = () => openModal(img.src);
			
			img.onload = () => {
				const resizedImage = document.getElementById("resized-image");
				resizeImage(reader.result).then((resizedImg) => {
					resizedImage.src = resizedImg;
					resizedImage.onclick = () => openModal(resizedImage.src);
					upload_image_url = resizedImg;
				});
			};
		};
		reader.readAsDataURL(file);
}

/* resize oeiginal image to specific size */
function resizeImage(imageSrc) {
	const img = new Image();
	img.src = imageSrc;
	return new Promise((resolve) => {
		img.onload = () => {
			const canvas = document.createElement("canvas");
			const ctx = canvas.getContext("2d");
			const targetSize = 64;
			canvas.width = targetSize;
			canvas.height = targetSize;
			ctx.drawImage(img, 0, 0, targetSize, targetSize);
			resolve(canvas.toDataURL("image/jpeg"));
		};
	});
}

/* upload image to the server(default pah: ./uploads/*.jpeg) */
function uploadFile(file_url) {
	return new Promise((resolve, reject) => {
		updateStatus("uploading image...", "#5630DC");
		
		const byteString = atob(file_url.split(',')[1]);
		const arrayBuffer = new ArrayBuffer(byteString.length);
		const uintArray = new Uint8Array(arrayBuffer);
		for (let i = 0; i < byteString.length; i++) {
			uintArray[i] = byteString.charCodeAt(i);
		}
		const blob = new Blob([uintArray], { type: "image/jpeg" });
		
		
		const formData = new FormData();
		formData.append("file", blob, "input.jpeg");

		fetch("/upload", {
			method: "POST",
			body: formData,
		})
		.then(response => response.json())
		.then(data => {
			if (data.uploaded_image_path) {
				updateStatus("image uploaded successfully!", "#45B08A");
				uploaded_image_path = data.uploaded_image_path;
				resolve();
			} else {
				updateStatus("image upload failed.", "#DC584F");
				error_occured = true;
				reject();
			}
		})
		.catch(error => {
			updateStatus("error uploading image.", "#DC584F");
			error_occured = true;
			reject(error);
		});
	});
	
}

/* check the file existence */
function isExist(filePath) {
	return fetch(`/check_file_exists?file_path=${encodeURIComponent(filePath)}`)
		.then(response => {
			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}
			return response.json();
		})
		.then(data => {
			if (data.exists) {
				return true;
			} else {
				return false;
			}
		})
		.catch(error => {
			console.error("error when checking file:", error);
			return false;
		});
}

/* download model weights */
async function downloadModelWeights() {

	const eventSource = new EventSource('/progress');

	eventSource.onmessage = function(event) {
		const percent = (Math.round(parseFloat(event.data) * 100) / 100);
		const formattedPercent = percent.toFixed(1); 
		
		const progressBar = document.getElementById('progress-bar');
		const progressLabel = document.getElementById('progress-label');
		
		progressBar.style.width = formattedPercent + '%';
		progressLabel.textContent = 'downloading model weights：' + formattedPercent + '%';

		if (percent >= 100) {
			eventSource.close();
		}
	};	
	
	const response = await fetch('/download_model_weights', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			dataset_train: globalState.modelSelected,
		}),
	})
	
	if (!response.ok) {
		const error = await response.json();
		alert('Error starting downloading: ' + error.error);
		return;
	}
	
	return new Promise((resolve, reject) => {
		const checkStatus = async () => {
			const statusResponse = await fetch('/download_result', {
				method: 'GET',
			});
			const download_complete = await statusResponse.json();

			if (download_complete.download_complete) {
				resolve(download_complete.download_complete);
			} else {
				setTimeout(checkStatus, 1000);
			}
		};

		checkStatus();
	});
}

/* sample image */
async function sampleImage(imgPath) {

	const eventSource = new EventSource('/progress');

	eventSource.onmessage = function(event) {
		const percent = (Math.round(parseFloat(event.data) * 100) / 100);
		const formattedPercent = percent.toFixed(1); 
		
		const progressBar = document.getElementById('progress-bar');
		const progressLabel = document.getElementById('progress-label');
		
		progressBar.style.width = formattedPercent + '%';
		progressLabel.textContent = 'sampling image：' + formattedPercent + '%';

		if (percent >= 100) {
			eventSource.close();
		}
	};	
	
	const response = await fetch('/sample', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			img_path: imgPath,
			dataset_train: globalState.modelSelected,
		}),
	})
	
	if (!response.ok) {
		const error = await response.json();
		alert('Error starting sampling: ' + error.error);
		return;
	}
	
	return new Promise((resolve, reject) => {
		const checkStatus = async () => {
			const statusResponse = await fetch('/sample_result', {
				method: 'GET',
			});
			const output_image_path = await statusResponse.json();

			if (output_image_path.output_image_path) {
				resolve(output_image_path.output_image_path);
			} else {
				setTimeout(checkStatus, 1000);
			}
		};

		checkStatus();
	});
}

/* show the upload container (including dropbox and button select) */
function showUploadContainer() {
	upload_container.style.display = "block";
	
	const offset = upload_container.offsetHeight + 20;
	preview_container.style.transition = "none";
	preview_container.style.transform = `translateY(${-offset}px)`;
	
	setTimeout(() => {
		upload_container.classList.add("show");
		preview_container.style.transition = "transform 0.5s ease";
		preview_container.style.transform = `translateY(0px)`;
	}, 500);
}

/* hide the upload container (including dropbox and button select) */
function hideUploadContainer() {
	const offset = upload_container.offsetHeight - 20;	
	preview_container.style.transition = "none";
	preview_container.style.transform = `translateY(${offset}px)`;
	
	upload_container.classList.remove("show");
	setTimeout(() => {
		
		preview_container.style.transition = "transform 0.5s ease";
		preview_container.style.transform = `translateY(0px)`;
		upload_container.style.display = "none";
	}, 500);
}

/* show the progress bar */
function showProgressBar() {
	globalState.showBar = true;
	const progress_bar_container = document.getElementById("progress-bar-container");
	const status_list = document.getElementById('status-list');
	progress_bar_container.classList.toggle('showBar');
	status_list.classList.toggle('showBar');
	adjustExpandBtnBottom();
}

/* close the progress bar */
function closeProgressBar() {
	globalState.showBar = false;
	const progress_bar_container = document.getElementById("progress-bar-container");
	const status_list = document.getElementById('status-list');
	progress_bar_container.classList.toggle('showBar');
	status_list.classList.toggle('showBar');
	adjustExpandBtnBottom();
}

/* update current status and save old status to status records */
function updateStatus(newStatusText, textColor) {
	addStatusItem();
	const currentStatusText = document.getElementById('status-text-current');
	currentStatusText.innerText = newStatusText;
	if (textColor != "") {
		currentStatusText.style.color = textColor;
	}
	updateStatusTime();
	wrapTextInSpans();
}


export {
	handleFiles,
	showUploadContainer,
};

window.scrollTo({
	top: 0,
	behavior: "smooth"
});

updateStatusTime();

wrapTextInSpans();
