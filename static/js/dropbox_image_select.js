/* funciton of image select dropbox */
import { handleFiles } from './script.js';

const dropZone = document.getElementById("drop-zone");

["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
	dropZone.addEventListener(eventName, e => e.preventDefault());
	document.body.addEventListener(eventName, e => e.preventDefault());
});

["dragenter", "dragover"].forEach(eventName => {
	dropZone.addEventListener(eventName, () => dropZone.classList.add("dragover"));
});

["dragleave", "drop"].forEach(eventName => {
	dropZone.addEventListener(eventName, () => dropZone.classList.remove("dragover"));
});

dropZone.addEventListener("drop", e => {
	const files = e.dataTransfer.files;
	handleFiles(files);
});
