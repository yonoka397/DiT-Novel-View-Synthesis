/* funciton of image reselect button */
import { globalState } from './global_var.js';
import { showUploadContainer } from './script.js';

const button_image_reselect = document.getElementById("button-image-reselect");

button_image_reselect.addEventListener("click", () => {
	if (globalState.disableButton){
		alert('Cannot reselect image when running progress.');
	} else {
		showUploadContainer();
	}
});
