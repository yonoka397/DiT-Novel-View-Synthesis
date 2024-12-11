/* funciton of dropdown that select which dataset the model train on */
import { globalState } from './global_var.js';
import { showUploadContainer } from './script.js';

const dropdown_model = document.getElementById("dropdown-model");
const dropdown_btn = document.getElementById("dropdown-btn");
const item1 = document.getElementById("item1");
const item2 = document.getElementById("item2");

dropdown_model.addEventListener("mouseleave", () => {
	dropdown_model.classList.remove('active');
});

dropdown_btn.addEventListener("click", () => {
	if (globalState.disableButton) {
		alert('Cannot reselect model weights when running progress.');
	} else {
		dropdown_model.classList.toggle('active');
	}
});

item1.addEventListener("click", (event) => {
	selectItem(event);
});

item2.addEventListener("click", (event) => {
	selectItem(event);
});

/* save which dataset train on */
function selectItem(event) {
	const isNotSelected = (globalState.modelSelected == null);
	event.preventDefault();
	const selectedItem = event.target.textContent;
	globalState.modelSelected = selectedItem.split("train on ")[1];
	dropdown_btn.innerHTML = `<span class="arrow">&#9662;</span> ${selectedItem}`;
	dropdown_model.classList.remove('active');
	if (isNotSelected) {
		showUploadContainer();
	}
}
