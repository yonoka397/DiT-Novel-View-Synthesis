/* funciton of image zoom-in modal */
const modal = document.getElementById("modal");
const modalContent = document.querySelector(".modal-content");
const close_btn = document.getElementById("btn-close-modal");
const modalImg = document.getElementById("modal-img");

close_btn.addEventListener("click", () => {
	closeModal();
});

modal.addEventListener("click", (event) => {
	closeModalOnOutsideClick(event);
});

function openModal(src) {
	modalImg.src = src;
	modal.classList.add("show");
}

function closeModal() {
	modal.classList.remove("show");
}

function closeModalOnOutsideClick(event) {
	if (!modalContent.contains(event.target)) {
		closeModal();
	}
}

export { openModal };
