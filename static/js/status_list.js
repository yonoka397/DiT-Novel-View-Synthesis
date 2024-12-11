/* funciton of status list */
import { globalState } from './global_var.js';

const expand_btn = document.getElementById('button-status-list-expand');
const status_list = document.getElementById('status-list');
const overlay_status_list = document.getElementById('overlay-status-list');
const statusRecords = document.getElementById('status-records');


/* expand/close status list when click expand button */
expand_btn.addEventListener('click', function () {
	status_list.classList.toggle('open');
	overlay_status_list.classList.toggle('active');
	this.classList.toggle('active');
	adjustExpandBtnBottom();
	statusRecords.scrollTop = statusRecords.scrollHeight;
});

/* close status list when click outside of status list */
overlay_status_list.addEventListener('click', function () {
	status_list.classList.remove('open');
	overlay_status_list.classList.remove('active');
	expand_btn.classList.toggle('active');
	adjustExpandBtnBottom();
	statusRecords.scrollTop = statusRecords.scrollHeight;
});

/* expand status list when click status list */
status_list.addEventListener('click', function () {
	// only expand, no close
	if (!status_list.classList.contains('open')) {
		status_list.classList.add('open');
		overlay_status_list.classList.add('active');
		expand_btn.classList.add('active');
		adjustExpandBtnBottom();
		statusRecords.scrollTop = statusRecords.scrollHeight;
	}
});

/* animation of expand button */
function adjustExpandBtnBottom() {
	const status_item_current = document.getElementById("status-item-current");
	const progress_bar_container = document.getElementById("progress-bar-container");
	let statusHeight;
	if (expand_btn.classList.contains('active')) {
		const maxHeight = window.innerHeight * 0.5; // get 50vh
		const scrollHeigh = status_item_current.scrollHeight + progress_bar_container.offsetHeight + statusRecords.scrollHeight;
		statusHeight = (scrollHeigh <= maxHeight) ? scrollHeigh + 4 : maxHeight + 4;
		expand_btn.style.transition = 'right 0.7s ease, bottom 0.5s ease, border 1.0s ease, transform 0.7s ease';
	}
	else {
		statusHeight = globalState.showBar ? 118 : 46;
		expand_btn.style.transition = 'right 0.7s ease, bottom 1.2s ease, border 1.0s ease, transform 0.7s ease';
	}
	expand_btn.style.bottom = `${statusHeight - 12}px`;
}

/* set current time to newest status item  */
function updateStatusTime() {
	const now = new Date();

	let hours = now.getHours();
	let minutes = now.getMinutes();
	const ampm = hours >= 12 ? 'PM' : 'AM';

	hours = hours % 12;
	hours = hours ? hours : 12;
	minutes = minutes < 10 ? '0' + minutes : minutes;

	const timeString = `${hours}&nbsp;:&nbsp;${minutes}&nbsp;&nbsp;${ampm}`;

	const timeElement = document.getElementById('status-time-current');
	timeElement.innerHTML = timeString;
}


/* status list text animation */
let text;

let animationTimeoutId;

function wrapTextInSpans() {
	const container = document.getElementById('status-text-current');

	text = container.innerText;

	container.innerHTML = '';

	// add each char into span
	for (let i = 0; i < text.length; i++) {
		const span = document.createElement('span');
		span.classList.add('char');

		// preserve space char
		if (text[i] === ' ') {
			span.innerHTML = '&nbsp;';
		} else {
			span.innerText = text[i];
		}
		container.appendChild(span);
	}

	startAnimationLoop();
}

/* text animation loop */
function startAnimationLoop() {
	const charTimeDelay = 0.15;
	const time_per_loop = charTimeDelay * 1000 * text.length + 2000;

	const chars = document.querySelectorAll('.char');

	// set each char time delay
	chars.forEach((char, index) => {
		char.style.animationDelay = `${index * charTimeDelay}s`;
	});

	// clear previous animation
	clearTimeout(animationTimeoutId);

	// set animation to every char
	animationTimeoutId = setTimeout(() => {
		chars.forEach(char => {
			char.style.animation = 'none';
			char.offsetHeight;
			char.style.animation = 'jump 1s ease-in-out';
		});

		// start animation loop
		startAnimationLoop();
	}, time_per_loop);
}

/* add current status into status list records */
function addStatusItem() {
	
	const currentText = document.getElementById('status-text-current');
	const currentTime = document.getElementById('status-time-current').textContent;
	
	const statusRecords = document.getElementById('status-records');

	const statusItem = document.createElement('div');
	statusItem.classList.add('status-item');

	// create status text
	const statusText = document.createElement('span');
	statusText.classList.add('status-text');
	statusText.textContent = currentText.textContent;
	statusText.style.color = currentText.style.color;

	// create status time
	const statusTime = document.createElement('span');
	statusTime.classList.add('status-time');
	statusTime.textContent = currentTime;

	// add status text and time into status item
	statusItem.appendChild(statusText);
	statusItem.appendChild(statusTime);

	// add status item into status records
	statusRecords.appendChild(statusItem);
}

window.addEventListener('resize', adjustExpandBtnBottom);

export {
	addStatusItem,
	adjustExpandBtnBottom,
	wrapTextInSpans,
	updateStatusTime,
};
