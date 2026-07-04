
// IMAGE PREVIEW (predict page)
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function () {
        const output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
}

// LOADING EFFECT (future upgrade)
function showLoading() {
    const btn = document.getElementById("predictBtn");
    if (btn) {
        btn.innerHTML = "Analyzing Image...";
        btn.disabled = true;
    }
}

// CONFIDENCE ANIMATION (result page)
function animateConfidence() {
    const el = document.getElementById("confidence");
    if (!el) return;

    let value = parseFloat(el.innerText);
    let start = 0;

    const interval = setInterval(() => {
        if (start >= value) clearInterval(interval);
        el.innerText = start.toFixed(1) + "%";
        start += 1;
    }, 15);
}

// AUTO RUN
window.onload = function () {
    animateConfidence();
};