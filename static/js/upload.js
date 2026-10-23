(function () {
  const form = document.getElementById("upload-form");
  const input = document.getElementById("file");
  const drop = document.getElementById("drop");
  const title = document.getElementById("drop-title");
  const sub = document.getElementById("drop-sub");
  const submit = document.getElementById("submit");
  const busy = document.getElementById("busy");
  if (!form) return;
  const OK = /\.(pdf|png|jpe?g)$/i;
  const MAX = 20 * 1024 * 1024;

  function show(file) {
    if (!file) { submit.disabled = true; return; }
    const kb = file.size / 1024;
    const size = kb > 1024 ? (kb / 1024).toFixed(1) + " MB" : Math.max(1, Math.round(kb)) + " KB";
    title.textContent = file.name;
    if (!OK.test(file.name)) { sub.textContent = "Unsupported type — choose a PDF, PNG, JPG or JPEG."; submit.disabled = true; }
    else if (file.size > MAX) { sub.textContent = size + " — exceeds the 20 MB limit."; submit.disabled = true; }
    else { sub.textContent = size + " · ready to analyse"; submit.disabled = false; }
    drop.classList.add("has-file");
  }

  input.addEventListener("change", () => show(input.files[0]));
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("is-over")));
  drop.addEventListener("drop", (ev) => {
    ev.preventDefault();
    if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; show(input.files[0]); }
  });
  form.addEventListener("submit", () => { submit.disabled = true; submit.textContent = "Analysing…"; busy.hidden = false; });
  window.addEventListener("pageshow", () => { submit.textContent = "Analyse Certificate"; busy.hidden = true; show(input.files[0]); });
})();
