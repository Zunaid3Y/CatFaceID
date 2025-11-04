// Same-origin API base when served by FastAPI at '/'
const API = ""; // use relative paths like /identify, /enroll

// Convert any File/Blob to a resized JPEG Blob via canvas
async function fileToJpegBlob(file, maxSide = 1024, quality = 0.9) {
  const dataURL = await new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
  const img = await new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = reject;
    el.src = dataURL;
  });
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  let outW = w, outH = h;
  if (Math.max(w, h) > maxSide) {
    const s = maxSide / Math.max(w, h);
    outW = Math.round(w * s);
    outH = Math.round(h * s);
  }
  const canvas = document.createElement('canvas');
  canvas.width = outW; canvas.height = outH;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, outW, outH);
  const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', quality));
  if (!blob) throw new Error('JPEG encode failed');
  return blob;
}

// HTTP helpers
async function enroll(petId, files) {
  const fd = new FormData();
  fd.append("pet_id", petId);
  for (const f of files) {
    if (!f || !f.size) continue; // skip empty
    let blob;
    try {
      blob = await fileToJpegBlob(f, 1024, 0.9);
    } catch (e) {
      console.warn('convert failed, sending raw file', e);
      blob = f; // fallback
    }
    if (blob.size < 10_000) continue; // skip tiny/blank frames
    const name = (f && f.name ? f.name : 'frame') + '.jpg';
    fd.append("images", new File([blob], name, { type: 'image/jpeg' }));
  }
  const res = await fetch(`${API}/enroll`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Enroll failed (${res.status}): ${await res.text()}`);
  return res.json();
}

async function identifyFile(file) {
  const fd = new FormData();
  fd.append("image", file); // field name must be "image"
  const res = await fetch(`${API}/identify`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Identify failed (${res.status}): ${await res.text()}`);
  return res.json(); // {label, score, top3}
}

// Continuously capture frames from a video element and identify.
function startCameraIdentify(videoEl, canvasEl, outEl) {
  const ctx = canvasEl.getContext("2d");
  let inFlight = false;
  const timer = setInterval(async () => {
    const vw = videoEl.videoWidth|0, vh = videoEl.videoHeight|0;
    if(!vw || !vh || inFlight) return;
    // Downscale frame to reduce upload and inference load
    const maxW = 480;
    const scale = Math.min(1, maxW / vw);
    const w = Math.max(1, Math.round(vw * scale));
    const h = Math.max(1, Math.round(vh * scale));
    canvasEl.width = w; canvasEl.height = h;
    ctx.drawImage(videoEl, 0, 0, w, h);
    const blob = await new Promise(r => canvasEl.toBlob(r, "image/jpeg", 0.7));
    if(!blob) return;
    const fd = new FormData();
    fd.append("image", blob, "frame.jpg");
    try{
      inFlight = true;
      const res = await fetch(`${API}/identify`, { method: "POST", body: fd });
      if (!res.ok) { outEl.textContent = `Identify failed (${res.status})`; return; }
      const { label, score } = await res.json();
      outEl.textContent = `${label} (${(score*100).toFixed(1)}%)`;
    }catch(err){ outEl.textContent = String(err.message||err); }
    finally{ inFlight = false; }
  }, 1200);
  return timer;
}

// Elements
const tabIdentify = document.getElementById("tab-identify");
const tabEnroll = document.getElementById("tab-enroll");
const panelIdentify = document.getElementById("identify");
const panelEnroll = document.getElementById("enroll");
const statusEl = document.getElementById("status");

// Identify elements
const idFile = document.getElementById("id-file");
const idUseCam = document.getElementById("id-use-camera");
const idCapture = document.getElementById("id-capture");
const idVideo = document.getElementById("id-video");
const idCanvas = document.getElementById("id-canvas");
const idPreview = document.getElementById("id-preview");
const idRun = document.getElementById("id-run");
const idResults = document.getElementById("id-results");

// Enroll elements
const enrPetId = document.getElementById("enr-pet-id");
const enrFiles = document.getElementById("enr-files");
const enrUseCam = document.getElementById("enr-use-camera");
const enrCapture = document.getElementById("enr-capture");
const enrVideo = document.getElementById("enr-video");
const enrCanvas = document.getElementById("enr-canvas");
const enrPreview = document.getElementById("enr-preview");
const enrAdd = document.getElementById("enr-add");
const enrUpload = document.getElementById("enr-upload");
const enrGallery = document.getElementById("enr-gallery");

// Camera state
let idStream = null;
let idIdentifyTimer = null;
let enrStream = null;
let enrollBlobs = []; // blobs to upload

function setStatus(msg, isError=false){
  statusEl.textContent = msg;
  statusEl.style.color = isError ? '#b91c1c' : '';
}

function switchTab(which){
  if(which === 'identify'){
    tabIdentify.classList.add('active');
    tabEnroll.classList.remove('active');
    panelIdentify.classList.add('active');
    panelEnroll.classList.remove('active');
  } else {
    tabEnroll.classList.add('active');
    tabIdentify.classList.remove('active');
    panelEnroll.classList.add('active');
    panelIdentify.classList.remove('active');
  }
}

tabIdentify.addEventListener('click', ()=>switchTab('identify'));
tabEnroll.addEventListener('click', ()=>switchTab('enroll'));

// Camera helpers
async function startStream(videoEl){
  const constraints = {
    video: {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 15 }
    },
    audio: false
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  videoEl.srcObject = stream;
  await videoEl.play().catch(()=>{});
  return stream;
}
function stopStream(stream){ if(stream){ stream.getTracks().forEach(t=>t.stop()); } }

function drawVideoToCanvas(videoEl, canvasEl){
  const w = videoEl.videoWidth;
  const h = videoEl.videoHeight;
  if(!w || !h) return false;
  canvasEl.width = w; canvasEl.height = h;
  const ctx = canvasEl.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, w, h);
  return true;
}

function blobFromCanvas(canvasEl, type='image/jpeg', quality=0.92){
  return new Promise((resolve)=>{
    canvasEl.toBlob(b=>resolve(b), type, quality);
  });
}

function blobFromFileInput(inputEl){
  const f = inputEl.files && inputEl.files[0];
  return f ? f : null;
}

function showPreviewFromFile(inputEl, imgEl){
  const f = blobFromFileInput(inputEl);
  if(!f){ imgEl.hidden = true; return; }
  const url = URL.createObjectURL(f);
  imgEl.src = url; imgEl.hidden = false;
}

// Identify tab logic
idUseCam.addEventListener('change', async (e)=>{
  if(e.target.checked){
    try{
      idStream = await startStream(idVideo);
      // Drive predictions from canvas; keep the <video> hidden
      idVideo.hidden = true;
      idCanvas.hidden = false;
      idCapture.disabled = false;
      idPreview.hidden = true;
      // Start continuous identify loop if not already running
      if(!idIdentifyTimer){
        idIdentifyTimer = startCameraIdentify(idVideo, idCanvas, statusEl);
      }
    }catch(err){
      setStatus('Camera permission denied or not available', true);
      e.target.checked = false;
    }
  } else {
    idVideo.hidden = true; idCapture.disabled = true; idCanvas.hidden = true;
    stopStream(idStream); idStream = null;
    if(idIdentifyTimer){ clearInterval(idIdentifyTimer); idIdentifyTimer = null; }
  }
});

idCapture.addEventListener('click', ()=>{
  if(drawVideoToCanvas(idVideo, idCanvas)){
    idCanvas.hidden = false; idPreview.hidden = true;
  }
});

idFile.addEventListener('change', ()=>{
  showPreviewFromFile(idFile, idPreview);
  idCanvas.hidden = true;
});

idRun.addEventListener('click', async ()=>{
  statusEl.textContent = 'Identifying...';
  try{
    let blob = null;
    if(!idCanvas.hidden && idCanvas.width>0){
      blob = await blobFromCanvas(idCanvas);
    } else {
      blob = blobFromFileInput(idFile);
    }
    if(!blob) { statusEl.textContent = 'Please capture or select an image'; return; }

    const res = await identifyFile(blob);
    renderIdentify(res);
    statusEl.textContent = 'Identify OK!';
  }catch(e){ statusEl.textContent = e.message; }
});

function renderIdentify(js){
  const label = js.label || 'UNKNOWN';
  const scoreNum = (typeof js.score === 'number') ? js.score : NaN;
  const score = Number.isFinite(scoreNum) ? scoreNum.toFixed(3) : '—';
  const top3 = Array.isArray(js.top3) ? js.top3 : [];
  const cls = (label==='UNKNOWN'||label==='NO_FACE'||label==='NO_GALLERY') ? 'unknown' : 'known';
  let html = `<div class="${cls}"><strong>${label}</strong>  <span>score=${score}</span></div>`;
  if(top3.length){
    html += '<ul>' + top3.map(([n,s])=>`<li>${n}: ${(s).toFixed(3)}</li>`).join('') + '</ul>';
  }
  idResults.innerHTML = html;

  // Draw overlay on the preview canvas
  overlayIdentifyOnCanvas(label, scoreNum);
}

function overlayIdentifyOnCanvas(label, scoreNum){
  // Ensure we have an image on the canvas: if preview img is shown, draw it into the canvas
  if(idCanvas.hidden){
    if(idPreview && !idPreview.hidden && idPreview.naturalWidth){
      idCanvas.width = idPreview.naturalWidth;
      idCanvas.height = idPreview.naturalHeight;
      const ctx = idCanvas.getContext('2d');
      ctx.drawImage(idPreview, 0, 0, idCanvas.width, idCanvas.height);
      idCanvas.hidden = false;
      idPreview.hidden = true;
    } else {
      // If neither canvas nor img has content, nothing to draw
      return;
    }
  }

  const ctx = idCanvas.getContext('2d');
  // Draw a semi-transparent rounded rectangle with label and score
  const unknown = (label === 'UNKNOWN' || label === 'NO_FACE' || label === 'NO_GALLERY');
  const bg = unknown ? 'rgba(220, 38, 38, 0.9)' : 'rgba(22, 163, 74, 0.9)';
  const padX = 10, padY = 8;
  const margin = 10;
  const fontSize = Math.max(16, Math.round(idCanvas.width * 0.03));
  ctx.save();
  ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`;
  ctx.textBaseline = 'top';
  const text = Number.isFinite(scoreNum) ? `${label}  sim=${scoreNum.toFixed(3)}` : `${label}`;
  const metrics = ctx.measureText(text);
  const textW = metrics.width;
  const textH = fontSize * 1.4;
  const x = margin, y = margin;

  // Rounded rect
  const w = textW + padX * 2;
  const h = textH + padY * 2;
  const r = Math.min(12, h / 2);
  ctx.fillStyle = bg;
  roundedRect(ctx, x, y, w, h, r);
  ctx.fill();

  // Text
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, x + padX, y + padY);
  ctx.restore();
}

function roundedRect(ctx, x, y, w, h, r){
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// Enroll tab logic
enrUseCam.addEventListener('change', async (e)=>{
  if(e.target.checked){
    try{
      enrStream = await startStream(enrVideo);
      enrVideo.hidden = false; enrCapture.disabled = false;
      enrPreview.hidden = true; enrCanvas.hidden = true;
    }catch(err){ setStatus('Camera permission denied or not available', true); e.target.checked=false; }
  } else {
    enrVideo.hidden = true; enrCapture.disabled = true; enrCanvas.hidden = true;
    stopStream(enrStream); enrStream = null;
  }
});

enrCapture.addEventListener('click', ()=>{
  if(drawVideoToCanvas(enrVideo, enrCanvas)){
    enrCanvas.hidden = false; enrPreview.hidden = true;
  }
});

enrFiles.addEventListener('change', async ()=>{
  const files = Array.from(enrFiles.files || []);
  for (const f of files) {
    if (!f || !f.size) continue;
    try {
      // Convert to JPEG for consistent preview and upload size
      const jpeg = await fileToJpegBlob(f, 1024, 0.9);
      enrollBlobs.push(new File([jpeg], (f.name || 'frame') + '.jpg', { type: 'image/jpeg' }));
    } catch (e) {
      console.warn('preview convert failed, using original', e);
      enrollBlobs.push(f);
    }
  }
  refreshEnrollGallery();
});

enrAdd.addEventListener('click', async ()=>{
  if(!enrCanvas.hidden && enrCanvas.width>0){
    const b = await blobFromCanvas(enrCanvas);
    enrollBlobs.push(b);
    refreshEnrollGallery();
  } else if (enrFiles.files && enrFiles.files.length){
    // Already handled on change, but keep as fallback
    const files = Array.from(enrFiles.files);
    files.forEach(f=>enrollBlobs.push(f));
    refreshEnrollGallery();
  } else {
    setStatus('No image to add. Capture or select files.', true);
  }
});

function refreshEnrollGallery(){
  enrGallery.innerHTML = '';
  enrollBlobs.forEach((b, i)=>{
    const url = URL.createObjectURL(b);
    const img = document.createElement('img');
    img.src = url; img.className = 'thumb'; img.title = `#${i+1}`;
    enrGallery.appendChild(img);
  });
  setStatus(`Enroll set: ${enrollBlobs.length} image(s).`);
}

enrUpload.addEventListener('click', async ()=>{
  const petId = (enrPetId.value||'').trim();
  if(!petId){ statusEl.textContent = 'Enter a pet_id first.'; return; }
  if(!enrollBlobs.length){ statusEl.textContent = 'No images to upload.'; return; }
  statusEl.textContent = 'Uploading & rebuilding gallery...';
  try{
    const files = enrollBlobs;
    const res = await enroll(petId, files);
    statusEl.textContent = `Enroll OK: saved ${res.saved}`;
    // Reset local set
    enrollBlobs = []; refreshEnrollGallery();
  }catch(e){ statusEl.textContent = e.message; }
});
