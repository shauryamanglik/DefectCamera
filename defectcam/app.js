/* DefectCam — on-device defect detection
 * Approach: MobileNet patch embeddings.
 *  - GOOD samples build a "normal" reference bank of patch embeddings.
 *  - BAD samples contribute painted-region patch embeddings as defect signatures (with labels).
 *  - Live: tile the frame, embed each patch, score = distance to nearest good patch,
 *    boosted if the patch is also near a known defect signature. A patch is flagged if it
 *    deviates from the good manifold beyond the threshold.
 */

const $ = id => document.getElementById(id);
const DB_KEY = 'defectcam_v1';

let model = null;
let stream = null;
let facing = 'environment';
let liveOn = false;
let threshold = 0.35;

// dataset kept in memory + persisted
// good: [{img(dataURL), emb:[Float32Array patch embeddings]}]
// bad:  [{img, mask(dataURL), label, emb:[patch embeddings inside mask]}]
let ds = { good: [], bad: [] };
let goodBank = [];         // flat array of Float32Array (normalized) — good patch embeddings
let defectBank = [];       // [{emb, label}]
let built = false;

const PATCH = 96;          // px size of a patch fed to embedder (resized internally)
const GRID = 5;            // live inspection grid (GRID x GRID patches)

/* ---------- model load ---------- */
async function loadModel(){
  try{
    await tf.setBackend('webgl'); await tf.ready();
    model = await mobilenet.load({version:2, alpha:0.5});
    // warmup
    tf.tidy(()=>model.infer(tf.zeros([1,224,224,3]), true));
    $('status').textContent = 'model ready';
  }catch(e){
    $('status').textContent = 'model failed';
    banner('err','Model failed to load. Check your internet on first run (model caches after that).');
    console.error(e);
  }
}

/* ---------- embedding helpers ---------- */
// embed one image element/canvas region -> normalized Float32Array
function embed(source){
  return tf.tidy(()=>{
    let t = tf.browser.fromPixels(source).toFloat();
    t = tf.image.resizeBilinear(t,[224,224]).div(255).expandDims(0);
    let f = model.infer(t, true);           // [1,1280] pooled features
    f = f.div(f.norm(2,-1,true));           // L2 normalize -> cosine space
    return f.squeeze();
  });
}
async function embedToArray(source){
  const t = embed(source);
  const a = await t.data();
  t.dispose();
  return Float32Array.from(a);
}
// cosine distance for normalized vectors = 1 - dot
function cosDist(a,b){
  let dot=0; for(let i=0;i<a.length;i++) dot+=a[i]*b[i];
  return 1-dot;
}
function minDist(v, bank){
  let m=2; for(const b of bank){ const d=cosDist(v,b); if(d<m)m=d; } return m;
}

/* crop a region from an image into an offscreen canvas of PATCH size */
const _c = document.createElement('canvas'); _c.width=PATCH; _c.height=PATCH;
const _cx = _c.getContext('2d',{willReadFrequently:true});
function cropPatch(imgEl, sx, sy, sw, sh){
  _cx.clearRect(0,0,PATCH,PATCH);
  _cx.drawImage(imgEl, sx,sy,sw,sh, 0,0,PATCH,PATCH);
  return _c;
}

/* ---------- build detector from dataset ---------- */
async function buildDetector(){
  if(!model){ banner('err','Model not ready yet.'); return; }
  $('buildBtn').innerHTML = '<span class="spin"></span> Building…';
  $('buildBtn').disabled = true;
  await new Promise(r=>setTimeout(r,30));

  goodBank = []; defectBank = [];

  // GOOD: tile each good image into GRID x GRID patches, embed all
  for(const g of ds.good){
    const img = await loadImg(g.img);
    const pw = img.naturalWidth/GRID, ph = img.naturalHeight/GRID;
    for(let r=0;r<GRID;r++){
      for(let c=0;c<GRID;c++){
        const patch = cropPatch(img, c*pw, r*ph, pw, ph);
        goodBank.push(await embedToArray(patch));
      }
    }
  }

  // BAD: use the painted mask to find defect regions, embed patches that overlap the mask
  for(const b of ds.bad){
    const img = await loadImg(b.img);
    const mask = await loadImg(b.mask);
    const mc = document.createElement('canvas');
    mc.width=GRID; mc.height=GRID;
    const mcx = mc.getContext('2d');
    mcx.drawImage(mask,0,0,GRID,GRID);
    const md = mcx.getImageData(0,0,GRID,GRID).data;
    const pw = img.naturalWidth/GRID, ph = img.naturalHeight/GRID;
    for(let r=0;r<GRID;r++){
      for(let c=0;c<GRID;c++){
        const alpha = md[(r*GRID+c)*4+3];
        if(alpha>40){ // this cell contains painted defect
          const patch = cropPatch(img, c*pw, r*ph, pw, ph);
          defectBank.push({emb:await embedToArray(patch), label:b.label});
        }
      }
    }
  }

  built = goodBank.length>0;
  $('buildBtn').innerHTML = 'Rebuild detector';
  $('buildBtn').disabled = false;
  $('modelState').textContent = built ? `ready · ${goodBank.length} good / ${defectBank.length} defect patches` : 'need good samples';
  $('modelState').style.color = built ? 'var(--ok)' : 'var(--dim)';
  if(built){
    banner('info', `Detector built from ${goodBank.length} good patches and ${defectBank.length} defect patches. Go to Inspect and start live.`);
    $('liveBtn').disabled = false;
    $('inspectHint').className='banner info';
    $('inspectHint').innerHTML='Detector ready. Point the camera at a part and press <b>Start live inspection</b>.';
  }
}

/* ---------- live inspection ---------- */
async function inspectFrame(){
  if(!liveOn || !built) return;
  const v = $('video');
  const ov = $('overlay');
  const ctx = ov.getContext('2d');
  if(v.videoWidth===0){ requestAnimationFrame(inspectFrame); return; }

  ov.width = v.videoWidth; ov.height = v.videoHeight;
  ctx.clearRect(0,0,ov.width,ov.height);

  const pw = v.videoWidth/GRID, ph = v.videoHeight/GRID;
  let worst = 0, flagged = 0, worstLabel = null;

  // embed all patches
  for(let r=0;r<GRID;r++){
    for(let c=0;c<GRID;c++){
      const patch = cropPatch(v, c*pw, r*ph, pw, ph);
      const e = await embedToArray(patch);
      const dGood = minDist(e, goodBank);         // higher = more abnormal
      let dDefect = 2, dlabel=null;
      if(defectBank.length){
        for(const d of defectBank){ const dd=cosDist(e,d.emb); if(dd<dDefect){dDefect=dd; dlabel=d.label;} }
      }
      // score: abnormality relative to good manifold; defect proximity nudges up
      let score = dGood;
      if(dDefect < 0.30) score = Math.max(score, 0.5 + (0.30-dDefect)); // clear defect match
      if(score > worst){ worst=score; worstLabel = (dDefect<0.35?dlabel:null); }

      if(score > threshold){
        flagged++;
        const inten = Math.min(1,(score-threshold)/0.4);
        ctx.fillStyle = `rgba(255,77,94,${0.18+inten*0.35})`;
        ctx.fillRect(c*pw, r*ph, pw, ph);
        ctx.strokeStyle = `rgba(255,77,94,0.9)`; ctx.lineWidth = Math.max(2,ov.width/300);
        ctx.strokeRect(c*pw+1, r*ph+1, pw-2, ph-2);
        if(dDefect<0.35 && dlabel){
          ctx.fillStyle='#fff'; ctx.font=`${Math.max(14,ov.width/34)}px -apple-system,sans-serif`;
          ctx.fillText(dlabel, c*pw+6, r*ph+ph-8);
        }
      }
    }
  }

  const bad = flagged>0;
  const vd = $('verdict');
  vd.className = 'verdict ' + (bad?'bad':'ok');
  $('verdictText').textContent = bad ? (worstLabel?`DEFECT · ${worstLabel}`:'DEFECT') : 'PASS';
  $('verdictScore').textContent = `dev ${worst.toFixed(2)} · ${flagged}/${GRID*GRID}`;

  requestAnimationFrame(inspectFrame);
}

/* ---------- camera ---------- */
async function startCam(videoEl){
  stopCam();
  try{
    stream = await navigator.mediaDevices.getUserMedia({
      video:{ facingMode:facing, width:{ideal:1280}, height:{ideal:960} }, audio:false
    });
    document.querySelectorAll('video').forEach(v=>{ v.srcObject=stream; v.play().catch(()=>{}); });
    $('flipBtn').disabled=false; $('capGood').disabled=false; $('capBad').disabled=false;
    $('status').textContent='camera live';
  }catch(e){
    banner('err','Camera blocked. On iPhone you must open this over HTTPS and allow camera access in Safari.');
    console.error(e);
  }
}
function stopCam(){ if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; } }

/* ---------- capture ---------- */
function grabFrame(videoEl){
  const c=document.createElement('canvas');
  c.width=videoEl.videoWidth; c.height=videoEl.videoHeight;
  c.getContext('2d').drawImage(videoEl,0,0);
  return c.toDataURL('image/jpeg',0.85);
}
async function captureGood(){
  const v = $('video2').videoWidth?$('video2'):$('video');
  if(!v.videoWidth){ banner('warn','Start the camera first.'); return; }
  ds.good.push({img: grabFrame(v)});
  persist(); refreshCounts(); refreshThumbs();
}
let pendingBad = null;
async function captureBad(fromDataURL){
  const src = fromDataURL || (function(){
    const v=$('video2').videoWidth?$('video2'):$('video');
    if(!v.videoWidth){ banner('warn','Start the camera first.'); return null; }
    return grabFrame(v);
  })();
  if(!src) return;
  pendingBad = src;
  openPaint(src);
}

/* ---------- paint modal ---------- */
let paintCtx, maskCtx, painting=false, tool='brush', curLabel='scratch', brush=30, paintImg=null;
function openPaint(dataURL){
  $('paint').classList.remove('hidden');
  paintImg = new Image();
  paintImg.onload = ()=>{
    const base=$('paintBase'), mask=$('paintMask');
    const maxW = Math.min(window.innerWidth-24, 900);
    const maxH = window.innerHeight-230;
    let w=paintImg.naturalWidth, h=paintImg.naturalHeight;
    const scale=Math.min(maxW/w, maxH/h, 1);
    w=Math.round(w*scale); h=Math.round(h*scale);
    base.width=w; base.height=h; mask.width=w; mask.height=h;
    paintCtx=base.getContext('2d'); maskCtx=mask.getContext('2d');
    paintCtx.drawImage(paintImg,0,0,w,h);
    maskCtx.clearRect(0,0,w,h);
  };
  paintImg.src = dataURL;
}
function paintAt(x,y){
  maskCtx.globalCompositeOperation = tool==='erase'?'destination-out':'source-over';
  maskCtx.fillStyle = 'rgba(255,77,94,0.55)';
  maskCtx.beginPath(); maskCtx.arc(x,y,brush/2,0,7); maskCtx.fill();
}
function maskPos(e){
  const mask=$('paintMask'); const r=mask.getBoundingClientRect();
  const p = e.touches?e.touches[0]:e;
  return { x:(p.clientX-r.left)*(mask.width/r.width), y:(p.clientY-r.top)*(mask.height/r.height) };
}

/* ---------- persistence ---------- */
function persist(){ try{ localStorage.setItem(DB_KEY, JSON.stringify(ds)); }catch(e){ banner('warn','Storage full — export your dataset to keep it.'); } }
function load(){ try{ const s=localStorage.getItem(DB_KEY); if(s) ds=JSON.parse(s); }catch(e){} }

/* ---------- UI wiring ---------- */
function banner(type,msg){ const b=$('inspectHint'); b.className='banner '+type; b.innerHTML=msg; }
function refreshCounts(){
  $('goodCount').textContent=ds.good.length;
  $('badCount').textContent=ds.bad.length;
  $('trainBadge').textContent=ds.good.length+ds.bad.length;
  $('buildBtn').disabled = ds.good.length<3;
  if(ds.good.length<3) $('modelState').textContent='need ≥3 good samples';
}
async function loadImg(src){ return new Promise((res)=>{ const i=new Image(); i.onload=()=>res(i); i.src=src; }); }
function refreshThumbs(){
  const wrap=$('thumbs'); wrap.innerHTML='';
  const all=[...ds.good.map((g,i)=>({...g,type:'good',i})),...ds.bad.map((b,i)=>({...b,type:'bad',i}))];
  $('noData').classList.toggle('hidden', all.length>0);
  all.forEach(item=>{
    const d=document.createElement('div'); d.className='thumb';
    d.innerHTML=`<img src="${item.img}"><span class="tag ${item.type}">${item.type==='good'?'GOOD':item.label||'BAD'}</span><span class="del">✕</span>`;
    d.querySelector('.del').onclick=()=>{ ds[item.type].splice(item.i,1); persist(); refreshCounts(); refreshThumbs(); built=false; };
    wrap.appendChild(d);
  });
  const labels=[...new Set(ds.bad.map(b=>b.label))];
  $('labelList').textContent = labels.length?labels.join(' · '):'—';
}

// tabs
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  ['inspect','train','data'].forEach(n=>$('tab-'+n).classList.toggle('hidden', n!==t.dataset.tab));
  if(t.dataset.tab==='data') refreshThumbs();
});

// inspect controls
$('camBtn').onclick=()=>{ startCam($('video')); $('camBtn').textContent='Restart camera'; };
$('flipBtn').onclick=()=>{ facing = facing==='environment'?'user':'environment'; startCam($('video')); };
$('liveBtn').onclick=()=>{
  liveOn=!liveOn;
  $('liveBtn').textContent = liveOn?'■ Stop live inspection':'▶ Start live inspection';
  if(liveOn) inspectFrame(); else { $('verdict').className='verdict idle'; $('verdictText').textContent='Stopped'; $('overlay').getContext('2d').clearRect(0,0,9999,9999); }
};
$('thresh').oninput=e=>{ threshold=parseFloat(e.target.value); $('threshVal').textContent=threshold.toFixed(2); };

// train controls
$('camBtn2').onclick=()=>{ startCam($('video2')); $('camBtn2').textContent='Restart camera'; };
$('capGood').onclick=captureGood;
$('capBad').onclick=()=>captureBad(null);
$('buildBtn').onclick=buildDetector;
$('importBtn').onclick=()=>$('fileIn').click();
$('fileIn').onchange=e=>{
  [...e.target.files].forEach(f=>{ const rd=new FileReader(); rd.onload=()=>{ // ask good or bad
    if(confirm('Is this a GOOD part? (Cancel = bad part, will open paint tool)')){ ds.good.push({img:rd.result}); persist(); refreshCounts(); refreshThumbs(); }
    else captureBad(rd.result);
  }; rd.readAsDataURL(f); });
  e.target.value='';
};

// paint controls
$('toolBrush').onclick=()=>{tool='brush';$('toolBrush').classList.add('on');$('toolErase').classList.remove('on');};
$('toolErase').onclick=()=>{tool='erase';$('toolErase').classList.add('on');$('toolBrush').classList.remove('on');};
$('brushSize').oninput=e=>brush=+e.target.value;
document.querySelectorAll('#labelChips .label').forEach(ch=>ch.onclick=()=>{
  if(ch.id==='customChip'){ const c=prompt('Custom defect label:'); if(!c)return; curLabel=c.toLowerCase(); ch.textContent='+ '+curLabel; }
  else curLabel=ch.dataset.label;
  document.querySelectorAll('#labelChips .label').forEach(x=>x.classList.remove('on'));
  ch.classList.add('on');
});
function bindPaint(){
  const mask=$('paintMask');
  const start=e=>{painting=true; const p=maskPos(e); paintAt(p.x,p.y); e.preventDefault();};
  const move=e=>{ if(!painting)return; const p=maskPos(e); paintAt(p.x,p.y); e.preventDefault(); };
  const end=()=>painting=false;
  mask.addEventListener('mousedown',start); mask.addEventListener('mousemove',move); window.addEventListener('mouseup',end);
  mask.addEventListener('touchstart',start,{passive:false}); mask.addEventListener('touchmove',move,{passive:false}); mask.addEventListener('touchend',end);
}
$('paintClear').onclick=()=>maskCtx&&maskCtx.clearRect(0,0,$('paintMask').width,$('paintMask').height);
$('paintCancel').onclick=()=>{ $('paint').classList.add('hidden'); pendingBad=null; };
$('paintSave').onclick=()=>{
  // check something was painted
  const md=maskCtx.getImageData(0,0,$('paintMask').width,$('paintMask').height).data;
  let any=false; for(let i=3;i<md.length;i+=4){ if(md[i]>0){any=true;break;} }
  if(!any){ alert('Paint over the defect first.'); return; }
  ds.bad.push({ img:pendingBad, mask:$('paintMask').toDataURL('image/png'), label:curLabel });
  persist(); refreshCounts(); refreshThumbs(); built=false;
  $('paint').classList.add('hidden'); pendingBad=null;
};

// data controls
$('exportBtn').onclick=()=>{
  const blob=new Blob([JSON.stringify(ds)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=`defectcam-dataset-${Date.now()}.json`; a.click();
};
$('loadBtn').onclick=()=>$('loadIn').click();
$('loadIn').onchange=e=>{ const rd=new FileReader(); rd.onload=()=>{ try{ ds=JSON.parse(rd.result); persist(); refreshCounts(); refreshThumbs(); built=false; banner('info','Dataset loaded. Rebuild the detector.'); }catch(x){alert('Bad file.');} }; rd.readAsText(e.target.files[0]); };
$('clearBtn').onclick=()=>{ if(confirm('Delete all samples and reset?')){ ds={good:[],bad:[]}; goodBank=[];defectBank=[];built=false; persist(); refreshCounts(); refreshThumbs(); } };

/* ---------- boot ---------- */
load(); refreshCounts(); bindPaint(); loadModel();
if('serviceWorker' in navigator){ navigator.serviceWorker.register('sw.js').catch(()=>{}); }
