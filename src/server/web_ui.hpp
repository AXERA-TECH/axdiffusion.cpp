// Single-page web UI served by axdiffusion-server at "/". Self-contained (no external assets), talks to the
// server's own OpenAI-compatible endpoints.
#pragma once

static const char *kWebUi = R"HTML(<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>axdiffusion</title>
<style>
:root{--bg:#f6f6f4;--panel:#fff;--ink:#1b1b1a;--muted:#6b6b66;--line:#e3e2dd;--accent:#d2552f;--accent-ink:#fff;--soft:#f0efea}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9a9993;--line:#2e2e2b;--accent:#e0704b;--accent-ink:#141413;--soft:#252522}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{display:flex;align-items:baseline;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.2px}
header .status{color:var(--muted);font-size:12.5px;margin-left:auto}
main{display:grid;grid-template-columns:360px 1fr;gap:0;min-height:calc(100vh - 52px)}
@media (max-width:860px){main{grid-template-columns:1fr}}
aside{border-right:1px solid var(--line);padding:16px 18px;display:flex;flex-direction:column;gap:14px;background:var(--panel)}
label{font-size:12.5px;color:var(--muted);display:block;margin-bottom:5px}
textarea,input,select{width:100%;font:inherit;color:var(--ink);background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
textarea{min-height:120px;resize:vertical}
.row{display:flex;gap:10px}.row>*{flex:1}
.check{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--muted);margin-top:6px}.check input{width:auto}
button{font:inherit;border:0;border-radius:8px;padding:10px 14px;cursor:pointer;background:var(--soft);color:var(--ink)}
button.primary{background:var(--accent);color:var(--accent-ink);font-weight:600;width:100%;padding:12px}
button:disabled{opacity:.55;cursor:default}
.lora{display:grid;grid-template-columns:1fr 70px 26px;gap:6px;align-items:center;margin-bottom:6px}
.lora input[type=range]{padding:0}
.lora .x{padding:4px;background:transparent;color:var(--muted)}
.small{font-size:12px;color:var(--muted)}
.refs{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}.refs img{width:64px;height:64px;object-fit:cover;border-radius:6px;border:1px solid var(--line)}
section{padding:18px 20px;overflow:auto}
.state{color:var(--muted);margin-bottom:12px;min-height:20px}
.err{color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card img{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;cursor:zoom-in}
.card .meta{padding:9px 11px;font-size:12px;color:var(--muted)}
.card .meta .p{color:var(--ink);font-size:12.5px;margin-bottom:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.card .meta a{color:var(--accent);text-decoration:none;margin-right:10px}
.empty{color:var(--muted);border:1px dashed var(--line);border-radius:10px;padding:40px;text-align:center}
#viewer{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9}
#viewer img{max-width:96vw;max-height:96vh}
</style>
</head>
<body>
<header><h1>axdiffusion</h1><span class="small" id="model"></span><span class="status" id="status">连接中…</span></header>
<main>
<aside>
  <div><label for="prompt">提示词</label><textarea id="prompt" placeholder="A photograph of an orange tabby cat sitting on a wooden windowsill in soft afternoon light"></textarea></div>
  <div class="row">
    <div><label for="size">尺寸</label><select id="size"><option value="1024x1024">1024 × 1024</option><option value="2048x2048">2048 × 2048</option><option value="custom">自定义…</option></select></div>
    <div><label for="n">张数</label><select id="n"><option>1</option><option>2</option><option>3</option><option>4</option></select></div>
  </div>
  <div class="row" id="customSize" style="display:none">
    <div><label for="cw">宽</label><input id="cw" type="number" min="256" step="32" value="1344"></div>
    <div><label for="ch">高</label><input id="ch" type="number" min="256" step="32" value="768"></div>
  </div>
  <div class="small" id="sizeHint" style="display:none"></div>
  <div><label for="seed">种子</label><input id="seed" type="number" min="0" value="42"><div class="check"><input type="checkbox" id="rand"><span>每次随机</span></div></div>
  <div><label>LoRA</label><div id="loras"></div><button id="addLora" type="button">+ 添加 LoRA</button> <span class="small" id="loraHint"></span></div>
  <div><label for="refs">参考图（编辑，可选）</label><input id="refs" type="file" accept="image/*" multiple><div class="refs" id="refPrev"></div></div>
  <button class="primary" id="go">生成</button>
  <div class="small">模型为 6 步蒸馏，无需设置步数与 CFG。同一时间只生成一张，多个请求会排队。</div>
</aside>
<section>
  <div class="state" id="state"></div>
  <div class="grid" id="grid"><div class="empty" id="empty">生成的图片会显示在这里</div></div>
</section>
</main>
<div id="viewer"><img alt=""></div>
<script>
const $ = s => document.querySelector(s);
let loraNames = [];
// Custom sizes: both sides rounded to a multiple of 32 (>= 256). The server answers with an error when the
// sequence (text + (W/32)*(H/32) image rows) falls between the installed kernel tiers.
const r32 = v => Math.max(256, Math.round((parseInt(v) || 0) / 32) * 32);
function chosenSize(){
  const v = $('#size').value;
  if (v !== 'custom') return v.split('x').map(Number);
  return [r32($('#cw').value), r32($('#ch').value)];
}
function sizeUi(){
  const custom = $('#size').value === 'custom';
  $('#customSize').style.display = custom ? '' : 'none';
  $('#sizeHint').style.display = custom ? '' : 'none';
  if (custom) {
    const [w, h] = chosenSize();
    $('#sizeHint').textContent = `实际 ${w} × ${h}（${(w * h / 1048576).toFixed(2)} MP，图像 ${(w / 32) * (h / 32)} 行；宽高取 32 的倍数）`;
  }
}
async function health(){
  try{
    const h = await (await fetch('/health')).json();
    $('#model').textContent = h.model || '';
    const sel = $('#size'), want = h.sizes || [];
    if (want.length && want.join() !== [...sel.options].map(o => o.value).join()) {
      const keep = sel.value, mp = v => { const [w, hh] = v.split('x').map(Number); return (w * hh / 1048576).toFixed(1); };
      sel.innerHTML = want.map(v => `<option value="${v}">${v.replace('x', ' × ')}（${mp(v)} MP）</option>`).join('') +
        '<option value="custom">自定义…</option>';
      sel.value = (want.includes(keep) || keep === 'custom') ? keep : (want.includes('1024x1024') ? '1024x1024' : want[0]);
    }
    $('#refs').disabled = !(h.tasks || []).includes('edit');
    $('#status').textContent = `就绪 · 已出图 ${h.requests} 张 · 排队 ${h.queued} · NPU 内存 ${h.cmm_used_gib.toFixed(1)} GiB`;
  }catch(e){ $('#status').textContent = '服务不可用'; }
}
async function loadLoras(){
  try{
    const d = await (await fetch('/v1/loras')).json();
    loraNames = (d.data||[]).map(x=>x.name);
    $('#loraHint').textContent = loraNames.length ? `可用 ${loraNames.length} 个` : '服务端未配置 LoRA 目录';
    $('#addLora').disabled = !loraNames.length;
  }catch(e){ $('#addLora').disabled = true; }
}
function addLora(){
  const row = document.createElement('div'); row.className='lora';
  const sel = document.createElement('select'); loraNames.forEach(n=>{const o=document.createElement('option');o.textContent=n;sel.appendChild(o);});
  const sc = document.createElement('input'); sc.type='number'; sc.step='0.05'; sc.value='1.0'; sc.title='强度';
  const x = document.createElement('button'); x.className='x'; x.type='button'; x.textContent='×'; x.onclick=()=>row.remove();
  row.append(sel, sc, x); $('#loras').appendChild(row);
}
$('#addLora').onclick = addLora;
$('#refs').onchange = () => {
  const p = $('#refPrev'); p.innerHTML='';
  [...$('#refs').files].forEach(f=>{const i=document.createElement('img');i.src=URL.createObjectURL(f);p.appendChild(i);});
};
function selectedLoras(){
  return [...document.querySelectorAll('#loras .lora')].map(r=>({name:r.querySelector('select').value, scale:parseFloat(r.querySelector('input').value)||1}));
}
function addCard(b64, prompt, info, loras){
  $('#empty')?.remove();
  const url = 'data:image/png;base64,' + b64;
  const c = document.createElement('div'); c.className='card';
  const img = document.createElement('img'); img.src=url; img.alt=prompt;
  img.onclick=()=>{ $('#viewer img').src=url; $('#viewer').style.display='flex'; };
  const m = document.createElement('div'); m.className='meta';
  const s = info && info.seconds ? info.seconds : {};
  const lt = loras.length ? ' · LoRA ' + loras.map(l=>`${l.name}@${l.scale}`).join(', ') : '';
  m.innerHTML = `<div class="p"></div>种子 ${info?info.seed:'-'} · ${info?(info.width+'×'+info.height):''} · 总 ${(s.total||0).toFixed(1)}s（DiT ${(s.dit||0).toFixed(1)}s）${lt}<br>`;
  m.querySelector('.p').textContent = prompt;
  const a = document.createElement('a'); a.href=url; a.download=`axdiffusion_${info?info.seed:Date.now()}.png`; a.textContent='下载'; m.appendChild(a);
  const r = document.createElement('a'); r.href='#'; r.textContent='用此种子'; r.onclick=e=>{e.preventDefault(); $('#seed').value=info.seed; $('#rand').checked=false;}; m.appendChild(r);
  c.append(img, m); $('#grid').prepend(c);
}
$('#viewer').onclick = () => $('#viewer').style.display='none';
const toB64 = f => new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = bad; r.readAsDataURL(f); });
const stageName = {lora:'构建 LoRA', text_encoder:'文本编码', denoise:'去噪', vae:'VAE 解码'};
let current = null;
async function runOne(body){
  const sub = await fetch('/v1/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const job = await sub.json();
  if(!sub.ok || job.error && !job.id) throw new Error(job.error ? job.error.message : sub.statusText);
  current = job.id;
  for(;;){
    await new Promise(r=>setTimeout(r, 400));
    const j = await (await fetch('/v1/jobs/' + job.id)).json();
    if(j.status === 'succeeded') return j;
    if(j.status === 'failed' || j.status === 'cancelled') throw new Error(j.error || j.status);
    const st = j.status === 'queued' ? '排队中' : (stageName[j.stage] || j.stage || '准备中');
    const pr = j.progress && j.progress.total ? ` ${j.progress.step}/${j.progress.total}` : '';
    $('#state').textContent = `${st}${pr}`;
  }
}
$('#go').onclick = async () => {
  const prompt = $('#prompt').value.trim();
  if(!prompt){ $('#state').innerHTML='<span class="err">请先输入提示词</span>'; return; }
  const loras = selectedLoras(), n = parseInt($('#n').value), [sw, sh] = chosenSize();
  const files = [...$('#refs').files];
  $('#go').disabled = true;
  const t0 = performance.now();
  try{
    const images = await Promise.all(files.map(toB64));
    for(let i = 0; i < n; i++){
      const seed = $('#rand').checked ? Math.floor(Math.random()*4294967295) : parseInt($('#seed').value||'0') + i;
      const j = await runOne({task: images.length ? 'edit' : 'txt2img', prompt, width:sw, height:sh, seed, loras, images});
      j.images.forEach(b64=>addCard(b64, prompt, Object.assign({}, j.info, {seconds: j.info.seconds}), loras));
    }
    $('#state').textContent = `完成，用时 ${((performance.now()-t0)/1000).toFixed(1)} 秒`;
  }catch(e){
    $('#state').innerHTML = '<span class="err"></span>'; $('#state .err').textContent = '出错：' + e.message;
  }finally{
    current = null; $('#go').disabled = false; health();
  }
};
document.addEventListener('keydown', e=>{ if((e.ctrlKey||e.metaKey) && e.key==='Enter') $('#go').click(); });
health(); loadLoras(); setInterval(health, 15000);
$('#size').addEventListener('change', sizeUi); $('#cw').addEventListener('input', sizeUi); $('#ch').addEventListener('input', sizeUi);
</script>
</body>
</html>
)HTML";
