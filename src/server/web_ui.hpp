// Single-page web UI served by axdiffusion-server at "/". Self-contained (no external assets), talks to the
// server's own /v1/jobs, /v1/loras and /health endpoints.
#pragma once

static const char *kWebUi = R"HTML(<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>axdiffusion</title>
<style>
:root{
  --paper:#edf1f3;--panel:#f8fafb;--field:#ffffff;--ink:#0e2a3f;--muted:#587084;--line:#c8d4db;
  --blue:#1f5f8b;--wash:#2b6f9e;--deep:#0c2f4a;--sun:#f0b429;--sun-ink:#2a1d00;--bad:#b3261e;
  --shadow:0 1px 0 rgba(12,47,74,.06),0 12px 32px -12px rgba(12,47,74,.35);
  --ease:cubic-bezier(.2,.8,.2,1);--in:150ms;--out:120ms;--press:90ms;--r-s:6px;--r-m:8px;--r-l:4px;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#071a28;--panel:#0b2435;--field:#0f2c40;--ink:#e3edf3;--muted:#8ba6b8;--line:#1e3d53;
  --blue:#62a8d4;--wash:#2b6f9e;--deep:#04121d;--sun:#f5c04a;--sun-ink:#231800;--bad:#ff8a80;
  --shadow:0 12px 36px -12px rgba(0,0,0,.7);
}}
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{height:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font:14px/1.55 "HarmonyOS Sans SC","PingFang SC","Microsoft YaHei UI","Microsoft YaHei","Noto Sans CJK SC",system-ui,sans-serif;
  font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
button,input,select,textarea{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid var(--blue);outline-offset:2px}
.app{display:grid;grid-template-columns:392px minmax(0,1fr);height:100vh}

/* ---------- control column ---------- */
.side{background:var(--panel);border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
.brand{padding:18px 22px 14px;display:flex;align-items:baseline;gap:10px}
.brand h1{margin:0;font-size:21px;font-weight:800;letter-spacing:-.4px}
.brand .model{font-size:12px;color:var(--muted)}
.form{flex:1;overflow:auto;padding:4px 22px 18px;display:flex;flex-direction:column;gap:17px}
.field>.name{display:flex;justify-content:space-between;align-items:baseline;font-weight:650;font-size:13px;margin-bottom:8px}
.field>.name .aux{font-weight:400;font-size:12px;color:var(--muted)}
textarea,input[type=number],select{width:100%;background:var(--field);border:1px solid var(--line);border-radius:var(--r-s);padding:9px 11px;transition:border-color 140ms var(--ease),box-shadow 140ms var(--ease)}
textarea:hover,input[type=number]:hover,select:hover{border-color:var(--blue)}
textarea:focus,input[type=number]:focus,select:focus{outline:0;border-color:var(--blue);box-shadow:0 0 0 3px color-mix(in srgb,var(--blue) 22%,transparent)}
#prompt{min-height:112px;resize:vertical;line-height:1.6}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{background:transparent;border:1px solid var(--line);border-radius:var(--r-s);padding:5px 11px;font-size:12.5px;line-height:1.4;transition:background var(--out) var(--ease),border-color var(--out) var(--ease),color var(--out) var(--ease),transform var(--press) var(--ease)}
.chip:active,.ghost:active{transform:translateY(1px) scale(.985)}
.chip:hover{border-color:var(--blue)}
.chip[aria-pressed=true]{background:var(--deep);border-color:var(--deep);color:#eaf3f8}
@media (prefers-color-scheme:dark){.chip[aria-pressed=true]{background:var(--blue);border-color:var(--blue);color:#04121d}}
.ratio{display:inline-flex;align-items:center;gap:7px}
.ratio i{display:block;border:1.5px solid currentColor;border-radius:1.5px;opacity:.8}
.sub{margin-top:9px}
.custom{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;margin-top:9px}
.note{font-size:12px;color:var(--muted);margin-top:7px}
.inline{display:flex;gap:10px;align-items:center}
.inline>input[type=number]{flex:1;min-width:0}
.ghost{background:transparent;border:1px solid var(--line);border-radius:var(--r-s);padding:8px 12px;font-size:13px;white-space:nowrap;transition:border-color var(--out) var(--ease),background var(--out) var(--ease),transform var(--press) var(--ease)}
@media (hover:hover) and (pointer:fine){.ghost:hover{border-color:var(--ink);background:color-mix(in srgb,var(--ink) 5%,transparent);transform:translateY(-1px)}}
.ghost[aria-pressed=true]{background:var(--deep);border-color:var(--deep);color:#eaf3f8}
.stepper{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:var(--field)}
.stepper button{border:0;background:transparent;width:34px;height:36px;font-size:16px}
.stepper button:hover{background:var(--paper)}
.stepper output{width:34px;text-align:center;line-height:36px;font-weight:650}

/* reference images */
.drop{border:1.5px dashed var(--line);border-radius:6px;padding:14px;text-align:center;color:var(--muted);font-size:12.5px;transition:border-color .15s,background .15s;cursor:pointer}
.drop:hover,.drop.over{border-color:var(--blue);background:color-mix(in srgb,var(--blue) 8%,transparent);color:var(--ink)}
.refs{display:flex;gap:8px;flex-wrap:wrap}
.refs:not(:empty){margin-bottom:9px}
.ref{position:relative;width:76px;height:76px;animation:pop .22s ease-out}
.ref img{width:100%;height:100%;object-fit:cover;border-radius:4px;border:1px solid var(--line);display:block}
.ref button{position:absolute;top:-7px;right:-7px;width:22px;height:22px;border-radius:50%;border:0;background:var(--deep);color:#fff;font-size:13px;line-height:22px;padding:0}
.ref button:hover{background:var(--bad)}

/* LoRA rows */
.loras:not(:empty){margin-bottom:9px;display:flex;flex-direction:column;gap:8px}
.lora{background:var(--field);border:1px solid var(--line);border-radius:6px;padding:9px 11px;animation:pop .22s ease-out}
.lora.out{animation:leave .18s ease-in forwards}
.lora .top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.lora .top b{font-weight:650;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lora .rm{border:0;background:transparent;color:var(--muted);font-size:12.5px;padding:2px 4px;border-radius:4px;white-space:nowrap}
.lora .rm:hover{color:var(--bad);background:color-mix(in srgb,var(--bad) 10%,transparent)}
.lora .str{display:grid;grid-template-columns:auto 1fr 46px;gap:10px;align-items:center;margin-top:6px;font-size:12px;color:var(--muted)}
.lora .str output{text-align:right;color:var(--ink);font-weight:650}
input[type=range]{-webkit-appearance:none;appearance:none;height:4px;border-radius:2px;background:var(--line);width:100%;padding:0;border:0}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--blue);border:2px solid var(--field);box-shadow:0 0 0 1px var(--blue);cursor:grab}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--blue);border:2px solid var(--field);cursor:grab}

.act{padding:14px 22px 18px;border-top:1px solid var(--line);background:var(--panel)}
.go{width:100%;border:0;border-radius:var(--r-s);padding:13px;font-weight:750;font-size:15px;background:var(--sun);color:var(--sun-ink);transition:transform var(--press) var(--ease),background var(--in) var(--ease),color var(--in) var(--ease)}
@media (hover:hover) and (pointer:fine){.go:hover{background:color-mix(in srgb,var(--sun) 86%,var(--ink));transform:translateY(-1px)}.go.stop:hover{background:color-mix(in srgb,var(--ink) 6%,transparent)}}
.go:active{transform:translateY(1px) scale(.985)}
.go.stop{background:transparent;color:var(--ink);box-shadow:inset 0 0 0 1.5px var(--ink)}
.act .note{text-align:center}

/* ---------- stage ---------- */
.main{display:flex;flex-direction:column;min-width:0;min-height:0}
.bar{display:flex;gap:18px;justify-content:flex-end;align-items:center;padding:14px 26px;font-size:12.5px;color:var(--muted)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3aa76d;margin-right:7px;vertical-align:1px}
.dot.busy{background:var(--sun);animation:pulse 1.2s ease-in-out infinite}
.dot.down{background:var(--bad)}
.stage{flex:1;min-height:0;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:6px 26px 14px;gap:16px}
.frame{position:relative;background:var(--field);box-shadow:var(--shadow);border-radius:var(--r-l);overflow:hidden;max-width:100%;max-height:100%;
  transition:width .35s cubic-bezier(.2,.8,.2,1),height .35s cubic-bezier(.2,.8,.2,1)}
.frame img.out{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block;cursor:zoom-in}
.frame img.out.develop{animation:develop 1.1s cubic-bezier(.3,.6,.2,1) both}
.frame .empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;color:var(--muted);text-align:center;padding:24px}
.frame .empty strong{font-size:17px;color:var(--ink);font-weight:750}
.frame .dims{position:absolute;left:12px;top:9px;z-index:1;font-size:11.5px;color:var(--muted)}
/* the exposure bath shown while an image is being generated */
.bath{position:absolute;inset:0;background:var(--deep);overflow:hidden;opacity:0;pointer-events:none;transition:opacity .3s}
.bath.on{opacity:1}
.bath .under{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;filter:grayscale(1) blur(14px) brightness(.7);opacity:.55;transform:scale(1.08)}
.bath .tint{position:absolute;inset:0;background:
  radial-gradient(120% 90% at 20% 15%,rgba(98,168,212,.55),transparent 60%),
  radial-gradient(90% 80% at 85% 90%,rgba(31,95,139,.8),transparent 65%),
  linear-gradient(160deg,#123f61,#0a2a43);mix-blend-mode:screen;animation:drift 9s ease-in-out infinite alternate}
.bath .grain{position:absolute;inset:-50%;opacity:.22;background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 .6 0'/></filter><rect width='160' height='160' filter='url(%23n)'/></svg>");animation:grain .9s steps(4) infinite}
.bath .light{position:absolute;left:0;right:0;top:0;height:100%;transform-origin:top;transform:scaleY(var(--p,0));
  background:linear-gradient(180deg,rgba(240,180,41,.0),rgba(240,180,41,.10) 60%,rgba(240,180,41,.34));
  border-bottom:2px solid var(--sun);transition:transform .25s linear}
.bath .big{position:absolute;left:18px;bottom:12px;color:#eaf3f8;font-weight:800;font-size:clamp(34px,7vw,76px);letter-spacing:-2px;line-height:1}
.bath .big small{font-size:.32em;letter-spacing:0;font-weight:600;opacity:.8;margin-left:4px}

.under-frame{width:100%;max-width:100%;min-height:74px;display:flex;flex-direction:column;gap:9px}
.prog{display:none;flex-direction:column;gap:8px}
.prog.on{display:flex}
.segs{display:flex;gap:4px;height:6px}
.seg{flex:var(--w,1);background:var(--line);border-radius:3px;overflow:hidden;position:relative}
.seg i{position:absolute;inset:0;background:var(--blue);transform-origin:left;transform:scaleX(var(--f,0));transition:transform .2s linear}
.seg.now i{background:var(--sun)}
.pline{display:flex;justify-content:space-between;gap:12px;font-size:13px}
.pline .what{font-weight:650}
.pline .time{color:var(--muted)}
.result{display:none;gap:12px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap}
.result.on{display:flex}
.result .txt{flex:1;min-width:240px}
.result .p{font-size:13.5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.result .m{font-size:12px;color:var(--muted);margin-top:3px;display:flex;flex-wrap:wrap;gap:4px 14px}
.result .btns{display:flex;gap:6px;flex-wrap:wrap}
.msg{display:none;font-size:13px;padding:10px 12px;border-radius:6px;border:1px solid color-mix(in srgb,var(--bad) 45%,transparent);color:var(--bad);background:color-mix(in srgb,var(--bad) 7%,transparent)}
.msg.on{display:block;animation:pop .2s ease-out}
.msg.calm{border-color:var(--line);color:var(--muted);background:transparent}

.strip{border-top:1px solid var(--line);padding:14px 26px 12px;display:flex;gap:8px;overflow-x:auto;min-height:92px;align-items:center}
.strip:empty{display:none}
.thumb{flex:none;position:relative;height:68px;animation:pop .25s ease-out;transition:transform var(--in) var(--ease)}
.thumb .pic{height:100%;border:2px solid transparent;border-radius:3px;padding:0;background:var(--field);overflow:hidden;display:block;transition:border-color var(--in) var(--ease)}
.thumb img{height:100%;display:block}
.thumb .del{position:absolute;top:4px;right:4px;width:20px;height:20px;border-radius:50%;border:0;background:var(--deep);color:#fff;font-size:12px;line-height:20px;padding:0;opacity:0;transition:opacity var(--in) var(--ease),background var(--in) var(--ease)}
.thumb:hover .del,.thumb .del:focus-visible{opacity:1}
.thumb .del:hover{background:var(--bad)}
@media (hover:none){.thumb .del{opacity:1}}
@media (hover:hover) and (pointer:fine){.thumb:hover{transform:translateY(-2px)}.thumb:hover img{transform:scale(1.03)}}
.thumb img{transition:transform 600ms var(--ease)}
.thumb[aria-current=true] .pic{border-color:var(--sun)}

#viewer{position:fixed;inset:0;background:rgba(4,18,29,.94);display:none;z-index:9;overflow:auto;cursor:zoom-out}
#viewer.on{display:block;animation:fade .18s ease-out}
#viewer img{display:block;margin:auto;max-width:100vw;max-height:100vh;position:absolute;inset:0}
#viewer.full{cursor:zoom-out}
#viewer.full img{position:static;max-width:none;max-height:none}
#viewer .tip{position:fixed;left:50%;bottom:14px;transform:translateX(-50%);color:#b9cbd8;font-size:12px;background:rgba(4,18,29,.7);padding:5px 12px;border-radius:99px}

@keyframes pop{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:none}}
@keyframes leave{to{opacity:0;transform:translateX(14px)}}
@keyframes fade{from{opacity:0}to{opacity:1}}
@keyframes pulse{50%{opacity:.35}}
@keyframes drift{from{transform:translate(-3%,-2%) scale(1.05)}to{transform:translate(3%,2%) scale(1.12)}}
@keyframes grain{0%{transform:translate(0,0)}25%{transform:translate(-4%,3%)}50%{transform:translate(3%,-5%)}75%{transform:translate(-2%,-3%)}100%{transform:translate(4%,2%)}}
@keyframes develop{0%{filter:grayscale(1) sepia(1) hue-rotate(165deg) saturate(2.2) brightness(.55) contrast(1.3);opacity:.2}
  55%{filter:grayscale(.7) sepia(.6) hue-rotate(165deg) saturate(1.5) brightness(.9);opacity:1}
  100%{filter:none;opacity:1}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01s!important;transition-duration:.01s!important}}
@media (max-width:880px){
  html,body{height:auto}
  .app{grid-template-columns:1fr;height:auto}
  .side{border-right:0;border-bottom:1px solid var(--line)}
  .form{overflow:visible}
  .act{position:sticky;bottom:0;z-index:2}
  .stage{min-height:70vh}
  .bar{justify-content:flex-start;flex-wrap:wrap;gap:6px 16px}
}
</style>
</head>
<body>
<div class="app">
<aside class="side">
  <div class="brand"><h1>axdiffusion</h1><span class="model" id="model"></span></div>
  <div class="form">
    <div class="field">
      <div class="name"><label for="prompt">画面描述</label><span class="aux" id="count"></span></div>
      <textarea id="prompt" placeholder="写下你想看到的画面，中英文都可以。例如：雨后的青石板小巷，一只橘猫蹲在木窗台上，傍晚暖光"></textarea>
    </div>

    <div class="field" id="refField">
      <div class="name"><span>参考图</span><span class="aux" id="refAux"></span></div>
      <div class="refs" id="refs"></div>
      <div class="drop" id="drop" tabindex="0" role="button">拖入、粘贴或点击选择图片。放了参考图就按描述改这张图</div>
      <input id="file" type="file" accept="image/*" multiple hidden>
    </div>

    <div class="field">
      <div class="name"><span>画幅</span><span class="aux" id="sizeAux"></span></div>
      <div class="chips" id="ratios"></div>
      <div class="chips sub" id="sizes"></div>
      <div class="custom" id="custom" hidden>
        <input id="cw" type="number" min="256" step="32" value="1344" aria-label="宽">
        <span>×</span>
        <input id="ch" type="number" min="256" step="32" value="768" aria-label="高">
      </div>
    </div>

    <div class="field">
      <div class="name"><span>风格 LoRA</span><span class="aux" id="loraAux"></span></div>
      <div class="loras" id="loras"></div>
      <select id="loraPick" aria-label="添加 LoRA"></select>
    </div>

    <div class="field">
      <div class="name"><label for="seed">种子</label><span class="aux">同一种子和描述会得到相近的图</span></div>
      <div class="inline">
        <input id="seed" type="number" min="0" value="42">
        <button class="ghost" id="rand" type="button" aria-pressed="false">每次随机</button>
      </div>
    </div>

    <div class="field">
      <div class="name"><span>连续生成</span></div>
      <div class="inline">
        <div class="stepper"><button type="button" id="nDown" aria-label="减少">−</button><output id="n">1</output><button type="button" id="nUp" aria-label="增加">+</button></div>
        <span class="note" style="margin:0">张，种子依次加 1</span>
      </div>
    </div>
  </div>
  <div class="act">
    <button class="go" id="go" type="button">生成图片</button>
    <div class="note" id="eta">Ctrl + Enter 也可以开始</div>
  </div>
</aside>

<main class="main">
  <div class="bar"><span><i class="dot" id="dot"></i><span id="status">连接中</span></span><span id="served"></span><span id="mem"></span></div>
  <div class="stage" id="stage">
    <div class="frame" id="frame">
      <div class="empty" id="empty"><strong>还没有图</strong><span>在左边写一段描述，然后点“生成图片”</span></div>
      <img class="out" id="out" alt="" hidden>
      <div class="bath" id="bath"><img class="under" id="under" alt="" hidden><div class="tint"></div><div class="grain"></div><div class="light"></div>
        <div class="big"><span id="pct">0</span><small>%</small></div></div>
      <span class="dims" id="dims"></span>
    </div>
    <div class="under-frame">
      <div class="prog" id="prog">
        <div class="segs" id="segs"></div>
        <div class="pline"><span class="what" id="what"></span><span class="time" id="time"></span></div>
      </div>
      <div class="result" id="result">
        <div class="txt"><div class="p" id="rp"></div><div class="m" id="rm"></div></div>
        <div class="btns">
          <a class="ghost" id="dl" download style="text-decoration:none;color:inherit">下载图片</a>
          <button class="ghost" id="reuse" type="button">沿用这组参数</button>
          <button class="ghost" id="asRef" type="button">拿它当参考图</button>
        </div>
      </div>
      <div class="msg" id="msg" role="alert"></div>
    </div>
  </div>
  <div class="strip" id="strip"></div>
</main>
</div>
<div id="viewer"><img alt=""><span class="tip">点击切换原始大小，Esc 关闭</span></div>

<script>
const $ = s => document.querySelector(s);
const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const store = {
  get(k, d){ try{ const v = localStorage.getItem('axd.' + k); return v === null ? d : JSON.parse(v); }catch(e){ return d; } },
  set(k, v){ try{ localStorage.setItem('axd.' + k, JSON.stringify(v)); }catch(e){} }
};
const S = {
  sizes: [], ratio: store.get('ratio', '1:1'), size: store.get('size', '1024x1024'),
  loras: store.get('loras', []), loraNames: [], refs: [], maxRefs: 3, canEdit: false,
  n: 1, running: false, stop: false, jobId: null, shots: [], current: -1
};
const RATIOS = [['1:1', 1], ['4:3', 4 / 3], ['3:4', 3 / 4], ['3:2', 3 / 2], ['2:3', 2 / 3], ['16:9', 16 / 9], ['9:16', 9 / 16], ['21:9', 21 / 9], ['9:21', 9 / 21]];
const ratioOf = (w, h) => RATIOS.reduce((best, r) => Math.abs(Math.log(w / h / r[1])) < Math.abs(Math.log(w / h / best[1])) ? r : best)[0];
const r32 = v => Math.max(256, Math.round((parseInt(v) || 0) / 32) * 32);
const secs = s => s < 60 ? s.toFixed(1) + ' 秒' : Math.floor(s / 60) + ' 分 ' + Math.round(s % 60) + ' 秒';

/* ---------- size ---------- */
function chosenSize(){
  if (S.ratio === 'custom') return [r32($('#cw').value), r32($('#ch').value)];
  return S.size.split('x').map(Number);
}
function drawSizes(){
  const groups = {};
  S.sizes.forEach(v => { const [w, h] = v.split('x').map(Number); (groups[ratioOf(w, h)] ||= []).push(v); });
  const ratios = RATIOS.map(r => r[0]).filter(r => groups[r]);
  if (S.sizes.length && S.ratio !== 'custom' && !groups[S.ratio]) S.ratio = ratios[0];
  $('#ratios').replaceChildren(...ratios.map(r => {
    const v = RATIOS.find(x => x[0] === r)[1], bw = v >= 1 ? 16 : Math.round(16 * v), bh = v >= 1 ? Math.round(16 / v) : 16;
    const icon = el('i'); icon.style.width = bw + 'px'; icon.style.height = bh + 'px';
    const b = el('button', {type: 'button', className: 'chip ratio'}, icon, r);
    b.setAttribute('aria-pressed', S.ratio === r);
    b.onclick = () => { S.ratio = r; S.size = nearest(groups[r]); drawSizes(); };
    return b;
  }), (() => { const b = el('button', {type: 'button', className: 'chip', textContent: '自定义'});
    b.setAttribute('aria-pressed', S.ratio === 'custom'); b.onclick = () => { S.ratio = 'custom'; drawSizes(); }; return b; })());
  const list = groups[S.ratio] || [];
  if (list.length && !list.includes(S.size)) S.size = nearest(list);
  $('#sizes').hidden = S.ratio === 'custom';
  $('#custom').hidden = S.ratio !== 'custom';
  $('#sizes').replaceChildren(...list.map(v => {
    const b = el('button', {type: 'button', className: 'chip', textContent: v.replace('x', ' × ')});
    b.setAttribute('aria-pressed', S.size === v);
    b.onclick = () => { S.size = v; drawSizes(); };
    return b;
  }));
  const [w, h] = chosenSize();
  $('#sizeAux').textContent = (w * h / 1e6).toFixed(1) + ' 百万像素' + (S.ratio === 'custom' ? '，宽高取 32 的倍数' : '');
  store.set('ratio', S.ratio); store.set('size', S.size);
  fitFrame(); showEta();
}
function nearest(list){
  const [w, h] = S.size.split('x').map(Number), area = w * h;
  return list.reduce((b, v) => { const [a, c] = v.split('x').map(Number), [p, q] = b.split('x').map(Number);
    return Math.abs(Math.log(a * c / area)) < Math.abs(Math.log(p * q / area)) ? v : b; }, list[0]);
}
function fitFrame(dim){
  const [w, h] = dim || (S.current >= 0 && !S.running ? [S.shots[S.current].info.width, S.shots[S.current].info.height] : chosenSize());
  const st = $('#stage'), availW = st.clientWidth - 52, availH = Math.max(220, st.clientHeight - 130);
  const k = Math.min(availW / w, availH / h);
  document.querySelector('.under-frame').style.width = Math.max(300, Math.round(w * k)) + 'px'; $('#frame').style.width = Math.round(w * k) + 'px'; $('#frame').style.height = Math.round(h * k) + 'px';
  $('#dims').textContent = (S.current < 0 && !S.running) ? w + ' × ' + h : '';
}

/* ---------- LoRA ---------- */
function drawLoraPick(){
  const free = S.loraNames.filter(n => !S.loras.some(l => l.name === n));
  const pick = $('#loraPick');
  pick.replaceChildren(el('option', {value: '', textContent: S.loraNames.length ? (free.length ? '添加一个 LoRA…' : '全部已添加') : '服务端没有配置 LoRA 目录'}),
    ...free.map(n => el('option', {value: n, textContent: n})));
  pick.disabled = !free.length;
  $('#loraAux').replaceChildren(S.loras.length ? Object.assign(el('button', {type: 'button', className: 'rm', textContent: '全部移除'}), {onclick: () => { S.loras = []; drawLoras(); }}) : (S.loraNames.length ? '可选 ' + S.loraNames.length + ' 个' : ''));
  const rm = $('#loraAux .rm'); if (rm) Object.assign(rm.style, {border: 0, background: 'transparent', color: 'var(--muted)', padding: 0});
}
function drawLoras(){
  $('#loras').replaceChildren(...S.loras.map(l => {
    const out = el('output', {textContent: l.scale.toFixed(2)});
    const range = el('input', {type: 'range', min: 0, max: 2, step: 0.05, value: l.scale});
    range.setAttribute('aria-label', l.name + ' 强度');
    range.oninput = () => { l.scale = parseFloat(range.value); out.textContent = l.scale.toFixed(2); store.set('loras', S.loras); };
    const rm = el('button', {type: 'button', className: 'rm', textContent: '移除'});
    const row = el('div', {className: 'lora'}, el('div', {className: 'top'}, el('b', {textContent: l.name, title: l.name}), rm),
      el('div', {className: 'str'}, el('span', {textContent: '强度'}), range, out));
    rm.onclick = () => { S.loras = S.loras.filter(x => x !== l); store.set('loras', S.loras); row.classList.add('out'); setTimeout(drawLoras, 170); };
    return row;
  }));
  store.set('loras', S.loras); drawLoraPick(); showEta();
}
$('#loraPick').onchange = e => { if (e.target.value) { S.loras.push({name: e.target.value, scale: 1}); drawLoras(); } };

/* ---------- reference images ---------- */
const readUrl = f => new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = bad; r.readAsDataURL(f); });
async function addRefs(files){
  for (const f of files) {
    if (!f.type.startsWith('image/')) continue;
    if (S.refs.length >= S.maxRefs) { say('最多放 ' + S.maxRefs + ' 张参考图，多出来的没有加入'); break; }
    S.refs.push(await readUrl(f));
  }
  drawRefs();
}
function drawRefs(){
  $('#refs').replaceChildren(...S.refs.map((u, i) => {
    const x = el('button', {type: 'button', textContent: '×', title: '移除这张参考图'});
    x.setAttribute('aria-label', '移除第 ' + (i + 1) + ' 张参考图');
    x.onclick = () => { S.refs.splice(i, 1); drawRefs(); };
    return el('div', {className: 'ref'}, el('img', {src: u, alt: ''}), x);
  }));
  $('#refAux').textContent = !S.canEdit ? '当前模型不支持' : S.refs.length ? S.refs.length + ' / ' + S.maxRefs + '，改图模式' : '可选，最多 ' + S.maxRefs + ' 张';
  $('#drop').hidden = !S.canEdit || S.refs.length >= S.maxRefs;
  $('#go').textContent = S.running ? '停止生成' : (S.refs.length ? '按描述改图' : '生成图片');
  showEta();
}
$('#drop').onclick = () => $('#file').click();
$('#drop').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('#file').click(); } };
$('#file').onchange = e => { addRefs([...e.target.files]); e.target.value = ''; };
['dragenter', 'dragover'].forEach(t => $('#drop').addEventListener(t, e => { e.preventDefault(); $('#drop').classList.add('over'); }));
['dragleave', 'drop'].forEach(t => $('#drop').addEventListener(t, e => { e.preventDefault(); $('#drop').classList.remove('over'); }));
$('#drop').addEventListener('drop', e => addRefs([...e.dataTransfer.files]));
document.addEventListener('paste', e => { const fs = [...(e.clipboardData?.files || [])]; if (fs.length && S.canEdit) { e.preventDefault(); addRefs(fs); } });

/* ---------- small controls ---------- */
$('#prompt').value = store.get('prompt', '');
const count = () => { const n = $('#prompt').value.trim().length; $('#count').textContent = n ? n + ' 字' : ''; store.set('prompt', $('#prompt').value); };
$('#prompt').oninput = count; count();
$('#seed').value = store.get('seed', 42);
$('#seed').oninput = () => store.set('seed', $('#seed').value);
function setRand(on){ $('#rand').setAttribute('aria-pressed', on); $('#seed').disabled = on; store.set('rand', on); }
$('#rand').onclick = () => setRand($('#rand').getAttribute('aria-pressed') !== 'true');
setRand(store.get('rand', false));
const setN = v => { S.n = Math.min(8, Math.max(1, v)); $('#n').textContent = S.n; showEta(); };
$('#nUp').onclick = () => setN(S.n + 1); $('#nDown').onclick = () => setN(S.n - 1);
$('#cw').oninput = $('#ch').oninput = () => drawSizes();
function say(text, calm){ const m = $('#msg'); m.textContent = text; m.classList.toggle('on', !!text); m.classList.toggle('calm', !!calm); }

/* ---------- time estimate, learned from earlier runs on this browser ---------- */
const estKey = () => { const [w, h] = chosenSize(); return (S.refs.length ? 'e' + S.refs.length : 't') + '|' + w + 'x' + h + '|' + (S.loras.length ? 'l' : ''); };
const estimate = () => store.get('est', {})[estKey()];
function showEta(){
  if (S.running) return;
  const e = estimate();
  $('#eta').textContent = e ? '上次同样设置用了 ' + secs(e) + (S.n > 1 ? '，' + S.n + ' 张约 ' + secs(e * S.n) : '') : 'Ctrl + Enter 也可以开始';
}

/* ---------- progress ---------- */
const STAGES = [['lora', '准备 LoRA', 1], ['text_encoder', '读懂描述', .5], ['denoise', '绘制', 9], ['vae', '显影成图', 1.5]];
const P = {};
function startProgress(hasLora, hasRef, idx, total){
  Object.assign(P, {t0: performance.now(), stage: '', step: 0, total: 0, stepAt: 0, stepDur: 0, stageAt: performance.now(), queued: true,
    list: STAGES.filter(s => s[0] !== 'lora' || hasLora), idx, of: total, est: estimate()});
  $('#segs').replaceChildren(...P.list.map(s => { const d = el('div', {className: 'seg'}, el('i')); d.style.setProperty('--w', s[2]); return d; }));
  $('#prog').classList.add('on'); $('#result').classList.remove('on'); say('');
  $('#bath').classList.add('on'); $('#out').hidden = true; $('#empty').hidden = true;
  const u = $('#under'); u.hidden = !hasRef; if (hasRef) u.src = S.refs[0];
  cancelAnimationFrame(P.raf); tick();
}
function onJob(j){
  const now = performance.now();
  P.queued = j.status === 'queued';
  if (j.stage && j.stage !== P.stage) { P.stage = j.stage; P.stageAt = now; P.step = 0; P.stepAt = now; }
  const pr = j.progress || {};
  if (pr.total) {
    P.total = pr.total;
    if (pr.step !== P.step) { if (pr.step > P.step && P.step > 0) P.stepDur = (now - P.stepAt) / (pr.step - P.step); P.step = pr.step; P.stepAt = now; }
  }
}
function tick(){
  const now = performance.now(), elapsed = (now - P.t0) / 1000;
  const cur = P.list.findIndex(s => s[0] === P.stage), sum = P.list.reduce((a, s) => a + s[2], 0);
  let done = 0, frac = 0;
  P.list.forEach((s, i) => {
    let f = i < cur ? 1 : 0;
    if (i === cur) {
      if (s[0] === 'denoise' && P.total) {
        const guess = P.stepDur || (P.est ? P.est * 1000 * .8 / P.total : 0);
        const part = guess ? Math.min(.92, (now - P.stepAt) / guess) : 0;
        f = Math.min(1, (P.step + part) / P.total);
      } else {
        const guess = P.est ? P.est * 1000 * s[2] / sum : 2500;
        f = Math.min(.9, (now - P.stageAt) / guess);
      }
    }
    const seg = $('#segs').children[i];
    seg.classList.toggle('now', i === cur); seg.firstChild.style.setProperty('--f', f.toFixed(3));
    done += f * s[2];
  });
  frac = done / sum;
  $('#bath').style.setProperty('--p', frac.toFixed(3));
  $('#pct').textContent = Math.round(frac * 100);
  const name = P.queued ? '排队中，前面还有任务' : cur < 0 ? '准备中' : P.list[cur][1] + (P.stage === 'denoise' && P.total ? '，第 ' + Math.min(P.total, P.step + 1) + ' / ' + P.total + ' 步' : '');
  $('#what').textContent = (P.of > 1 ? '第 ' + (P.idx + 1) + ' / ' + P.of + ' 张：' : '') + name;
  let left = '';
  if (!P.queued && P.est && P.est - elapsed > 1) left = '，大约还要 ' + secs(P.est - elapsed);
  else if (!P.queued && P.stage === 'denoise' && P.stepDur) left = '，大约还要 ' + secs((P.total - P.step) * P.stepDur / 1000 + 1);
  $('#time').textContent = '已用 ' + secs(elapsed) + left;
  P.raf = requestAnimationFrame(tick);
}
function endProgress(){ cancelAnimationFrame(P.raf); $('#prog').classList.remove('on'); $('#bath').classList.remove('on'); }

/* ---------- results ---------- */
function b64Blob(b64){ const bin = atob(b64), a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return new Blob([a], {type: 'image/png'}); }
function addShot(b64, prompt, info, loras){
  const blob = b64Blob(b64), url = URL.createObjectURL(blob);
  S.shots.push({url, blob, prompt, info, loras});
  drawStrip(); show(S.shots.length - 1, true);
}
function drawStrip(){
  $('#strip').replaceChildren(...S.shots.map((s, i) => {
    const pic = el('button', {type: 'button', className: 'pic', title: s.prompt}, el('img', {src: s.url, alt: s.prompt}));
    pic.onclick = () => show(i, false);
    const del = el('button', {type: 'button', className: 'del', textContent: '×', title: '从历史里删掉这张'});
    del.setAttribute('aria-label', '从历史里删掉这张');
    del.onclick = () => removeShot(i);
    const t = el('div', {className: 'thumb'}, pic, del); t.setAttribute('aria-current', i === S.current); return t;
  }).reverse());
}
function removeShot(i){
  URL.revokeObjectURL(S.shots[i].url); S.shots.splice(i, 1);
  if (!S.shots.length) {
    S.current = -1; drawStrip(); $('#result').classList.remove('on');
    if (!S.running) { $('#out').hidden = true; $('#empty').hidden = false; fitFrame(); }
    return;
  }
  const was = S.current; S.current = i < was ? was - 1 : Math.min(was, S.shots.length - 1);
  drawStrip(); if (i === was && !S.running) show(S.current, false);
}
function show(i, fresh){
  if (S.running && !fresh) { /* allow browsing while the next one is cooking */ }
  S.current = i; const s = S.shots[i], inf = s.info, t = inf.seconds || {};
  [...$('#strip').children].forEach((c, k) => c.setAttribute('aria-current', S.shots.length - 1 - k === i));
  const img = $('#out'); img.hidden = false; img.src = s.url; img.alt = s.prompt; $('#empty').hidden = true;
  img.classList.remove('develop'); if (fresh) { void img.offsetWidth; img.classList.add('develop'); }
  fitFrame([inf.width, inf.height]); $('#dims').textContent = '';
  $('#rp').textContent = s.prompt;
  $('#rm').replaceChildren(...[inf.width + ' × ' + inf.height, '种子 ' + inf.seed, '用时 ' + secs(t.total || 0),
    inf.task === 'edit' ? '改图' : '', ...s.loras.map(l => l.name + ' ' + l.scale.toFixed(2))].filter(Boolean).map(x => el('span', {textContent: x})));
  $('#dl').href = s.url; $('#dl').download = 'axdiffusion_' + inf.seed + '_' + inf.width + 'x' + inf.height + '.png';
  $('#asRef').hidden = !S.canEdit;
  if (!S.running || fresh) $('#result').classList.add('on');
}
$('#reuse').onclick = () => {
  const s = S.shots[S.current]; if (!s) return;
  $('#prompt').value = s.prompt; count(); $('#seed').value = s.info.seed; setRand(false); store.set('seed', s.info.seed);
  const v = s.info.width + 'x' + s.info.height;
  if (S.sizes.includes(v)) { S.ratio = ratioOf(s.info.width, s.info.height); S.size = v; } else { S.ratio = 'custom'; $('#cw').value = s.info.width; $('#ch').value = s.info.height; }
  S.loras = s.loras.map(l => ({...l})); drawLoras(); drawSizes();
};
$('#asRef').onclick = async () => {
  const s = S.shots[S.current]; if (!s) return;
  if (S.refs.length >= S.maxRefs) return say('参考图已经放满 ' + S.maxRefs + ' 张，先移除一张');
  S.refs.push(await readUrl(s.blob)); drawRefs(); $('#refField').scrollIntoView({behavior: 'smooth', block: 'nearest'});
};
$('#out').onclick = () => { const v = $('#viewer'); v.querySelector('img').src = $('#out').src; v.classList.remove('full'); v.classList.add('on'); };
$('#viewer').onclick = e => { const v = $('#viewer'); if (e.target.tagName === 'IMG') v.classList.toggle('full'); else v.classList.remove('on'); };

/* ---------- running jobs ---------- */
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function runOne(body, idx, total){
  startProgress(body.loras.length > 0, body.images.length > 0, idx, total);
  const sub = await fetch('/v1/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const job = await sub.json();
  if (!sub.ok || !job.id) throw new Error(job.error && job.error.message || sub.statusText);
  S.jobId = job.id;
  for (;;) {
    await sleep(250);
    const j = await (await fetch('/v1/jobs/' + job.id)).json();
    if (j.status === 'succeeded') return j;
    if (j.status === 'cancelled') throw Object.assign(new Error('cancelled'), {cancelled: true});
    if (j.status === 'failed') throw new Error(j.error || '生成失败');
    onJob(j);
  }
}
function explain(msg){
  if (/tier|sequence|no kernel|rows/i.test(msg)) return '这个尺寸暂时出不了：' + msg + '。换一个预设画幅，或把宽高调小一些。';
  if (/Failed to fetch|NetworkError/i.test(msg)) return '连不上服务，检查板子上的 axdiffusion-server 是否还在运行。';
  return '没有生成成功：' + msg;
}
async function generate(){
  const prompt = $('#prompt').value.trim();
  if (!prompt) { say('先写一段画面描述'); $('#prompt').focus(); return; }
  const [w, h] = chosenSize(), loras = S.loras.map(l => ({...l})), images = [...S.refs], n = S.n, rand = $('#rand').getAttribute('aria-pressed') === 'true';
  const key = estKey(), base = parseInt($('#seed').value || '0');
  S.running = true; S.stop = false; $('#go').classList.add('stop'); $('#go').textContent = '停止生成'; $('#dot').className = 'dot busy'; $('#status').textContent = '生成中';
  $('#eta').textContent = '停止会在当前这一步结束后生效'; fitFrame([w, h]);
  if (innerWidth <= 880) $('#stage').scrollIntoView({behavior: 'smooth', block: 'start'});
  try {
    for (let i = 0; i < n && !S.stop; i++) {
      const seed = rand ? Math.floor(Math.random() * 4294967295) : base + i;
      const j = await runOne({task: images.length ? 'edit' : 'txt2img', prompt, width: w, height: h, seed, loras, images}, i, n);
      const est = store.get('est', {}); est[key] = j.info.seconds.total; store.set('est', est);
      endProgress(); j.images.forEach(b => addShot(b, prompt, j.info, loras));
      if (i + 1 < n && !S.stop) await sleep(1200);
    }
  } catch (e) {
    endProgress();
    if (e.cancelled) say('已停止生成', true); else say(explain(e.message));
    if (S.current >= 0) show(S.current, false); else { $('#empty').hidden = false; }
  } finally {
    endProgress(); S.running = false; S.jobId = null;
    $('#go').classList.remove('stop'); drawRefs(); health();
    if (S.current >= 0) $('#result').classList.add('on');
  }
}
$('#go').onclick = () => {
  if (!S.running) return generate();
  S.stop = true; $('#go').textContent = '正在停止';
  if (S.jobId) fetch('/v1/jobs/' + S.jobId, {method: 'DELETE'}).catch(() => {});
};
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !S.running) generate();
  if (e.key === 'Escape') $('#viewer').classList.remove('on');
});

/* ---------- server state ---------- */
async function health(){
  try {
    const h = await (await fetch('/health')).json();
    $('#model').textContent = h.model || '';
    S.maxRefs = h.max_references || 3; S.canEdit = (h.tasks || []).includes('edit');
    if ((h.sizes || []).join() !== S.sizes.join()) { S.sizes = h.sizes || []; drawSizes(); }
    if (!S.running) { $('#dot').className = h.queued ? 'dot busy' : 'dot'; $('#status').textContent = h.queued ? '别处有 ' + h.queued + ' 个任务在排队' : '空闲，可以生成'; }
    $('#served').textContent = '累计出图 ' + h.requests + ' 张';
    $('#mem').textContent = 'NPU 内存 ' + h.cmm_used_gib.toFixed(1) + ' GiB';
    drawRefs();
  } catch (e) { $('#dot').className = 'dot down'; $('#status').textContent = '连不上服务'; }
}
async function loadLoras(){
  try { const d = await (await fetch('/v1/loras')).json(); S.loraNames = (d.data || []).map(x => x.name); } catch (e) { S.loraNames = []; }
  if (S.loraNames.length) S.loras = S.loras.filter(l => S.loraNames.includes(l.name));
  drawLoras();
}
drawSizes(); drawRefs(); health(); loadLoras(); setInterval(health, 10000);
addEventListener('resize', () => fitFrame());
</script>
</body>
</html>
)HTML";
