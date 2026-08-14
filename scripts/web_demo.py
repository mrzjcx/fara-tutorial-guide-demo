#!/usr/bin/env python3
"""
Fara Tutorial Guide RAG — Web 可视化界面
左侧: 测试网页 (iframe, 同源)  |  右侧: 控制面板 + 结果输出

浏览器端 html2canvas 截图 → 服务器端 RAG+Fara+Checker → 返回结果

用法:
    python web_demo.py --port <WEB_DEMO_PORT>   # 端口见 config.py
    浏览器打开 http://localhost:<WEB_DEMO_PORT>
"""
import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 项目根加入 sys.path：scripts.config 通过包路径导入（与 fara/checker/rag 的 config 无冲突）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.config as pc
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from rag import RAGConfig, RAGRetriever
from fara import FaraConfig, FaraActor
from checker import CheckerConfig, Checker
from loop import FaraCheckerLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_demo")

# ============================================================
# HTML 模板
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fara RAG Demo</title>
<script src="{{ test_url }}/vendor/html-to-image.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { display: flex; height: 100vh; font-family: "Microsoft YaHei",sans-serif; background: #1a1a2e; }

#left { flex: 0 0 55%; border-right: 2px solid #333; position: relative; }
#left iframe { width: 100%; height: 100%; border: none; }

#right { flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 10px; overflow-y: auto; }
#right h2 { color: #e0e0e0; font-size: 17px; border-bottom: 1px solid #444; padding-bottom: 8px; margin-bottom: 4px; }

.row { display: flex; gap: 8px; align-items: center; }
.row label { color: #aaa; font-size: 13px; white-space: nowrap; min-width: 50px; }
.row input, .row select { flex: 1; padding: 8px 10px; border-radius: 6px; border:1px solid #444; background:#16213e; color:#e0e0e0; font-size:13px; }

.btn { padding: 10px 16px; border-radius: 8px; border: none; cursor: pointer; font-size: 14px; font-weight: bold; transition: .2s; }
.btn-go { background: #0f3460; color: #e0e0e0; flex: 1; }
.btn-go:hover { background: #1a508b; }
.btn-go:disabled { opacity: .4; cursor: not-allowed; }
.btn-rag { background: #333; color: #aaa; flex: 0 0 90px; font-size: 12px; }
.btn-rag:hover { color: #fff; }

#output { flex: 1; background: #16213e; border-radius: 8px; padding: 12px; overflow-y: auto; color: #ccc; font-size: 13px; line-height: 1.7; min-height: 250px; border: 1px solid #333; }
#output .h { color: #4ecca3; font-weight: bold; margin-top: 10px; }
#output .coord { color: #f0a500; font-size: 16px; font-weight: bold; }
#output .dim { color: #888; margin-left: 8px; }
#output .ok { color: #4ecca3; }
#output .fail { color: #e84545; }
#output .err { color: #e84545; }
#output .info { color: #888; font-size: 11px; border-top: 1px solid #333; margin-top: 10px; padding-top: 6px; }

.spinner { display: none; text-align: center; padding: 20px; color: #888; }
.spinner.on { display: block; }
#screenshotPreview { display: none; width: 100%; border-radius: 6px; margin-top: 4px; border: 1px solid #444; }
#debug-panel { margin-top: 8px; border-top: 1px solid #333; }
#debug-toggle { background: none; border: none; color: #888; cursor: pointer; font-size: 12px; padding: 4px 0; }
#debug-toggle:hover { color: #ccc; }
#debug-log { max-height: 300px; overflow-y: auto; background: #111; border-radius: 6px; padding: 8px; font-size: 11px; font-family: monospace; line-height: 1.5; color: #888; display: none; }
#debug-log.show { display: block; }
.d-step { color: #4ecca3; } .d-fara { color: #f0a500; } .d-checker { color: #00b894; } .d-rag { color: #74b9ff; } .d-time { color: #555; }
</style>
</head>
<body>

<div id="left">
  <iframe id="appFrame" src="{{ test_url }}/register_app.html"></iframe>
</div>

<div id="right">
  <h2>Fara RAG 控制面板</h2>

  <div class="row">
    <label>意图:</label>
    <input id="intent" value="{{ default_intent }}" placeholder="输入用户意图">
  </div>

  <div class="row">
    <label>已完成步骤:</label>
    <input id="stepCtx" placeholder="可选，如：已选日期和科室">
  </div>

  <div class="row">
    <label>页面:</label>
    <select id="pageUrl" onchange="document.getElementById('appFrame').src=this.value">
      <option value="{{ test_url }}/register_app.html">在线挂号</option>
      <option value="{{ test_url }}/payment_app.html">在线缴费</option>
      <option value="{{ test_url }}/pharmacy_app.html">门诊取药</option>
      <option value="{{ test_url }}/home.html">首页</option>
    </select>
  </div>

  <div class="row">
    <label>Fara 模型:</label>
    <select id="faraModel">
      {% for m in fara_model_options %}<option value="{{ m }}"{% if m == fara_model_default %} selected{% endif %}>{{ m }}</option>{% endfor %}
    </select>
  </div>
  <div class="row">
    <label>Fara 地址:</label>
    <input id="faraApi" value="{{ fara_api_default }}" placeholder="http://host:port/v1/chat/completions">
  </div>
  <div class="row">
    <label>Checker 模型:</label>
    <select id="checkerModel">
      {% for m in checker_model_options %}<option value="{{ m }}"{% if m == checker_model_default %} selected{% endif %}>{{ m }}</option>{% endfor %}
    </select>
  </div>
  <div class="row">
    <label>Checker 地址:</label>
    <input id="checkerApi" value="{{ checker_api_default }}" placeholder="http://host:port/v1/chat/completions">
  </div>

  <div style="display:flex; gap:8px;">
    <button class="btn btn-go" id="btnRun" onclick="runFull()">执行 (截图+Fara+Checker)</button>
    <button class="btn btn-rag" onclick="runRagOnly()">RAG</button>
    <button class="btn btn-rag" onclick="removeAnnotationDot()" style="flex:0 0 70px;">清除</button>
  </div>

  <div class="spinner" id="spinner">推理中...</div>
  <div id="output"><div class="info">点击「执行」开始。左侧操作测试网页后，输入意图点执行。</div></div>
  <img id="screenshotPreview" alt="截图预览">

  <div id="debug-panel">
    <button id="debug-toggle" onclick="toggleDebug()">调试日志 ▼</button>
    <div id="debug-log"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const output = $('output'), spinner = $('spinner'), btnRun = $('btnRun'), debugLog = $('debug-log');
let lastCoord = null, lastCanvasW = 720, lastCanvasH = 900;

function toggleDebug() {
  debugLog.classList.toggle('show');
  $('debug-toggle').textContent = debugLog.classList.contains('show') ? '调试日志 ▲' : '调试日志 ▼';
}

function dlog(msg, cls) {
  const t = new Date().toLocaleTimeString();
  debugLog.innerHTML += '<div><span class=\"d-time\">' + t + '</span> <span class=\"' + cls + '\">' + msg + '</span></div>';
  debugLog.scrollTop = debugLog.scrollHeight;
}

function setLoading(v) {
  btnRun.disabled = v;
  spinner.className = 'spinner' + (v ? ' on' : '');
}

async function captureFrame() {
  removeAnnotationDot();  // 截图前先清除标注，防止干扰 Fara
  const iframe = $('appFrame');
  const doc = iframe.contentDocument || iframe.contentWindow.document;
  // 真实渲染截图（SVG foreignObject，浏览器原生渲染，无 html2canvas 重绘偏移）
  const dataUrl = await htmlToImage.toPng(doc.body, { pixelRatio: 1 });
  // 取真实尺寸（与红圈换算基准 getBoundingClientRect 一致）
  const img = new Image();
  img.src = dataUrl;
  await new Promise(r => img.onload = r);
  lastCanvasW = img.width;
  lastCanvasH = img.height;
  // 预览
  $('screenshotPreview').src = dataUrl;
  $('screenshotPreview').style.display = 'block';
  return dataUrl.split(',')[1];
}

async function runFull() {
  const intent = $('intent').value.trim();
  if (!intent) { alert('请输入意图'); return; }
  const stepCtx = $('stepCtx').value.trim();

  setLoading(true);
  debugLog.innerHTML = '';
  dlog('[启动] ' + intent, 'd-step');
  output.innerHTML = '<div class="h">截图 + RAG检索 + Fara推理 + Checker验证...</div>';

  try {
    dlog('[截图] 进行中...', 'd-rag');
    const screenshot = await captureFrame();
    dlog('[截图] 完成', 'd-rag');

    dlog('[请求] 发送到服务器...', 'd-step');
    const resp = await fetch('/api/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        intent, screenshot, step_context: stepCtx,
        fara_model: $('faraModel').value.trim(),
        fara_api: $('faraApi').value.trim(),
        checker_model: $('checkerModel').value.trim(),
        checker_api: $('checkerApi').value.trim()
      })
    });
    const d = await resp.json();
    dlog('[响应] 收到', 'd-step');
    renderFull(d);
  } catch(e) {
    dlog('[错误] ' + e, 'd-step');
    output.innerHTML += '<div class="err">[错误] ' + e + '</div>';
  }
  setLoading(false);
}

async function runRagOnly() {
  const intent = $('intent').value.trim();
  if (!intent) { alert('请输入意图'); return; }

  output.innerHTML = '<div class="h">RAG-Only 检索中...</div>';
  try {
    const resp = await fetch('/api/rag_only', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ intent })
    });
    const d = await resp.json();
    renderRag(d);
  } catch(e) {
    output.innerHTML += `<div class="err">[错误] ${e}</div>`;
  }
}

function renderFull(d) {
  if (d.error) { output.innerHTML = `<div class="err">[错误] ${esc(d.error)}</div>`; dlog('[错误] ' + d.error, 'd-step'); return; }

  dlog('[Fara] 推演: ' + esc((d.keywords||'').substring(0,80)), 'd-fara');
  dlog('[RAG] 切片: ' + (d.snippets||[]).length + ' 块', 'd-rag');
  dlog('[Fara] 坐标: ' + JSON.stringify(d.coordinate), 'd-fara');
  if (d.checker) {
    dlog('[Checker] ' + (d.checker.verified ? '正确' : '错误') + ' ' + esc(d.checker.reason||'').substring(0,60), 'd-checker');
  }
  dlog('[耗时] ' + (d.elapsed||'-'), 'd-time');
  let h = '';
  // ④ 降级：文字指导模式
  if (d.degraded) {
    h += `<div class="h" style="color:#f5a623">📝 文字指导模式（未能确定坐标，请按下方引导语操作）</div>`;
  } else {
    h += `<div class="h">Fara 推演:</div><div style="font-size:12px">${esc(d.keywords||'-')}</div>`;
  }
  h += `<div class="h">RAG 切片 (${(d.snippets||[]).length} 块):</div>`;
  for (const s of (d.snippets||[])) h += `<div style="font-size:12px">  [${s.chunk_id}] ${s.step} (${s.score})</div>`;
  // ① 引导语（面向用户的操作指引）
  h += `<div class="h">引导语（请照做）:</div>`;
  h += `<div class="coord" style="font-size:13px;color:#1a73e8;white-space:pre-wrap">${esc(d.reasoning||'')}</div>`;
  // 坐标（降级时无）
  if (d.degraded) {
    h += `<div class="info">坐标: 无（模型未确定，请按引导语手动操作）</div>`;
  } else {
    h += `<div class="h">Fara 坐标:</div>`;
    h += `<div class="coord">${JSON.stringify(d.coordinate)}</div>`;
  }
  if (d.checker) {
    const c = d.checker;
    h += `<div class="h">Checker:</div>`;
    h += `<div class="${c.verified?'ok':'fail'}">${c.verified?'正确':'错误'}: ${esc(c.reason||'')}</div>`;
    // Checker 判错但坐标照常返回：提示用户自行核对（0.8B 仅供参考）
    if (!c.verified) {
      h += `<div class="fail" style="border:2px solid #f5a623;color:#f5a623">⚠️ Checker 认为该坐标可能不正确（仅供参考），请对照引导语和红圈自行核对后再点击</div>`;
    }
  }
  // ⑤ 人机确认点
  if (d.needs_confirmation) {
    h += `<div class="fail" style="border:2px solid #f5a623;color:#f5a623">⚠️ 该操作涉及提交/确认（不可逆），请确认无误后再执行！</div>`;
  }
  h += `<div class="info">${d.elapsed||'-'} | Token: ${d.tokens||'-'}</div>`;
  output.innerHTML = h;

  // 画红圈标注（降级不画；需人工确认的先弹窗）
  if (!d.degraded && d.coordinate && Array.isArray(d.coordinate) && typeof d.coordinate[0]==='number') {
    if (d.needs_confirmation) {
      if (confirm('⚠️ 该操作涉及提交/确认（不可逆操作），确认继续吗？')) {
        drawAnnotation(d.coordinate[0], d.coordinate[1]);
      }
    } else {
      drawAnnotation(d.coordinate[0], d.coordinate[1]);
    }
  }
}

function renderRag(d) {
  let h = `<div class="h">摘要 (${d.summary_len}字)</div>`;
  h += `<div style="color:#888;font-size:11px;white-space:pre-wrap;max-height:200px;overflow-y:auto;">${esc(d.summary||'')}</div>`;
  h += `<div class="h">BM25 Top-5:</div>`;
  for (const b of (d.bm25||[])) h += `<div style="font-size:12px">  [${b.score}] ${b.chunk_id} ${b.step}</div>`;
  h += `<div class="h">FAISS + 邻块扩展:</div>`;
  for (const r of (d.results||[])) h += `<div style="font-size:12px">  [${r.score}] ${r.chunk_id} ${r.step}</div>`;
  h += `<div class="h">覆盖: ${(d.covered||[]).join(', ')}</div>`;
  output.innerHTML = h;
}

function esc(s) { return String(s||'').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ========== 红圈标注（Fara 默认 1000×1000 归一化 → body 像素） ==========
// Fara1.5 训练使用 MagenticLite 1000×1000 坐标空间，需转换到实际像素
const FARA_SPACE = {{ fara_space }};  // Fara 内部坐标空间尺寸（来自 config.py）

function drawAnnotation(x, y) {
  lastCoord = [x, y];
  try {
    const iframe = $('appFrame');
    const doc = iframe.contentDocument || iframe.contentWindow.document;
    if (!doc) return;
    removeAnnotationDot();

    const body = doc.body;
    const bw = body.getBoundingClientRect().width;
    const bh = body.getBoundingClientRect().height;

    // Fara 1000×1000 归一化 → body 实际像素
    const fx = x * (bw / FARA_SPACE);
    const fy = y * (bh / FARA_SPACE);

    console.log(`Fara归一化(${x},${y}) -> body像素(${fx.toFixed(0)},${fy.toFixed(0)}) body:${bw.toFixed(0)}x${bh.toFixed(0)}`);

    const dot = doc.createElement('div');
    dot.id = 'fara-guide-dot';
    dot.style.cssText = 'position:fixed;z-index:99999;pointer-events:none;'
      + `left:${fx-22}px;top:${fy-22}px;width:44px;height:44px;`
      + 'border:3px solid #e94560;border-radius:50%;animation:fara-pulse 1.2s ease-in-out infinite;';
    const inner = doc.createElement('div');
    inner.style.cssText = 'position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:10px;height:10px;background:#e94560;border-radius:50%;';
    dot.appendChild(inner);
    const label = doc.createElement('div');
    label.style.cssText = 'position:absolute;left:50%;top:-28px;transform:translateX(-50%);background:#e94560;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;white-space:nowrap;';
    label.textContent = '点这里';
    dot.appendChild(label);
    if (!doc.getElementById('fara-anim-style')) {
      const s = doc.createElement('style');
      s.id = 'fara-anim-style';
      s.textContent = '@keyframes fara-pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.6;transform:scale(1.15)}}';
      doc.head.appendChild(s);
    }
    doc.body.appendChild(dot);
  } catch(e) { console.error('标注失败:', e); }
}

function removeAnnotationDot() {
  try {
    const doc = $('appFrame').contentDocument || $('appFrame').contentWindow.document;
    if (!doc) return;
    const dot = doc.getElementById('fara-guide-dot');
    if (dot) dot.remove();
  } catch(e) {}
}
</script>
</body>
</html>"""

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)

rag = None
actor = None
checker = None
loop = None
HTML_DIR = None
SAVE_DIR = None


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        test_url="",
        default_intent=pc.DEFAULT_INTENT,
        fara_space=pc.FARA_SPACE,
        fara_model_options=pc.FARA_MODEL_OPTIONS,
        checker_model_options=pc.CHECKER_MODEL_OPTIONS,
        fara_model_default=pc.FARA_MODEL_NAME,
        checker_model_default=pc.CHECKER_MODEL_NAME,
        fara_api_default=pc.FARA_API_URL,
        checker_api_default=pc.CHECKER_API_URL,
    )


@app.route("/<path:filename>")
def serve_test_app(filename):
    """同源提供测试 HTML 文件（避免跨域 iframe 问题）。"""
    return send_from_directory(HTML_DIR, filename)


@app.route("/api/rag_only", methods=["POST"])
def api_rag_only():
    data = request.get_json()
    intent = data.get("intent", "")

    summary = rag.get_summary()
    results = rag.query(intent, top_k=3)

    import jieba
    tokens = list(jieba.cut(intent))
    bm25_scores = rag._index._bm25.get_scores(tokens)
    ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    bm25_list = []
    for i, s in ranked[:5]:
        if s > 0:
            c = rag._index.get_chunk(i)
            bm25_list.append({"chunk_id": c.chunk_id, "step": c.step_title[:40], "score": round(float(s), 4)})

    all_ids = set()
    for r in results:
        all_ids.add(r["chunk_id"])
        for n in r.get("neighbors", []):
            all_ids.add(n["chunk_id"])
    covered = sorted(all_ids, key=lambda x: rag._index._id_to_idx.get(x, 999))

    return jsonify({
        "summary_len": len(summary),
        "summary": summary[:1000],
        "bm25": bm25_list,
        "results": [{"chunk_id": r["chunk_id"], "step": r["step"], "score": r["score"]} for r in results],
        "covered": covered,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    intent = data.get("intent", "")
    screenshot_b64 = data.get("screenshot", "")
    step_context = data.get("step_context", "")

    if not screenshot_b64:
        return jsonify({"error": "缺少截图"}), 400

    # 前端可实时指定模型与地址（缺省用 config 默认）
    fara_model = (data.get("fara_model") or "").strip() or pc.FARA_MODEL_NAME
    fara_api = (data.get("fara_api") or "").strip() or pc.FARA_API_URL
    checker_model = (data.get("checker_model") or "").strip() or pc.CHECKER_MODEL_NAME
    checker_api = (data.get("checker_api") or "").strip() or pc.CHECKER_API_URL

    # 动态构造（支持前端实时切换模型/地址；不污染全局默认实例）
    fara_actor = FaraActor(FaraConfig(
        api_url=fara_api, model_path=fara_model, model_name=fara_model))
    fara_checker = Checker(CheckerConfig(
        api_url=checker_api, model_path=checker_model, model_name=checker_model, max_retries=3))
    loop = FaraCheckerLoop(rag, fara_actor, fara_checker, max_retries=3)

    t0 = time.time()

    # 运行主流程
    try:
        result, history = loop.run(
            screenshot_b64=screenshot_b64,
            user_intent=intent,
            step_context=step_context,
            top_k_rag=5,
        )
    except Exception as e:
        return jsonify({"error": f"推理失败: {e}"})

    elapsed = f"{time.time() - t0:.1f}s"

    # 组装响应：优先展示 loop 实际喂给模型的切片（含补全），否则退回旧查询
    snippets = getattr(loop, "_last_snippets", None) or rag.query(intent, top_k=3)
    # keywords 用 plan_rag_query 结果（来自 loop 内部缓存），回退用首轮 Fara reasoning
    keywords = loop._query_cache.get((intent, step_context), "")
    if not keywords and history:
        fr = history[0].get("fara")
        if fr and fr.reasoning:
            keywords = fr.reasoning[:200]
    logger.info(f"keywords debug: cache_size={len(loop._query_cache)}, intent_key={'HIT' if intent in loop._query_cache else 'MISS'}, len={len(keywords)}")
    response = {
        "keywords": keywords,
        "snippets": [{"chunk_id": s["chunk_id"], "step": s["step"], "score": s["score"]} for s in snippets],
        "coordinate": result.coordinate,
        "reasoning": result.reasoning[:200],
        "elapsed": elapsed,
        "tokens": f"prompt={result.prompt_tokens} comp={result.completion_tokens}",
        "degraded": getattr(result, "degraded", False),
        "needs_confirmation": getattr(result, "needs_confirmation", False),
    }
    for entry in history:
        cr = entry.get("checker")
        if cr and cr.enabled:
            response["checker"] = {"verified": cr.verified, "reason": cr.reason[:200]}
            break

    # 保存结果
    save_analysis(screenshot_b64, result, history, intent)
    return jsonify(response)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "fara": actor._cfg.api_url,
        "checker": checker.enabled,
    })


def save_analysis(screenshot_b64, result, history, intent):
    """保存截图和分析文本。"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    img_path = os.path.join(SAVE_DIR, f"{ts}_screenshot.png")
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(screenshot_b64))

    txt_path = os.path.join(SAVE_DIR, f"{ts}_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"时间: {ts}\n意图: {intent}\n")
        f.write(f"坐标: {result.coordinate}\n")
        f.write(f"推理: {result.reasoning}\n")
        for entry in history:
            cr = entry.get("checker")
            if cr and cr.enabled:
                f.write(f"Checker: {'正确' if cr.verified else '错误'} {cr.reason}\n")
    logger.info("已保存: %s", txt_path)


def main():
    parser = argparse.ArgumentParser(description="Fara RAG Web Demo")
    parser.add_argument("--port", type=int, default=int(pc.WEB_DEMO_PORT))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--fara-api", default=pc.FARA_API_URL)
    parser.add_argument("--fara-model", default=pc.FARA_MODEL_PATH)
    parser.add_argument("--checker-api", default=pc.CHECKER_API_URL)
    parser.add_argument("--checker-model", default=pc.CHECKER_MODEL_PATH)
    parser.add_argument("--pdf", default=pc.PDF_PATH)
    parser.add_argument("--models-cache", default=pc.MODELS_CACHE)
    parser.add_argument("--html-dir", default=str(pc.PROJECT_ROOT / "html"), help="测试 HTML 页面目录")
    parser.add_argument("--save-dir", default=str(pc.PROJECT_ROOT / "output"), help="分析结果保存目录")
    parser.add_argument("--no-checker", action="store_true")
    args = parser.parse_args()

    global HTML_DIR, SAVE_DIR, rag, actor, checker, loop
    HTML_DIR = os.path.abspath(args.html_dir)
    SAVE_DIR = os.path.abspath(args.save_dir)

    rag = RAGRetriever(RAGConfig(
        pdf_path=args.pdf,
        models_cache_dir=args.models_cache,
        context_window=1, device="cpu",
    ))

    actor = FaraActor(FaraConfig(
        api_url=args.fara_api,
        model_path=args.fara_model,
    ))

    checker = Checker(CheckerConfig(
        api_url="" if args.no_checker else args.checker_api,
        model_path=args.checker_model,
        max_retries=3,
    ))

    loop = FaraCheckerLoop(rag, actor, checker, max_retries=3)

    print(f"""
╔══════════════════════════════════════════════╗
║   Fara RAG Web Demo                         ║
║   打开: http://localhost:{args.port}              ║
║   Fara:  {args.fara_api}  ║
║   Checker: {'启用' if checker.enabled else '禁用'}                              ║
║   HTML:  {HTML_DIR}          ║
╚══════════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
