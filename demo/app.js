// 安步 Amble · 步态分析 Demo (W1)
// MediaPipe PoseLandmarker + 正面视角 2D 关键点指标，纯端侧，无后端。
import { PoseLandmarker, FilesetResolver } from "./vendor/vision_bundle.mjs";

const MODEL_LOCAL = "models/pose_landmarker_lite.task";
const MODEL_REMOTE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_LOCAL = "wasm";
const WASM_REMOTE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MAX_SECONDS = 20;

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const screens = { home: $("screen-home"), capture: $("screen-capture"), report: $("screen-report") };
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
const statusBar = $("status-bar"), timerEl = $("timer");

let landmarker = null;
let analyzing = false, rafId = 0, startTs = 0, samples = [];
let mediaRecorder = null, recordedChunks = [];
let mode = "camera"; // "camera" | "file"

// 骨架连线（MediaPipe 33 关键点索引）
const BONES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],
               [23,25],[25,27],[24,26],[26,28],[27,31],[28,32]];
const DRAW_POINTS = [11,12,13,14,15,16,23,24,25,26,27,28];

function show(name) {
  Object.values(screens).forEach(s => s.classList.remove("active"));
  screens[name].classList.add("active");
}
function setStatus(text, cls) {
  statusBar.textContent = text;
  statusBar.className = "status " + cls;
}
function homeError(msg) {
  const el = $("home-error");
  el.textContent = msg; el.hidden = !msg;
}

// ---- 模型加载（本地 wasm+本地模型优先，逐级回退公网）----
async function initLandmarker() {
  if (landmarker) return landmarker;
  setStatus("正在加载姿态模型…", "idle");
  const mkOpts = (asset) => ({
    baseOptions: { modelAssetPath: asset },
    runningMode: "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.4, minTrackingConfidence: 0.4,
  });
  const attempts = [
    [WASM_LOCAL, MODEL_LOCAL],
    [WASM_REMOTE, MODEL_LOCAL],
    [WASM_REMOTE, MODEL_REMOTE],
  ];
  let lastErr = null;
  for (const [wasm, model] of attempts) {
    try {
      const vision = await FilesetResolver.forVisionTasks(wasm);
      landmarker = await PoseLandmarker.createFromOptions(vision, mkOpts(model));
      console.log("姿态模型加载成功：wasm=" + wasm + ", model=" + model);
      return landmarker;
    } catch (e) {
      console.warn("模型加载失败（wasm=" + wasm + ", model=" + model + "），尝试下一来源", e);
      lastErr = e;
    }
  }
  throw lastErr;
}

// ---- 摄像头模式 ----
async function startCamera() {
  homeError("");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
    // MediaRecorder 同步录制（demo 仅保存在内存，不上传）
    try {
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => e.data.size && recordedChunks.push(e.data);
      mediaRecorder.start();
    } catch (e) { console.warn("MediaRecorder 不可用，仅做实时分析", e); }
    mode = "camera";
    beginAnalysis();
  } catch (e) {
    console.error(e);
    homeError("无法访问摄像头：" + (e.name === "NotAllowedError"
      ? "权限被拒绝，请在浏览器设置中允许相机访问（手机需 HTTPS）。"
      : e.message || "未知错误") + " 也可以改用「上传本地视频」。");
  }
}

// ---- 上传视频模式 ----
// 根据 play() 拒绝原因 / video.error 给出可操作的提示
function fileErrorText(e) {
  const code = video.error && video.error.code; // 4 = MEDIA_ERR_SRC_NOT_SUPPORTED
  const name = e && e.name;
  console.error("视频加载/播放失败:", name, e && e.message, "mediaErrorCode:", code);
  if (name === "NotAllowedError") {
    return "浏览器拦截了自动播放，请再点一次「上传本地视频」重试。";
  }
  // NotSupportedError / code 4 / 其他：多半是编码格式问题
  return "浏览器无法解码该视频（最常见原因：iPhone「高效」模式录的是 H.265/HEVC，桌面版 Chrome/Edge 不支持）。"
    + "可任选其一：① 改用 Safari 打开本页重试；② 手机设置 → 相机 → 格式 → 改为「兼容性最佳」后重拍；"
    + "③ 把视频转成 MP4/H.264 后再上传。";
}

function startFile(file) {
  homeError("");
  const url = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = url;
  video.onerror = () => {
    stopAnalysis();
    show("home");
    homeError(fileErrorText(null));
    URL.revokeObjectURL(url);
  };
  video.onended = () => { if (analyzing) finishAnalysis(); };
  // 关键：先等模型就绪再开始播放。
  // 否则短视频（如 5 秒）可能在模型加载期间就放完了，采样不足被误判「测量失败」。
  initLandmarker()
    .then(() => video.play())
    .then(() => { mode = "file"; beginAnalysis(); })
    .catch((e) => {
      show("home");
      if (e && (e.name === "NotSupportedError" || e.name === "NotAllowedError")) {
        homeError(fileErrorText(e));
      } else {
        homeError("姿态模型加载失败：" + (e && e.message || "网络异常") + "，请检查网络后刷新重试。");
      }
      URL.revokeObjectURL(url);
    });
}

// ---- 分析主循环（两种模式共用）----
function beginAnalysis() {
  show("capture");
  samples = [];
  analyzing = true;
  startTs = performance.now();
  overlay.width = video.videoWidth || 640;
  overlay.height = video.videoHeight || 480;
  initLandmarker().then(() => { if (analyzing) loop(); })
    .catch((e) => { stopAnalysis(); show("home"); homeError("姿态模型加载失败：" + e.message); });
}

function loop() {
  if (!analyzing) return;
  const now = performance.now();
  const elapsed = (now - startTs) / 1000;
  timerEl.textContent = elapsed.toFixed(1) + " / " + MAX_SECONDS + "s";
  if (elapsed >= MAX_SECONDS) return finishAnalysis();

  if (video.readyState >= 2 && landmarker) {
    try {
      const res = landmarker.detectForVideo(video, now);
      const lm = res.landmarks && res.landmarks[0];
      samples.push({ t: elapsed, lm: lm || null });
      drawSkeleton(lm);
      updateQC(lm);
    } catch (e) { console.warn(e); }
  }
  rafId = requestAnimationFrame(loop);
}

function stopAnalysis() {
  analyzing = false;
  cancelAnimationFrame(rafId);
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
  video.pause();
}

function finishAnalysis() {
  stopAnalysis();
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const metrics = computeMetrics(samples, parseFloat($("distance").value) || 5);
  renderReport(metrics);
  show("report");
}

// ---- 质检状态条 ----
function vis(p) { return p ? (p.visibility ?? 1) : 0; }
function updateQC(lm) {
  if (!lm) return setStatus("未检测到人体", "bad");
  const shVis = (vis(lm[11]) + vis(lm[12])) / 2;
  const ankVis = (vis(lm[27]) + vis(lm[28])) / 2;
  const shoulderW = Math.abs(lm[11].x - lm[12].x);
  if (shVis < 0.5) setStatus("未检测到人体", "bad");
  else if (shoulderW < 0.08) setStatus("请走近一点", "warn");
  else if (ankVis < 0.4) setStatus("请站远一点，需要看到脚踝", "warn");
  else setStatus("已检测到人体 ✓ 请自然行走", "ok");
}

// ---- 骨架叠加 ----
function drawSkeleton(lm) {
  const w = overlay.width, h = overlay.height;
  octx.clearRect(0, 0, w, h);
  if (!lm) return;
  octx.strokeStyle = "#4A9B8E"; octx.lineWidth = Math.max(3, w / 180);
  octx.lineCap = "round";
  for (const [a, b] of BONES) {
    if (vis(lm[a]) < 0.3 || vis(lm[b]) < 0.3) continue;
    octx.beginPath();
    octx.moveTo(lm[a].x * w, lm[a].y * h);
    octx.lineTo(lm[b].x * w, lm[b].y * h);
    octx.stroke();
  }
  octx.fillStyle = "#F5A623";
  for (const i of DRAW_POINTS) {
    if (vis(lm[i]) < 0.3) continue;
    octx.beginPath();
    octx.arc(lm[i].x * w, lm[i].y * h, Math.max(4, w / 110), 0, Math.PI * 2);
    octx.fill();
  }
}

// ---- 指标计算 ----
const smooth = (arr, k = 2) => arr.map((_, i) => {
  let s = 0, n = 0;
  for (let j = Math.max(0, i - k); j <= Math.min(arr.length - 1, i + k); j++) { s += arr[j]; n++; }
  return s / n;
});
const range = (arr) => arr.length ? Math.max(...arr) - Math.min(...arr) : 0;

function computeMetrics(samples, distanceM) {
  const valid = samples.filter(s => s.lm && vis(s.lm[11]) > 0.4 && vis(s.lm[12]) > 0.4);
  const dur = samples.length ? samples[samples.length - 1].t : 0;
  if (valid.length < 15 || dur < 2) {
    return { failed: true, reason: `未检测到足够的人体画面（有效帧 ${valid.length}/${samples.length}）。请在光线充足处让全身入镜、距离镜头 3–5 米后重试。` };
  }
  const t = valid.map(s => s.t);
  const durValid = t[t.length - 1] - t[0] || dur;

  // 1. 步频：左右踝横向间距的峰值 = 每次迈步
  const ank = valid.filter(s => vis(s.lm[27]) > 0.35 && vis(s.lm[28]) > 0.35);
  const sep = smooth(ank.map(s => Math.abs(s.lm[27].x - s.lm[28].x)));
  let steps = 0, lastPeak = -1;
  for (let i = 1; i < sep.length - 1; i++) {
    if (sep[i] > sep[i-1] && sep[i] >= sep[i+1] &&
        (lastPeak < 0 || ank[i].t - ank[lastPeak].t > 0.3) &&
        sep[i] > 0.03) { steps++; lastPeak = i; }
  }
  const cadence = durValid > 0 ? Math.round(steps / durValid * 60) : 0;

  // 2. 步速：用户输入距离 ÷ 分析时长
  const speed = dur > 0 ? distanceM / dur : 0;

  // 3. 对称性：左右踝相对身体中线摆动幅度之比
  const devL = [], devR = [], sw = [], midSh = [];
  for (const s of valid) {
    const l = s.lm;
    const mid = ((l[11].x + l[12].x) / 2 + (l[23].x + l[24].x) / 2) / 2;
    if (vis(l[27]) > 0.35) devL.push(l[27].x - mid);
    if (vis(l[28]) > 0.35) devR.push(l[28].x - mid);
    sw.push(Math.abs(l[11].x - l[12].x));
    midSh.push((l[11].x + l[12].x) / 2);
  }
  const ampL = range(devL), ampR = range(devR);
  const symmetry = (ampL > 0.005 && ampR > 0.005)
    ? Math.min(ampL, ampR) / Math.max(ampL, ampR) * 100 : 0;

  // 4. 躯干摇晃：双肩中点横向峰峰值 ÷ 平均肩宽
  const avgSW = sw.reduce((a, b) => a + b, 0) / (sw.length || 1);
  const sway = avgSW > 0 ? range(smooth(midSh)) / avgSW : 0;

  return { failed: false, cadence, speed, symmetry, sway, duration: dur };
}

// ---- 评级与报告 ----
const COLORS = { green: "绿", yellow: "黄", red: "红" };
function rate(m) {
  const r = {};
  r.speed = m.speed > 1.2 ? "green" : m.speed >= 1.0 ? "yellow" : "red";
  r.cadence = (m.cadence >= 90 && m.cadence <= 115) ? "green"
    : (m.cadence >= 80 && m.cadence <= 125) ? "yellow" : "red";
  r.symmetry = m.symmetry >= 90 ? "green" : m.symmetry >= 80 ? "yellow" : "red";
  r.sway = m.sway < 0.15 ? "green" : m.sway <= 0.30 ? "yellow" : "red";
  return r;
}

const METRIC_META = [
  { key: "speed", name: "步速", fmt: m => m.speed.toFixed(2), unit: "m/s",
    explain: "步行速度反映整体活动能力与身体机能。" },
  { key: "cadence", name: "步频", fmt: m => m.cadence, unit: "步/分",
    explain: "每分钟迈出的步数，节奏稳定很重要。" },
  { key: "symmetry", name: "左右对称性", fmt: m => Math.round(m.symmetry), unit: "%",
    explain: "左右脚摆动幅度是否均衡，越接近 100% 越好。" },
  { key: "sway", name: "躯干摇晃", fmt: m => (m.sway * 100).toFixed(0), unit: "%肩宽",
    explain: "走路时上身左右晃动的幅度，越小越稳。" },
];
const ADVICE = {
  sway: "躯干晃动偏大：试试每天靠墙站 2 分钟——后脑、肩胛、臀部贴墙，轻轻收下巴。",
  symmetry: "左右不太对称：注意两侧均衡用力，可扶椅背做单腿站立练习，每侧 30 秒。",
  speed: "步速偏慢：在安全环境下做短距离快走练习，每次 5–10 分钟，循序渐进。",
  cadence: "步频不在常见区间：可以跟着节拍器或节奏稳定的音乐走路，目标每分钟 100 步左右。",
  allGood: "各项指标都不错，继续保持规律的步行习惯！",
};

function renderReport(m) {
  const cards = $("cards");
  cards.innerHTML = "";
  if (m.failed) {
    $("overall-light").className = "light red";
    $("overall-title").textContent = "本次测量失败";
    $("overall-sub").textContent = m.reason;
    $("advice-text").textContent = "调整机位与光线后，点击下面按钮重新测量。";
    return;
  }
  const r = rate(m);
  const order = { red: 0, yellow: 1, green: 2 };
  const worst = METRIC_META.reduce((a, b) => order[r[a.key]] <= order[r[b.key]] ? a : b);
  const overallColor = r[worst.key];
  $("overall-light").className = "light " + overallColor;
  $("overall-title").textContent = "总体信号：" + COLORS[overallColor] + "灯";
  $("overall-sub").textContent = overallColor === "green"
    ? "四项指标都在常见区间内，走得很稳！"
    : "最需要关注的是「" + worst.name + "」。";

  for (const meta of METRIC_META) {
    const c = r[meta.key];
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML =
      `<div class="metric-head"><h3>${meta.name}</h3>
       <div class="metric-val">${meta.fmt(m)} <small>${meta.unit}</small></div></div>
       <div class="band">
         <span class="seg-r${c === "red" ? " on" : ""}"></span>
         <span class="seg-y${c === "yellow" ? " on" : ""}"></span>
         <span class="seg-g${c === "green" ? " on" : ""}"></span>
       </div>
       <div class="band-note">当前：${COLORS[c]}灯区间</div>
       <p class="explain">${meta.explain}</p>`;
    cards.appendChild(div);
  }
  $("advice-text").textContent = overallColor === "green" ? ADVICE.allGood : ADVICE[worst.key];
}

// ---- 事件绑定 ----
$("btn-camera").addEventListener("click", startCamera);
$("btn-upload").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) startFile(e.target.files[0]);
  e.target.value = "";
});
$("btn-stop").addEventListener("click", () => { if (analyzing) finishAnalysis(); });
$("btn-cancel").addEventListener("click", () => { stopAnalysis(); show("home"); });
$("btn-retry").addEventListener("click", () => { show("home"); });
