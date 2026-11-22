/* Figures of the page, drawn on canvas by engine.js from window.STERN (assets/data.js, written by the notebook).
   Colours: blue = effective update of 0.7x or 1x, green = 2x to 3x, orange = 7x, grey = random init / reference. */
(() => {
const A = window.Anim, D = window.STERN;
if (!A || !D) return;
const { TAU, clamp, lerp, seg, E, text, alpha, Plot } = A;
const lvl = m => (m == null ? "grey" : m <= 1 ? "blue" : m <= 3 ? "green" : "orange");
const DASH = [7, 5];
const isNarrow = W => W < 620;
function panels(W, H, n, { padL = 58, padR = 18, gap = 64, top = 34, bottom = 48, vgap = 70 } = {}) {
  if (isNarrow(W)) { const h = (H - top - bottom - vgap * (n - 1)) / n; return Array.from({ length: n }, (_, i) => ({ x: padL, w: W - padL - padR, y: top + i * (h + vgap), h })); }
  const w = (W - padL - padR - gap * (n - 1)) / n; return Array.from({ length: n }, (_, i) => ({ x: padL + i * (w + gap), w, y: top, h: H - top - bottom }));
}
const title = (ctx, s, x, y, th) => text(ctx, s, x, y, { size: 12.5, w: 650, color: th.ink });
const note = (ctx, s, x, y, th, o = {}) => text(ctx, s, x, y, { size: 11.5, color: th.muted, ...o });
/* monotone cubic through pixel points: smooth, no overshoot, passes through every measured point */
function trace(ctx, p) {
  const n = p.length, h = [], d = [], m = new Array(n).fill(0);
  ctx.beginPath(); ctx.moveTo(p[0][0], p[0][1]); if (n < 3) { for (let i = 1; i < n; i++) ctx.lineTo(p[i][0], p[i][1]); return; }
  for (let i = 0; i < n - 1; i++) { h[i] = p[i + 1][0] - p[i][0]; d[i] = (p[i + 1][1] - p[i][1]) / h[i]; }
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) if (d[i - 1] * d[i] > 0) { const a = 2 * h[i] + h[i - 1], b = h[i] + 2 * h[i - 1]; m[i] = (a + b) / (a / d[i - 1] + b / d[i]); }
  for (let i = 0; i < n - 1; i++) ctx.bezierCurveTo(p[i][0] + h[i] / 3, p[i][1] + m[i] * h[i] / 3, p[i + 1][0] - h[i] / 3, p[i + 1][1] - m[i + 1] * h[i] / 3, p[i + 1][0], p[i + 1][1]);
}
function series(ctx, P, xs, ys, color, { width = 2.2, dash = null, r = 2.7 } = {}) {
  const pts = xs.map((x, i) => [P.X(x), P.Y(ys[i])]);
  trace(ctx, pts); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.lineJoin = ctx.lineCap = "round"; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]);
  if (r) pts.forEach(q => dot(ctx, q[0], q[1], r, color));
  return pts;
}
function dot(ctx, x, y, r, color, hollow) { ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); if (hollow) { ctx.fillStyle = hollow; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.stroke(); } else { ctx.fillStyle = color; ctx.fill(); } }
function seg2(ctx, x1, y1, x2, y2, color, width = 1, dash = null) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); }
const hline = (ctx, P, v, color, width = 1, dash = null) => seg2(ctx, P.x, P.Y(v), P.x + P.w, P.Y(v), color, width, dash);
/* a panel cut into a big-picture plot and a close-up under it, on the same x axis */
const split = (c, f = .52, mid = 56) => ({ t: { x: c.x, w: c.w, y: c.y, h: (c.h - mid) * f }, b: { x: c.x, w: c.w, y: c.y + (c.h - mid) * f + mid, h: (c.h - mid) * (1 - f) } });
function lens(ctx, r, th, s) { ctx.strokeStyle = th.faint; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.strokeRect(r.x - .5, r.y - .5, r.w + 1, r.h + 1); ctx.setLineDash([]); text(ctx, s, r.x, r.y - 9, { size: 11.5, color: th.muted }); }
const spread = (items, gap) => { items.sort((a, b) => a.y - b.y); for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + gap); return items; };
const armStyle = (a, th) => ({ color: th.c[lvl(a.multiplier)], dash: a.route === "plain" ? DASH : null });
const BLOCKS = Array.from({ length: 12 }, (_, i) => i + 1);

/* ---------- 1. DiLoCo, round by round (plays with the scroll) ---------- */
const OUTER = (lr, m, n = 40) => { let b = 0; const o = []; for (let j = 0; j < n; j++) { b = m * b + 1; o.push(lr * (1 + m * b)); } return o; };
const SETTINGS = [[.7, .9, "orange"], [.45, .8, "green"], [.2, .9, "green"], [.1, .9, "blue"], [.7, 0, "blue"]];
function arrow(ctx, x1, y1, x2, y2, color, width = 3) {
  const a = Math.atan2(y2 - y1, x2 - x1), L = Math.hypot(x2 - x1, y2 - y1), hd = Math.min(11, L * .6); if (L < 1) return;
  seg2(ctx, x1, y1, x2 - Math.cos(a) * hd * .7, y2 - Math.sin(a) * hd * .7, color, width);
  ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - hd * Math.cos(a - .42), y2 - hd * Math.sin(a - .42)); ctx.lineTo(x2 - hd * Math.cos(a + .42), y2 - hd * Math.sin(a + .42)); ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}
function drawDiloco(ctx, u, W, H, th) {
  const narrow = isNarrow(W), R = 3, s = OUTER(.7, .9), f = u * R, k = Math.min(R - 1, Math.floor(f)), t = u >= 1 ? 1 : f - k;
  const L = narrow ? { x: 14, y: 78, w: W - 28, h: H * .46 - 78 } : { x: 22, y: 84, w: W * .57 - 30, h: H - 120 };
  const C = narrow ? { x: 50, y: H * .46 + 64, w: W - 160, h: H * .54 - 116 } : { x: W * .57 + 58, y: 84, w: W * .43 - 176, h: H - 150 };
  const total = s.slice(0, R).reduce((a, v) => a + v, 0), ell = (L.w - 36) / (total + .35), y0 = L.y + L.h * .5, x0 = L.x + 16;
  const pos = j => x0 + ell * s.slice(0, j).reduce((a, v) => a + v, 0);
  const beat = t < .42 ? 0 : t < .62 ? 1 : 2;
  text(ctx, `Round ${k + 1}`, L.x, 30, { size: 17, w: 700 });
  text(ctx, (narrow ? ["each machine trains alone on its own data", "the updates are averaged", `outer step with momentum: ×${s[k].toFixed(2)} this round`] : ["each machine starts from the shared weights and trains alone on its own data", "the machines send their updates; the updates are averaged", `the outer step applies the average with momentum: ×${s[k].toFixed(2)} this round`])[beat], L.x, 52, { size: narrow ? 12 : 13.5, color: th.muted });
  // ruler in units of one averaged update
  const yr = L.y + L.h - 6; seg2(ctx, x0, yr, x0 + ell * total, yr, th.faint, 1);
  for (let i = 0; i <= Math.floor(total); i++) { seg2(ctx, x0 + ell * i, yr - 4, x0 + ell * i, yr + 4, th.faint, 1); text(ctx, String(i), x0 + ell * i, yr + 17, { size: 10.5, color: th.muted, align: "center" }); }
  note(ctx, "distance moved, in averaged updates", x0 + ell * total, yr - 9, th, { align: "right", size: 10.5 });
  // rounds already played
  for (let j = 0; j < k; j++) { arrow(ctx, pos(j), y0, pos(j + 1), y0, alpha(th.c.orange, .45), 3); dot(ctx, pos(j), y0, 4, th.faint); text(ctx, `×${s[j].toFixed(2)}`, (pos(j) + pos(j + 1)) / 2, y0 + 20, { size: 11.5, color: th.muted, align: "center" }); }
  // the current round
  const px = pos(k), fan = [[-.16, -1], [.1, -.5], [.12, 0], [.1, .5], [-.16, 1]], grow = E.out(seg(t, .02, .4)), fade = lerp(1, .32, seg(t, .42, .6));
  fan.forEach(([dx, dy], i) => {
    const ex = px + ell * (1 + dx), ey = y0 + ell * .92 * dy * (k % 2 ? -1 : 1), bx = (px + ex) / 2 + (ey - y0) * .22 * (i % 2 ? 1 : -1), by = (y0 + ey) / 2 - ell * .12 * (i % 2 ? 1 : -1);
    ctx.beginPath(); ctx.moveTo(px, y0);
    for (let q = 1; q <= 24; q++) { const v = grow * q / 24, w = 1 - v; ctx.lineTo(w * w * px + 2 * w * v * bx + v * v * ex, w * w * y0 + 2 * w * v * by + v * v * ey); }
    ctx.lineWidth = 2; ctx.strokeStyle = alpha(th.c.grey, fade); ctx.lineCap = "round"; ctx.stroke();
    const w = 1 - grow; dot(ctx, w * w * px + 2 * w * grow * bx + grow * grow * ex, w * w * y0 + 2 * w * grow * by + grow * grow * ey, 3.6, alpha(th.c.grey, fade));
  });
  const avg = E.inOut(seg(t, .42, .6)), out = E.inOut(seg(t, .64, .96)), tip = px + ell * lerp(1, s[k], out);
  if (out > 0) arrow(ctx, px, y0, tip, y0, th.c.orange, 3.4);
  if (avg > 0) arrow(ctx, px, y0, px + ell * avg, y0, th.c.blue, 3.4);
  dot(ctx, px, y0, 6.5, th.ink);
  if (beat === 0) note(ctx, "shared weights", px, y0 + ell * .92 + 26, th, { align: px < L.x + 60 ? "left" : "center" });
  if (beat === 1) text(ctx, "average of the updates", px + ell * .5, y0 + 22, { size: 11.5, color: th.ink, align: "center", a: avg });
  if (beat === 2) text(ctx, `×${lerp(1, s[k], out).toFixed(2)}`, (px + tip) / 2, y0 + 21, { size: 12.5, w: 700, align: "center" });
  // the multiplier, round by round
  const P = new Plot(ctx, { ...C, xr: [1, 40], yr: [0, 7.6] });
  title(ctx, "Outer step ÷ averaged update, by round", C.x - 34, C.y - 22, th);
  P.grid({ xt: [1, 10, 20, 30, 40], yt: [0, 1, 2, 3, 4, 5, 6, 7], xlabel: "round", size: 11 });
  const ends = spread(SETTINGS.map(([lr, m, c]) => { const y = OUTER(lr, m); series(ctx, P, y.map((_, i) => i + 1), y, th.c[c], { r: 0, width: lr === .7 && m === .9 ? 2.8 : 1.8 }); return { y: P.Y(y[39]), s: `${+(lr / (1 - m)).toFixed(2)}×  (${lr} / ${m})` }; }), 13);
  ends.forEach(e => text(ctx, e.s, C.x + C.w + 6, e.y, { size: 10.5, color: th.muted, base: "middle" }));
  const cur = k + (beat === 2 ? out : 0), cy = beat === 2 ? lerp(k ? s[k - 1] : 0, s[k], out) : (k ? s[k - 1] : 0);
  if (k || beat === 2) { dot(ctx, P.X(Math.max(1, cur)), P.Y(cy), 5.5, th.c.orange, th.panel); text(ctx, `round ${k + (beat === 2 ? 1 : 0)}: ×${cy.toFixed(2)}`, P.X(Math.max(1, cur)) + 10, P.Y(cy) + 4, { size: 11.5, w: 650 }); }
}

/* ---------- 2. the symptom ---------- */
function drawLoss(ctx, u, W, H, th) {
  const c = panels(W, H, 1, { top: 22, bottom: 46 })[0], L = D.loss, P = new Plot(ctx, { ...c, xr: [0, 236], yr: [.2, .66] }), n = L.val.length - 1;
  P.grid({ xt: [0, 50, 100, 150, 200], yt: [.2, .3, .4, .5, .6], yf: v => v.toFixed(1), xlabel: "DiLoCo round", ylabel: "validation loss" });
  series(ctx, P, L.round, L.val, th.c.blue, { r: 0, width: 2.4 });
  text(ctx, L.val[0].toFixed(3), P.X(L.round[0]) + 9, P.Y(L.val[0]) + 2, { size: 12 }); text(ctx, L.val[n].toFixed(3), P.X(L.round[n]), P.Y(L.val[n]) - 10, { size: 12, align: "right" });
}
function drawSkip(ctx, u, W, H, th) {
  const c = panels(W, H, 1, { top: 30, bottom: 48 })[0], { t, b } = split(c), xr = [.6, 12.4], band = .013;
  const P = new Plot(ctx, { ...t, xr, yr: [-.08, 1.2] }), Q = new Plot(ctx, { ...b, xr, yr: [-.03, .03] }), mag = Math.round((b.h / .06) / (t.h / 1.28));
  const all = (R, r) => { D.profile.forEach(a => { const st = armStyle(a, th); series(ctx, R, BLOCKS, a.skip_cost_mse, st.color, { dash: st.dash, r }); }); series(ctx, R, BLOCKS, D.random.skip_cost_mse, th.c.grey, { r }); };
  P.grid({ yt: [0, .2, .4, .6, .8, 1, 1.2], yf: v => v.toFixed(1), ylabel: "rise in held-out error" });
  ctx.fillStyle = alpha(th.faint, .16); ctx.fillRect(P.x, P.Y(.03), P.w, P.Y(-.03) - P.Y(.03));
  all(P, 2.4);
  const b1 = D.profile.map(a => a.skip_cost_mse[0]), deep = Math.max(...D.profile.flatMap(a => a.skip_cost_mse.slice(3)));
  text(ctx, `block 1: ${Math.min(...b1).toFixed(2)} to ${Math.max(...b1).toFixed(2)}`, P.X(1) + 14, P.Y(1.06), { size: 12, w: 650 });
  note(ctx, "shaded strip: the range magnified below", P.x + P.w, P.Y(.03) - 7, th, { align: "right" });
  lens(ctx, b, th, isNarrow(W) ? `close-up ×${mag}: same curves, magnified` : `close-up ×${mag}: the same curves, vertical scale magnified`);
  ctx.fillStyle = alpha(th.c.grey, .2); ctx.fillRect(Q.x, Q.Y(band), Q.w, Q.Y(-band) - Q.Y(band));
  Q.grid({ xt: BLOCKS, yt: [-.02, 0, .02], yf: v => v.toFixed(2), xlabel: "encoder block skipped", ylabel: "same, magnified" });
  hline(ctx, Q, 0, th.muted); Q.clip(); all(Q, 2.6); ctx.restore();
  note(ctx, `grey band: what a random initialisation shows (±${band})`, Q.x + Q.w - 4, Q.y + 13, th, { align: "right" });
  text(ctx, `blocks 4 to 12: at most ${deep.toFixed(3)}`, Q.x + Q.w - 4, Q.y + Q.h - 7, { size: 12, w: 650, align: "right" });
}

/* ---------- 3. the cause ---------- */
function drawCause(ctx, u, W, H, th) {
  const ps = panels(W, H, 2, { top: 44 }), P = new Plot(ctx, { ...ps[0], xr: [.6, 12.4], yr: [0, 480] }), ln = Math.log;
  title(ctx, "Residual-stream norm after each block", ps[0].x, ps[0].y - 18, th);
  P.grid({ xt: BLOCKS, yt: [0, 100, 200, 300, 400], xlabel: "encoder block", ylabel: "stream norm" });
  D.profile.forEach(a => { const st = armStyle(a, th); series(ctx, P, BLOCKS, a.stream_norm, st.color, { dash: st.dash, r: 2.4 }); });
  series(ctx, P, BLOCKS, D.random.stream_norm, th.c.grey, { r: 2.4 });
  const end = m => D.profile.filter(a => lvl(a.multiplier) === m).map(a => a.stream_norm[11]), rng = v => `${Math.min(...v).toFixed(0)} to ${Math.max(...v).toFixed(0)}`;
  [["orange", "7×"], ["green", "2× to 3×"], ["blue", "0.7× and 1×"]].forEach(([m, s]) => { const v = end(m); text(ctx, `${s}: ${rng(v)}`, P.X(11.9), P.Y(Math.max(...v)) - 9, { size: 11.5, w: 650, align: "right" }); });
  const Q = new Plot(ctx, { ...ps[1], xr: [ln(.5), ln(9)], yr: [.37, .7] }), R0 = D.dose_random;
  title(ctx, "Motion information kept at the last block", ps[1].x, ps[1].y - 18, th);
  Q.grid({ xt: [.7, 1, 2, 3, 7].map(ln), xf: v => `${+Math.exp(v).toFixed(2)}×`, yt: [.4, .5, .6, .7], yf: v => v.toFixed(1), xlabel: "effective update multiplier (log scale)", ylabel: "motion R²" });
  hline(ctx, Q, R0, th.c.grey, 2); note(ctx, `random init ${R0.toFixed(2)}`, Q.X(ln(3.3)), Q.Y(R0) - 7, th);
  D.dose.forEach(d => { const x = Q.X(ln(d.multiplier)), y = Q.Y(d.r2), c = th.c[lvl(d.multiplier)]; if (d.route === "diloco") dot(ctx, x, y, 6, c); else { ctx.lineWidth = 2.2; ctx.strokeStyle = c; ctx.fillStyle = th.panel; ctx.beginPath(); ctx.rect(x - 5.5, y - 5.5, 11, 11); ctx.fill(); ctx.stroke(); } });
  const at = r => { const v = D.dose.filter(d => d.multiplier === 7 && d.route === r).map(d => d.r2); return v.reduce((a, b) => a + b, 0) / v.length; };
  text(ctx, "DiLoCo default", Q.X(ln(7)) - 13, Q.Y(at("diloco")) + 4, { size: 11.5, align: "right" }); text(ctx, "plain AdamW, lr × 7", Q.X(ln(7)) - 13, Q.Y(at("plain")) + 4, { size: 11.5, align: "right" });
  dot(ctx, Q.x + 12, Q.y + Q.h - 30, 5, th.muted); note(ctx, "DiLoCo outer step", Q.x + 24, Q.y + Q.h - 26, th);
  ctx.lineWidth = 2; ctx.strokeStyle = th.muted; ctx.strokeRect(Q.x + 7.5, Q.y + Q.h - 17.5, 9, 9); note(ctx, "plain AdamW", Q.x + 24, Q.y + Q.h - 9, th);
}

/* ---------- 4. the correction ---------- */
const whisk = (ctx, P, v, sd, y, color, w = 2) => seg2(ctx, P.X(v - sd), y, P.X(v + sd), y, color, w);
function drawLadder(ctx, u, W, H, th) {
  const narrow = isNarrow(W), lw = narrow ? 14 : 250, NAME = { 1: ["From scratch: ViT-B, 9 machines at 1×", "this project"], 2: ["From scratch: an earlier short run", "this project, superseded"], 3: ["Public ViT-B, supervised weights", "reference"], 4: ["Released LeVJEPA ViT-L", "reference"] };
  const c = { x: lw, y: 30, w: W - lw - 60, h: H - 30 - 46 }, P = new Plot(ctx, { ...c, xr: [0, .27], yr: [0, 1] }), rows = D.ladder, rh = c.h / rows.length;
  P.grid({ xt: [0, .05, .1, .15, .2, .25], xf: v => v.toFixed(2), xlabel: "frozen-probe CCC on AFEW-VA (dot: mean of 3 seeds, bar: ± 1 SD)" });
  [0, .05, .1, .15, .2, .25].forEach(v => seg2(ctx, P.X(v), c.y, P.X(v), c.y + c.h, th.grid));
  rows.forEach((r, i) => {
    const y = c.y + rh * (i + .5) + (narrow ? 8 : 0), col = r.rung === 1 ? th.c.blue : r.kind === "reference" ? th.c.green : th.c.grey;
    if (narrow) text(ctx, NAME[r.rung][0], c.x, y - 16, { size: 11.5, w: r.rung === 1 ? 700 : 500 });
    else { text(ctx, NAME[r.rung][0], 16, y - 1, { size: 12.5, w: r.rung === 1 ? 700 : 500 }); note(ctx, NAME[r.rung][1], 16, y + 14, th, { size: 11 }); }
    seg2(ctx, P.X(r.ctrl), y, P.X(r.ccc), y, th.grid, 3);
    whisk(ctx, P, r.ctrl, r.ctrl_sd, y, th.c.grey); dot(ctx, P.X(r.ctrl), y, 4.5, th.c.grey, th.panel);
    whisk(ctx, P, r.ccc, r.sd, y, col, 2.6); dot(ctx, P.X(r.ccc), y, 6, col);
    text(ctx, r.ccc.toFixed(3), P.X(r.ccc + r.sd) + 8, y + 4, { size: 12, w: 650 });
  });
}
function drawGate(ctx, u, W, H, th) {
  const c = panels(W, H, 1, { top: 24, bottom: 46 })[0], P = new Plot(ctx, { ...c, xr: [-1, 86], yr: [-.035, .065] }), G = D.gate;
  P.grid({ xt: [0, 20, 40, 60, 80], yt: [-.02, 0, .02, .04, .06], yf: v => (v > 0 ? "+" : "") + v.toFixed(2), xlabel: "training round of the snapshot", ylabel: "CCC minus random init" });
  hline(ctx, P, 0, th.muted);
  series(ctx, P, G.before.map(g => g.round), G.before.map(g => g.diff), th.c.grey, { r: 3.4, width: 2 });
  G.after.forEach(g => { const x = P.X(g.round); seg2(ctx, x, P.Y(g.lo), x, P.Y(g.hi), th.c.blue, 2.6); seg2(ctx, x - 5, P.Y(g.lo), x + 5, P.Y(g.lo), th.c.blue, 2.6); seg2(ctx, x - 5, P.Y(g.hi), x + 5, P.Y(g.hi), th.c.blue, 2.6); dot(ctx, x, P.Y(g.diff), 6, th.c.blue); text(ctx, (g.diff > 0 ? "+" : "") + g.diff.toFixed(3), x + 11, P.Y(g.diff) + 4, { size: 12, w: 650 }); });
}

/* ---------- 5. what is still open ---------- */
const SHORT = { "LayerScale init 1e-4": "LayerScale 1e-4", "LayerNorm scaling + embedding LayerNorm": "LN scaling + embedding LN", "lr_mult 0.5": "half the learning rate" };
function drawFixes(ctx, u, W, H, th) {
  const narrow = isNarrow(W), lw = narrow ? 14 : 232, base = D.profile.filter(a => a.multiplier === 7 && a.route === "diloco"), mean = k => base.reduce((s, a) => s + k(a), 0) / base.length;
  const b1 = a => a.skip_cost_mse[0], dp = a => Math.max(...a.skip_cost_mse.slice(3));
  const rows = [{ label: "7× recipe, unchanged (2 seeds)", b1: mean(b1), dp: mean(dp), ref: true }, ...D.fixes.map(f => ({ label: SHORT[f.label] || f.label, b1: b1(f), dp: dp(f), norm: f.stream_norm[11] }))];
  const top = 48, bottom = 44, gap = narrow ? 34 : 60, w = (W - lw - gap - 50) / 2, rh = (H - top - bottom) / rows.length;
  const P = new Plot(ctx, { x: lw, y: top, w, h: H - top - bottom, xr: [0, 1.4], yr: [0, 1] }), Q = new Plot(ctx, { x: lw + w + gap, y: top, w, h: H - top - bottom, xr: [0, .03], yr: [0, 1] }), mag = Math.round(1.4 / .03);
  title(ctx, "Cost of skipping block 1", P.x, top - 22, th); lens(ctx, { x: Q.x, y: top, w, h: H - top - bottom }, th, narrow ? `close-up ×${mag}: blocks 4 to 12` : `close-up ×${mag}: largest cost among blocks 4 to 12`);
  P.grid({ xt: [0, .5, 1], xf: v => v.toFixed(1) }); Q.grid({ xt: [0, .01, .02, .03], xf: v => v.toFixed(2) });
  ctx.fillStyle = alpha(th.c.grey, .2); ctx.fillRect(Q.x, top, Q.X(.013) - Q.x, H - top - bottom); note(ctx, "random-init range", Q.X(.0065), H - bottom + 34, th, { align: "center", size: 10.5 });
  rows.forEach((r, i) => {
    const y = top + rh * (i + .5) + (narrow ? 6 : 0), col = r.ref ? th.c.grey : th.c.orange, bar = (R, v) => { const x = R.X(Math.max(v, 0)); ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(R.x, y - 4, Math.max(x - R.x, 2), 8, [0, 4, 4, 0]); ctx.fill(); return x; };
    if (narrow) text(ctx, r.label, lw, y - 9, { size: 10.5, w: r.ref ? 700 : 500 }); else text(ctx, r.label, 16, y + 4, { size: 12.5, w: r.ref ? 700 : 500 });
    text(ctx, r.b1.toFixed(2), bar(P, r.b1) + 6, y + 4, { size: 11, color: th.muted }); text(ctx, r.dp.toFixed(3), bar(Q, r.dp) + 6, y + 4, { size: 11, color: th.muted });
  });
}
function drawMotion(ctx, u, W, H, th) {
  const narrow = isNarrow(W), c = panels(W, H, 1, { top: 24, bottom: 46, padR: narrow ? 104 : 190 })[0], P = new Plot(ctx, { ...c, xr: [.6, 12.4], yr: [-.2, .75] });
  P.grid({ xt: BLOCKS, yt: [-.2, 0, .2, .4, .6], yf: v => v.toFixed(1), xlabel: "encoder block read out", ylabel: "motion R², split by clip" });
  hline(ctx, P, 0, th.muted);
  const ends = D.motion.map(a => { const st = armStyle(a, th); series(ctx, P, BLOCKS, a.r2, st.color, { dash: a.long ? null : st.dash, width: a.long ? 3.2 : 2, r: a.long ? 3.4 : 2.4 }); return { y: P.Y(a.r2[11]), s: (narrow ? a.label.replace(", 60 rounds", "").replace(" AdamW", "") : a.label).replace(/(\d)x/, "$1×"), w: a.long ? 700 : 500 }; });
  series(ctx, P, BLOCKS, D.motion_random, th.c.grey, { r: 2.4 }); ends.push({ y: P.Y(D.motion_random[11]), s: "random init", w: 500 });
  spread(ends, 15).forEach(e => text(ctx, e.s, c.x + c.w + 10, e.y + 4, { size: narrow ? 10.5 : 12, w: e.w }));
}

/* ---------- 6. continued pretraining ---------- */
function drawCpt(ctx, u, W, H, th) {
  const c = panels(W, H, 1, { top: 26, bottom: 44 })[0], { t, b } = split(c, .46, 64), P = new Plot(ctx, { ...t, xr: [-.03, .41], yr: [0, 1] });
  const ARMS = [["random", "Random init", "grey"], ["hf", "Released ViT-L", "orange"], ["cpt", "Released ViT-L + continued pretraining", "blue"]], mean = v => v.reduce((a, x) => a + x, 0) / v.length;
  P.grid({ xt: [0, .1, .2, .3, .4], xf: v => v.toFixed(1), xlabel: "fine-tuned CCC on AFEW-VA (dot: one seed, bar: mean)" });
  ARMS.forEach(([k, name, col], i) => {
    const v = D.cpt[k], y = t.y + t.h * (i + .62) / 3, m = mean(v);
    v.forEach((x, j) => dot(ctx, P.X(x), y + ((j * .618) % 1 - .5) * t.h * .13, 3.2, alpha(th.c[col], .55)));
    seg2(ctx, P.X(m), y - t.h * .12, P.X(m), y + t.h * .12, th.ink, 2.6);
    text(ctx, name, P.X(m) - (i ? 0 : -14), y - t.h * .16, { size: 12, w: k === "cpt" ? 700 : 500, align: i ? "right" : "left" }); text(ctx, m.toFixed(3), P.X(m) + 7, y - t.h * .16, { size: 12, w: 650, align: "left", a: i ? 1 : 0 });
    if (!i) text(ctx, m.toFixed(3), P.X(m) - 7, y - t.h * .16, { size: 12, w: 650, align: "right" });
  });
  const d = D.cpt.cpt.map((x, j) => x - D.cpt.hf[j]).sort((p, q) => p - q), n = d.length, m = mean(d), sd = Math.sqrt(d.reduce((a, x) => a + (x - m) ** 2, 0) / (n - 1)), hw = 2.045 * sd / Math.sqrt(n);
  const Q = new Plot(ctx, { ...b, xr: [-.8, n - .2], yr: [-.03, .12] });
  lens(ctx, b, th, isNarrow(W) ? "close-up: the gain, seed by seed, sorted" : "close-up: continued pretraining minus the released model, seed by seed, sorted");
  Q.grid({ yt: [0, .05, .1], yf: v => (v > 0 ? "+" : "") + v.toFixed(2), xlabel: `the ${n} seeds`, ylabel: "gain in CCC" });
  ctx.fillStyle = alpha(th.c.blue, .14); ctx.fillRect(Q.x, Q.Y(m + hw), Q.w, Q.Y(m - hw) - Q.Y(m + hw)); hline(ctx, Q, m, th.c.blue, 1.4, [5, 4]); hline(ctx, Q, 0, th.ink);
  d.forEach((x, j) => { const col = x > 0 ? th.c.blue : th.c.grey; seg2(ctx, Q.X(j), Q.Y(0), Q.X(j), Q.Y(x), col, 2.4); dot(ctx, Q.X(j), Q.Y(x), 3.6, col); });
  text(ctx, isNarrow(W) ? `mean +${m.toFixed(3)}; ${d.filter(x => x > 0).length} of ${n} improve` : `mean ${m > 0 ? "+" : ""}${m.toFixed(3)}, 95% interval [+${(m - hw).toFixed(3)}, +${(m + hw).toFixed(3)}]; ${d.filter(x => x > 0).length} of ${n} seeds improve`, Q.x + 8, Q.y + 14, { size: 12, w: 650 });
}

/* ---------- 7. second collapse ---------- */
function drawRank(ctx, u, W, H, th) {
  const ps = panels(W, H, 2, { top: 44 });
  [["rank", "Effective rank of the final token", [0, 11.5], [0, 2, 4, 6, 8, 10], v => String(v)], ["spk", "Speaker-identity accuracy (118 speakers)", [.1, .235], [.1, .15, .2], v => v.toFixed(2)]].forEach(([k, name, yr, yt, yf], i) => {
    const P = new Plot(ctx, { ...ps[i], xr: [-1, 31], yr }); title(ctx, name, ps[i].x, ps[i].y - 18, th);
    P.grid({ xt: [0, 10, 20, 30], yt, yf, xlabel: "training round" });
    const r0 = D.rank[0][k][0]; hline(ctx, P, r0, th.c.grey, 2); note(ctx, "random init", P.x + P.w - 4, P.Y(r0) + (k === "rank" ? -7 : 15), th, { align: "right" });
    D.rank.forEach(a => series(ctx, P, a.round, a[k], a.pooled ? th.c.blue : th.c.orange, { r: 3 }));
    const pooled = D.rank.filter(a => a.pooled).map(a => a[k][a[k].length - 1]), small = D.rank.find(a => !a.pooled)[k];
    text(ctx, "pooled batch 256 (2 seeds)", P.X(30), P.Y(Math.max(...pooled)) + 36, { size: 12, w: 650, align: "right" }); text(ctx, "batch 16", P.X(30), P.Y(small[small.length - 1]) + (k === "rank" ? -10 : 18), { size: 12, w: 650, align: "right" });
  });
}

/* ---------- 8. the system ---------- */
function drawThroughput(ctx, u, W, H, th) {
  const c = panels(W, H, 1, { top: 24, bottom: 46, padR: 76 })[0], ln = Math.log, P = new Plot(ctx, { ...c, xr: [ln(.85), ln(9.4)], yr: [0, 27] });
  P.grid({ xt: [1, 2, 4, 8].map(ln), xf: v => Math.round(Math.exp(v)), yt: [0, 5, 10, 15, 20, 25], xlabel: "machines training synchronously (1 GPU each, 1 GbE)", ylabel: "samples per second, all machines" });
  D.throughput.forEach((s, i) => { const col = th.c[["grey", "green", "blue"][i]], n = s.sps.length - 1; series(ctx, P, s.machines.map(ln), s.sps, col, { r: 4 }); text(ctx, `batch ${s.batch}`, c.x + c.w + 10, P.Y(s.sps[n]) + 4, { size: 12, w: s.batch === 4 ? 700 : 500 }); });
  const b4 = D.throughput[D.throughput.length - 1]; text(ctx, b4.sps[0].toFixed(1), P.X(0) + 10, P.Y(b4.sps[0]) + 4, { size: 12, w: 650 }); text(ctx, b4.sps[b4.sps.length - 1].toFixed(1), P.X(ln(8)), P.Y(b4.sps[b4.sps.length - 1]) - 11, { size: 12, w: 650, align: "center" });
}

const still = draw => ({ draw });
const SCENES = { diloco: { tau: .16, draw: drawDiloco }, loss: still(drawLoss), skip: still(drawSkip), cause: still(drawCause), ladder: still(drawLadder), gate: still(drawGate),
  fixes: still(drawFixes), motion: still(drawMotion), cpt: still(drawCpt), rank: still(drawRank), throughput: still(drawThroughput) };
document.querySelectorAll("canvas[data-scene]").forEach(cv => { const spec = SCENES[cv.dataset.scene]; try { if (spec) cv.__scene = new A.Scene(cv, spec); } catch (e) { console.error("scene", cv.dataset.scene, e); } });

/* ---------- contents panel at the top left: section titles, the current one marked ---------- */
const el = document.querySelector("#resume"), box = document.querySelector("#resume-body");
if (el && box) {
  const heads = [...document.querySelectorAll("main h2[id]")], wide = matchMedia("(min-width:1180px)");
  box.innerHTML = `<nav class="rs-nav" aria-label="Sections"><a href="#top">Summary</a>${heads.map(h => `<a href="#${h.id}">${h.querySelector(".n").textContent}</a>`).join("")}</nav>`;
  el.open = wide.matches; wide.addEventListener("change", () => { el.open = wide.matches; });
  box.addEventListener("click", ev => { if (ev.target.closest("a") && !wide.matches) el.open = false; });
  const links = [...box.querySelectorAll("a")]; let spy = 0;
  const mark = () => { spy = 0; let k = 0; heads.forEach((h, i) => { if (h.getBoundingClientRect().top < innerHeight * .35) k = i + 1; }); links.forEach((a, i) => a.classList.toggle("cur", i === k)); };
  addEventListener("scroll", () => { if (!spy) spy = requestAnimationFrame(mark); }, { passive: true }); mark();
}
})();
