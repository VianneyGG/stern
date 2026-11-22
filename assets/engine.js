/* Animation engine: scroll-driven canvas scenes, pinned on screen while they play. No controls, no inputs. */
(() => {
const TAU = Math.PI * 2;
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const seg = (t, a, b) => clamp((t - a) / (b - a));
const E = { out: t => 1 - Math.pow(1 - t, 3), inOut: t => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2), smooth: t => t * t * (3 - 2 * t), outBack: t => 1 + 2.4 * Math.pow(t - 1, 3) + 1.4 * Math.pow(t - 1, 2) };
let TH = null;
function theme() {
  if (TH) return TH;
  const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim();
  TH = { bg: g("--bg"), panel: g("--panel"), ink: g("--ink"), muted: g("--muted"), faint: g("--faint"), grid: g("--grid"), accent: g("--accent"), warm: g("--warm"), good: g("--good"), bad: g("--bad"),
         c: { blue: g("--s-blue"), green: g("--s-green"), orange: g("--s-orange"), grey: g("--s-grey") } };
  return TH;
}
const FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const font = (px, w = 400) => `${w} ${px}px ${FONT}`;
function alpha(hex, a) { // "#rrggbb" -> rgba
  const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}
function text(ctx, s, x, y, { size = 12, color, w = 400, align = "left", base = "alphabetic", a = 1 } = {}) {
  ctx.font = font(size, w); ctx.fillStyle = color || theme().ink; ctx.textAlign = align; ctx.textBaseline = base;
  ctx.globalAlpha = a; ctx.fillText(s, x, y); ctx.globalAlpha = 1;
}
class Plot { // data-space -> pixel-space helper
  constructor(ctx, { x, y, w, h, xr, yr, ylog = false }) { Object.assign(this, { ctx, x, y, w, h, xr, yr, ylog }); }
  X(v) { return this.x + this.w * (v - this.xr[0]) / (this.xr[1] - this.xr[0]); }
  Y(v) { const [a, b] = this.yr, f = this.ylog ? (Math.log(v) - Math.log(a)) / (Math.log(b) - Math.log(a)) : (v - a) / (b - a); return this.y + this.h * (1 - f); }
  grid({ xt = [], yt = [], xf = String, yf = String, xlabel, ylabel, size = 11, yoff = 40 } = {}) {
    const { ctx } = this, th = theme();
    ctx.lineWidth = 1; ctx.strokeStyle = th.grid;
    yt.forEach(v => { ctx.beginPath(); ctx.moveTo(this.x, this.Y(v)); ctx.lineTo(this.x + this.w, this.Y(v)); ctx.stroke(); text(ctx, yf(v), this.x - 7, this.Y(v), { size, color: th.muted, align: "right", base: "middle" }); });
    xt.forEach(v => text(ctx, xf(v), this.X(v), this.y + this.h + 16, { size, color: th.muted, align: "center" }));
    if (xlabel) text(ctx, xlabel, this.x + this.w / 2, this.y + this.h + 34, { size, color: th.muted, align: "center" });
    if (ylabel) { ctx.save(); ctx.translate(this.x - yoff, this.y + this.h / 2); ctx.rotate(-Math.PI / 2); text(ctx, ylabel, 0, 0, { size, color: th.muted, align: "center" }); ctx.restore(); }
  }
  clip() { this.ctx.save(); this.ctx.beginPath(); this.ctx.rect(this.x - 2, this.y - 6, this.w + 4, this.h + 12); this.ctx.clip(); }
}
/* line + CI band revealed up to fraction p of the x-extent */
function reveal(ctx, P, xs, mid, lo, hi, color, p, { width = 2.2, dash = null, fill = .16, head = true } = {}) {
  const n = xs.length, f = p * (n - 1), k = Math.floor(f), t = f - k;
  const pt = (arr, i) => (i < n - 1 ? lerp(arr[i], arr[i + 1], i === k ? t : 0) : arr[i]);
  const idx = []; for (let i = 0; i <= Math.min(k, n - 1); i++) idx.push(i);
  const px = i => (i === k && k < n - 1 ? lerp(xs[k], xs[k + 1], t) : xs[i]);
  if (p <= 0) return;
  const last = idx.length - 1, sample = (arr, i) => (i === last && k < n - 1 ? pt(arr, k) : arr[i]);
  if (lo && hi) {
    ctx.beginPath(); idx.forEach((i, j) => { ctx[j ? "lineTo" : "moveTo"](P.X(px(i)), P.Y(sample(hi, i))); });
    for (let j = last; j >= 0; j--) ctx.lineTo(P.X(px(idx[j])), P.Y(sample(lo, idx[j])));
    ctx.closePath(); ctx.fillStyle = alpha(color, fill); ctx.fill();
  }
  ctx.beginPath(); idx.forEach((i, j) => { ctx[j ? "lineTo" : "moveTo"](P.X(px(i)), P.Y(sample(mid, i))); });
  ctx.lineWidth = width; ctx.strokeStyle = color; ctx.lineJoin = "round"; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]);
  const hx = P.X(px(last)), hy = P.Y(sample(mid, last));
  if (head) { ctx.beginPath(); ctx.arc(hx, hy, 3.4, 0, TAU); ctx.fillStyle = color; ctx.fill(); }
  return [hx, hy];
}
function ellipsePath(ctx, cx, cy, a, b, ang) { ctx.beginPath(); ctx.ellipse(cx, cy, Math.max(a, .3), Math.max(b, .3), ang, 0, TAU); }
function eig2(C) { // symmetric 2x2 -> {l1>=l2, angle of major axis}
  const a = C[0][0], b = C[0][1], d = C[1][1], t = (a + d) / 2, r = Math.hypot((a - d) / 2, b);
  return { l1: t + r, l2: Math.max(t - r, 1e-12), ang: 0.5 * Math.atan2(2 * b, a - d) };
}
const logTicks = (lo, hi) => { const o = []; for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++) [1, 2, 5].forEach(m => { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) o.push(v); }); return o; };
const fmtN = v => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2).replace(/\.?0+$/, "") : v.toPrecision(2).replace(/\.?0+$/, ""));

/* One frame loop for every scene. The page scrolls natively (no wheel handler, so the browser keeps its own
   compositor-driven scrolling); each pinned scene follows the scroll position through ONE time-based easing, so a
   wheel notch or a jump to an anchor glides instead of stepping. Geometry is cached and re-read only when the layout
   changes; a scene repaints only when its eased progress moved and it is on screen. */
const LIVE = new Set(); let RAF = 0, LAST = 0;
function frame(now) {
  RAF = 0; const dt = Math.min(.05, LAST ? (now - LAST) / 1000 : 1 / 60), y = scrollY, vh = innerHeight; let moving = false;
  for (const s of LIVE) {
    const on = !(y + vh < s.top - 80 || y > s.top + s.height + 80);
    if (!on) { s.on = false; continue; }
    const target = s.progress(y, vh);
    if (!s.on) { s.on = true; s.cur = target; s.dirty = true; }   // arriving on screen: no catch-up animation
    else if (target !== s.cur) {
      const next = s.cur + (target - s.cur) * (1 - Math.exp(-dt / s.tau));
      s.cur = Math.abs(target - next) < 2e-4 ? target : next; s.dirty = true;
    }
    if (s.dirty) { s.dirty = false; s.paint(); }
    if (s.cur !== target) moving = true;
  }
  LAST = moving ? now : 0;
  if (moving) schedule();
}
function schedule() { if (!RAF) RAF = requestAnimationFrame(frame); }
let MEASURE = 0;
function remeasure() { if (MEASURE) return; MEASURE = requestAnimationFrame(() => { MEASURE = 0; for (const s of LIVE) s.measure(); schedule(); }); }
addEventListener("scroll", schedule, { passive: true });
addEventListener("resize", remeasure, { passive: true });
addEventListener("load", remeasure);
if (window.ResizeObserver) new ResizeObserver(remeasure).observe(document.body);   // late layout shifts (fonts, images, panels) keep the cached geometry right
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { TH = null; for (const s of LIVE) s.dirty = true; schedule(); });

class Scene { // a pinned scene (inside .story) plays with the scroll; a free-standing canvas is drawn once, complete
  constructor(canvas, spec) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.spec = spec; this.story = canvas.closest(".story");
    const fz = new URLSearchParams(location.search).get("freeze");
    this.freeze = !this.story ? 1 : fz !== null ? clamp(parseFloat(fz)) : (matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : null);
    this.tau = spec.tau || .14; this.on = false; this.dirty = true; this.W = this.H = 1;
    this.resize(); this.measure(); this.cur = this.progress(scrollY, innerHeight);
    new ResizeObserver(() => { if (this.resize()) { this.dirty = true; schedule(); } }).observe(canvas);
    LIVE.add(this); schedule();
  }
  resize() { // returns true when the backing store changed
    const r = this.c.getBoundingClientRect(), W = Math.max(1, Math.round(r.width)), H = Math.max(1, Math.round(r.height));
    const dpr = Math.min(devicePixelRatio || 1, 2, Math.sqrt(3.2e6 / (W * H)));   // cap the backing store (~3.2 Mpx) so a repaint stays cheap
    if (W === this.W && H === this.H && dpr === this.dpr) return false;
    this.W = W; this.H = H; this.dpr = dpr;
    this.c.width = Math.round(W * dpr); this.c.height = Math.round(H * dpr); this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return true;
  }
  measure() { // document-relative geometry of the scroll track, cached (layout is read here only, never per frame)
    const r = (this.story || this.c).getBoundingClientRect();
    this.top = r.top + scrollY; this.height = r.height;
  }
  progress(y, vh) { // 0 while the figure is arriving, plays only while it is pinned, 1 just before it leaves
    if (this.freeze !== null) return this.freeze;
    const span = Math.max(1, this.height - vh);
    return clamp(((y - this.top) / span - .03) / .9);
  }
  paint() {
    const th = theme(), { ctx } = this; ctx.restore(); ctx.save(); ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.globalAlpha = 1; ctx.setLineDash([]);
    ctx.clearRect(0, 0, this.W, this.H);
    try { this.spec.draw(ctx, this.cur, this.W, this.H, th, this); } catch (e) { if (!this.failed) { this.failed = true; console.error("scene draw", e); } }
  }
}
window.Anim = { TAU, clamp, lerp, seg, E, theme, text, font, alpha, Plot, reveal, ellipsePath, eig2, logTicks, fmtN, Scene };
})();
