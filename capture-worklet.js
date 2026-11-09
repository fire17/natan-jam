/* Jam phase 1: the mic tap. Collects the input (mono: channels averaged) into ~43 ms blocks, stamps each with the
   context frame of its first sample, and posts them straight to the analysis Worker over a MessagePort (no main-thread
   hop).
   Its output is inaudible: ±1e-9 (−180 dBFS, far below the last bit of any DAC, so the speaker gets digital zero).
   Not exact zeros on purpose: Chromium swaps a context whose output is all zeros to a slow fake sink after ~30 s
   (seen in headless Chrome: the audio clock — and so the mic stream — fell to ~¼ speed at 30 s). */
const BATCH = 2048;
class JamCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(BATCH); this.n = 0; this.f0 = 0; this.out = null; this.alive = true;
    this.port.onmessage = (e) => { const m = e.data || {}; if (m.worker) this.out = m.worker; if (m.stop) { this.alive = false; this.out = null; } };
  }
  process(inputs, outputs) {
    if (!this.alive) return false;
    const o = outputs[0] && outputs[0][0]; if (o) for (let i = 0; i < o.length; i++) o[i] = (i & 1) ? 1e-9 : -1e-9;
    const inp = inputs[0];
    if (!inp || !inp.length || !this.out) return true;
    const a = inp[0], b = inp.length > 1 ? inp[1] : null, q = a.length;
    if (this.n === 0) this.f0 = currentFrame;
    for (let i = 0; i < q; i++) this.buf[this.n + i] = b ? 0.5 * (a[i] + b[i]) : a[i];
    this.n += q;
    if (this.n >= BATCH) { const d = this.buf; this.out.postMessage({ f0: this.f0, d }, [d.buffer]); this.buf = new Float32Array(BATCH); this.n = 0; }
    return true;
  }
}
registerProcessor('jam-capture', JamCapture);
