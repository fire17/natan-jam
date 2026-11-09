/* Jam phase 1 — the live music analyzer. Pure DSP, no DOM: runs in a Worker (importScripts) and in Node (require),
   so the very same code that runs on the phone is tuned and tested offline on known clips.

   Feed it mono samples with push(Float32Array) at any sample rate. Every ~0.5 s of audio it refreshes this.snap:
     { t, level, active, key, tempo, beat, meter, feel, inst, chord }
   Nothing here plays sound; nothing leaves the device.

   The pieces (each is a small, classical, well-understood method — no network model):
   - Spectrum every ~10.7 ms (FFT 4×hop, Hann) folded into 48 log-spaced bands.
   - Onset strength = rectified rise of the band levels in dB (spectral flux with a max filter across bands, the
     "superflux" trick, so a singer's vibrato doesn't look like new notes).
   - Tempo = autocorrelation of the onset strength over the last 12 s, scored as a comb (the period and its multiples),
     a soft preference for 60–180 BPM, smoothed over time, with hysteresis against octave flips.
   - Beat = the phase of that comb on the latest seconds (a phase-locked loop), so the page can pulse on each beat.
   - Meter = which beat of a 3- or 4-beat cycle gets the accents (low onsets = kick, harmonic change = new bar).
   - Feel = straight vs swung (onset energy at 1/2 vs 2/3 of the beat), density = notes per beat.
   - Key = a pitch-class profile (chroma from spectral peaks of a long FFT, tuning-corrected) accumulated with a
     ~20 s memory, correlated with 24 key profiles.
   - Instruments = a light harmonic/percussive split (median filters over time and over bands) → energy by band and
     kind → a small logistic model per instrument (weights in INST_MODEL; fitted offline on separated stems). */
(function (G) {
  'use strict';

  const PC_EN = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

  // Key profiles (index 0 = tonic). Krumhansl–Kessler and Temperley (Kostka–Payne corpus); the mix is used.
  const KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const TP_MAJ = [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400];
  const TP_MIN = [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330];

  function zprof(p) { const m = p.reduce((a, b) => a + b, 0) / 12; const d = p.map((v) => v - m); const s = Math.sqrt(d.reduce((a, b) => a + b * b, 0)); return d.map((v) => v / s); }
  function mixProf(a, b) { const za = zprof(a), zb = zprof(b); return zprof(za.map((v, i) => v + zb[i])); }
  const PROF = { major: mixProf(KK_MAJ, TP_MAJ), minor: mixProf(KK_MIN, TP_MIN) };

  // Chord templates: major / minor triads (root, third, fifth), z-scored.
  const CHORD_T = [];
  for (let r = 0; r < 12; r++) for (const q of ['maj', 'min']) {
    const t = new Array(12).fill(0); t[r] = 1; t[(r + (q === 'maj' ? 4 : 3)) % 12] = 1; t[(r + 7) % 12] = 1;
    CHORD_T.push({ root: r, q, v: zprof(t) });
  }

  // Instrument model: logistic regression over the feature vector built in _instFeatures(), fitted offline on
  // separated stems; the weights load from inst-model.js (falls back to hand-set rules without it).
  let INST_MODEL = G.JAM_INST_MODEL || null;

  function makeFFT(n) {
    const lev = Math.round(Math.log2(n)), rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) { let r = 0, x = i; for (let b = 0; b < lev; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
    const cs = new Float64Array(n / 2), sn = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cs[i] = Math.cos(2 * Math.PI * i / n); sn[i] = -Math.sin(2 * Math.PI * i / n); }
    return function (re, im) {
      for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half, c = cs[k], s = sn[k];
          const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        }
      }
    };
  }
  function hann(n) { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); return w; }
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const sigm = (x) => 1 / (1 + Math.exp(-x));

  class JamAnalyzer {
    constructor(sr, opts) {
      opts = opts || {};
      this.sr = sr;
      this.H = 1 << Math.round(Math.log2(sr * 0.0107)); // hop ≈ 10.7 ms (512 at 44.1/48 kHz)
      this.N1 = this.H * 4; this.N2 = this.H * 32;      // 2048 (onsets, bands) and 16384 (pitch) at 48 kHz
      this.fps = sr / this.H;
      this.RING = 1; while (this.RING < this.N2 * 2) this.RING <<= 1; this.mask = this.RING - 1;
      this.ring = new Float32Array(this.RING); this.w = 0; this.pend = 0;
      this.win1 = hann(this.N1); this.win2 = hann(this.N2);
      let s1 = 0; for (const v of this.win1) s1 += v * v; this.norm1 = 4 / (this.N1 * s1);
      let s2 = 0; for (const v of this.win2) s2 += v; this.norm2 = 2 / s2; // amplitude: a sine of amplitude A peaks at A
      this.fft1 = makeFFT(this.N1); this.fft2 = makeFFT(this.N2);
      this.re1 = new Float64Array(this.N1); this.im1 = new Float64Array(this.N1);
      this.re2 = new Float64Array(this.N2); this.im2 = new Float64Array(this.N2);
      // bands
      const NB = 48, fLo = 30, fHi = Math.min(16000, sr * 0.45), df = sr / this.N1;
      this.NB = NB; this.bLo = new Int32Array(NB); this.bHi = new Int32Array(NB); this.bFc = new Float64Array(NB);
      for (let b = 0; b < NB; b++) {
        const a = fLo * Math.pow(fHi / fLo, b / NB), c = fLo * Math.pow(fHi / fLo, (b + 1) / NB);
        let lo = Math.max(1, Math.floor(a / df)), hi = Math.floor(c / df); if (hi <= lo) hi = lo + 1;
        this.bLo[b] = lo; this.bHi[b] = hi; this.bFc[b] = Math.sqrt(a * c);
      }
      this.region = (f) => (f < 180 ? 0 : f < 500 ? 1 : f < 2000 ? 2 : f < 5000 ? 3 : 4); // bass, low-mid, mid, presence, air
      this.bReg = Int8Array.from(this.bFc, (f) => this.region(f));
      this.bandDb = new Float64Array(NB); this.prevDb = new Float64Array(NB).fill(-120); this.peakDb = -60;
      // onset strength history (24 s)
      this.OL = Math.ceil(this.fps * 24);
      this.odf = new Float32Array(this.OL); this.odfLow = new Float32Array(this.OL); this.odfHigh = new Float32Array(this.OL);
      this.frame = 0;
      // harmonic / percussive split
      this.HK = 17; this.hist = []; for (let i = 0; i < this.HK; i++) this.hist.push(new Float64Array(NB));
      this.eH = new Float64Array(5); this.eP = new Float64Array(5); this.eAlpha = 1 - Math.exp(-1 / (this.fps * 1.5));
      this.eHs = new Float64Array(5); this.eAlphaS = 1 - Math.exp(-1 / (this.fps * 4)); // slower, for the instrument features
      this.eTot = 0;
      // 16 coarse bands (3 of the 48 each): harmonic / percussive energy and the band level's mean + spread over ~2 s
      this.NB16 = 16; this.hb = new Float64Array(16); this.pb = new Float64Array(16); this.dbm = new Float64Array(16).fill(-100); this.dbv = new Float64Array(16);
      this.a2 = 1 - Math.exp(-1 / (this.fps * 2));
      this.onL = 0; this.onH = 0; this.onM = 0;
      // pitch / chroma
      this.CH_EVERY = 4; this.chFps = this.fps / this.CH_EVERY;
      this.CL = Math.ceil(this.chFps * 24); this.chroma = []; for (let i = 0; i < this.CL; i++) this.chroma.push(new Float32Array(12));
      this.chFrame = 0; this.chE = new Float32Array(this.CL);
      this.keyAcc = new Float64Array(12); this.keyN = 0; this.keyDecay = Math.exp(-1 / (this.chFps * (opts.keyMemory || (G.JAM_TUNE && G.JAM_TUNE.km) || 30)));
      this.tunRe = 0; this.tunIm = 0; this.tuning = 0;
      this.mag2 = new Float64Array(this.N2 / 2);
      this.chordAcc = new Float64Array(12);
      // modulation of the presence band (voice-like syllables) and spectral shape
      this.envPres = new Float32Array(this.OL); this.flat = 0; this.cent = 0;
      // tempo state
      this.bpmGrid = []; for (let b = 40; b <= 220; b += 0.25) this.bpmGrid.push(b);
      this.tAcc = new Float64Array(this.bpmGrid.length);
      this.bpm = 0; this.bpmCand = 0; this.bpmCount = 0; this.tempoConf = 0;
      this.beatPh = null; // { last (sample), period (samples) }
      this.keyCur = null; this.keyCand = null; this.keyCandN = 0;
      this.meterCur = 0; this.instLvl = { drums: 0, bass: 0, chords: 0, voice: 0 };
      this.UPD = Math.round(this.fps / 2); this.seq = 0; this.snap = null;
      this.levelDb = -120; this.activeS = 0; this.silentS = 0;
      this.rmsAcc = 0; this.rmsN = 0;
    }

    push(x) {
      const r = this.ring, m = this.mask, H = this.H;
      for (let i = 0; i < x.length; i++) {
        const v = x[i]; r[this.w & m] = v; this.w++; this.rmsAcc += v * v; this.rmsN++;
        if (++this.pend >= H) { this.pend = 0; this._hop(); }
      }
    }

    _hop() {
      const N = this.N1, re = this.re1, im = this.im1, r = this.ring, m = this.mask, w0 = this.w - N, win = this.win1;
      for (let i = 0; i < N; i++) { re[i] = r[(w0 + i) & m] * win[i]; im[i] = 0; }
      this.fft1(re, im);
      const NB = this.NB, db = this.bandDb, lin = this.hist[this.frame % this.HK];
      let mx = -200;
      for (let b = 0; b < NB; b++) {
        let p = 0; for (let k = this.bLo[b]; k < this.bHi[b]; k++) p += re[k] * re[k] + im[k] * im[k];
        p *= this.norm1; lin[b] = Math.sqrt(p); db[b] = 10 * Math.log10(p + 1e-14); if (db[b] > mx) mx = db[b];
      }
      // a slowly falling peak: band levels more than 60 dB under it count as silence (no onsets from noise)
      this.peakDb = Math.max(mx, this.peakDb - 0.02); const floor = Math.max(-100, this.peakDb - 60);
      let fl = 0, flLow = 0, flHigh = 0, nL = 0, nH = 0;
      const pd = this.prevDb;
      for (let b = 0; b < NB; b++) {
        const cur = Math.max(db[b], floor);
        const prev = Math.max(pd[b], b > 0 ? pd[b - 1] : -200, b < NB - 1 ? pd[b + 1] : -200, floor);
        const d = Math.max(0, cur - prev);
        fl += d; const f = this.bFc[b];
        if (f < 160) { flLow += d; nL++; } else if (f > 5000) { flHigh += d; nH++; }
      }
      for (let b = 0; b < NB; b++) pd[b] = db[b];
      const j = this.frame % this.OL;
      this.odf[j] = fl / NB; this.odfLow[j] = nL ? flLow / nL : 0; this.odfHigh[j] = nH ? flHigh / nH : 0;
      // level (for the meter and the silence gate)
      const rms = Math.sqrt(this.rmsAcc / Math.max(1, this.rmsN)); this.rmsAcc = 0; this.rmsN = 0;
      const ldb = 20 * Math.log10(rms + 1e-9); this.levelDb = Math.max(ldb, this.levelDb - 0.5);
      const active = ldb > -58; const dt = 1 / this.fps;
      if (active) { this.activeS += dt; this.silentS = 0; } else { this.silentS += dt; if (this.silentS > 4) this.activeS = 0; }
      this._hpss(active);
      if (this.frame % this.CH_EVERY === 0) this._chroma(active);
      this.frame++;
      if (this.frame % this.UPD === 0) this._update();
    }

    _hpss(active) {
      // the centre frame of the last HK: harmonic = median over time per band; percussive = median over 7 bands
      if (this.frame < this.HK) return;
      const NB = this.NB, HK = this.HK, c = this.hist[(this.frame - (HK >> 1)) % HK], tmp = new Float64Array(HK), tb = new Float64Array(7);
      const eH = new Float64Array(5), eP = new Float64Array(5), hb16 = new Float64Array(16), pb16 = new Float64Array(16);
      let flatNum = 0, flatDen = 0, cNum = 0, cDen = 0;
      for (let b = 0; b < NB; b++) {
        for (let k = 0; k < HK; k++) tmp[k] = this.hist[k][b];
        tmp.sort(); const h = tmp[HK >> 1];
        let n = 0; for (let q = Math.max(0, b - 3); q <= Math.min(NB - 1, b + 3); q++) tb[n++] = c[q];
        const sub = tb.subarray(0, n); sub.sort(); const p = sub[n >> 1];
        const mH = (h * h) / (h * h + p * p + 1e-20), e = c[b] * c[b], reg = this.bReg[b];
        eH[reg] += e * mH; eP[reg] += e * (1 - mH);
        hb16[b / 3 | 0] += e * mH; pb16[b / 3 | 0] += e * (1 - mH);
        flatNum += Math.log(e + 1e-14); flatDen += e; cNum += e * Math.log2(this.bFc[b]); cDen += e;
      }
      const a = this.eAlpha, as = this.eAlphaS;
      if (!active) return;
      for (let i = 0; i < 5; i++) { this.eH[i] += a * (eH[i] - this.eH[i]); this.eP[i] += a * (eP[i] - this.eP[i]); this.eHs[i] += as * (eH[i] - this.eHs[i]); }
      const a2 = this.a2;
      for (let i = 0; i < 16; i++) {
        this.hb[i] += a2 * (hb16[i] - this.hb[i]); this.pb[i] += a2 * (pb16[i] - this.pb[i]);
        const d = 10 * Math.log10(hb16[i] + pb16[i] + 1e-14), dm = d - this.dbm[i];
        this.dbm[i] += a2 * dm; this.dbv[i] = (1 - a2) * (this.dbv[i] + a2 * dm * dm);
      }
      { const j = (this.frame - 1) % this.OL; this.onL += a2 * (this.odfLow[j] - this.onL); this.onH += a2 * (this.odfHigh[j] - this.onH); this.onM += a2 * (this.odf[j] - this.onM); }
      const fl = Math.exp(flatNum / NB) / (flatDen / NB + 1e-14); this.flat += a * (fl - this.flat);
      const ce = cDen > 0 ? cNum / cDen : 0; this.cent += a * (ce - this.cent);
      // presence band envelope (1–4 kHz harmonic) for syllable-rate modulation
      let pr = 0; for (let b = 0; b < NB; b++) if (this.bFc[b] > 1000 && this.bFc[b] < 4000) pr += c[b] * c[b];
      this.envPres[(this.frame - (HK >> 1)) % this.OL] = 10 * Math.log10(pr + 1e-14);
    }

    _chroma(active) {
      const N = this.N2, re = this.re2, im = this.im2, r = this.ring, m = this.mask, w0 = this.w - N, win = this.win2;
      const ci = this.chFrame % this.CL, ch = this.chroma[ci]; ch.fill(0); this.chE[ci] = 0; this.chFrame++;
      if (this.w < N || !active) return;
      for (let i = 0; i < N; i++) { re[i] = r[(w0 + i) & m] * win[i]; im[i] = 0; }
      this.fft2(re, im);
      const df = this.sr / N, k0 = Math.ceil(70 / df), k1 = Math.min(N / 2 - 2, Math.floor(4200 / df)), mag = this.mag2;
      let mx = 0;
      for (let k = k0 - 1; k <= k1 + 1; k++) { mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]) * this.norm2; if (mag[k] > mx) mx = mag[k]; }
      if (mx <= 1e-6) return;
      const thr = mx * 0.003; // peaks within 50 dB of the strongest
      let tot = 0, tr = 0, ti = 0;
      for (let k = k0; k <= k1; k++) {
        const b = mag[k]; if (b < thr || b <= mag[k - 1] || b < mag[k + 1]) continue;
        // a local-mean check removes the broadband "peaks" of noise and cymbals
        let lm = 0; for (let q = k - 8; q <= k + 8; q++) lm += mag[q] || 0; lm /= 17; if (b < 2.2 * lm) continue;
        const a = Math.log(mag[k - 1] + 1e-12), bb = Math.log(b + 1e-12), c = Math.log(mag[k + 1] + 1e-12), den = a - 2 * bb + c;
        const dl = den < 0 ? clamp(0.5 * (a - c) / den, -0.5, 0.5) : 0, f = (k + dl) * df;
        const midi = 69 + 12 * Math.log2(f / 440) - this.tuning, nr = Math.round(midi), dev = midi - nr;
        const TU = G.JAM_TUNE || {}, wgt = Math.pow(b, TU.kp || 0.5) * (f < 250 ? (TU.bw || 1.6) : f < 1800 ? 1.0 : 0.6) * Math.cos(Math.PI * dev) ** 2;
        ch[((nr % 12) + 12) % 12] += wgt; tot += wgt;
        const ang = 2 * Math.PI * (midi + this.tuning - Math.round(midi + this.tuning)); tr += wgt * Math.cos(ang); ti += wgt * Math.sin(ang);
      }
      if (tot <= 0) return;
      for (let i = 0; i < 12; i++) ch[i] /= tot;
      this.chE[ci] = 1;
      // tuning (slow): the circular mean of the peaks' offsets from A440's semitone grid
      this.tunRe = 0.995 * this.tunRe + tr / tot; this.tunIm = 0.995 * this.tunIm + ti / tot;
      if (this.chFrame > 40) this.tuning = Math.atan2(this.tunIm, this.tunRe) / (2 * Math.PI);
      const d = this.keyDecay; for (let i = 0; i < 12; i++) this.keyAcc[i] = d * this.keyAcc[i] + ch[i]; this.keyN = d * this.keyN + 1;
    }

    _odfWin(arr, n) { // the last n values of an onset ring, oldest first
      const o = new Float64Array(n), f = this.frame;
      for (let i = 0; i < n; i++) o[i] = arr[((f - n + i) % this.OL + this.OL) % this.OL];
      return o;
    }

    _tempo() {
      const fps = this.fps, n = Math.min(this.frame, Math.round(fps * 12));
      if (n < fps * 4) return;
      const o = this._odfWin(this.odf, n), oL = this._odfWin(this.odfLow, n);
      for (let i = 0; i < n; i++) o[i] = o[i] + 0.5 * oL[i];
      let mean = 0; for (let i = 0; i < n; i++) mean += o[i]; mean /= n;
      for (let i = 0; i < n; i++) o[i] -= mean;
      const L = Math.min(n - 2, Math.ceil(fps * 4.2)), rr = new Float64Array(L + 2);
      for (let l = 0; l <= L + 1; l++) { let s = 0; for (let i = 0; i + l < n; i++) s += o[i] * o[i + l]; rr[l] = s / (n - l); }
      if (rr[0] <= 0) return;
      for (let l = L + 1; l >= 0; l--) rr[l] /= rr[0];
      const at = (x) => { if (x > L) return 0; const i = Math.floor(x), f = x - i; return rr[i] * (1 - f) + rr[i + 1] * f; };
      const TU = G.JAM_TUNE || {};
      const g = this.bpmGrid, sc = new Float64Array(g.length);
      for (let i = 0; i < g.length; i++) {
        const T = 60 * fps / g[i]; let s = 0, wsum = 0;
        for (let mlt = 1; mlt <= 4; mlt++) { if (mlt * T > L) break; const wm = 1 / mlt; s += wm * at(mlt * T); wsum += wm; }
        // the half-period: a real beat usually has some energy there too (8th notes); a wrong double-time has its
        // "beats" alternate strong/weak — penalise a period whose odd multiples of T/2... (kept mild)
        sc[i] = wsum > 0 ? s / wsum : 0;
      }
      this.tN = (this.tN || 0) + 1; const al = Math.max(TU.al || 0.12, 1 / this.tN); // running mean at first, then ~4 s memory
      const acc = this.tAcc; for (let i = 0; i < g.length; i++) acc[i] = (1 - al) * acc[i] + al * sc[i];
      const PC0 = TU.pc || 110, PSD = TU.psd || 0.8, HW = TU.hw != null ? TU.hw : 0.25;
      const prior = (b) => Math.exp(-0.5 * Math.pow(Math.log2(b / PC0) / PSD, 2));
      // subdivision support: a true beat usually has onsets at its half (8ths); the comb alone can't tell T from 1.5 T
      const f = new Float64Array(g.length);
      let bi = -1, bv = -1e9;
      for (let i = 0; i < g.length; i++) {
        const T = 60 * fps / g[i];
        f[i] = Math.max(0, acc[i] + HW * Math.max(0, at(T / 2))) * prior(g[i]);
        if (f[i] > bv) { bv = f[i]; bi = i; }
      }
      if (bi < 0 || bv <= 0) return;
      // octave fold: most songs are felt between ~78 and ~160 BPM; a winner outside that range yields to its double /
      // half when that one is nearly as strong (the comb can't tell a beat from its half-time on its own)
      const FL = TU.fold != null ? TU.fold : 0.6;
      const near = (bpm) => { let m = -1, mi = -1; for (let d = -8; d <= 8; d++) { const i = Math.round((bpm - 40) / 0.25) + d; if (i >= 0 && i < f.length && f[i] > m) { m = f[i]; mi = i; } } return mi; };
      if (FL > 0 && g[bi] < 78) { const j = near(2 * g[bi]); if (j >= 0 && g[j] <= 160 && f[j] >= FL * bv) bi = j; }
      else if (FL > 0 && g[bi] > 160) { const j = near(g[bi] / 2); if (j >= 0 && g[j] >= 78 && f[j] >= FL * bv) bi = j; }
      bv = f[bi];
      let b = g[bi];
      if (bi > 0 && bi < g.length - 1) { const y0 = f[bi - 1], y1 = f[bi], y2 = f[bi + 1], den = y0 - 2 * y1 + y2; if (den < 0) b += 0.25 * clamp(0.5 * (y0 - y2) / den, -0.5, 0.5); }
      // pulse clarity: how far the winner stands above the typical score
      const med = Array.from(f).sort((a, c) => a - c)[f.length >> 1];
      const clar = clamp((bv - med) / (bv + 1e-9), 0, 1) * clamp(acc[bi] * 3, 0, 1);
      this.tempoConf += 0.3 * (clar - this.tempoConf);
      const fAt = (bpm) => { let m = 0; for (let d = -2; d <= 2; d++) { const i = Math.round((bpm - 40) / 0.25) + d; if (i >= 0 && i < f.length && f[i] > m) m = f[i]; } return m; };
      if (!this.bpm) { this.bpm = b; this.bpmCount = 0; return; }
      if (Math.abs(b - this.bpm) / this.bpm < 0.04) { this.bpm += 0.3 * (b - this.bpm); this.bpmCount = 0; this.bpmCand = 0; return; }
      // a different tempo must win clearly and steadily before the display follows; a metrically related one
      // (×2, ×½, ×3/2, ×2/3 — the usual confusions) has to win by more and for longer
      const ratio = b / this.bpm, related = [2, 0.5, 1.5, 2 / 3, 3, 1 / 3].some((q) => Math.abs(ratio / q - 1) < 0.05);
      const need = related ? 1.25 : 1.12, needN = related ? 10 : 5;
      // the first seconds: the old value has little evidence, so it may still move freely
      const young = this.frame < fps * 10;
      if (fAt(b) > (young ? 1.02 : need) * fAt(this.bpm) * (related && b > 160 ? 1.3 : 1) && this.bpmCand && Math.abs(b - this.bpmCand) / b < 0.04) this.bpmCount++;
      else { this.bpmCand = b; this.bpmCount = fAt(b) > need * fAt(this.bpm) ? 1 : 0; }
      if (this.bpmCount >= (young ? 2 : needN)) { this.bpm = b; this.bpmCount = 0; this.bpmCand = 0; }
    }

    _beat() {
      // the beat phase: a comb with the current period over the last 6 s of onsets, newest beats weighted most
      if (!this.bpm) return;
      const fps = this.fps, T = 60 * fps / this.bpm, n = Math.min(this.frame, Math.round(fps * 6));
      if (n < 2 * T) return;
      const o = this._odfWin(this.odf, n), oL = this._odfWin(this.odfLow, n);
      for (let i = 0; i < n; i++) o[i] = o[i] + 0.5 * oL[i];
      const at = (x) => { if (x < 0 || x > n - 1) return 0; const i = Math.floor(x), f = x - i; return o[i] * (1 - f) + (i + 1 < n ? o[i + 1] : 0) * f; };
      let bestPh = 0, best = -1;
      for (let ph = 0; ph < T; ph += 0.5) {
        let s = 0; for (let k = 0, x = n - 1 - ph; x >= 0; k++, x -= T) s += Math.pow(0.88, k) * (at(x) + 0.5 * (at(x - 1) + at(x + 1)));
        if (s > best) { best = s; bestPh = ph; }
      }
      const lastFrame = this.frame - 1 - bestPh; // onset frames are stamped at their window centre
      // (measured on clips with a known grid: a fresh attack peaks the flux ~17 ms after it starts)
      const lastS = lastFrame * this.H + this.N1 / 2 - this.H - Math.round(0.0167 * this.sr), perS = T * this.H;
      if (this.beatPh && Math.abs(this.beatPh.period - perS) / perS < 0.05) {
        // phase-locked: move the predicted grid a little toward the measurement
        const pr = this.beatPh.last + Math.round((lastS - this.beatPh.last) / perS) * perS;
        let e = lastS - pr; if (e > perS / 2) e -= perS; if (e < -perS / 2) e += perS;
        this.beatPh = { last: pr + 0.25 * e, period: perS };
      } else this.beatPh = { last: lastS, period: perS };
    }

    _rhythm() {
      // meter + downbeat (accent pattern over the last ~24 s of beats), feel (swing) and density
      if (!this.beatPh || !this.bpm) return null;
      const fps = this.fps, T = 60 * fps / this.bpm, n = Math.min(this.frame, this.OL - 1);
      const o = this._odfWin(this.odf, n), oL = this._odfWin(this.odfLow, n), oH = this._odfWin(this.odfHigh, n);
      const lastF = (this.beatPh.last - this.N1 / 2 + this.H) / this.H, f0 = this.frame - n; // frame index of o[0]
      const beats = []; for (let x = lastF; x - f0 >= 2 && beats.length < 48; x -= T) beats.unshift(x - f0);
      const nb = beats.length; if (nb < 8) return null;
      const pk = (arr, x, r) => { let m = 0; for (let i = Math.round(x - r); i <= Math.round(x + r); i++) if (i >= 0 && i < n && arr[i] > m) m = arr[i]; return m; };
      // harmonic novelty per beat: chroma of the beat vs the beat before
      const chAt = (fr) => { // mean chroma over frames [fr, fr+T) (fr in onset frames)
        const v = new Float64Array(12); let c = 0;
        const a = Math.ceil((fr + f0) / this.CH_EVERY), b = Math.floor((fr + f0 + T) / this.CH_EVERY);
        for (let q = a; q <= b; q++) { if (q < this.chFrame - this.CL + 1 || q >= this.chFrame) continue; const ci = q % this.CL; if (!this.chE[ci]) continue; const ch = this.chroma[ci]; for (let i = 0; i < 12; i++) v[i] += ch[i]; c++; }
        return c ? v : null;
      };
      const cos = (a, b) => { if (!a || !b) return 0; let s = 0, na = 0, nb2 = 0; for (let i = 0; i < 12; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb2 += b[i] * b[i]; } return na && nb2 ? s / Math.sqrt(na * nb2) : 0; };
      const accL = [], accN = [], accT = [];
      let prev = null;
      for (let i = 0; i < nb; i++) {
        accL.push(pk(oL, beats[i], 2)); accT.push(pk(o, beats[i], 2));
        const c = chAt(beats[i]); accN.push(prev && c ? 1 - cos(prev, c) : 0); prev = c;
      }
      const z = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; const s = Math.sqrt(a.reduce((x, y) => x + (y - m) * (y - m), 0) / a.length) || 1; return a.map((v) => (v - m) / s); };
      const zl = z(accL), zn = z(accN), zt = z(accT), acc = zl.map((v, i) => v + 1.2 * zn[i] + 0.5 * zt[i]);
      const contrast = (M) => {
        let best = -1e9, bp = 0;
        for (let p = 0; p < M; p++) { let s = 0, c = 0, so = 0, co = 0; for (let i = 0; i < nb; i++) { if ((nb - 1 - i) % M === p) { s += acc[i]; c++; } else { so += acc[i]; co++; } } const v = (c ? s / c : 0) - (co ? so / co : 0); if (v > best) { best = v; bp = p; } }
        return { v: best, p: bp };
      };
      const c3 = contrast(3), c4 = contrast(4);
      let meter = 4, conf = clamp((c4.v - c3.v) / 1.0, -1, 1);
      if (c3.v > c4.v * 1.25 + 0.15) { meter = 3; conf = clamp((c3.v - c4.v) / 1.0, 0, 1); }
      else conf = clamp(conf, 0, 1);
      this.meterCur = this.meterCur ? (this.meterCur === meter ? meter : (this._meterFlip = (this._meterFlip || 0) + 1) >= 6 ? (this._meterFlip = 0, meter) : this.meterCur) : meter;
      if (this.meterCur === meter) this._meterFlip = 0;
      const cm = this.meterCur === 3 ? c3 : c4;
      // beats since the downbeat for the newest beat: cm.p = how many beats ago the "1" was (mod M)
      // feel: onset energy at 1/2 vs 2/3 of each beat (minus the local floor)
      let eH2 = 0, eT = 0, eOn = 0, cnt = 0;
      for (let i = Math.max(0, nb - 24); i < nb - 1; i++) {
        const x = beats[i], r = Math.max(1, T * 0.04);
        const fl = Math.min(pk(o, x + T * 0.25, r), pk(o, x + T * 0.85, r));
        eOn += pk(o, x, 2); eH2 += Math.max(0, pk(o, x + T / 2, r) - fl); eT += Math.max(0, pk(o, x + 2 * T / 3, r) - fl); cnt++;
      }
      const off = (eH2 + eT) / (eOn + 1e-9), swing = eT / (eH2 + eT + 1e-9);
      // density: onset peaks per beat over the last 8 s
      const m8 = Math.min(n, Math.round(fps * 8)), a8 = o.subarray(n - m8);
      let mu = 0; for (const v of a8) mu += v; mu /= m8; let sd = 0; for (const v of a8) sd += (v - mu) * (v - mu); sd = Math.sqrt(sd / m8);
      let peaks = 0, lastP = -99;
      for (let i = 3; i < m8 - 3; i++) { const v = a8[i]; if (v > mu + 0.5 * sd && v >= a8[i - 1] && v >= a8[i + 1] && v >= a8[i - 2] && v >= a8[i + 2] && i - lastP > fps * 0.06) { peaks++; lastP = i; } }
      const perBeat = peaks / (m8 / T);
      let hiShare = 0; { let sH = 0, sA = 0; for (let i = 0; i < n; i++) { sH += oH[i]; sA += o[i]; } hiShare = sA ? sH / sA : 0; }
      return { meter: this.meterCur, meterConf: +conf.toFixed(2), downbeatAgo: cm.p, swing: +swing.toFixed(2), offbeat: +off.toFixed(2), perBeat: +perBeat.toFixed(2), hiShare: +hiShare.toFixed(2) };
    }

    _key() {
      if (this.keyN < this.chFps * 3) return null;
      const a = Array.from(this.keyAcc), s = a.reduce((x, y) => x + y, 0); if (s <= 0) return null;
      const m = s / 12, d = a.map((v) => v - m), nd = Math.sqrt(d.reduce((x, y) => x + y * y, 0)) || 1;
      const scores = [];
      for (let t = 0; t < 12; t++) for (const mode of ['major', 'minor']) {
        const p = PROF[mode]; let c = 0; for (let i = 0; i < 12; i++) c += d[(i + t) % 12] * p[i];
        scores.push({ tonic: t, mode, r: c / nd });
      }
      scores.sort((x, y) => y.r - x.r);
      const best = scores[0];
      // the runner-up that isn't the relative key (relative major/minor share all notes — that one is an alternative, not a rival)
      const rel = (k) => (k.mode === 'major' ? { tonic: (k.tonic + 9) % 12, mode: 'minor' } : { tonic: (k.tonic + 3) % 12, mode: 'major' });
      const isRel = (a1, b1) => { const r = rel(a1); return r.tonic === b1.tonic && r.mode === b1.mode; };
      const rival = scores.find((k) => k !== best && !isRel(best, k));
      const relK = scores.find((k) => isRel(best, k));
      const conf = clamp((best.r - 0.35) / 0.4, 0, 1) * clamp((best.r - rival.r) / 0.12, 0, 1);
      // hysteresis: a new key must lead for 3 updates in a row (1.5 s) — the 20 s memory already smooths it
      const same = (x, y) => x && y && x.tonic === y.tonic && x.mode === y.mode;
      if (!this.keyCur) this.keyCur = best;
      else if (!same(best, this.keyCur)) {
        if (same(best, this.keyCand)) this.keyCandN++; else { this.keyCand = best; this.keyCandN = 1; }
        const curR = scores.find((k) => same(k, this.keyCur)).r;
        if (this.keyCandN >= 3 && best.r - curR > 0.02) { this.keyCur = best; this.keyCandN = 0; this.keyCand = null; }
      } else { this.keyCandN = 0; this.keyCand = null; }
      const cur = scores.find((k) => same(k, this.keyCur));
      return { tonic: cur.tonic, mode: cur.mode, name: PC_EN[cur.tonic] + (cur.mode === 'major' ? ' major' : ' minor'), r: +cur.r.toFixed(3), conf: +conf.toFixed(2),
        relR: relK ? +relK.r.toFixed(3) : 0, alt: relK ? { tonic: relK.tonic, mode: relK.mode } : null, tuning: +(this.tuning * 100).toFixed(0),
        top: scores.slice(0, 3).map((k) => PC_EN[k.tonic] + (k.mode === 'major' ? '' : 'm') + ':' + k.r.toFixed(2)), chroma: a.map((v) => +(v / s).toFixed(3)) };
    }

    _chord() {
      // the chord of the last ~1.2 s (or one beat when the tempo is known), triads only
      const span = Math.max(4, Math.round(this.chFps * (this.bpm ? clamp(60 / this.bpm * 1.0, 0.5, 1.5) : 1.2)));
      const v = new Float64Array(12); let c = 0;
      for (let q = this.chFrame - span; q < this.chFrame; q++) { if (q < 0) continue; const ci = q % this.CL; if (!this.chE[ci]) continue; const ch = this.chroma[ci]; for (let i = 0; i < 12; i++) v[i] += ch[i]; c++; }
      if (c < span / 2) return null;
      const m = v.reduce((a, b) => a + b, 0) / 12, d = Array.from(v, (x) => x - m), nd = Math.sqrt(d.reduce((a, b) => a + b * b, 0)) || 1;
      let best = null, bv = -2, second = -2;
      for (const t of CHORD_T) { let s = 0; for (let i = 0; i < 12; i++) s += d[i] * t.v[i]; s /= nd; if (s > bv) { second = bv; bv = s; best = t; } else if (s > second) second = s; }
      if (!best || bv < 0.55) return null;
      return { root: best.root, q: best.q, name: PC_EN[best.root] + (best.q === 'min' ? 'm' : ''), r: +bv.toFixed(2), margin: +(bv - second).toFixed(2) };
    }

    _instFeatures() {
      const eH = this.eH, eP = this.eP, tot = eH.reduce((a, b) => a + b, 0) + eP.reduce((a, b) => a + b, 0) + 1e-14;
      const totH = eH.reduce((a, b) => a + b, 0) + 1e-14;
      const L = (x) => Math.log10(x + 1e-6);
      // presence-band modulation at syllable rate (3–8 Hz) relative to its total variance, over the last 3 s
      const fps = this.fps, n = Math.min(this.frame - this.HK, Math.round(fps * 3));
      let modS = 0;
      if (n > fps) {
        const e = this._odfWin(this.envPres, n + (this.HK >> 1)).subarray(0, n); let mu = 0; for (const v of e) mu += v; mu /= n;
        let tv = 0; for (const v of e) tv += (v - mu) * (v - mu);
        let bandv = 0; // a crude band-pass: difference of two moving averages (≈ 3–8 Hz)
        const ma = (k) => { const out = new Float64Array(n); let s = 0; for (let i = 0; i < n; i++) { s += e[i]; if (i >= k) s -= e[i - k]; out[i] = s / Math.min(i + 1, k); } return out; };
        const a1 = ma(Math.max(1, Math.round(fps / 16))), a2 = ma(Math.max(2, Math.round(fps / 3)));
        for (let i = 0; i < n; i++) { const v = a1[i] - a2[i]; bandv += v * v; }
        modS = tv > 0 ? bandv / tv : 0;
      }
      let k = null; if (this.keyN > 0) { const s = this.keyAcc.reduce((a, b) => a + b, 0); k = s; }
      // chroma "polyphony": how many pitch classes hold ≥ 40 % of the strongest, over the last second
      const span = Math.round(this.chFps * 1); const v = new Float64Array(12); let c = 0;
      for (let q = this.chFrame - span; q < this.chFrame; q++) { if (q < 0) continue; const ci = q % this.CL; if (!this.chE[ci]) continue; for (let i = 0; i < 12; i++) v[i] += this.chroma[ci][i]; c++; }
      let poly = 0; if (c) { const mx = Math.max(...v); for (const x of v) if (x >= 0.4 * mx) poly++; }
      return [
        eH[0] / tot, eH[1] / tot, eH[2] / tot, eH[3] / tot, eH[4] / tot,
        eP[0] / tot, eP[1] / tot, eP[2] / tot, eP[3] / tot, eP[4] / tot,
        L(eH[0] / totH), L(eH[3] / totH), L(eP[4] / tot + eP[0] / tot),
        this.flat, this.cent / 14, modS, poly / 12,
        this.eHs[0] / (this.eHs.reduce((a, b) => a + b, 0) + 1e-14),
      ].concat(this._bandFeat(), [this.onL / 10, this.onH / 10, this.onM / 10]);
    }

    _bandFeat() {
      // relative band levels (dB under the loudest band), harmonic share per band, spread of the band level
      let mx = 1e-14; const tot = new Float64Array(16);
      for (let i = 0; i < 16; i++) { tot[i] = this.hb[i] + this.pb[i]; if (tot[i] > mx) mx = tot[i]; }
      const rel = [], hs = [], sp = [];
      for (let i = 0; i < 16; i++) { rel.push(Math.max(-60, 10 * Math.log10((tot[i] + 1e-14) / mx)) / 60); hs.push(this.hb[i] / (tot[i] + 1e-14)); sp.push(Math.sqrt(this.dbv[i]) / 10); }
      return rel.concat(hs, sp);
    }

    _inst() {
      const x = this._instFeatures(), out = {};
      if (INST_MODEL) {
        for (const name of Object.keys(INST_MODEL)) {
          const md = INST_MODEL[name]; let z = md.b;
          for (let i = 0; i < x.length && i < md.w.length; i++) z += md.w[i] * (x[i] - md.mu[i]) / md.sd[i];
          out[name] = sigm(z);
        }
      } else {
        // hand-set fallback (before a fitted model exists)
        out.drums = clamp((x[9] + x[5] * 0.5) * 12, 0, 1);
        out.bass = clamp((x[17] - 0.05) * 3, 0, 1);
        out.chords = clamp(((x[1] + x[2]) * 2) * (x[16] * 4), 0, 1);
        out.voice = clamp(x[15] * 2 * (x[3] * 10), 0, 1);
      }
      const live = this.activeS > 1.5;
      for (const k of Object.keys(out)) { const p = this.instLvl[k] || 0, v = live ? out[k] : 0; this.instLvl[k] = p + 0.35 * (v - p); }
      return { lvl: Object.fromEntries(Object.entries(this.instLvl).map(([k, v]) => [k, +v.toFixed(2)])), feat: x.map((v) => +v.toFixed(5)) };
    }

    _update() {
      const t = this.w / this.sr;
      const active = this.activeS > 1.5;
      this._tempo(); this._beat();
      const rh = this._rhythm();
      const key = this._key();
      const chord = this._chord();
      const inst = this._inst();
      this.seq++;
      this.snap = {
        seq: this.seq, t: +t.toFixed(2), level: +this.levelDb.toFixed(1), active, listenS: +this.activeS.toFixed(1),
        key, chord,
        tempo: this.bpm ? { bpm: +this.bpm.toFixed(1), conf: +this.tempoConf.toFixed(2) } : null,
        beat: this.beatPh ? { last: Math.round(this.beatPh.last), period: Math.round(this.beatPh.period) } : null,
        rhythm: rh, inst: inst.lvl, feat: inst.feat, sr: this.sr,
      };
    }
  }

  JamAnalyzer.setModel = (m) => { INST_MODEL = m; };
  JamAnalyzer.PC_EN = PC_EN;
  G.JamAnalyzer = JamAnalyzer;
  if (typeof module !== 'undefined' && module.exports) module.exports = JamAnalyzer;
})(typeof self !== 'undefined' ? self : typeof globalThis !== 'undefined' ? globalThis : this);
