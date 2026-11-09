/* Jam phase 1 — the page. Mic → AudioWorklet tap → Worker (analyzer.js) → this page shows the snapshots.
   It never plays a sound: the tap's output is silence, and nothing is sent anywhere (no network calls at all). */
(function () {
  'use strict';
  const VER = 'v0.1.3';
  const $ = (id) => document.getElementById(id);
  const HE = ['דו', 'דו דיאז', 'רה', 'מי במול', 'מי', 'פה', 'פה דיאז', 'סול', 'לה במול', 'לה', 'סי במול', 'סי'];
  const EN = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
  const INST = [
    { id: 'drums', he: 'תופים', ico: '🥁' },
    { id: 'bass', he: 'בס', ico: '🎸' },
    { id: 'chords', he: 'אקורדים · גיטרה / פסנתר', ico: '🎹' },
    { id: 'voice', he: 'שירה / כלי סולו', ico: '🎤' },
  ];
  const keyHe = (t, mode) => HE[t] + ' ' + (mode === 'major' ? "מז'ור" : 'מינור');
  const keyEn = (t, mode) => EN[t] + ' ' + (mode === 'major' ? 'major' : 'minor');
  const S = { ctx: null, stream: null, node: null, worker: null, src: null, on: false, snap: null, f0: 0, t0: 0, raf: 0, lastBeatK: null,
    hist: [], shown: {} };
  window.__jam = { snaps: [], shown: S.shown, ver: VER }; // (read by the headless tests)
  $('ver').textContent = VER;
  if ('serviceWorker' in navigator && location.protocol !== 'file:') navigator.serviceWorker.register('sw.js').catch(() => {});
  if (/[?&]debug\b/.test(location.search)) $('dbg').hidden = false;

  // the instrument list
  const ul = $('instList');
  for (const it of INST) {
    const li = document.createElement('li'); li.className = 'inst-item off'; li.id = 'i-' + it.id;
    li.innerHTML = '<span class="i-ico" aria-hidden="true">' + it.ico + '</span><span class="i-name">' + it.he + '</span>' +
      '<span class="i-word"></span><span class="i-bar"><span></span></span>';
    ul.appendChild(li);
  }

  function setStatus(t, cls) { $('statusTxt').textContent = t; $('status').className = 'status' + (cls ? ' ' + cls : ''); }

  async function start() {
    if (S.on) return stop();
    const go = $('go'); go.disabled = true;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('nomic');
      S.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
      const AC = window.AudioContext || window.webkitAudioContext;
      S.ctx = new AC({ latencyHint: 'interactive' });
      if (S.ctx.state === 'suspended') await S.ctx.resume();
      S.worker = new Worker('worker.js');
      S.worker.onmessage = onWorker;
      S.worker.postMessage({ start: { sr: S.ctx.sampleRate } });
      S.src = S.ctx.createMediaStreamSource(S.stream);
      const mute = S.ctx.createGain(); mute.gain.value = 1; // the tap's output is already ±1e-9 (inaudible; see capture-worklet.js)
      if (S.ctx.audioWorklet) {
        await S.ctx.audioWorklet.addModule('capture-worklet.js');
        S.node = new AudioWorkletNode(S.ctx, 'jam-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
        const ch = new MessageChannel();
        S.node.port.postMessage({ worker: ch.port1 }, [ch.port1]);
        S.worker.postMessage({ port: ch.port2 }, [ch.port2]);
      } else { // old browsers: a ScriptProcessor, blocks via the page
        S.node = S.ctx.createScriptProcessor(2048, 1, 1); let frame = 0;
        S.node.onaudioprocess = (e) => { const d = new Float32Array(e.inputBuffer.getChannelData(0)); S.worker.postMessage({ block: { f0: frame, d } }, [d.buffer]); frame += d.length; };
      }
      S.src.connect(S.node); S.node.connect(mute); mute.connect(S.ctx.destination);
      S.on = true; go.classList.add('on'); $('goTxt').textContent = 'עצור'; $('goIco').textContent = '⏹';
      setStatus('מקשיב… צריך כמה שניות של מוזיקה', 'live');
      document.body.classList.add('listening');
      S.raf = requestAnimationFrame(tick);
    } catch (e) {
      const n = e && e.name;
      setStatus(n === 'NotAllowedError' ? 'צריך לאשר גישה למיקרופון (בהגדרות הדפדפן) ולנסות שוב'
        : n === 'NotFoundError' ? 'לא נמצא מיקרופון במכשיר' : 'המיקרופון לא נפתח. נסו לרענן את הדף', 'err');
      cleanup();
    }
    go.disabled = false;
  }
  function cleanup() {
    try { if (S.node && S.node.port) S.node.port.postMessage({ stop: true }); } catch (e) { /* closed */ }
    try { S.src && S.src.disconnect(); S.node && S.node.disconnect(); } catch (e) { /* closed */ }
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    if (S.worker) S.worker.terminate();
    if (S.ctx) S.ctx.close().catch(() => {});
    S.ctx = S.stream = S.node = S.worker = S.src = null; S.on = false; cancelAnimationFrame(S.raf);
  }
  function stop() {
    cleanup(); $('go').classList.remove('on'); $('goTxt').textContent = 'התחל להקשיב'; $('goIco').textContent = '🎧'; document.body.classList.remove('listening');
    setStatus('הפסקתי להקשיב. אפשר להתחיל שוב בכל רגע');
  }
  $('go').addEventListener('click', start);
  document.addEventListener('visibilitychange', () => { if (document.hidden && S.on) stop(); });

  // ---- the snapshots → the screen (with display-side hysteresis: a value must be confident before it shows) ----
  function onWorker(e) {
    const m = e.data || {}; if (!m.snap) return;
    S.snap = m.snap; S.f0 = m.f0; window.__jam.diag = Object.assign({}, m.diag, { ctxF: S.ctx ? Math.round(S.ctx.currentTime * S.ctx.sampleRate) : 0, state: S.ctx && S.ctx.state });
    const s = m.snap;
    window.__jam.snaps.push(s); if (window.__jam.snaps.length > 600) window.__jam.snaps.shift();
    render(s);
  }

  function word(v) { return v >= 0.62 ? 'כן' : v >= 0.38 ? 'אולי' : 'לא נשמע'; }

  function render(s) {
    const sh = S.shown;
    // level meter
    const lv = Math.max(0, Math.min(1, (s.level + 70) / 60)); $('meterBar').style.width = (lv * 100).toFixed(0) + '%';
    if (!s.active) {
      setStatus(s.listenS > 0 ? 'מקשיב…' : 'שקט. מחכה למוזיקה…', 'live');
    } else {
      setStatus('מקשיב ומנתח · ' + Math.round(s.t) + ' שניות', 'live');
    }
    const ready = s.active && s.listenS >= 4;
    // key
    const k = s.key;
    if (ready && k && k.conf >= 0.15) {
      sh.key = keyEn(k.tonic, k.mode); sh.keyConf = k.conf;
      $('keyHe').textContent = keyHe(k.tonic, k.mode); $('keyEn').textContent = sh.key;
      $('keyConf').style.width = Math.round(15 + 85 * k.conf) + '%';
      $('keyConfTxt').textContent = k.conf >= 0.6 ? 'בטוח' : k.conf >= 0.3 ? 'די בטוח' : 'לא בטוח';
      $('keyAlt').textContent = k.alt && k.relR > k.r - 0.08 ? 'או ' + keyHe(k.alt.tonic, k.alt.mode) + ' (אותם צלילים)' : '';
      $('cKey').classList.remove('wait');
    } else if (!sh.key) { $('keyHe').textContent = 'מקשיב…'; $('cKey').classList.add('wait'); }
    // tempo
    const tp = s.tempo;
    if (ready && tp && tp.conf >= 0.12 && s.listenS >= 5) {
      sh.bpm = Math.round(tp.bpm); sh.tempoConf = tp.conf;
      $('bpm').textContent = sh.bpm; $('cTempo').classList.remove('wait');
      // a hint only when the pulse is really unclear, in plain words (no second number)
      const alt = tp.bpm >= 120 ? 'אולי חצי מזה' : tp.bpm <= 85 ? 'אולי כפול מזה' : 'עוד לא בטוח';
      $('tempoNote').textContent = tp.conf < 0.2 ? 'הקצב עוד לא חד · ' + alt : tempoWord(tp.bpm);
    } else if (!sh.bpm) { $('bpm').textContent = '…'; $('cTempo').classList.add('wait'); $('tempoNote').textContent = 'מקשיב…'; }
    // rhythm
    const r = s.rhythm;
    if (ready && r && sh.bpm) {
      const sure = r.meterConf >= 0.25;
      sh.meter = r.meter + '/4'; sh.meterSure = sure;
      $('meter').textContent = sure ? r.meter + '/4' : r.meter + '/4 ?';
      sh.feel = r.swing > 0.62 && r.offbeat > 0.25 ? 'swung' : r.offbeat > 0.12 ? 'straight' : 'plain';
      $('feel').textContent = sh.feel === 'swung' ? 'מתנדנד (סווינג)' : sh.feel === 'straight' ? 'ישר' : 'פשוט, על הפעמות';
      sh.density = r.perBeat;
      $('density').textContent = r.perBeat < 1.5 ? 'דליל' : r.perBeat < 3 ? 'בינוני' : r.perBeat < 5 ? 'צפוף' : 'צפוף מאוד';
      $('rhythmNote').textContent = sure ? '' : 'המשקל עוד לא ברור';
      $('cRhythm').classList.remove('wait'); buildBeats(r.meter);
    } else if (!sh.meter) { $('meter').textContent = '…'; $('cRhythm').classList.add('wait'); }
    // chord
    const c = s.chord;
    if (s.active && c && c.r >= 0.6) {
      sh.chord = c.name; $('chordHe').textContent = HE[c.root] + (c.q === 'min' ? ' מינור' : " מז'ור"); $('chordEn').textContent = c.name;
      $('cChord').className = 'chord-now';
    } else if (sh.chord) $('cChord').className = 'chord-now stale'; else { $('chordHe').textContent = 'מקשיב…'; $('cChord').className = 'chord-now wait'; }
    // instruments
    sh.inst = {};
    for (const it of INST) {
      const v = s.active ? (s.inst[it.id] || 0) : 0, li = $('i-' + it.id);
      sh.inst[it.id] = v;
      li.className = 'inst-item ' + (v >= 0.62 ? 'yes' : v >= 0.38 ? 'maybe' : 'off');
      li.querySelector('.i-word').textContent = s.listenS >= 3 ? word(v) : '…';
      li.querySelector('.i-bar span').style.width = Math.round(v * 100) + '%';
    }
    if (!$('dbg').hidden) $('dbg').textContent = JSON.stringify({ t: s.t, key: k && k.top, tuning: k && k.tuning, tempo: tp, rh: r, inst: s.inst }, null, 1);
  }
  function tempoWord(b) { return b < 76 ? 'איטי' : b < 108 ? 'בינוני' : b < 140 ? 'מהיר' : 'מהיר מאוד'; }

  let beatsM = 0;
  function buildBeats(M) {
    if (M === beatsM) return; beatsM = M; const el = $('beats'); el.innerHTML = '';
    for (let i = 0; i < M; i++) { const d = document.createElement('span'); d.className = 'bt' + (i === 0 ? ' one' : ''); el.appendChild(d); }
  }

  // the beat pulse: predict the beats from the last phase-locked beat and the period, light the dot on each one
  function tick() {
    S.raf = requestAnimationFrame(tick);
    const s = S.snap; if (!s || !s.beat || !S.ctx || !S.shown.bpm || !s.active) return;
    const sr = s.sr, now = S.ctx.currentTime * sr - S.f0; // in analyzer samples
    const k = Math.floor((now - s.beat.last) / s.beat.period);
    if (k === S.lastBeatK && S.lastBeatLast === s.beat.last) return;
    const beatAt = s.beat.last + k * s.beat.period;
    if (now - beatAt > 0.12 * sr) return; // only light within ~120 ms of the beat
    const abs = Math.round(beatAt / s.beat.period);
    if (abs === S.lastAbs) return; S.lastAbs = abs; S.lastBeatK = k; S.lastBeatLast = s.beat.last;
    const M = beatsM || 4, pos = s.rhythm ? ((s.rhythm.downbeatAgo + k) % M + M) % M : 0;
    const dots = $('beats').children;
    for (let i = 0; i < dots.length; i++) dots[i].classList.toggle('hit', i === pos);
    const card = $('cTempo'); card.classList.remove('pulse'); void card.offsetWidth; card.classList.add('pulse');
    window.__jam.beats = (window.__jam.beats || 0) + 1;
  }
})();
