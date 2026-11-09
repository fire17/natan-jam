/* Jam phase 1: the analysis Worker. Blocks from the mic tap in → JamAnalyzer → a snapshot to the page ~twice a second. */
importScripts('analyzer.js', 'inst-model.js');
let A = null, f0 = null, inPort = null, lastSeq = 0, nextF = null, gaps = 0, gapFrames = 0, blocks = 0;
function onBlock(m) {
  if (!A) return;
  if (f0 === null) f0 = m.f0;
  blocks++;
  if (nextF !== null && m.f0 !== nextF) { gaps++; gapFrames += m.f0 - nextF; }
  nextF = m.f0 + m.d.length;
  A.push(m.d);
  if (A.snap && A.snap.seq !== lastSeq) { lastSeq = A.snap.seq; postMessage({ snap: A.snap, f0, diag: { blocks, gaps, gapFrames, lastF: nextF } }); }
}
onmessage = (e) => {
  const m = e.data || {};
  if (m.start) { A = new JamAnalyzer(m.start.sr); if (self.JAM_INST_MODEL) JamAnalyzer.setModel(self.JAM_INST_MODEL); f0 = null; lastSeq = 0; nextF = null; gaps = 0; gapFrames = 0; blocks = 0; }
  if (m.port) { if (inPort) inPort.close(); inPort = m.port; inPort.onmessage = (ev) => onBlock(ev.data); }
  if (m.block) onBlock(m.block); // (ScriptProcessor fallback: blocks via the page)
  if (m.stop) { A = null; if (inPort) { inPort.close(); inPort = null; } }
};
