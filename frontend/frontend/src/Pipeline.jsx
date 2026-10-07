import React, { useEffect, useRef, useState } from 'react';

// Logo mark (waveform + fusion)
export function Logo({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs><linearGradient id="vfLg"><stop stopColor="#7c6cff" /><stop offset="1" stopColor="#35d0e6" /></linearGradient></defs>
      <path d="M3 16h4l3-9 5 18 4-14 3 5h7" fill="none" stroke="url(#vfLg)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Architecture diagram. NO data values: it shows the pipeline structure only.
// Weights (35 / 30 / ±12) come from src/fusion.py in the repo.
export function FusionDiagram() {
  const RM = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const N = [['ECAPA-TDNN', 'speaker', 20, '#35d0e6'], ['AASIST', 'deepfake', 76, '#7c6cff'], ['Whisper', 'transcript', 132, '#35d0e6'], ['Scam Intent V1', 'rules', 188, '#7c6cff'], ['Scam Intent V2', 'classifier', 244, '#35d0e6']];
  return (
    <svg className="vfSvg" viewBox="0 0 720 290" width="100%" role="img" aria-label="Five signals (ECAPA-TDNN, AASIST, Whisper, Scam Intent V1 and V2) converge into the fusion engine, producing a threat assessment">
      {N.map((n, i) => {
        const y = n[2] + 20, d = `M170 ${y} C 330 ${y}, 330 145, 440 145`;
        return (
          <g key={n[0]}>
            <path d={d} className="vfFlow" fill="none" stroke={n[3]} strokeWidth="2" opacity=".8" />
            {!RM && <circle r="3.2" fill={n[3]}><animateMotion dur="2.6s" begin={`${i * 0.4}s`} repeatCount="indefinite" path={d} /></circle>}
            <rect x="10" y={n[2]} width="160" height="40" rx="10" fill="var(--surface2)" stroke="var(--line)" />
            <text x="24" y={n[2] + 17} fontWeight="600">{n[0]}</text><text className="mu" x="24" y={n[2] + 31}>{n[1]}</text>
          </g>
        );
      })}
      <path d="M522 145H580" className="vfFlow" stroke="#7c6cff" strokeWidth="3" fill="none" />
      <circle cx="481" cy="145" r="42" fill="var(--surface)" stroke="#7c6cff" strokeWidth="2" />
      <circle cx="481" cy="145" r="54" fill="none" stroke="#7c6cff" opacity=".3" className="vfPulse" />
      <text x="481" y="150" textAnchor="middle" fontWeight="600">FUSION</text><text className="mu" x="481" y="158" textAnchor="middle"></text>
      <rect x="580" y="115" width="130" height="60" rx="12" fill="var(--surface2)" stroke="#7c6cff" />
      <text x="645" y="142" textAnchor="middle" fontWeight="600">THREAT</text><text className="mu" x="645" y="159" textAnchor="middle">assessment</text>
    </svg>
  );
}

// Honest busy indicator: the backend reports NO per-model progress, so nothing here
// pretends to track stages. Chips simply pulse while the single request is in flight.
export function PipelineBusy({ label = 'Running the VoxFusion pipeline…' }) {
  return (
    <div className="panel" role="status" aria-live="polite">
      <b>{label}</b>
      <div className="vfBusy">{['ECAPA-TDNN', 'AASIST', 'Whisper', 'Scam Intent V1', 'Scam Intent V2', 'Fusion'].map((m) => <span key={m}>{m}</span>)}</div>
    </div>
  );
}

// Count-up for a REAL value coming from the backend (renders instantly if reduced motion / non-number).
export function AnimatedNumber({ value, digits = 1, duration = 1400 }) {
  const [v, setV] = useState(typeof value === 'number' ? 0 : value);
  useEffect(() => {
    if (typeof value !== 'number') { setV(value); return; }
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { setV(value); return; }
    let raf; const t0 = performance.now();
    const f = (n) => { const k = Math.min((n - t0) / duration, 1); setV(value * (1 - Math.pow(1 - k, 3))); if (k < 1) raf = requestAnimationFrame(f); };
    raf = requestAnimationFrame(f); return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  return <>{typeof v === 'number' ? v.toFixed(digits) : v}</>;
}

export const MODEL_INFO = {
  'AASIST': 'Acoustic deepfake countermeasure. Produces a deepfake/spoof score and prediction. In fusion it contributes up to 35 points.',
  'ECAPA-TDNN': 'Speaker verification against the enrolled reference samples of the target family member. In fusion it acts as a contextual modifier (+12 if spoofed audio matches the enrolled speaker, +6 on mismatch), not an independent vote.',
  'Whisper (small)': 'Automatic speech recognition. Produces the transcript that feeds both scam-intent models.',
  'Scam Intent V1': 'Rule-based urgency and extortion detection over the transcript. Returns detected categories and a risk level.',
  'Scam Intent V2': 'TF-IDF + Logistic Regression classifier returning a scam probability. Scam intent contributes up to 30 points in fusion.',
  'Fusion Engine': 'Provisional multi-factor synthesis of the signals above into a 0-100 index with reasons and evidence. Experimental; not a calibrated probability.',
};

export function Drawer({ open, title, onClose, children }) {
  useEffect(() => {
    if (!open) return;
    const k = (e) => e.key === 'Escape' && onClose();
    addEventListener('keydown', k);
    return () => removeEventListener('keydown', k);
  }, [open, onClose]);
  return (
    <>
      <div className={'vfScrim' + (open ? ' open' : '')} onClick={onClose} />
      <aside className={'vfDrawer' + (open ? ' open' : '')} role="dialog" aria-modal="true" aria-label={title} inert={!open}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ margin: 0 }}>{title}</h2>
          <button className="ghost" onClick={onClose} aria-label="Close">✕</button>
        </div>
        {children}
      </aside>
    </>
  );
}

/* ---------- Upload zone (wraps the existing file state; no API change) ---------- */
const AUDIO_EXT = /\.(wav|mp3|m4a|flac|ogg|webm)$/i;
export function UploadZone({ file, onFile }) {
  const [over, setOver] = useState(false);
  const [err, setErr] = useState('');
  const inp = useRef(null);
  const take = (f) => {
    if (!f) return;
    if (!(f.type.startsWith('audio/') || AUDIO_EXT.test(f.name))) { setErr(`${f.name} isn’t a supported audio file. Use WAV, MP3, M4A, FLAC, OGG or WEBM.`); return; }
    setErr(''); onFile(f);
  };
  const key = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inp.current.click(); } };
  return (
    <div style={{ flex: '1 1 100%' }}>
      <div className={'vfDrop' + (over ? ' over' : '') + (file ? ' has' : '')} role="button" tabIndex={0} aria-label="Choose or drop an audio file"
        onClick={() => inp.current.click()} onKeyDown={key}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files?.[0]); }}>
        <input ref={inp} type="file" accept="audio/*" hidden onChange={(e) => { take(e.target.files?.[0]); e.target.value = ''; }} />
        {file ? (
          <>
            <span className="vfTick" aria-hidden="true">✓</span>
            <div style={{ flex: 1, minWidth: 0 }}><b style={{ wordBreak: 'break-all' }}>{file.name}</b><div className="note">{(file.size / 1024).toFixed(0)} KB · ready to store</div></div>
            <button type="button" className="ghost" onClick={(e) => { e.stopPropagation(); onFile(null); setErr(''); }}>Remove</button>
          </>
        ) : (
          <div><b>Drop an audio file here</b><div className="note">or click to browse · WAV, MP3, M4A, FLAC, OGG, WEBM</div></div>
        )}
      </div>
      {err && <p role="alert" style={{ color: 'var(--danger)', margin: '8px 0 0', fontSize: 13 }}>{err}</p>}
    </div>
  );
}

/* ---------- Waveform computed from the REAL audio file served by the backend ---------- */
const peakCache = new Map();
async function computePeaks(src, n = 90) {
  if (peakCache.has(src)) return peakCache.get(src);
  const res = await fetch(src);
  if (!res.ok) throw new Error('audio fetch failed');
  const buf = await res.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  try {
    const audio = await new Promise((ok, no) => ctx.decodeAudioData(buf, ok, no));
    const ch = audio.getChannelData(0), step = Math.max(1, Math.floor(ch.length / n)), skip = Math.max(1, Math.floor(step / 200)), peaks = [];
    for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step; j < (i + 1) * step && j < ch.length; j += skip) m = Math.max(m, Math.abs(ch[j])); peaks.push(m); }
    const mx = Math.max(...peaks) || 1, out = peaks.map((x) => x / mx);
    peakCache.set(src, out);
    return out;
  } finally { if (ctx.close) ctx.close(); }
}

export function SampleAudio({ src }) {
  const [peaks, setPeaks] = useState(null);
  const [err, setErr] = useState(false);
  const [prog, setProg] = useState(0);
  const a = useRef(null);
  useEffect(() => { let live = true; setPeaks(null); setErr(false); computePeaks(src).then((p) => live && setPeaks(p)).catch(() => live && setErr(true)); return () => { live = false; }; }, [src]);
  const seek = (e) => { const el = a.current; if (!el || !el.duration) return; const r = e.currentTarget.getBoundingClientRect(); el.currentTime = ((e.clientX - r.left) / r.width) * el.duration; };
  return (
    <div className="vfSample">
      {!err && (
        <div className="vfWave" onClick={seek} role="img" aria-label="Audio waveform. Click to seek.">
          {peaks ? peaks.map((v, i) => <i key={i} className={i / peaks.length < prog ? 'on' : ''} style={{ height: `${Math.max(8, v * 100)}%` }} />) : <span className="vfSkel" />}
        </div>
      )}
      <audio ref={a} controls preload="metadata" src={src} onTimeUpdate={(e) => setProg(e.target.duration ? e.target.currentTime / e.target.duration : 0)} onEnded={() => setProg(0)} />
    </div>
  );
}

/* ---------- Error boundary: shows the real error instead of a blank white page ---------- */
export class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { err: null }; }
  static getDerivedStateFromError(err) { return { err }; }
  componentDidCatch(err, info) { console.error('VoxFusion UI error:', err, info?.componentStack); }
  render() {
    if (!this.state.err) return this.props.children;
    return (
      <div className="panel" role="alert" style={{ margin: 24, borderLeft: '4px solid var(--danger)' }}>
        <h3>This view hit an error</h3>
        <p className="note">The backend and your data are not affected. Error message:</p>
        <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--danger)' }}>{(String(this.state.err?.message || this.state.err) + '\n' + String(this.state.err?.stack || '')).slice(0, 900)}</pre>
        <button className="ghost" onClick={() => this.props.inline ? this.setState({ err: null }) : location.reload()}>{this.props.inline ? 'Dismiss' : 'Reload'}</button>
      </div>
    );
  }
}