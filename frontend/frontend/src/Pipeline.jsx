import React, { useEffect, useState } from 'react';

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
      <text x="481" y="142" textAnchor="middle" fontWeight="600">FUSION</text><text className="mu" x="481" y="158" textAnchor="middle">35 · 30 · ±12</text>
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