import React, { useEffect, useRef } from 'react';
 
// Ambient background: sparse node network + faint waveform + cursor light.
// Also sets --mx/--my on hovered cards for the glow. Purely decorative (aria-hidden).
export default function Background() {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current, cx = cv.getContext('2d');
    const RM = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const SM = innerWidth < 760;
    let W, H, pts = [], mx = -999, my = -999, raf, col = '53,208,230';
    const color = () => {
      const h = getComputedStyle(document.documentElement).getPropertyValue('--accent2').trim().replace('#', '');
      if (h.length === 6) col = [0, 2, 4].map((i) => parseInt(h.substr(i, 2), 16)).join(',');
    };
    const size = () => {
      W = cv.width = innerWidth; H = cv.height = innerHeight;
      pts = Array.from({ length: SM ? 30 : 75 }, () => ({ x: Math.random() * W, y: Math.random() * H, z: 0.3 + Math.random() * 0.7, vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3 }));
    };
    const frame = (t) => {
      cx.clearRect(0, 0, W, H);
      const ox = (mx - W / 2) * 0.015, oy = (my - H / 2) * 0.015;
      for (const p of pts) {
        if (!RM) { p.x = (p.x + p.vx * p.z + W) % W; p.y = (p.y + p.vy * p.z + H) % H; }
        p.px = p.x + (SM || RM ? 0 : ox * p.z); p.py = p.y + (SM || RM ? 0 : oy * p.z);
        cx.fillStyle = `rgba(${col},${0.95 * p.z})`; cx.beginPath(); cx.arc(p.px, p.py, 2.1 * p.z, 0, 6.3); cx.fill();
      }
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i], b = pts[j], d = Math.hypot(a.px - b.px, a.py - b.py);
        if (d < 150) { cx.strokeStyle = `rgba(${col},${(1 - d / 150) * 0.4 * Math.min(a.z, b.z)})`; cx.beginPath(); cx.moveTo(a.px, a.py); cx.lineTo(b.px, b.py); cx.stroke(); }
      }
      cx.strokeStyle = `rgba(${col},.22)`; cx.lineWidth = 1.5; cx.beginPath();
      for (let x = 0; x <= W; x += 8) { const y = H * 0.85 + Math.sin(x * 0.008 + t * 0.0006) * 26 * Math.sin(x * 0.002 + t * 0.0003); x ? cx.lineTo(x, y) : cx.moveTo(x, y); }
      cx.stroke();
      if (!RM && !SM && mx > 0) { const g = cx.createRadialGradient(mx, my, 0, mx, my, 260); g.addColorStop(0, `rgba(${col},.12)`); g.addColorStop(1, 'transparent'); cx.fillStyle = g; cx.fillRect(mx - 260, my - 260, 520, 520); }
      if (!RM) raf = requestAnimationFrame(frame);
    };
    const move = (e) => {
      mx = e.clientX; my = e.clientY;
      const c = e.target.closest && e.target.closest('.stat,.panel,.row,.model,.card,.compareCol');
      if (c) { const r = c.getBoundingClientRect(); c.style.setProperty('--mx', e.clientX - r.left + 'px'); c.style.setProperty('--my', e.clientY - r.top + 'px'); }
    };
    const resize = () => { size(); if (RM) frame(0); };
    const mo = new MutationObserver(color); mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    color(); size(); raf = requestAnimationFrame(frame);
    addEventListener('pointermove', move, { passive: true }); addEventListener('resize', resize);
    return () => { cancelAnimationFrame(raf); mo.disconnect(); removeEventListener('pointermove', move); removeEventListener('resize', resize); };
  }, []);
  return <canvas id="vf-bg" ref={ref} aria-hidden="true" />;
}
 
