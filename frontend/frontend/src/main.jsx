import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = 'http://127.0.0.1:8001';

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : await response.blob();
  if (!response.ok) throw new Error(data?.detail || 'Request failed.');
  return data;
}

function App() {
  const [page, setPage] = useState('Overview');
  const [members, setMembers] = useState([]);
  const [allSamples, setAllSamples] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [health, setHealth] = useState(null);
  const [message, setMessage] = useState('');
  const [theme, setTheme] = useState(() => localStorage.getItem('voice-analysis-theme') || 'light');

  // Voice samples state
  const [selectedMember, setSelectedMember] = useState('');
  const [file, setFile] = useState(null);
  const [samples, setSamples] = useState([]);

  // Family edit state
  const [name, setName] = useState('');
  const [relation, setRelation] = useState('');
  const [editing, setEditing] = useState(null);
  const [editName, setEditName] = useState('');
  const [editRelation, setEditRelation] = useState('');

  // Analysis workflow state
  const [selectedSampleId, setSelectedSampleId] = useState('');
  const [targetMemberId, setTargetMemberId] = useState('');
  const [analysisMode, setAnalysisMode] = useState('combined');
  const [activeAnalysis, setActiveAnalysis] = useState(null);

  // Comparison workflow state
  const [compareSampleId, setCompareSampleId] = useState('');
  const [compareTargetId, setCompareTargetId] = useState('');
  const [comparisonResult, setComparisonResult] = useState(null);

  const [busy, setBusy] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('voice-analysis-theme', theme);
  }, [theme]);

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (selectedMember) loadSamples(selectedMember);
    else setSamples([]);
  }, [selectedMember]);

  async function load() {
    try {
      const [m, a, h] = await Promise.all([
        api('/api/family-members'),
        api('/api/analyses'),
        api('/api/health'),
      ]);
      setMembers(m);
      setAnalyses(a);
      setHealth(h);

      // Gather all samples across all members for selection menus
      const samplePromises = m.map(mem => api(`/api/voice-samples/${mem.id}`).catch(() => []));
      const nested = await Promise.all(samplePromises);
      const flat = [];
      m.forEach((mem, idx) => {
        (nested[idx] || []).forEach(s => flat.push({ ...s, memberName: mem.name }));
      });
      setAllSamples(flat);
      if (flat.length && !selectedSampleId) setSelectedSampleId(String(flat[0].id));
      if (flat.length && !compareSampleId) setCompareSampleId(String(flat[0].id));
    } catch (e) {
      setHealth(null);
      setMessage(e.message || 'Backend unavailable. Start FastAPI on port 8001.');
    }
  }

  async function loadSamples(memberId) {
    try {
      setSamples(await api(`/api/voice-samples/${memberId}`));
    } catch (e) {
      setMessage(e.message);
    }
  }

  function notify(text) {
    setMessage(text);
    window.clearTimeout(window.__voiceToast);
    window.__voiceToast = window.setTimeout(() => setMessage(''), 4000);
  }

  async function addMember(e) {
    e.preventDefault();
    if (!name.trim()) return notify('Enter a family member name.');
    const fd = new FormData();
    fd.append('name', name);
    fd.append('relation', relation);
    try {
      await api('/api/family-members', { method: 'POST', body: fd });
      setName('');
      setRelation('');
      notify('Family member added.');
      await load();
    } catch (e) {
      notify(e.message);
    }
  }

  function startEdit(member) {
    setEditing(member.id);
    setEditName(member.name);
    setEditRelation(member.relation || '');
  }

  async function saveEdit(memberId) {
    const fd = new FormData();
    fd.append('name', editName);
    fd.append('relation', editRelation);
    try {
      await api(`/api/family-members/${memberId}`, { method: 'PATCH', body: fd });
      setEditing(null);
      notify('Family details updated.');
      await load();
    } catch (e) {
      notify(e.message);
    }
  }

  async function deleteMember(member) {
    if (!window.confirm(`Delete ${member.name} and ALL associated recordings/analyses? This cannot be undone.`)) return;
    try {
      await api(`/api/family-members/${member.id}`, { method: 'DELETE' });
      if (String(selectedMember) === String(member.id)) {
        setSelectedMember('');
        setSamples([]);
      }
      notify(`${member.name} deleted.`);
      await load();
    } catch (e) {
      notify(e.message);
    }
  }

  async function clearMemberSamples(member) {
    if (!window.confirm(`Delete all voice recordings for ${member.name}? The member profile will remain.`)) return;
    try {
      await api(`/api/family-members/${member.id}/samples`, { method: 'DELETE' });
      if (String(selectedMember) === String(member.id)) await loadSamples(member.id);
      notify('Voice data cleared.');
      await load();
    } catch (e) {
      notify(e.message);
    }
  }

  async function uploadSample(e) {
    e.preventDefault();
    if (!selectedMember || !file) return notify('Choose a family member and an audio file.');
    const fd = new FormData();
    fd.append('family_member_id', selectedMember);
    fd.append('file', file);
    try {
      const data = await api('/api/voice-samples', { method: 'POST', body: fd });
      setFile(null);
      e.target.reset();
      notify(`Stored: ${data.filename}`);
      await load();
      await loadSamples(selectedMember);
    } catch (e) {
      notify(e.message);
    }
  }

  async function deleteSample(sample) {
    if (!window.confirm(`Delete recording “${sample.filename}”?`)) return;
    try {
      await api(`/api/voice-samples/${sample.id}`, { method: 'DELETE' });
      notify('Recording deleted.');
      await load();
      await loadSamples(selectedMember);
    } catch (e) {
      notify(e.message);
    }
  }

  async function runAnalysis(e) {
    if (e) e.preventDefault();
    if (!selectedSampleId) return notify('Please select an audio sample to analyze.');
    setBusy(true);
    try {
      const fd = new FormData();
      if (targetMemberId) fd.append('target_member_id', targetMemberId);
      fd.append('mode', analysisMode);
      const res = await api(`/api/analyze/${selectedSampleId}`, { method: 'POST', body: fd });
      setActiveAnalysis(res);
      notify('Analysis complete.');
      await load();
    } catch (e) {
      notify(`Analysis failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function viewAnalysis(analysisId) {
    setBusy(true);
    try {
      const data = await api(`/api/analyses/${analysisId}`);
      setActiveAnalysis(data);
    } catch (e) {
      notify(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function deleteAnalysis(id) {
    if (!window.confirm(`Delete analysis #${id}?`)) return;
    try {
      await api(`/api/analyses/${id}`, { method: 'DELETE' });
      if (activeAnalysis?.id === id) setActiveAnalysis(null);
      notify('Analysis deleted.');
      await load();
    } catch (e) {
      notify(e.message);
    }
  }

  async function runComparison(e) {
    if (e) e.preventDefault();
    if (!compareSampleId) return notify('Select an audio sample for comparison.');
    setBusy(true);
    try {
      const fd = new FormData();
      if (compareTargetId) fd.append('target_member_id', compareTargetId);
      const res = await api(`/api/compare/${compareSampleId}`, { method: 'POST', body: fd });
      setComparisonResult(res);
      notify('Comparison completed.');
    } catch (e) {
      notify(`Comparison failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function clearDatabase() {
    const phrase = window.prompt('This deletes EVERY family member, recording, and analysis. Type CLEAR to continue.');
    if (phrase !== 'CLEAR') return;
    setBusy(true);
    try {
      await api('/api/database', { method: 'DELETE' });
      setMembers([]);
      setAnalyses([]);
      setSamples([]);
      setAllSamples([]);
      setSelectedMember('');
      setActiveAnalysis(null);
      setComparisonResult(null);
      notify('Entire local database cleared.');
    } catch (e) {
      notify(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function exportDatabase() {
    setBusy(true);
    try {
      const blob = await api('/api/database/export');
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `voxfusion-backup-${new Date().toISOString().slice(0, 10)}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      notify('Backup exported.');
    } catch (e) {
      notify(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function importDatabase(e) {
    const selected = e.target.files?.[0];
    if (!selected) return;
    if (!window.confirm('Importing replaces the current local database and stored recordings. Continue?')) {
      e.target.value = '';
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append('file', selected);
      await api('/api/database/import', { method: 'POST', body: fd });
      notify('Backup imported. Reloading data…');
      await load();
      setSelectedMember('');
      setSamples([]);
      setActiveAnalysis(null);
      setComparisonResult(null);
    } catch (e) {
      notify(e.message);
    } finally {
      setBusy(false);
      e.target.value = '';
    }
  }

  const totalSamples = useMemo(() => members.reduce((n, m) => n + (m.sample_count || 0), 0), [members]);
  const modelReady = health && Object.values(health.models || {}).every(Boolean);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <b>VOXFUSION</b>
          <span> / AI SUITE</span>
        </div>
        <nav>
          {['Overview', 'Family', 'Voice Samples', 'Analyses', 'Comparisons', 'Settings'].map((x) => (
            <button key={x} onClick={() => setPage(x)} className={page === x ? 'nav active' : 'nav'}>
              {x}
            </button>
          ))}
        </nav>
        <div className="backend">
          <span className={health?.status === 'ok' ? 'dot on' : 'dot'}></span> Backend{' '}
          {health?.status === 'ok' ? 'online' : 'offline'}
        </div>
      </aside>

      <main className="main">
        <div className="topline">
          <span>{page.toUpperCase()}</span>
          <span className="badge">{modelReady ? 'MODELS CONNECTED' : 'INITIALIZING'}</span>
        </div>

        {/* OVERVIEW PAGE */}
        {page === 'Overview' && (
          <section>
            <h1>
              Evidence, not
              <br />
              certainty.
            </h1>
            <p className="sub">Multi-modal voice deepfake detection, biometric verification & scam-intent analysis.</p>
            <div className="stats">
              <Stat label="Family Members" value={members.length} />
              <Stat label="Voice Samples" value={totalSamples} />
              <Stat label="Completed Analyses" value={analyses.length} />
              <Stat label="AI Pipeline Status" value={modelReady ? 'Online' : 'Ready'} />
            </div>
            <div className="panel">
              <div className="panelhead">
                <h2>Connected AI Modules</h2>
                <span className="demo">VOXFUSION PIPELINE</span>
              </div>
              <div className="modelgrid">
                <div className="model">
                  <b>AASIST</b>
                  <span>Acoustic deepfake countermeasure ({health?.models?.aasist ? 'Active' : 'Offline'})</span>
                </div>
                <div className="model">
                  <b>ECAPA-TDNN</b>
                  <span>Biometric speaker identity ({health?.models?.ecapa ? 'Active' : 'Offline'})</span>
                </div>
                <div className="model">
                  <b>Whisper (small)</b>
                  <span>Automatic speech recognition ({health?.models?.whisper ? 'Active' : 'Offline'})</span>
                </div>
                <div className="model">
                  <b>Scam Intent V1</b>
                  <span>Rule-based urgency & extortion detection ({health?.models?.scam_intent_v1 ? 'Active' : 'Offline'})</span>
                </div>
                <div className="model">
                  <b>Scam Intent V2</b>
                  <span>TF-IDF + Logistic Regression ML classifier ({health?.models?.scam_intent_v2 ? 'Active' : 'Offline'})</span>
                </div>
                <div className="model">
                  <b>Fusion Engine</b>
                  <span>Provisional multi-factor synthesis ({health?.models?.fusion ? 'Active' : 'Offline'})</span>
                </div>
              </div>
              <p className="note" style={{ marginTop: '20px' }}>
                Note: The output risk score is a structured multi-modal synthesis index. It is experimental evidence and not a calibrated probability.
              </p>
            </div>
          </section>
        )}

        {/* FAMILY PAGE */}
        {page === 'Family' && (
          <section>
            <h2 className="pageTitle">Family members</h2>
            <form className="form panel" onSubmit={addMember}>
              <input placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
              <input placeholder="Relation (optional)" value={relation} onChange={(e) => setRelation(e.target.value)} />
              <button>Add member</button>
            </form>
            <div className="list">
              {members.map((m) =>
                editing === m.id ? (
                  <div className="row editRow" key={m.id}>
                    <div className="editFields">
                      <input value={editName} onChange={(e) => setEditName(e.target.value)} />
                      <input value={editRelation} onChange={(e) => setEditRelation(e.target.value)} placeholder="Relation" />
                    </div>
                    <div className="actions">
                      <button className="secondary" onClick={() => saveEdit(m.id)}>Save</button>
                      <button className="ghost" onClick={() => setEditing(null)}>Cancel</button>
                    </div>
                  </div>
                ) : (
                  <div className="row memberRow" key={m.id}>
                    <div>
                      <b>{m.name}</b>
                      <span>{m.relation || 'No relation specified'} · {m.sample_count} voice samples</span>
                    </div>
                    <div className="actions">
                      <button className="ghost" onClick={() => startEdit(m)}>Edit</button>
                      <button className="ghost danger" onClick={() => clearMemberSamples(m)}>Clear voice data</button>
                      <button className="ghost danger" onClick={() => deleteMember(m)}>Delete member</button>
                    </div>
                  </div>
                )
              )}
            </div>
            {!members.length && <div className="empty">No family members registered yet.</div>}
          </section>
        )}

        {/* VOICE SAMPLES PAGE */}
        {page === 'Voice Samples' && (
          <section>
            <h2 className="pageTitle">Voice samples</h2>
            <form className="form panel" onSubmit={uploadSample}>
              <select value={selectedMember} onChange={(e) => setSelectedMember(e.target.value)}>
                <option value="">Select family member</option>
                {members.map((m) => (
                  <option value={m.id} key={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
              <input type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              <button>Store voice sample</button>
            </form>
            <p className="note">Supported: WAV, MP3, M4A, FLAC, OGG, WEBM. Original recordings stored locally for ECAPA reference.</p>
            {selectedMember && (
              <div className="list sampleList">
                {samples.map((s) => (
                  <div className="row sampleRow" key={s.id}>
                    <div>
                      <b>{s.filename}</b>
                      <span>{new Date(s.created_at).toLocaleString()}</span>
                      <audio controls preload="metadata" src={`${API}/api/voice-samples/${s.id}/audio`} />
                    </div>
                    <button className="ghost danger" onClick={() => deleteSample(s)}>Delete</button>
                  </div>
                ))}
                {!samples.length && <div className="empty">No recordings for this family member yet.</div>}
              </div>
            )}
          </section>
        )}

        {/* ANALYSES PAGE */}
        {page === 'Analyses' && (
          <section>
            <h2 className="pageTitle">Voice analysis</h2>

            {/* Analysis Launcher Form */}
            <div className="panel">
              <div className="panelhead">
                <h2>Run new analysis</h2>
                <span className="demo">MULTI-MODAL EVALUATION</span>
              </div>
              <form className="form" onSubmit={runAnalysis} style={{ marginTop: '16px' }}>
                <select value={selectedSampleId} onChange={(e) => setSelectedSampleId(e.target.value)}>
                  <option value="">-- Choose Audio Sample to Analyze --</option>
                  {allSamples.map((s) => (
                    <option value={s.id} key={s.id}>
                      {s.memberName}: {s.filename}
                    </option>
                  ))}
                </select>

                <select value={targetMemberId} onChange={(e) => setTargetMemberId(e.target.value)}>
                  <option value="">-- Reference Speaker: None (Impersonation check only) --</option>
                  {members.map((m) => (
                    <option value={m.id} key={m.id}>
                      Compare with {m.name}'s voice reference
                    </option>
                  ))}
                </select>

                <select value={analysisMode} onChange={(e) => setAnalysisMode(e.target.value)}>
                  <option value="combined">Mode: Full VoxFusion Pipeline (Multi-modal)</option>
                  <option value="aasist_only">Mode: AASIST-Only Baseline (Acoustic only)</option>
                </select>

                <button type="submit" disabled={busy || !selectedSampleId}>
                  {busy ? 'Analyzing audio...' : 'Run Analysis'}
                </button>
              </form>
              {!allSamples.length && (
                <p className="note" style={{ marginTop: '12px' }}>
                  No audio samples found. Please upload recordings on the <b>Voice Samples</b> page first.
                </p>
              )}
            </div>

            {/* Active Analysis Detailed Results */}
            {activeAnalysis && (
              <AnalysisDetailView analysis={activeAnalysis} onClose={() => setActiveAnalysis(null)} />
            )}

            {/* Analysis History List */}
            <h3 style={{ marginTop: '45px', marginBottom: '16px' }}>Analysis History</h3>
            <div className="list">
              {analyses.length ? (
                analyses.map((a) => (
                  <div className="row" key={a.id}>
                    <div>
                      <b>Analysis #{a.id}</b>
                      <span>{a.verdict} · Mode: {a.mode} · {new Date(a.created_at).toLocaleString()}</span>
                    </div>
                    <div className="actions">
                      <button className="ghost" onClick={() => viewAnalysis(a.id)}>View Details</button>
                      <button className="ghost danger" onClick={() => deleteAnalysis(a.id)}>Delete</button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty">No completed analyses yet. Select a sample above and click "Run Analysis".</div>
              )}
            </div>
          </section>
        )}

        {/* COMPARISONS PAGE */}
        {page === 'Comparisons' && (
          <section>
            <h2 className="pageTitle">Model comparison</h2>
            <div className="compare panel">
              <div>
                <b>AASIST-Only Baseline</b>
                <span>Evaluates acoustic synthetic artifacts directly from the raw waveform. Susceptible to compression artifacts on mobile/telephony channels.</span>
              </div>
              <div>
                <b>Full VoxFusion Pipeline</b>
                <span>Multi-modal synthesis combining acoustic anti-spoofing (AASIST), biometric speaker matching (ECAPA-TDNN), speech recognition (Whisper), and dual semantic intent analysis (V1 & V2).</span>
              </div>
            </div>

            {/* Comparison Trigger Form */}
            <div className="panel" style={{ marginTop: '20px' }}>
              <div className="panelhead">
                <h2>Run Side-by-Side Comparison</h2>
                <span className="demo">BASELINE VS VOXFUSION</span>
              </div>
              <form className="form" onSubmit={runComparison} style={{ marginTop: '16px' }}>
                <select value={compareSampleId} onChange={(e) => setCompareSampleId(e.target.value)}>
                  <option value="">-- Select audio sample to compare --</option>
                  {allSamples.map((s) => (
                    <option value={s.id} key={s.id}>
                      {s.memberName}: {s.filename}
                    </option>
                  ))}
                </select>

                <select value={compareTargetId} onChange={(e) => setCompareTargetId(e.target.value)}>
                  <option value="">-- Target reference: None (Unverified speaker) --</option>
                  {members.map((m) => (
                    <option value={m.id} key={m.id}>
                      Target: {m.name}
                    </option>
                  ))}
                </select>

                <button type="submit" disabled={busy || !compareSampleId}>
                  {busy ? 'Running comparison...' : 'Compare Paradigms'}
                </button>
              </form>
            </div>

            {/* Comparison Side-by-Side View */}
            {comparisonResult && (
              <div style={{ marginTop: '25px' }}>
                <div className="compareColumns">
                  {/* Left Column: AASIST Baseline */}
                  <div className="compareCol">
                    <div className="compareColHeader">
                      <h3>AASIST-Only Baseline</h3>
                      <span className={`riskBadge ${getRiskClass(comparisonResult.baseline_aasist?.risk_level)}`}>
                        {comparisonResult.baseline_aasist?.risk_level || 'EVALUATED'}
                      </span>
                    </div>
                    <div className="resultScoreBlock">
                      <b>{comparisonResult.baseline_aasist?.risk_score}</b>
                      <span>/ 100 Risk Score</span>
                    </div>
                    <p style={{ marginTop: '12px', fontWeight: 600 }}>{comparisonResult.baseline_aasist?.verdict}</p>
                    <div className="analysisCardsGrid" style={{ gridTemplateColumns: '1fr', marginTop: '14px' }}>
                      <div className="card">
                        <h4>Acoustic Detection</h4>
                        <div className="cardMetric">Prediction: {comparisonResult.baseline_aasist?.prediction}</div>
                        <div className="cardDesc">Deepfake raw score: {comparisonResult.baseline_aasist?.deepfake_score?.toFixed(4)}</div>
                        <div className="cardDesc">Bona-fide raw score: {comparisonResult.baseline_aasist?.bona_fide_score?.toFixed(4)}</div>
                      </div>
                    </div>
                    <h4 style={{ marginTop: '18px', color: 'var(--muted)', fontSize: '13px' }}>Reasoning</h4>
                    <ul className="evidenceUl">
                      {(comparisonResult.baseline_aasist?.reasons || []).map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Right Column: VoxFusion Combined */}
                  <div className="compareCol">
                    <div className="compareColHeader">
                      <h3>Full VoxFusion Multi-Modal</h3>
                      <span className={`riskBadge ${getRiskClass(comparisonResult.voxfusion_combined?.risk_level)}`}>
                        {comparisonResult.voxfusion_combined?.risk_level || 'EVALUATED'}
                      </span>
                    </div>
                    <div className="resultScoreBlock">
                      <b>{comparisonResult.voxfusion_combined?.risk_score}</b>
                      <span>/ 100 Risk Score</span>
                    </div>
                    <p style={{ marginTop: '12px', fontWeight: 600 }}>{comparisonResult.voxfusion_combined?.verdict}</p>
                    <div className="analysisCardsGrid" style={{ gridTemplateColumns: '1fr', marginTop: '14px' }}>
                      <div className="card">
                        <h4>Multi-Tier Evidence</h4>
                        <div className="cardDesc">AASIST deepfake: {comparisonResult.voxfusion_combined?.models?.aasist?.deepfake_score?.toFixed(4) || 'N/A'} ({comparisonResult.voxfusion_combined?.models?.aasist?.prediction || 'N/A'})</div>
                        <div className="cardDesc">ECAPA speaker verification: {comparisonResult.voxfusion_combined?.models?.ecapa?.status} ({comparisonResult.voxfusion_combined?.models?.ecapa?.similarity_score !== null ? `similarity: ${comparisonResult.voxfusion_combined?.models?.ecapa?.similarity_score?.toFixed(4)}` : 'no reference'})</div>
                        <div className="cardDesc">Scam Intent V1 (Rules): {comparisonResult.voxfusion_combined?.models?.scam_intent_v1?.risk_level} (Score: {comparisonResult.voxfusion_combined?.models?.scam_intent_v1?.scam_intent_score})</div>
                        <div className="cardDesc">Scam Intent V2 (ML Prob): {comparisonResult.voxfusion_combined?.models?.scam_intent_v2?.scam_probability?.toFixed(4)}</div>
                      </div>
                    </div>
                    <h4 style={{ marginTop: '18px', color: 'var(--muted)', fontSize: '13px' }}>Multi-Modal Reasoning</h4>
                    <ul className="evidenceUl">
                      {(comparisonResult.voxfusion_combined?.reasons || []).map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Comparison Insights */}
                <div className="panel" style={{ marginTop: '20px' }}>
                  <h3>Comparison Insights</h3>
                  <ul className="evidenceUl" style={{ color: 'var(--text)' }}>
                    {(comparisonResult.comparison_insights || []).map((item, idx) => (
                      <li key={idx}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </section>
        )}

        {/* SETTINGS PAGE */}
        {page === 'Settings' && (
          <Settings
            theme={theme}
            setTheme={setTheme}
            onExport={exportDatabase}
            onImport={importDatabase}
            onClear={clearDatabase}
            busy={busy}
          />
        )}

        {message && <div className="toast">{message}</div>}
      </main>
    </div>
  );
}

function AnalysisDetailView({ analysis, onClose }) {
  const models = analysis.models || {};
  const evidenceData = analysis.evidence || {};
  const reasons = analysis.reasons || (Array.isArray(evidenceData.reasons) ? evidenceData.reasons : []);
  const evidenceList = analysis.evidence?.evidence || (Array.isArray(evidenceData) ? evidenceData : []);

  const aasist = models.aasist || {};
  const ecapa = models.ecapa || {};
  const whisper = models.whisper || {};
  const v1 = models.scam_intent_v1 || evidenceData.v1 || {};
  const v2 = models.scam_intent_v2 || evidenceData.v2 || {};

  const transcript = whisper.transcript || evidenceData.transcript || '';
  const riskLevel = analysis.risk_level || evidenceData.risk_level || 'LOW';
  const riskScore = analysis.risk_score !== undefined ? analysis.risk_score : (evidenceData.risk_score ?? 0);

  return (
    <div className="panel" style={{ marginTop: '24px', borderLeft: '4px solid var(--text)' }}>
      <div className="resultHeader">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h2 style={{ margin: 0 }}>Analysis #{analysis.id || 'Current'} Result</h2>
            <span className={`riskBadge ${getRiskClass(riskLevel)}`}>{riskLevel} RISK</span>
          </div>
          <p style={{ margin: '8px 0 0', color: 'var(--muted)', fontSize: '14px' }}>
            {analysis.verdict || 'Evaluation Complete'}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div className="resultScoreBlock">
            <b>{typeof riskScore === 'number' ? riskScore.toFixed(1) : riskScore}</b>
            <span>/ 100 Fusion Score</span>
          </div>
          {onClose && <button className="ghost" onClick={onClose}>Close</button>}
        </div>
      </div>

      {/* Model Cards Grid */}
      <h3>Sub-Model Assessments</h3>
      <div className="analysisCardsGrid">
        {/* AASIST */}
        <div className="card">
          <h4>AASIST Deepfake Detector</h4>
          <div className="cardMetric" style={{ textTransform: 'capitalize' }}>
            {aasist.prediction || evidenceData.prediction || (aasist.spoof_score >= 0.5 ? 'deepfake' : 'bona_fide')}
          </div>
          <div className="cardDesc">
            Deepfake Raw Score: {aasist.spoof_score !== undefined ? aasist.spoof_score.toFixed(4) : (analysis.aasist_score ? (1 - analysis.aasist_score).toFixed(4) : 'N/A')}
          </div>
          <div className="cardDesc">
            Bona-fide Raw Score: {aasist.authentic_score !== undefined ? aasist.authentic_score.toFixed(4) : (analysis.aasist_score?.toFixed(4) || 'N/A')}
          </div>
          <div className="cardDesc">
            Raw CM Score: {aasist.raw_bona_fide_score !== undefined ? aasist.raw_bona_fide_score.toFixed(4) : (evidenceData.raw_bona_fide_score ?? 'N/A')}
          </div>
        </div>

        {/* ECAPA */}
        <div className="card">
          <h4>ECAPA-TDNN Speaker Verification</h4>
          <div className="cardMetric" style={{ textTransform: 'capitalize' }}>
            {ecapa.status || (analysis.ecapa_score !== null && analysis.ecapa_score !== undefined ? 'verified' : 'not_verified')}
          </div>
          <div className="cardDesc">
            Speaker Match: {ecapa.same_speaker ? 'Same Speaker' : (ecapa.status === 'not_verified' ? 'No Reference Provided' : 'Different Speaker')}
          </div>
          <div className="cardDesc">
            Voice Match Score: {ecapa.similarity_score !== null && ecapa.similarity_score !== undefined ? ecapa.similarity_score.toFixed(4) : (analysis.ecapa_score?.toFixed(4) ?? 'N/A')}
          </div>
          <div className="cardDesc">{ecapa.detail || 'Reference speaker comparison'}</div>
        </div>

        {/* Whisper */}
        <div className="card">
          <h4>Whisper Speech Recognition</h4>
          <div className="cardMetric">
            {whisper.language ? whisper.language.toUpperCase() : 'EN'} Speech
          </div>
          <div className="cardDesc">
            Language Prob: {whisper.language_probability !== undefined ? whisper.language_probability?.toFixed(4) : (analysis.whisper_score?.toFixed(4) || '1.0000')}
          </div>
          <div className="cardDesc">
            Status: {transcript ? 'Speech Detected' : 'No speech or silent'}
          </div>
        </div>

        {/* Scam Intent V1 */}
        <div className="card">
          <h4>Scam Intent V1 (Rules)</h4>
          <div className="cardMetric">Risk: {v1.risk_level || 'LOW'}</div>
          <div className="cardDesc">Score: {v1.scam_intent_score ?? 0} / 100</div>
          <div className="cardTagList">
            {(v1.matched_categories || []).length > 0 ? (
              v1.matched_categories.map((c) => (
                <span className="cardTag" key={c}>{c}</span>
              ))
            ) : (
              <span className="cardDesc">No scam keywords detected</span>
            )}
          </div>
        </div>

        {/* Scam Intent V2 */}
        <div className="card">
          <h4>Scam Intent V2 (ML Model)</h4>
          <div className="cardMetric">Risk: {v2.risk_level || 'LOW'}</div>
          <div className="cardDesc">
            ML Scam Probability: {v2.scam_probability !== undefined ? v2.scam_probability.toFixed(4) : 'N/A'}
          </div>
          <div className="cardDesc">Score: {v2.scam_intent_score !== undefined ? v2.scam_intent_score.toFixed(1) : 0} / 100</div>
        </div>
      </div>

      {/* Transcript Block */}
      <h3 style={{ marginTop: '24px' }}>Speech Transcript</h3>
      <div className="transcriptQuote">
        {transcript ? `“${transcript}”` : <em>No transcript available for this recording.</em>}
      </div>

      {/* Explanatory Reasons and Evidence */}
      <h3 style={{ marginTop: '24px' }}>Fusion Reasoning & Evidence</h3>
      {reasons.length > 0 && (
        <div>
          <h4 style={{ margin: '10px 0 5px', fontSize: '13px', color: 'var(--muted)' }}>Key Factors:</h4>
          <ul className="evidenceUl">
            {reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      {Array.isArray(evidenceList) && evidenceList.length > 0 && (
        <div style={{ marginTop: '12px' }}>
          <h4 style={{ margin: '10px 0 5px', fontSize: '13px', color: 'var(--muted)' }}>Evidence Points:</h4>
          <ul className="evidenceUl">
            {evidenceList.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="disclaimerBox">
        <b>Evaluation Notice:</b> The final assessment is experimental evidence based on provisional multi-modal weights.
        Raw scores and composite indices are not calibrated probabilities or legal identity determinations.
      </div>
    </div>
  );
}

function getRiskClass(level) {
  if (!level) return 'riskLow';
  const l = String(level).toUpperCase();
  if (l === 'HIGH') return 'riskHigh';
  if (l === 'MEDIUM') return 'riskMedium';
  return 'riskLow';
}

function Settings({ theme, setTheme, onExport, onImport, onClear, busy }) {
  return (
    <section>
      <h2 className="pageTitle">Settings</h2>
      <div className="settingsGrid">
        <div className="panel settingCard">
          <div className="settingTitle">
            <div>
              <h3>Appearance</h3>
              <p>Choose how VOXFUSION looks on this device.</p>
            </div>
            <span className="settingIcon">◐</span>
          </div>
          <div className="segmented">
            <button className={theme === 'light' ? 'selected' : ''} onClick={() => setTheme('light')}>
              Light
            </button>
            <button className={theme === 'dark' ? 'selected' : ''} onClick={() => setTheme('dark')}>
              Dark
            </button>
          </div>
        </div>

        <div className="panel settingCard">
          <div className="settingTitle">
            <div>
              <h3>Database backup</h3>
              <p>Export the SQLite database together with all stored voice samples.</p>
            </div>
            <span className="settingIcon">↥</span>
          </div>
          <button className="wideButton" disabled={busy} onClick={onExport}>
            Export complete backup
          </button>
          <label className="wideButton secondaryButton">
            Import complete backup
            <input type="file" accept=".zip,application/zip" onChange={onImport} hidden />
          </label>
          <p className="micro">Import replaces the current local database and recordings with the selected backup.</p>
        </div>

        <div className="panel settingCard dangerCard">
          <div className="settingTitle">
            <div>
              <h3>Danger zone</h3>
              <p>Permanently remove every family member, recording, and analysis from this local installation.</p>
            </div>
            <span className="settingIcon">!</span>
          </div>
          <button className="wideButton dangerButton" disabled={busy} onClick={onClear}>
            Clear entire database
          </button>
          <p className="micro">A confirmation phrase is required. Export a backup first if you may need the data later.</p>
        </div>

        <div className="panel settingCard">
          <div className="settingTitle">
            <div>
              <h3>Storage & AI Pipeline</h3>
              <p>Current application configuration.</p>
            </div>
          </div>
          <div className="kv">
            <span>API Server</span>
            <b>127.0.0.1:8001</b>
            <span>Database</span>
            <b>Local SQLite (voice_analysis.db)</b>
            <span>Audio Storage</span>
            <b>Local uploads (frontend/backend/data/uploads)</b>
            <span>AI Modules</span>
            <b>AASIST + ECAPA-TDNN + Whisper + Intent V1/V2 + Fusion</b>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
