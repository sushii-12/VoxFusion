import json
import shutil
import uuid
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

from .db import Base, engine, get_db, UPLOAD_DIR, DATA_DIR
from .models import FamilyMember, VoiceSample, Analysis
from .services.schemas import result_dict
from .services.aasist import AASISTAdapter
from .services.ecapa import ECAPAAdapter
from .services.whisper import WhisperAdapter
from .services.fusion import ScamIntentAdapter, FusionAdapter, FusionResult

Base.metadata.create_all(bind=engine)

app = FastAPI(title="VOICE / ANALYSIS API - VoxFusion", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_BASE = "http://127.0.0.1:8000"

aasist = AASISTAdapter()
ecapa = ECAPAAdapter()
whisper = WhisperAdapter()
scam_intent = ScamIntentAdapter()
fusion_adapter = FusionAdapter()


def _delete_sample_files(samples):
    for sample in samples:
        try:
            Path(sample.stored_path).unlink(missing_ok=True)
        except OSError:
            pass


def _clear_all_storage():
    for path in UPLOAD_DIR.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health():
    aasist_ok = aasist.is_available()
    ecapa_ok = ecapa.is_available()
    whisper_ok = whisper.is_available()
    all_ready = aasist_ok and ecapa_ok and whisper_ok

    return {
        "status": "ok",
        "demo_mode": False,
        "models": {
            "aasist": aasist_ok,
            "ecapa": ecapa_ok,
            "whisper": whisper_ok,
            "scam_intent_v1": True,
            "scam_intent_v2": True,
            "fusion": True,
        },
        "message": (
            "VoxFusion AI pipeline loaded and operational."
            if all_ready
            else "VoxFusion core models initializing..."
        ),
    }


@app.get("/api/family-members")
def list_family_members(db: Session = Depends(get_db)):
    members = db.scalars(select(FamilyMember).order_by(FamilyMember.id)).all()
    return [
        {"id": m.id, "name": m.name, "relation": m.relation, "sample_count": len(m.samples)}
        for m in members
    ]


@app.post("/api/family-members")
def create_family_member(name: str = Form(...), relation: str | None = Form(None), db: Session = Depends(get_db)):
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name is required.")
    member = FamilyMember(name=name, relation=(relation or "").strip() or None)
    db.add(member)
    db.commit()
    db.refresh(member)
    return {"id": member.id, "name": member.name, "relation": member.relation, "sample_count": 0}


@app.patch("/api/family-members/{member_id}")
def update_family_member(member_id: int, name: str = Form(...), relation: str | None = Form(None), db: Session = Depends(get_db)):
    member = db.get(FamilyMember, member_id)
    if not member:
        raise HTTPException(404, "Family member not found.")
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name is required.")
    member.name = name
    member.relation = (relation or "").strip() or None
    db.commit()
    db.refresh(member)
    return {"id": member.id, "name": member.name, "relation": member.relation, "sample_count": len(member.samples)}


@app.delete("/api/family-members/{member_id}")
def delete_family_member(member_id: int, db: Session = Depends(get_db)):
    member = db.get(FamilyMember, member_id)
    if not member:
        raise HTTPException(404, "Family member not found.")
    samples = list(member.samples)
    _delete_sample_files(samples)
    db.query(Analysis).filter(Analysis.target_member_id == member_id).delete(synchronize_session=False)
    sample_ids = [s.id for s in samples]
    if sample_ids:
        db.query(Analysis).filter(Analysis.sample_id.in_(sample_ids)).delete(synchronize_session=False)
    db.delete(member)
    db.commit()
    return {"ok": True, "message": f"Deleted family member '{member.name}' and all associated data."}


@app.delete("/api/family-members/{member_id}/samples")
def delete_family_member_samples(member_id: int, db: Session = Depends(get_db)):
    member = db.get(FamilyMember, member_id)
    if not member:
        raise HTTPException(404, "Family member not found.")
    samples = list(member.samples)
    sample_ids = [s.id for s in samples]
    _delete_sample_files(samples)
    if sample_ids:
        db.query(Analysis).filter(Analysis.sample_id.in_(sample_ids)).delete(synchronize_session=False)
    for sample in samples:
        db.delete(sample)
    db.commit()
    return {"ok": True, "deleted_samples": len(samples)}


@app.post("/api/voice-samples")
async def upload_voice_sample(family_member_id: int = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    member = db.get(FamilyMember, family_member_id)
    if not member:
        raise HTTPException(404, "Family member not found.")
    original = Path(file.filename or "audio")
    allowed = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
    if original.suffix.lower() not in allowed:
        raise HTTPException(400, f"Unsupported audio type. Allowed: {', '.join(sorted(allowed))}")
    safe_name = f"{uuid.uuid4().hex}{original.suffix.lower()}"
    destination = UPLOAD_DIR / safe_name
    content = await file.read()
    destination.write_bytes(content)
    sample = VoiceSample(family_member_id=member.id, filename=original.name, stored_path=str(destination))
    db.add(sample)
    db.commit()
    db.refresh(sample)
    return {"id": sample.id, "family_member_id": member.id, "filename": sample.filename, "size_bytes": len(content), "message": "Voice sample stored successfully."}


@app.get("/api/voice-samples/{family_member_id}")
def list_voice_samples(family_member_id: int, db: Session = Depends(get_db)):
    if not db.get(FamilyMember, family_member_id):
        raise HTTPException(404, "Family member not found.")
    samples = db.scalars(select(VoiceSample).where(VoiceSample.family_member_id == family_member_id).order_by(VoiceSample.id.desc())).all()
    return [{"id": s.id, "filename": s.filename, "created_at": s.created_at.isoformat(), "audio_url": f"{API_BASE}/api/voice-samples/{s.id}/audio"} for s in samples]


@app.get("/api/voice-samples/{sample_id}/audio")
def get_voice_sample_audio(sample_id: int, db: Session = Depends(get_db)):
    sample = db.get(VoiceSample, sample_id)
    if not sample:
        raise HTTPException(404, "Voice sample not found.")
    path = Path(sample.stored_path)
    if not path.exists():
        raise HTTPException(404, "Stored audio file not found.")
    return FileResponse(path, filename=sample.filename)


@app.delete("/api/voice-samples/{sample_id}")
def delete_voice_sample(sample_id: int, db: Session = Depends(get_db)):
    sample = db.get(VoiceSample, sample_id)
    if not sample:
        raise HTTPException(404, "Voice sample not found.")
    _delete_sample_files([sample])
    db.query(Analysis).filter(Analysis.sample_id == sample_id).delete(synchronize_session=False)
    db.delete(sample)
    db.commit()
    return {"ok": True, "message": f"Deleted '{sample.filename}'."}


def _run_pipeline_on_path(audio_path: str, target_member_id: int | None, db: Session, mode: str = "combined"):
    # 1. Run AASIST deepfake detection
    aasist_result = aasist.analyze(audio_path)

    if mode == "aasist_only":
        spoof_score = aasist_result.spoof_score or 0.0
        prediction = aasist_result.prediction or "unknown"
        risk_score = round(spoof_score * 100.0, 2)
        risk_level = "HIGH" if risk_score >= 60 else ("MEDIUM" if risk_score >= 30 else "LOW")
        reasons = [
            f"AASIST single-model detection: {prediction} (deepfake score: {spoof_score:.4f})"
        ]
        evidence = [
            f"AASIST deepfake score: {spoof_score:.4f} ({prediction})",
            "Single-model acoustic countermeasure path only.",
        ]
        fused = FusionResult(
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            verdict=f"AASIST Baseline: {prediction.upper()} (Deepfake score: {spoof_score:.4f})",
            confidence=spoof_score,
            evidence=evidence,
            experimental=True,
            fusion_status="single_model_baseline",
        )
        return aasist_result, None, None, None, None, fused

    # 2. Run ECAPA speaker verification against registered reference
    references = []
    if target_member_id is not None:
        refs = db.scalars(select(VoiceSample).where(VoiceSample.family_member_id == target_member_id)).all()
        references = [r.stored_path for r in refs if Path(r.stored_path).exists()]
    ecapa_result = ecapa.compare(audio_path, references)

    # 3. Run Whisper transcription
    whisper_result = whisper.transcribe(audio_path)
    transcript = whisper_result.transcript or ""

    # 4. Run Scam Intent V1 & V2
    v1_result = scam_intent.analyze_v1(transcript)
    v2_result = scam_intent.analyze_v2(transcript)

    # 5. Run authoritative FusionEngine
    fused = fusion_adapter.run_fusion(aasist_result, ecapa_result, v1_result, v2_result)

    return aasist_result, ecapa_result, whisper_result, v1_result, v2_result, fused


@app.post("/api/analyze/{sample_id}")
def analyze_sample(
    sample_id: int,
    target_member_id: int | None = Form(None),
    mode: str = Form("combined"),
    db: Session = Depends(get_db)
):
    sample = db.get(VoiceSample, sample_id)
    if not sample:
        raise HTTPException(404, "Voice sample not found.")
    if target_member_id is not None and not db.get(FamilyMember, target_member_id):
        raise HTTPException(404, "Target family member not found.")

    aasist_res, ecapa_res, whisper_res, v1_res, v2_res, fused = _run_pipeline_on_path(
        sample.stored_path, target_member_id, db, mode=mode
    )

    # Save to SQLite Analysis table
    evidence_payload = {
        "reasons": fused.reasons,
        "evidence": fused.evidence,
        "risk_score": fused.risk_score,
        "risk_level": fused.risk_level,
        "raw_bona_fide_score": getattr(aasist_res, "raw_bona_fide_score", None),
        "prediction": getattr(aasist_res, "prediction", None),
        "transcript": getattr(whisper_res, "transcript", "") if whisper_res else "",
        "v1": result_dict(v1_res) if v1_res else None,
        "v2": result_dict(v2_res) if v2_res else None,
    }

    analysis = Analysis(
        sample_id=sample.id,
        target_member_id=target_member_id,
        mode=mode,
        aasist_score=aasist_res.authentic_score if aasist_res else None,
        ecapa_score=ecapa_res.similarity_score if ecapa_res else None,
        whisper_score=whisper_res.speech_score if whisper_res else None,
        verdict=fused.verdict,
        confidence=fused.confidence,
        evidence=json.dumps(evidence_payload),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return {
        "id": analysis.id,
        "sample_id": sample.id,
        "target_member_id": target_member_id,
        "mode": mode,
        "experimental": fused.experimental,
        "verdict": fused.verdict,
        "risk_score": fused.risk_score,
        "risk_level": fused.risk_level,
        "confidence": fused.confidence,
        "reasons": fused.reasons,
        "models": {
            "aasist": result_dict(aasist_res) if aasist_res else None,
            "ecapa": result_dict(ecapa_res) if ecapa_res else None,
            "whisper": result_dict(whisper_res) if whisper_res else None,
            "scam_intent_v1": result_dict(v1_res) if v1_res else None,
            "scam_intent_v2": result_dict(v2_res) if v2_res else None,
        },
        "evidence": fused.evidence,
        "disclaimer": "Provisional experimental assessment. Not an exact identity determination and not a calibrated probability.",
    }


@app.get("/api/analyses")
def list_analyses(db: Session = Depends(get_db)):
    rows = db.scalars(select(Analysis).order_by(Analysis.id.desc()).limit(50)).all()
    results = []
    for a in rows:
        evidence_data = None
        if a.evidence:
            try:
                evidence_data = json.loads(a.evidence)
            except Exception:
                evidence_data = {"raw": a.evidence}

        results.append({
            "id": a.id,
            "sample_id": a.sample_id,
            "target_member_id": a.target_member_id,
            "mode": a.mode,
            "aasist_score": a.aasist_score,
            "ecapa_score": a.ecapa_score,
            "whisper_score": a.whisper_score,
            "verdict": a.verdict,
            "confidence": a.confidence,
            "evidence": evidence_data,
            "created_at": a.created_at.isoformat(),
        })
    return results


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found.")
    evidence_data = None
    if analysis.evidence:
        try:
            evidence_data = json.loads(analysis.evidence)
        except Exception:
            evidence_data = {"raw": analysis.evidence}
    return {
        "id": analysis.id,
        "sample_id": analysis.sample_id,
        "target_member_id": analysis.target_member_id,
        "mode": analysis.mode,
        "aasist_score": analysis.aasist_score,
        "ecapa_score": analysis.ecapa_score,
        "whisper_score": analysis.whisper_score,
        "verdict": analysis.verdict,
        "confidence": analysis.confidence,
        "evidence": evidence_data,
        "created_at": analysis.created_at.isoformat(),
    }


@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found.")
    db.delete(analysis)
    db.commit()
    return {"ok": True}


@app.get("/api/comparisons")
def comparison_structure():
    return {
        "available": True,
        "message": "VoxFusion comparison framework active.",
        "methods": [
            {
                "id": "aasist_only",
                "name": "AASIST-only Baseline",
                "description": "Single-factor acoustic countermeasure evaluating synthetic artifacts only.",
                "strengths": "Fast raw audio spoof detection without needing reference voice or speech transcript.",
                "limitations": "Susceptible to false alarms on poor channels (GSM/telephony compression) and cannot detect genuine voice scams or speaker identity mismatch.",
            },
            {
                "id": "voxfusion_combined",
                "name": "Full VoxFusion Pipeline",
                "description": "Multi-modal fusion integrating AASIST (acoustic deepfake), ECAPA-TDNN (biometric speaker verification), Whisper (transcription), and Scam Intent V1 & V2 (semantic risk analysis).",
                "strengths": "Distinguishes authorized speakers, detects synthetic spoofing, and catches conversational scam triggers with explanatory reasoning.",
                "limitations": "Requires reference sample for speaker verification and speech content for intent modeling.",
            },
        ],
        "metrics": ["deepfake_score", "speaker_similarity", "scam_intent_score", "fused_risk_score", "reasoning"],
    }


@app.post("/api/compare/{sample_id}")
def compare_sample(
    sample_id: int,
    target_member_id: int | None = Form(None),
    db: Session = Depends(get_db)
):
    sample = db.get(VoiceSample, sample_id)
    if not sample:
        raise HTTPException(404, "Voice sample not found.")
    if target_member_id is not None and not db.get(FamilyMember, target_member_id):
        raise HTTPException(404, "Target family member not found.")

    # 1. Run AASIST baseline
    a_aasist, _, _, _, _, a_fused = _run_pipeline_on_path(
        sample.stored_path, target_member_id, db, mode="aasist_only"
    )

    # 2. Run Full VoxFusion Pipeline
    v_aasist, v_ecapa, v_whisper, v_v1, v_v2, v_fused = _run_pipeline_on_path(
        sample.stored_path, target_member_id, db, mode="combined"
    )

    return {
        "sample_id": sample.id,
        "filename": sample.filename,
        "target_member_id": target_member_id,
        "baseline_aasist": {
            "mode": "aasist_only",
            "verdict": a_fused.verdict,
            "risk_score": a_fused.risk_score,
            "risk_level": a_fused.risk_level,
            "deepfake_score": a_aasist.spoof_score,
            "bona_fide_score": a_aasist.authentic_score,
            "prediction": a_aasist.prediction,
            "reasons": a_fused.reasons,
        },
        "voxfusion_combined": {
            "mode": "combined",
            "verdict": v_fused.verdict,
            "risk_score": v_fused.risk_score,
            "risk_level": v_fused.risk_level,
            "confidence": v_fused.confidence,
            "reasons": v_fused.reasons,
            "evidence": v_fused.evidence,
            "models": {
                "aasist": result_dict(v_aasist),
                "ecapa": result_dict(v_ecapa),
                "whisper": result_dict(v_whisper),
                "scam_intent_v1": result_dict(v_v1),
                "scam_intent_v2": result_dict(v_v2),
            },
        },
        "comparison_insights": [
            f"AASIST classified audio as '{a_aasist.prediction}' with deepfake score {a_aasist.spoof_score:.4f}.",
            f"ECAPA speaker verification status: '{v_ecapa.status}' (similarity: {f'{v_ecapa.similarity_score:.4f}' if v_ecapa.similarity_score is not None else 'N/A'}).",
            f"Semantic intent analysis detected risk '{v_v1.risk_level}' with categories: {', '.join(v_v1.matched_categories) or 'None'}.",
            f"Multi-modal fusion yielded overall Risk: {v_fused.risk_level} (Score: {v_fused.risk_score}/100).",
        ],
    }


@app.delete("/api/database")
def clear_database(db: Session = Depends(get_db)):
    _clear_all_storage()
    db.query(Analysis).delete(synchronize_session=False)
    db.query(VoiceSample).delete(synchronize_session=False)
    db.query(FamilyMember).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "message": "Database and stored audio cleared."}


@app.get("/api/database/export")
def export_database(db: Session = Depends(get_db)):
    db.commit()
    zip_path = DATA_DIR / f"voice_analysis_backup_{uuid.uuid4().hex}.zip"
    db_file = DATA_DIR / "voice_analysis.db"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(db_file, arcname="voice_analysis.db")
        if UPLOAD_DIR.exists():
            for p in UPLOAD_DIR.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=f"uploads/{p.name}")
    return FileResponse(zip_path, filename="voice_analysis_backup.zip", media_type="application/zip", background=None)


@app.post("/api/database/import")
async def import_database(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Import file must be a .zip backup created by this app.")
    content = await file.read()
    with TemporaryDirectory() as tmp:
        archive = Path(tmp) / "backup.zip"
        archive.write_bytes(content)
        try:
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                if "voice_analysis.db" not in names or any(n.startswith("../") or n.startswith("/") for n in names):
                    raise HTTPException(400, "Invalid backup archive.")
                zf.extractall(tmp)
        except zipfile.BadZipFile:
            raise HTTPException(400, "Invalid backup archive.")
        incoming_db = Path(tmp) / "voice_analysis.db"
        db.close()
        engine.dispose()
        target_db = DATA_DIR / "voice_analysis.db"
        shutil.copy2(incoming_db, target_db)
        _clear_all_storage()
        for item in Path(tmp).glob("uploads/*"):
            if item.is_file():
                shutil.copy2(item, UPLOAD_DIR / item.name)
    return {"ok": True, "message": "Database backup imported. Refresh the app."}
