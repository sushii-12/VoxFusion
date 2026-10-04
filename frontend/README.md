# VoxFusion Demo

Application foundation for future AASIST, ECAPA-TDNN and Whisper integration.

## Features

- Local SQLite database for family‑member profiles and uploaded audio (runtime data, not committed to Git)
- Manage family members (create, edit, delete)
- Upload, play back, and delete voice reference recordings
- AASIST deepfake detection
- ECAPA‑TDNN speaker verification against registered family‑member references
- Whisper transcription of uploaded audio
- Scam‑intent analysis (V1 and V2)
- VoxFusion fusion layer producing a risk result; includes AASIST‑only vs combined comparison
- Analysis history with results view

## Running the Demo

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Backend docs: http://127.0.0.1:8001/docs
Frontend UI: http://localhost:5173/

## Important

The real AASIST, ECAPA-TDNN and Whisper inference implementations are not included yet. Their adapters remain integration points for the user's working model code. Fusion output is experimental evidence, not a calibrated probability or exact identity determination.
