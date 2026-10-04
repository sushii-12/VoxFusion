Update ONLY the root README.md for the final current state of VoxFusion.

The current README is outdated because it does not document the implemented React/Vite + FastAPI web application.

Rewrite the README while preserving the existing research question, technical terminology, evaluation methodology, limitations, and honest/non-overclaiming framing.

The README must accurately document:

1. Project
- VoxFusion
- Team AltF4
- Voice-cloning family-emergency scam detection
- Research question exactly as currently defined

2. Architecture
- ECAPA-TDNN personalized speaker verification
- AASIST deepfake detection
- faster-whisper transcription
- Scam Intent V1 rule-based analyzer
- Scam Intent V2 TF-IDF + Logistic Regression
- interpretable fusion layer
- GSM/telephone codec simulation
- missing ECAPA reference does NOT prevent other branches from running

3. Web Application
Explain that the project now includes:
- React + Vite frontend
- FastAPI backend
- local SQLite storage for the demo
- local audio upload storage
- family-member/reference voice management
- analysis workflow
- AASIST-only vs combined comparison
- component-level results
- fusion risk score and reasons
- health/readiness endpoint
- settings/backup functionality if actually present

IMPORTANT:
The web application is located at:
frontend/
├── backend/
└── frontend/

Do not rename the directories.

4. Project Structure
Update the tree to accurately include:
- src/
- frontend/backend/
- frontend/frontend/
- scripts/
- tests/fixtures/
- aasist/
- data/ (ignored)
- models/ (ignored)
- results/ (ignored)

5. Setup
Document:
- cloning with AASIST submodule
- Python 3.11 environment
- root requirements installation
- AASIST checkpoint requirement
- FFmpeg requirement
- frontend npm install
- backend startup command:
  python -m uvicorn frontend.backend.app.main:app --host 127.0.0.1 --port 8001
- frontend startup:
  cd frontend/frontend
  npm run dev

Clearly state that the backend runs on port 8001 and the frontend on Vite's port 5173.

6. Local Runtime Data
Clearly explain:
- SQLite database lives under frontend/backend/data/
- uploaded audio is stored under frontend/backend/data/uploads/
- these are local demo/runtime data
- they are intentionally ignored by Git
- datasets, model weights and generated results are also ignored

7. Evaluation
Accurately summarize the completed AASIST and ECAPA evaluation results already established in the project.

For AASIST:
- ASVspoof 2019 LA local subset: 200 samples, Accuracy 0.98, Precision 1.00, Recall 0.96, F1 0.9796, FNR 0.04
- ASVspoof 2021 LA local subset: 200 samples, Accuracy 0.845, Precision 0.7717, Recall 0.98, F1 0.8634, FNR 0.02

Clearly call these local prepared subsets, NOT complete datasets.

For ECAPA:
- speaker-disjoint LibriSpeech calibration/held-out evaluation
- calibration EER 1.27%, threshold 0.2951
- held-out FAR 4.4%
- held-out FRR 0.8%
- held-out Accuracy 97.4%
- held-out Precision 98.90%
- held-out Recall 99.2%
- held-out F1 99.05%

Clearly state this is clean read-speech evaluation and does NOT establish telephony-domain performance.

8. Case-study findings
Explain that observed case studies show the multimodal system can provide contextual evidence beyond AASIST alone, including reducing some observed false-alarm severity when speaker verification/intent evidence is benign and retaining HIGH risk when multiple signals agree.

DO NOT claim statistically proven end-to-end superiority.
DO NOT claim VoxFusion has higher accuracy than AASIST.
DO NOT call fusion scores calibrated probabilities.
DO NOT claim production readiness.
DO NOT claim live-call interception.

9. Limitations
Keep the honest limitations:
- AASIST false positives under some codec/phone-quality examples
- provisional fusion weights/thresholds
- ECAPA scores are not probabilities
- voice mismatch alone is not proof of fraud
- intent model limitations and dataset imbalance
- no combined ground-truth dataset for end-to-end VoxFusion accuracy
- audio-file analysis prototype, not live-call interception

10. Team / development notes
Keep the project framed as an academic capstone.

Do not modify any other files.
Do not git add.
Do not commit.
Do not push.
After editing, show:
git diff -- README.md