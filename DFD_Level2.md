# Level 2 Data Flow Diagram (DFD) — Upload & Check Process (Detailed)

This Level‑2 DFD decomposes the `Upload & Check Process` from Level‑1 into detailed subprocesses and data stores. The layout mirrors the example you provided: a central set of processing bubbles (ingest, validate, preproc, inference, verdict), with auxiliary subprocesses (query, report, storage) arranged around it.

```mermaid
flowchart LR
  %% Left: external actor
  User[End User]

  %% Central process group
  subgraph Central
    Ingest[(1) Receive Upload]
    Validate[(2) Validate / Sanitize]
    FaceDetect[(3) Face Detect & Crop]
    FeatureExtract[(4) Feature Extraction]
    Detect[(5) Detector Inference]
    Verdict[(6) Postprocess & Verdict]
  end

  %% Demo and attack playground
  Demo[(D) Demo / Attack Playground]
  Attacks[(Adversarial Attacks)]

  %% Right/top: management & reporting
  Query[(7) Query / Admin Ops]
  Report[(8) Report Generation]
  Storage[(Raw Media / Processed Frames)]
  Artifacts[(Model Artifacts)]
  Logs[(Logs & Metrics)]

  %% flows: user upload path
  User -->|multipart/form-data| Ingest
  Ingest -->|bytes| Validate
  Validate -->|clean image| FaceDetect
  FaceDetect -->|face crop(s)| FeatureExtract
  FeatureExtract -->|tensor / features| Detect
  Detect -->|score, logits| Verdict
  Verdict -->|json verdict| User

  %% demo flows
  User -->|view demo| Demo
  Demo -->|demo images| User
  Artifacts -->|sample models| Demo
  Attacks -->|generate attacked image| Demo
  Storage -->|serve demo images| Demo

  %% storage and reporting flows
  Validate -->|store original (opt-in)| Storage
  FaceDetect -->|store crops (optional)| Storage
  Detect -->|log inference| Logs
  Verdict -->|write event| Logs
  Query -->|admin request| Report
  Report -->|report| Query
  Query -->|manage models| Artifacts
  Artifacts -->|model version| Detect

  style Ingest fill:#cde,stroke:#333
  style Validate fill:#def,stroke:#333
  style FaceDetect fill:#fde,stroke:#333
  style FeatureExtract fill:#edf,stroke:#333
  style Detect fill:#fbd,stroke:#333
  style Verdict fill:#cfe,stroke:#333
  style Storage fill:#bfb,stroke:#333
  style Artifacts fill:#bfb,stroke:#333
  style Logs fill:#eee,stroke:#333
  style Query fill:#fef,stroke:#333
  style Report fill:#ffd,stroke:#333
  style Demo fill:#eef,stroke:#333
  style Attacks fill:#fdd,stroke:#333
```

## Process descriptions (detailed)
1) Receive Upload (Ingest)
 - Purpose: accept multipart upload, return immediate acknowledgement. Validate authentication (if required) and enforce file-size limits.
 - Inputs: multipart/form-data (field `file`), headers (user id/session). Output: raw image bytes (stream or temp file reference).

2) Validate / Sanitize
 - Purpose: confirm image format (JPEG/PNG), dimensions, size; attempt simple malware checks (reject files with unusual headers); verify image decodes.
 - Output: canonical RGB image bytes or error.

3) Face Detect & Crop
 - Purpose: run face detection (MTCNN or fallback) and produce one or more aligned crops. If no face found, fallback to center-crop.
 - Output: one or more cropped images resized to model input.

4) Feature Extraction
 - Purpose: convert image crop to tensor, apply normalization and any feature preprocessing (e.g., embeddings). Can optionally cache embeddings for identical images.
 - Output: tensor (1,3,224,224) or features vector.

5) Detector Inference
 - Purpose: run the detector model (TorchScript/ONNX) and compute p(attacked). Use the manifest to ensure preprocessing parity.
 - Output: raw scores {logits, prob_attacked, pred}.

6) Postprocess & Verdict
 - Purpose: apply calibrated threshold to p(attacked), produce final verdict and human-friendly diagnostics (confidence, model_version, note).
 - Output: JSON verdict returned to user; event logged.

7) Query / Admin Ops
 - Purpose: support admin queries (generate reports, list artifacts, run audit) and manage models (register new artifact, rollback).

8) Report Generation
 - Purpose: periodically build aggregate metrics (daily inference counts, FPR, FNR by attack type) for QA and research.

## Data contracts / example payloads
- Upload request (multipart): field `file` (image), optional `user_id` or session token.
- Inference request / internal tensor: shape [1,3,224,224], dtype=float32.
- Verdict response JSON:
```json
{
  "verdict": "attacked",
  "prob_attacked": 0.87,
  "model_version": "v1",
  "preprocessing": {"resize": [224,224]},
  "notes": "low confidence"
}
```

## File mapping (where these processes live in the repo)
- `server.py` — orchestrates `Ingest`, `Validate`, lightweight `FaceDetect` (runtime path), calls `Detect`.
- `tools/face_preprocess.py` — dataset preprocessing and the full face detection pipeline used offline.
- `tools/export_inference.py` — export the model artifact to `artifacts/`.
- `models.py` — model definitions and any embedding/feature extractors.
- `attacks.py` — for synthetic adversarial example generation used in training and sample creation.

## Acceptance / test points
- Unit tests: each subprocess should have unit tests for normal and edge cases (corrupt images, no face found, large files).
- End-to-end smoke: upload test image and verify a valid JSON verdict and logs entry.
- Admin flows: registering an artifact should update `artifacts/manifest.json` and be reflected in `Detect` on reload.

## Security & privacy notes
- Avoid storing raw uploads unless user explicitly consents. If storing, encrypt at rest and retain minimal metadata.
- Enforce file size limits and rate limiting to prevent DoS attacks.
- Sanitize filenames and use random storage keys for temp files.

---

If you'd like, I can now:
1. Render `DFD_Level2.md` to `DFD_Level2.svg` and add it to the repo, or
2. Implement the `Checkpoint Manager` and the manifest verification in `server.py` (so the server refuses to serve if no valid artifact exists).

Which one should I do next?
