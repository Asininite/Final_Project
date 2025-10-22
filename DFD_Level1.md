# Level 1 Data Flow Diagram (DFD) — Upload Checker & Demo (Adversarial Deepfake Detector)

This Level‑1 diagram decomposes the Level‑0 system (`Adversarial Deepfake Detector`) into its main runtime components and shows high‑level data flows between them. The Level‑1 view is intended for developers and testers: it shows the web front‑end and upload flow, the preprocessing and inference pipeline, the model / checkpoint manager, the adversarial demo component (victim model), and logging/monitoring.

## Mermaid diagram (Level‑1)

```mermaid
flowchart LR
  %% Left: external actor
  EndUser[End User]

  %% Center: main processes (arranged like the example layout)
  Upload[Upload & Check Process]
  Preproc[Preprocessor]
  Inference[Inference Engine]
  Demo[Adversarial Demo Process]

  %% Right: external services and data stores
  Victim[Victim Pretrained Model]
  Registry[(Model Registry / Artifacts)]
  DataStore[(Raw Media / Datasets)]
  Logs[(Logs & Metrics)]

  %% Layout hints
  classDef proc fill:#cfe,stroke:#333,stroke-width:1px;
  class Upload,Preproc,Inference,Demo proc;

  %% Flows (mimic radial layout of example)
  EndUser -->|uploads image| Upload
  EndUser -->|requests demo| Demo

  Upload -->|image bytes| Preproc
  Preproc -->|tensor| Inference
  Inference -->|verdict,score| Upload
  Upload -->|verdict json| EndUser

  Demo -->|demo image request| Preproc
  Demo -->|query| Victim
  Victim -->|label| Demo
  Demo -->|demo results| EndUser

  %% Management & storage flows to the right (like outputs in the example)
  Inference -->|log inference| Logs
  Upload -->|write event| Logs
  Preproc -->|reads/writes| DataStore
  Upload -->|model version/read| Registry
  Upload -->|save model artifact (admin)| Registry

  style EndUser fill:#eef,stroke:#333
  style Upload fill:#bbf,stroke:#333
  style Preproc fill:#ffd,stroke:#333
  style Inference fill:#f8b,stroke:#333
  style Demo fill:#fdd,stroke:#333
  style Victim fill:#fef,stroke:#333
  style Registry fill:#bfb,stroke:#333
  style DataStore fill:#bfb,stroke:#333
  style Logs fill:#eee,stroke:#333
```

## Component descriptions and data shapes
- Browser / User UI (Frontend)
  - Inputs: image files (JPEG/PNG), user actions (upload, request demo)
  - Outputs: JSON verdicts, demo pages, visualizations

- Web API / Controller (`server.py`)
  - Responsibilities: receive uploads, validate inputs, orchestrate preprocessing and inference, return JSON results, serve static demo UI.
  - Data shapes: multipart/form-data for uploads; JSON responses: {verdict: string, score: float, model_version: str, details: { ... }}

- Preprocessor (`tools/face_preprocess.py` or server preprocess)
  - Responsibilities: decode image, crop/align faces (optional), resize to model input shape, normalize.
  - Output: torch.Tensor (C, H, W) or batched tensor (1, C, H, W).

- Inference Engine (Detector) (`server.py` loads `artifacts/*.pt`)
  - Responsibilities: run model forward, compute detection score/probability, apply decision threshold, return verdict.
  - Input: tensor (1, 3, 224, 224). Output: {prob_attacked: float, verdict: 'attacked'|'not_attacked', pred_class: int}

- Checkpoint Manager / Model Registry
  - Responsibilities: load specified model version, export new TorchScript artifacts, handle retention (keep best N), store metadata/manifest.
  - Storage: local `artifacts/` or remote object store. Manifest JSON: {model_version, trained_on, metrics, git_commit, preprocessing}.

- Adversarial Demo Module (`static/` + server integration)
  - Responsibilities: provide demo images (clean and attacked), query victim model service for labels to illustrate misclassification, present side‑by‑side UI.
  - Note: victim model can be local or remote; treat as a separate service for Level‑1.

- Victim Pretrained Model Service
  - Responsibilities: return classification label(s) for demo images. Could be a third-party model endpoint or a local model loaded separately to show misbehavior.

- Logs & Metrics
  - Capture: inference requests, verdicts, latencies, user uploads (paths only), model versions. Avoid storing raw PII in logs.

## Edge cases and non-functional requirements
- Input validation: reject non-image uploads, limit file size (e.g., 5MB), handle corrupted images gracefully.
- Concurrency: inference must be threadsafe; use a queue or worker pool for high throughput.
- Performance: single inference should aim for <200ms on GPU, <1s on CPU for demo scale.
- Security: sanitize filenames, rate-limit uploads, store uploads temporarily with TTL or process in-memory.
- Privacy: do not persist user uploads unless user consents; store paths or hashed references if needed.

## Mapping to repository files
- `server.py` — Web API, orchestrator, loads `artifacts/detector.pt`.
- `static/index.html` — front-end demo and upload form.
- `tools/face_preprocess.py` — dataset preprocessing for training; server includes a lightweight runtime preprocess step.
- `tools/export_inference.py` — helper to export TorchScript artifacts.
- `models.py` — model definitions (SmallCNN and potential backbones).
- `attacks.py` — adversarial attack implementations for dataset generation and demo (FGSM/PGD; C&W documented).

## Acceptance criteria (for Level‑1 implementation)
- Upload endpoint responds with a JSON verdict within timeout and uses model listed in artifact manifest.
- Demo page shows side‑by‑side clean vs attacked images and victim model label for the attacked image.
- Checkpoint manager can load/export a model artifact and write a manifest.
- System logs inference events with model_version and latency.

## Next steps I can implement
1. Add `DFD_Level1.svg` (rendered image) and include it in repo.
2. Implement checkpoint manager functions in `utils.py` and wire simple versioning into `server.py`.
3. Replace the server's heuristic with a binary detector model by training a small experiment and exporting the artifact.

Which of the next steps do you want me to take now? I can start with (1) render and save the Level‑1 diagram, or (2) implement checkpoint-management wiring in code and a smoke export flow.
