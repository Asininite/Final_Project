# Level 0 Data Flow Diagram (DFD) — Adversarial Deepfake Detector

This document contains a Level‑0 (context) Data Flow Diagram for the Adversarial Deepfake Detector prototype and a short textual explanation. The diagram is drawn using a Mermaid flowchart so you can preview it in editors that render Mermaid.

## Mermaid diagram (finalized)

```mermaid
%%{init: {"themeVariables": {
  "primaryTextColor":"#000000",
  "lineColor":"#000000",
  "fontFamily":"Arial"
}}}%%
flowchart TB
  %% External entities (finalized web demo context)
  V[Visitor / Demo User]
  U[Uploader / End User]
  VP[Victim Pretrained Model]
  DP[Data Provider - FF++, Celeb-DF, etc.]

  %% Single-system bubble (strict Level-0 / Context)
  SYS[("Adversarial Deepfake Detector - Web Demo & Upload Checker")]

  # Level 0 Data Flow Diagram (DFD) — Adversarial Deepfake Detector (minimal)

  This is a Level‑0 (context) DFD in the classic single‑bubble style. It intentionally shows only the primary external actors and the single system to keep the high‑level boundary clear.

  ```mermaid
  flowchart LR
    EndUser[End User]
    SYS((Adversarial Deepfake Detector))
    Admin[Admin / Researcher]

    EndUser -->|views demo / uploads image| SYS
    SYS -->|returns verdict & diagnostics| EndUser

    Admin -->|provides datasets / manages models| SYS
    SYS -->|writes logs & metrics| Admin

    style SYS fill:#0a6,stroke:#033,stroke-width:2px,color:#fff
    style EndUser fill:#2b8,stroke:#033,stroke-width:1px
    style Admin fill:#2b8,stroke:#033,stroke-width:1px
  ```

  Keep implementation details (victim model, model artifacts, preprocessing, attack modules) for Level‑1 diagrams. This Level‑0 is intentionally simple — similar to the example 0‑level image you provided — and suitable for a report cover or executive summary.