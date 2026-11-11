# GenAI Designer

Code for **GenAI Designer: Reusable Designer Style Profiles for Personalized Fashion Generation**.

GenAI Designer extracts nine garment attributes from historical clothing images, aggregates shared traits and individual variants into an archive, and retrieves these records for later design briefs. The archive is an inference-time representation; this release does not train model weights.

## Included

- `core/`: nine-dimensional schema, observation/aggregation prompts, API client and archive analysis.
- `config/`, `train/`: configurable Training Assistant and Gradio interface.
- `evaluate/`: API-based generation/evaluation workflow, archive reuse experiments, budget controls and offline tests. These are research scripts, not a one-command hosted image editor.
- `reproduction/compact_study/`: task specifications, portable protocol, **128 final corrected scores**, and dimension-level judgments for the main experiment.
- `tools/recompute_main_results.py`: offline reproduction of the manuscript's all-valid and complete-matched-subset tables.
- Docker files for the Training Assistant; see `DEPLOYMENT.md`.

The legacy local SDXL editor, vendored model repositories, checkpoints, private credentials, raw API logs, author photographs, and third-party garment images are not distributed here. The main experiment uses API generation; Docker packages the Training Assistant only.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/todreamboat/design-style.git
cd design-style
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the main numerical table offline

No images, API credentials, paid requests or GPU are needed for this step:

```bash
python tools/recompute_main_results.py
python -m unittest discover -s evaluate -p "test_*.py"
```

Scores are first averaged within task and source, then averaged equally across sources. A complete matched block has the same task and repeat ID available in all three conditions. The included corrected style scores use four historical reference images followed by the generated image as image 5. Early, incorrectly indexed style judgments are **not** included as final data.

| Scope | Condition | Images | Task compliance (%) | Style retention (%) |
|---|---|---:|---:|---:|
| All valid | Task only | 41 | 100.00 | 40.63 |
| All valid | Free summary | 45 | 100.00 | 75.50 |
| All valid | Nine-dimensional archive | 42 | 100.00 | 83.49 |
| Complete matched | Task only | 24 | 100.00 | 43.68 |
| Complete matched | Free summary | 24 | 100.00 | 73.32 |
| Complete matched | Nine-dimensional archive | 24 | 100.00 | 83.88 |

`style_score` is the mean of eligible attribute judgments (retained=1, partial=0.5, not_retained/unknown=0). `task_score` is the core-task constraint adherence. These are automated evaluation outcomes, not human satisfaction scores.

## Training Assistant

Copy `.env.example` to `.env` and set your own API key locally. Never commit credentials.

```bash
python train/train.py
# http://localhost:12002
```

For containers: `docker compose up --build`. This calls external models and may incur provider charges. No API experiment is run by installation or by the offline analysis above.

## Generating new experimental images

The historical model identifiers and endpoints are recorded in `evaluate/workflow_config.json` and the frozen protocol. Availability is provider-dependent. The scripts require `OPENROUTER_API_KEY` in the environment and explicit `--execute-api` before live phases.

```bash
python evaluate/run_compact_saved_style.py --help
python evaluate/rescore_compact_style.py --help
```

For an exact image-level rerun, obtain lawful access to the source garments, place them in a sibling `data/` directory using the relative paths in the protocol, and supply the source images and generated outputs referenced by each phase. The frozen script also checks source-code/image hashes and cached artifacts; simply copying the portable protocol to a new directory is not sufficient to resume the original paid run. For a new study, prepare and freeze a new local protocol rather than silently changing the original experiment. The public snapshot supports offline table reproduction; it does not bundle every original run artifact.

## Data and licensing

The processed scores, tasks and judgments distributed here support the main numerical analysis. Garment images come from third-party brand collections and are excluded: permission to use an image in research is not assumed to grant unrestricted redistribution. API keys, raw provider responses and personal author materials are also excluded. Do not interpret this repository as public access to the full original image dataset.

The existing project's **GNU GPL version 3** license is preserved in `LICENSE`; third-party dependencies and model services retain their own terms. No model weights are redistributed. This is a research code release and requires author verification before downstream deployment.
