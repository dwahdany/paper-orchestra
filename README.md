# <div align="center">🎻 PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing</div>
<div align="center">Yiwen Song<sup>1</sup>, Yale Song<sup>1</sup>, Tomas Pfister<sup>1</sup>, and Jinsung Yoon<sup>1</sup></div>
<div align="center"><sup>1</sup>Google Cloud AI Research</div>
<br><br>

PaperOrchestra is a multi-agent framework for automated AI research paper writing. It turns unconstrained pre-writing materials (an idea and an experimental log) into a submission-ready LaTeX manuscript, including literature synthesis and generated figures.

A team of specialized agents handles outline generation, literature review, plotting, section writing and content refinement.

<div align="center">
  <img src="assets/overview.png" alt="PaperOrchestra Overview" width="90%"/>
</div>

## Key features

- **Multi-agent pipeline** with per-stage retry and resume: outline, literature review, plotting, section writing, content refinement.
- **Verified literature**: candidates found by web search are checked against Semantic Scholar before they can be cited.
- **Claude Fable 5.2 by default**, authenticated with an API key or keyless Workload Identity Federation. Gemini and OpenAI models remain available as optional providers.
- **Compute backends**: run model-written figure code and LaTeX locally, or in network-isolated [Modal](https://modal.com) sandboxes. The whole pipeline can also run on Modal.
- **A wiki as agent memory**: every stage works through a context harness that records transcripts, keeps notes, consolidates them into linked wiki pages, and loads a small index into each session. Lessons about the process carry over to the next paper.
- **Self-repairing LaTeX**: drafts are compiled and checked (citation keys, figure files, environments), and the errors are fed back to the writer.

## Installation

PaperOrchestra is managed with [uv](https://docs.astral.sh/uv/) and needs Python 3.11 or newer.

```bash
uv sync                      # core: Claude models, local compute
uv sync --extra modal        # + Modal compute backend
uv sync --extra ui           # + Streamlit demo
uv sync --all-extras         # everything, including Gemini and OpenAI providers
```

Local LaTeX compilation needs TeX Live (`pdflatex` and `bibtex` on `PATH`). With `--compute modal` no local TeX install is required.

Copy `.env.example` to `.env` and fill in what you use, then check the setup:

```bash
cp .env.example .env
uv run paper-orchestra doctor
```

`doctor` reports which credential the model provider will use (never the secret itself), whether TeX and Modal are available, and where memory is stored.

## Quick start

```bash
uv run paper-orchestra write \
  --raw-materials-dir path/to/raw_materials \
  --template cvpr2025 \
  --output-dir paper_output \
  --plotting
```

`raw_materials/` contains:

| File | Purpose |
|---|---|
| `idea_sparse.md` | The method, contributions and framing (`--idea-filename` to change). |
| `experimental_log.md` | Raw results: tables, ablations, metrics (`--experimental-log-filename`). |
| `figures/` + `figures/info.json` | Optional ready-made figures, as `[{"name": "fig.png", "caption": "..."}]`. Needed when running without `--plotting`. |

`--template` takes the name of a bundled template (`cvpr2025`, `iclr2025`), or a path to your own template directory containing `template.tex` and `guidelines.md`. A name is also looked up under `$PAPER_ORCHESTRA_TEMPLATES` and `./templates`.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--plotting / --no-plotting` | off | Generate figures from the outline's plotting plan. |
| `--model` | `claude-fable-5-2` | Model for every role without a flag of its own. |
| `--writer-model`, `--reflection-model`, `--plotting-model` | `--model` | Model per role. |
| `--fast-model` | `--model` | Model for high-volume calls such as literature discovery (run at low effort). |
| `--image-model` | unset | Image-generation model for diagrams (for example a Gemini image model). When unset, diagrams are drawn from model-written plotting code. |
| `--effort` | `high` | Reasoning effort for the main calls. |
| `--research-cutoff` | current month | `YYYY-MM`; papers published later are not cited as prior work. |
| `--compute` | `local` | `local` or `modal`. |
| `--memory / --no-memory` | on | The wiki memory harness. |
| `--lab-memory / --no-lab-memory` | on | Share process lessons across papers. Turn off for benchmark runs. |
| `--dream` | `stages` | When memory is consolidated by a model: `off`, `end`, or `stages`. |
| `--resume / --no-resume` | on | Skip stages already completed in the output directory. |

The flag names of the original CLI (`--raw_materials_dir`, `--latex_template_dir`, `--use_plotting true`, ...) are still accepted.

## Models and authentication

All agents default to `claude-fable-5-2`. Two ways to authenticate:

**API key.** Set `ANTHROPIC_API_KEY`.

**Workload Identity Federation (keyless).** Set the federation variables and provide an OIDC identity token from your platform:

```bash
ANTHROPIC_FEDERATION_RULE_ID=fdrl_...
ANTHROPIC_ORGANIZATION_ID=...
ANTHROPIC_SERVICE_ACCOUNT_ID=svac_...
ANTHROPIC_WORKSPACE_ID=wrkspc_...          # only if the rule spans several workspaces
ANTHROPIC_IDENTITY_TOKEN_FILE=/path/to/oidc-token
```

The token is exchanged for a short-lived access token and refreshed automatically. An `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in the environment takes precedence over federation, so unset them when you want federation; `doctor` warns when this happens. Identity tokens are usually single-use, so each process needs its own: for local batch runs with several worker processes use an API key, or fan out on Modal, where every container gets its own token.

Claude Fable 5.2 requires an organization with data retention enabled. Requests the model declines are retried on a fallback model by the API (server-side fallbacks are enabled).

To use other providers, install the extra and pass the model name:

```bash
uv sync --extra gemini   # GEMINI_API_KEY, or VERTEX_AI_PROJECT + VERTEX_AI_LOCATION
uv sync --extra openai   # OPENAI_API_KEY
uv run paper-orchestra write ... --writer-model gemini-3.1-pro-preview
```

Providers without tool support use the memory harness in "push" mode: relevant wiki pages are placed in the prompt instead of being opened on demand.

`SEMANTIC_SCHOLAR_API_KEY` is optional but raises the rate limit of literature verification considerably.

## Compute backends

Two kinds of work leave the orchestrating process: running model-written Python (figure code) and compiling model-written LaTeX.

| Backend | Figure code | LaTeX | Isolation |
|---|---|---|---|
| `local` | subprocess with a scrubbed environment, resource limits and a timeout | local `pdflatex`/`bibtex`, shell escape disabled | Keeps credentials out of the child's environment, but it is not a sandbox. |
| `modal` | Modal Sandbox, network blocked, no secrets | Modal Sandbox with TeX Live, network blocked | Nothing model-written runs on your machine. |

```bash
uv sync --extra modal
uv run modal setup
uv run paper-orchestra write ... --compute modal
```

### Running the whole pipeline on Modal

```bash
uv run paper-orchestra modal run \
  --raw-materials-dir path/to/raw_materials \
  --template cvpr2025 \
  --output-dir paper_output \
  --plotting
```

This is `modal run -m paper_orchestra.modal_app` underneath and must be run from a checkout of this repository (the image is built from `uv.lock`). It uploads the raw materials, runs the pipeline in a container with TeX Live, and downloads `final_paper.pdf`, `final_paper.tex` and `usage.json`. `paper-orchestra modal deploy` deploys the app, and `paper-orchestra modal batch` writes many papers with it, one container per paper.

Outputs and the lab memory live on a Modal Volume (`paper-orchestra` by default), so a preempted or interrupted run resumes where it stopped. Model-written figure code runs in a nested sandbox with no network, no secret and no identity token.

Credentials come from a Modal Secret (named `paper-orchestra` by default, `PAPER_ORCHESTRA_MODAL_SECRET` to change). Two setups are supported:

- **API key**: `modal secret create paper-orchestra ANTHROPIC_API_KEY=sk-ant-...` (add `SEMANTIC_SCHOLAR_API_KEY` if you have one).
- **Keyless**: Modal gives every function an OIDC identity token (issuer `https://oidc.modal.com`, audience `oidc.modal.com`), which is exchanged for short-lived Anthropic access tokens, so no Anthropic secret is stored anywhere. The container only needs the non-secret identifiers `ANTHROPIC_FEDERATION_RULE_ID` and `ANTHROPIC_ORGANIZATION_ID` (plus `ANTHROPIC_SERVICE_ACCOUNT_ID` / `ANTHROPIC_WORKSPACE_ID` if your rule needs them). Put them in the Secret, or export them in the shell that launches the run and set `PAPER_ORCHESTRA_MODAL_SECRET=none`. Do not also provide an API key: it would take precedence, and the run refuses to start when both are present.

## Memory

Agents do not get everything stuffed into their prompt, and they do not start from zero each time. Every model call goes through a context harness whose memory is a wiki on disk:

| Layer | Where | Written by | Used for |
|---|---|---|---|
| Transcripts | `memory/transcripts/*.jsonl` | the harness, automatically | The record of what happened; notes are checked against it. |
| Notes | `memory/notes/*.md` | agents and pipeline code, during a stage | Salient observations: a failure and its cause, a decision and its reason. |
| Wiki pages | `memory/wiki/<section>/<page>.md` | pipeline code (records) and consolidation sessions (lessons, synthesis) | One page per thing, linked with `[[section/page]]`. |
| Index | `memory/MEMORY.md` | regenerated automatically; a few curated standing notes | Loaded into every session (capped at 200 lines / 25 KB). |

Between stages a separate consolidation session ("dreaming") verifies notes against the transcripts and sources, folds what holds up into pages, repairs links and keeps the index within budget. The refinement loop uses this as a learning loop: each accepted or rejected revision is recorded with its score changes and turned into a rule the next iteration can see. A rejected revision is discarded and the remaining iterations start again from the last accepted version, now with that lesson (`retry_after_rejection`; the original behaviour of stopping at the first rejection is available by turning it off in `PipelineConfig`).

There are two memories:

- **Project memory** (`<output-dir>/memory`): everything about this paper. Sources are stored verbatim and are read-only ground truth.
- **Lab memory** (`~/.paper_orchestra/memory`, or `PAPER_ORCHESTRA_MEMORY_DIR` / `--lab-memory-dir`): persists across runs and holds only process lessons (LaTeX pitfalls, venue rules, which revisions helped) plus a cache of public bibliographic metadata. A lesson is copied there only after a mechanical scan confirms it carries no numbers or wording from the paper's own materials, and the session that maintains the lab memory never sees a paper. Use `--no-lab-memory` for strict benchmark runs.

Inspect and maintain a run's memory:

```bash
uv run paper-orchestra memory show   --run-dir paper_output
uv run paper-orchestra memory search --run-dir paper_output "undefined control sequence"
uv run paper-orchestra memory lint   --run-dir paper_output
uv run paper-orchestra memory dream  --run-dir paper_output
```

With Claude models the wiki is exposed through the native memory tool, so the model opens pages and writes notes itself. Reviewers and bulk calls never see memory.

## Outputs

Everything is written under `--output-dir`:

| Path | Content |
|---|---|
| `final_paper.pdf`, `final_paper.tex` | The result. |
| `outline.json` | Plotting plan, literature search plan and section plan. |
| `literature_agent_output/` | `references.bib`, `citation_map.json`, the outline with citation candidates, the template with Introduction and Related Work written. |
| `plotting_results.json`, `figures/` | Generated figures with captions and critic history. |
| `latex_writeup/` | The LaTeX workspace (`raw_draft_paper.tex`, figures, bibliography). |
| `content_refinement_workdir/` | Each revision, reviews, worklogs and page screenshots. |
| `memory/` | The project memory described above. |
| `state.json` | Per-stage status; drives resume and the demo's progress display. |
| `usage.json`, `time_breakdown.json` | Token usage with an estimated cost, and time per stage. |
| `memory_report.json` | What the end-of-run consolidation did. |

A failed stage is retried on its own; re-running the same command resumes from the last completed stage.

## Batch runs

```bash
uv run paper-orchestra batch \
  --root-input-folder datasets/cvpr2025/papers \
  --template cvpr2025 \
  --output-dir batch_output
```

Each paper gets its own output directory; `paper_writing_log.json` records the result per paper and a re-run skips papers that already succeeded.

## Interactive demo

A Streamlit demo is in `frontend/`; see [frontend/README.md](frontend/README.md).

```bash
uv sync --extra ui
uv run streamlit run frontend/app.py
```

## Evaluation

`paper_orchestra.eval` contains the autoraters used in the paper: citation F1, literature-review quality, and side-by-side comparisons of literature reviews and overall paper quality. They work with any configured provider.

## Dataset

The dataset used for paper generation is **not included** in this code repository. It will be released separately at a later date. Once available, download it and place it under the `datasets/` directory.

## Project structure

```
src/paper_orchestra/
  agents/      outline, literature, plotting, section writing, refinement
  prompts/     prompts for those agents
  llm/         provider-neutral model layer (Anthropic, Gemini, OpenAI)
  memory/      wiki, transcripts, context harness, consolidation
  compute/     local and Modal backends
  eval/        autoraters
  pipeline.py  stage orchestration, retry and resume
  modal_app.py the pipeline as a Modal app
  cli.py       the paper-orchestra command
frontend/      Streamlit demo
src/paper_orchestra/templates/   bundled LaTeX templates (shipped in the wheel)
tests/         offline test suite
```

## Development

```bash
uv sync --all-extras
uv run pytest            # offline; tests that need TeX are skipped without it
uv run ruff check .
uv run ruff format .
```

## Citation

If you find this repo or our paper helpful, please cite it as follows:

```bibtex
@article{song2026paperorchestra,
  title={PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing},
  author={Song, Yiwen and Song, Yale and Pfister, Tomas and Yoon, Jinsung},
  journal={arXiv preprint arXiv:2604.05018},
  year={2026}
}
```

# Disclaimer

This is not an officially supported Google product.
