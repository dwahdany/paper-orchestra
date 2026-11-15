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
| `figures/` + `figures/info.json` | Optional ready-made figures, as `[{"name": "fig.png", "caption": "..."}]`. Used when running without `--plotting`. With `--plotting` every figure is generated and these are ignored, unless you pass `--keep-provided-figures`. |

`--template` takes the name of a bundled template (`cvpr2025`, `iclr2025`), or a path to your own template directory containing `template.tex` and `guidelines.md`. A name is also looked up under `$PAPER_ORCHESTRA_TEMPLATES` and `./templates`.

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--plotting / --no-plotting` | off | Generate figures from the outline's plotting plan. |
| `--keep-provided-figures` | off | With `--plotting`, also give the writer the figures in `raw_materials/figures`. |
| `--model` | `claude-fable-5-2` | Model for every role without a flag of its own. |
| `--writer-model`, `--reflection-model`, `--plotting-model` | `--model` | Model per role. |
| `--fast-model` | `--model`, else the writer model | Model for high-volume calls such as literature discovery (run at low effort). Discovery works best on a model with web search; pass this flag when the writer's provider has none. |
| `--image-model` | unset | Image-generation model for diagrams (for example a Gemini image model). When unset, diagrams are drawn from model-written plotting code. |
| `--effort` | `high` | Reasoning effort for the main calls. |
| `--research-cutoff` | current month | `YYYY-MM`; papers published later are not cited as prior work. |
| `--compute` | `local` | `local` or `modal`. |
| `--memory / --no-memory` | on | The wiki memory harness. |
| `--lab-memory / --no-lab-memory` | on | Share process lessons across papers. Turn off for benchmark runs. |
| `--dream` | `stages` | When memory is consolidated by a model: `off`, `end`, or `stages`. |
| `--resume / --no-resume` | on | Skip stages already completed in the output directory, while the inputs and the settings that shape the paper are unchanged (see [Outputs](#outputs)). |

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

A temporary failure of the token exchange (a network error, HTTP 408, 429 or 5xx) is retried with backoff and, if it persists, fails the stage as a retryable error. A denied exchange (400, 401, 403) stops the run as a configuration error; the reason is in the federation history of the Claude Console.

Claude Fable 5.2 requires an organization with data retention enabled. Requests the model declines are retried on a fallback model by the API (server-side fallbacks are enabled). A reply cut off at the default output limit is requested once more with the model's maximum. At `--effort xhigh` or `max` a full-paper reply can still exceed that; lower the effort if a stage fails with a truncation.

To use other providers, install the extra and pass the model name:

```bash
uv sync --extra gemini   # GEMINI_API_KEY, or VERTEX_AI_PROJECT + VERTEX_AI_LOCATION
uv sync --extra openai   # OPENAI_API_KEY
uv run paper-orchestra write ... --writer-model gemini-3.1-pro-preview
```

The Gemini and OpenAI clients have no memory tool, so they use the memory harness in "push" mode: the lessons and syntheses in memory are placed in the prompt instead of being opened on demand. Consolidation sessions, which write those pages, need a model with the memory tool. Keep a Claude model in the reflection role if you want the learning loop described under [Memory](#memory); without one, notes are kept but never turned into lessons.

`SEMANTIC_SCHOLAR_API_KEY` is optional but raises the rate limit of literature verification considerably.

## Compute backends

Two kinds of work leave the orchestrating process: running model-written Python (figure code) and compiling model-written LaTeX.

| Backend | Figure code | LaTeX | Isolation |
|---|---|---|---|
| `local` | subprocess with a scrubbed environment, resource limits and a timeout | local `pdflatex`/`bibtex`, shell escape disabled, file access limited to the workspace | Keeps credentials out of the child's environment, but it is not a sandbox. |
| `modal` | Modal Sandbox, network blocked, no secrets | Modal Sandbox with TeX Live, network blocked | Nothing model-written runs on your machine. |

```bash
uv sync --extra modal
uv run modal setup
uv run paper-orchestra write ... --compute modal
```

Model-written LaTeX is screened before it is compiled, on every backend. A document that uses constructs which read or write files or run code (shell escape, `\openin`/`\openout`, `\pdfobj` file streams, absolute or parent paths in `\input` and `\includegraphics`, and similar) is not compiled; the reasons go back to the writer as errors to fix. The screen exists because TeX's own file-access limits do not cover every primitive. It is a filter, not a proof: for materials you do not trust, use `--compute modal`, where a document can reach nothing but its own workspace.

A backend that cannot work is reported before the first model call is paid for: missing TeX Live for `local`, a missing Modal token for `modal`. If a stage later fails while the backend reports itself unusable, the run stops with that message instead of retrying, and the same command resumes there once the environment is fixed.

### Running the whole pipeline on Modal

```bash
uv run paper-orchestra modal run \
  --raw-materials-dir path/to/raw_materials \
  --template cvpr2025 \
  --output-dir paper_output \
  --plotting
```

This is `modal run -m paper_orchestra.modal_app` underneath. The Modal commands must be run from a checkout of this repository with `uv sync --extra modal` (the image is built from `uv.lock`); from an installed wheel they fail with a message that says so. The command uploads the raw materials, runs the pipeline in a container, and downloads `final_paper.pdf`, `final_paper.tex` and `usage.json`. It takes the same model options as `write` (`--model`, `--writer-model`, ...); `paper-orchestra modal run --help` lists them, including `--run-id` to resume or re-download a run.

Outputs and the lab memory live on a Modal Volume (`paper-orchestra` by default), so a preempted or interrupted run resumes where it stopped. A run id is written by one call at a time: a second call for a run that is in flight is refused, and after a run was cancelled or its container killed the same run id can be refused for a couple of minutes, until its lease expires.

Nothing model-written runs next to a credential. Figure code and LaTeX both run in nested sandboxes with no network, no Secret and no identity token; one LaTeX sandbox serves a whole run. Set `PAPER_ORCHESTRA_MODAL_LATEX=local` when running or deploying to compile inside the pipeline container instead, which is faster and still screened as described above. The TeX images cover the bundled templates and common font packages; `PAPER_ORCHESTRA_MODAL_TEX_PACKAGES="pkg1 pkg2"` adds Debian packages for a template that needs more.

Runs on Modal only add their lessons to the lab memory as notes. A separate function, `consolidate_memory`, is the single writer that folds them into pages, because a Volume keeps the last write of each file.

#### Batches on Modal

```bash
uv run paper-orchestra modal deploy
uv run paper-orchestra modal batch \
  --root-input-folder datasets/cvpr2025/papers \
  --template cvpr2025 \
  --output-dir batch_output
```

Each paper is its own Modal call, downloaded and logged as soon as it finishes. Ctrl-C or a lost connection stops nothing on Modal: the papers are recorded in `paper_writing_log.json` as `submitted`, and running the same command again waits for them and collects the results (Modal keeps a result for seven days). `--cancel` stops the submitted papers instead. Each paper runs under `<paper id>-<8 hex digits>`, where the digits stand for the options and the template: the same batch submitted again resumes its runs, while another model, effort, option or template gets runs of its own, so batches that compare settings never return each other's papers.

#### Credentials on Modal

Credentials come from a Modal Secret (named `paper-orchestra` by default, `PAPER_ORCHESTRA_MODAL_SECRET` to change). Two setups are supported:

- **API key**: `modal secret create paper-orchestra ANTHROPIC_API_KEY=sk-ant-...` (add `SEMANTIC_SCHOLAR_API_KEY` if you have one).
- **Keyless**: Modal gives every function an OIDC identity token (issuer `https://oidc.modal.com`, audience `oidc.modal.com`), which is exchanged for short-lived Anthropic access tokens, so no Anthropic secret is stored anywhere. The container only needs the non-secret identifiers `ANTHROPIC_FEDERATION_RULE_ID` and `ANTHROPIC_ORGANIZATION_ID` (plus `ANTHROPIC_SERVICE_ACCOUNT_ID` / `ANTHROPIC_WORKSPACE_ID` if your rule needs them). Put them in the Secret, or export them in the shell that launches the run and set `PAPER_ORCHESTRA_MODAL_SECRET=none`. Do not also provide an API key: it would take precedence, and the run refuses to start when both are present.

The keyless setup could not be exercised against the live services while this was written, so check three things on the Anthropic side when you set up the federation rule. If the first exchange is refused, the error lists the claims of the container's token to compare against the rule.

1. The rule must admit both functions of the app, `write_paper` and `consolidate_memory`. Match on the Modal workspace and app name, or list both function names; a rule pinned to `write_paper` alone lets papers be written but never folds their lessons into the lab memory.
2. The issuer's maximum token lifetime must be at least the lifetime of Modal's identity tokens, which can be much longer than the one-hour default.
3. The access tokens the rule issues should live as long as a run can take (`PAPER_ORCHESTRA_MODAL_TIMEOUT`, six hours by default), because a single-use identity token cannot be exchanged a second time to renew them.

## Memory

Agents do not get everything stuffed into their prompt, and they do not start from zero each time. Every model call goes through a context harness whose memory is a wiki on disk:

| Layer | Where | Written by | Used for |
|---|---|---|---|
| Transcripts | `memory/transcripts/*.jsonl` | the harness, automatically | The record of what happened; notes are checked against it. |
| Notes | `memory/notes/*.md` | agents and pipeline code, during a stage | Salient observations: a failure and its cause, a decision and its reason. |
| Wiki pages | `memory/wiki/<section>/<page>.md` | pipeline code (records) and consolidation sessions (lessons, synthesis) | One page per thing, linked with `[[section/page]]`. |
| Index | `memory/MEMORY.md` | regenerated automatically; a few curated standing notes | Loaded into every session (capped at 200 lines / 25 KB). |

Between stages a separate consolidation session ("dreaming") verifies notes against the transcripts and sources, folds what holds up into pages, repairs links and keeps the index within budget. It archives only the notes it actually read; the rest wait for the next consolidation. Its pages may state only what this run's sources, bibliography and transcripts say, and it does not see the lab memory. The refinement loop uses this as a learning loop: each accepted or rejected revision is recorded with its score changes and turned into a rule the next iteration can see. A rejected revision is discarded and the remaining iterations start again from the last accepted version, now with that lesson (`retry_after_rejection`; the original behaviour of stopping at the first rejection is available by turning it off in `PipelineConfig`).

There are two memories:

- **Project memory** (`<output-dir>/memory`): everything about this paper. Sources are stored verbatim and are read-only ground truth.
- **Lab memory** (`~/.paper_orchestra/memory`, or `PAPER_ORCHESTRA_MEMORY_DIR` / `--lab-memory-dir`): persists across runs and holds only process lessons (LaTeX pitfalls, venue rules, which revisions helped). Runs read it and never rewrite it.

A lesson reaches the lab memory in two steps, so that the session maintaining it never sees a paper. First, pipeline code copies each lesson that a consolidation marked as portable into the lab as a note, but only if a mechanical scan finds nothing of the paper in it: no result-like number, no distinctive name (a method, dataset or system) and no run of wording from the paper's inputs, outline, figure captions, reviews or drafts. What the template directory ships (the template and guidelines) is the venue's and may be quoted; everything else in the run is the paper, including the template file once sections have been written into it. Nothing is shared when there is nothing to scan against. Second, a session that has only the lab memory in front of it folds those notes into pages; until then other runs cannot see them.

The scan is a filter on form. It stops lessons that carry their evidence along, in the spellings that occur in practice (a result as a percentage or rounded, a name in lower case or inside a file name). It does not stop a paraphrase, a number that was computed or rounded to an integer, or a name that is an ordinary capitalised word. `memory_report.json` says how many lessons a run shared and how many the scan kept back. Use `--no-lab-memory` for strict benchmark runs.

Beside the lab wiki, `cache/` keeps Semantic Scholar records across runs. It is read by pipeline code only: no model session can open or search it, and every cached record is checked against the current run's research cutoff before it is used.

Inspect and maintain a run's memory:

```bash
uv run paper-orchestra memory show   --run-dir paper_output
uv run paper-orchestra memory search --run-dir paper_output "undefined control sequence"
uv run paper-orchestra memory lint   --run-dir paper_output
uv run paper-orchestra memory dream  --run-dir paper_output
```

`memory dream` consolidates with the reflection model the run used (`--model` to override).

With Claude models the wiki is exposed through the native memory tool, so the model opens pages and writes notes itself. Reviewers and bulk calls never see memory.

## Outputs

Everything is written under `--output-dir`:

| Path | Content |
|---|---|
| `final_paper.pdf`, `final_paper.tex` | The result. |
| `outline.json` | Plotting plan, literature search plan and section plan. |
| `literature_agent_output/` | `references.bib`, `citation_map.json`, the outline with citation candidates, the template with Introduction and Related Work written, and `collected_papers.json` (the verified papers, kept so that a retried literature stage does not search again). |
| `plotting_results.json`, `figures/` | Generated figures with captions and critic history. |
| `latex_writeup/` | The LaTeX workspace (`raw_draft_paper.tex`, figures, bibliography). |
| `content_refinement_workdir/` | Each revision, reviews, worklogs and page screenshots. |
| `memory/` | The project memory described above. |
| `state.json` | Per-stage status; drives resume and the demo's progress display. |
| `usage.json`, `time_breakdown.json` | Token usage with an estimated cost (refreshed every few seconds while stages run), and time per stage. |
| `memory_report.json` | What the end-of-run consolidation did, including how many lessons were shared with the lab memory and how many were kept back. |

A failed stage is retried on its own; re-running the same command resumes from the last completed stage. Details worth knowing:

- **What restarts a run.** Resume applies while the inputs and the settings that shape the paper are unchanged: the models and efforts, plotting and its options, the refinement options, memory on or off (and, with memory on, lab memory and the dream mode), and the input file names. If one changes, a warning names it and every stage runs again in that directory, replacing the earlier paper, so use a new output directory to compare settings. The compute backend, timeouts, worker counts and a later research cutoff do not restart a run.
- **Ctrl-C.** The run stops issuing model calls, waits up to ten seconds for literature and plotting when they run side by side, and records itself and the stopped stages as `interrupted` in `state.json`. The same command resumes it.
- **Plotting is optional.** If it fails, the paper is written without generated figures. Such a failure is tried again when the run is resumed, unless that run had already delivered its paper.
- **Refinement needs a review.** If the review of the draft fails (a provider outage, a rate limit), the refinement stage fails and is retried; it never passes the unrefined draft off as refined. An error in one refinement iteration costs that iteration only.

## Batch runs

```bash
uv run paper-orchestra batch \
  --root-input-folder datasets/cvpr2025/papers \
  --template cvpr2025 \
  --output-dir batch_output
```

Each paper gets its own output directory and is written in a worker process of its own (`--max-workers` at a time). `paper_writing_log.json` records the result per paper, and a re-run skips papers that already succeeded. If a worker dies, for example killed for using too much memory, only its paper is logged as an error and the others continue.

Ctrl-C, or SIGTERM from `kill`, `timeout` or a scheduler, starts no further paper and stops the papers in flight: each run closes its `state.json` as `interrupted`, and running the same command again continues the batch. A single `write` behaves the same way. Exit codes: 0 success, 1 the pipeline or a check failed, 2 a usage error, 130 interrupted.

To write a batch on Modal instead, see [Batches on Modal](#batches-on-modal).

## Interactive demo

A Streamlit demo is in `frontend/`; see [frontend/README.md](frontend/README.md).

```bash
uv sync --extra ui
uv run streamlit run frontend/app.py
```

## Evaluation

`paper_orchestra.eval` contains the autoraters used in the paper: citation F1, literature-review quality, and side-by-side comparisons of literature reviews and overall paper quality. They work with any configured provider.

## Differences from the original implementation

This rework keeps the method of the paper, but a few behaviours changed. If you compare results with the paper or the original repository, these are the ones that matter:

- **Models.** Every role defaults to `claude-fable-5-2`; literature discovery runs on the fast model at low effort. The original used Gemini models, which remain available through `--model` and the per-role flags.
- **Refinement.** A rejected revision is discarded and the remaining iterations continue from the last accepted version, with the lesson from the rejection (`retry_after_rejection`, on by default); the original stopped at the first rejection. A revision that introduces new LaTeX errors is rejected before it is reviewed.
- **Reviewer.** The review form no longer asks for a free-form reasoning section before the JSON; the models reason on their own before answering.
- **Author block.** The section writer is told to keep the manuscript anonymous, where the original asked it to fill in author names if they were missing.
- **Diagrams.** Without `--image-model`, diagrams are drawn by model-written plotting code instead of an image-generation model. Generated figures are PNG files at 200 dpi.
- **PaperBanana.** Few-shot figure examples and style guides are read from the checkout named by `PAPER_BANANA_DIR`. The original looked in a sibling `../PaperBanana` directory; without the variable, figures are planned zero-shot with a built-in style guide.
- **Literature.** Papers without a publication date, and papers dated only by year in the year of the research cutoff, are not cited, since they cannot be shown to precede the cutoff.
- **Memory.** The memory harness and its consolidation sessions (including a synthesis of the collected literature) are new. `--no-memory` runs the stages on their prompts alone, and `--no-lab-memory` keeps runs independent of each other.
- **Knowledge isolation.** The rule that the paper is written only from the run's materials is unchanged in substance and reworded: it now explains the reason and says how memory fits in (lessons guide how to write; what the paper claims comes from the run's materials).

## Dataset

The dataset used for paper generation is **not included** in this code repository. It will be released separately at a later date. Once available, download it and place it under the `datasets/` directory.

## Project structure

```
src/paper_orchestra/
  agents/      outline, literature, plotting, section writing, refinement
  prompts/     prompts for those agents
  llm/         provider-neutral model layer (Anthropic, Gemini, OpenAI)
  memory/      wiki, transcripts, context harness, consolidation, cross-run cache
  compute/     local and Modal backends
  eval/        autoraters
  pipeline.py  stage orchestration, retry and resume
  modal_app.py the pipeline as a Modal app
  modal_batch.py  the local side of batches on Modal
  cli.py       the paper-orchestra command
  templates/   bundled LaTeX templates (shipped in the wheel)
frontend/      Streamlit demo
tests/         offline test suite
```

## Development

```bash
uv sync --all-extras
uv run pytest            # offline; tests that need TeX are skipped without it
uv run ruff check .
uv run ruff format .
```

The test suite never opens a network connection (the test configuration blocks sockets) and uses scripted models. With TeX Live installed it also compiles real documents, including an end-to-end run of the whole pipeline; CI installs TeX Live so that those tests run there.

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
