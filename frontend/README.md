# Paper Orchestra Frontend

This directory contains the Streamlit frontend for Paper Orchestra.

## How to Launch the Demo Locally

1. From the project root, install the project with the UI extra (add `--extra gemini`,
   `--extra openai` or `--extra modal` if you want those providers or the Modal compute backend).
   The page needs Streamlit 1.49 or newer:
   ```bash
   cd paper-orchestra
   uv sync --extra ui
   ```
2. Make sure the model credentials are available, either in the environment or in a `.env` file
   in the project root. `uv run paper-orchestra doctor` reports what is configured.
3. Run the Streamlit application:
   ```bash
   uv run streamlit run frontend/app.py
   ```
4. Open the URL provided in the terminal (usually `http://localhost:8501` or similar) in your browser.

## What the Sidebar Controls

- **Select Model**: writes, reviews and plots. `claude-fable-5-2` is the default.
- **Diagram Image Model**: leave it at "None" to have diagrams drawn by model-written plotting
  code; pick a Gemini image model to generate them as images (needs Gemini credentials).
- **Compute Backend**: `local` runs model-written figure code and LaTeX on this machine;
  `modal` runs them in network-isolated Modal sandboxes (needs `uv sync --extra modal` and a
  Modal token).
- **Wiki Memory** and **Memory Consolidation (Dreaming)**: whether the run keeps a wiki of what
  it learns, and when notes are consolidated into wiki pages. The **Memory** tab shows the
  run's `MEMORY.md` and lets you open any wiki page.
- **Your Email**: optional, one address. A finished run is e-mailed when `SMTP_EMAIL` and
  `SMTP_PASSWORD` (and optionally `SMTP_HOST` / `SMTP_PORT`, default Gmail over SSL) are set.

Each run is written to `../demo_logs/paper_orchestra_run_<timestamp>_<id>/` next to the
repository (set `PAPER_ORCHESTRA_DEMO_DIR` to use another folder). The complete pipeline output
is kept in `pipeline.log` inside the run folder; the values of credential environment variables
(`*_API_KEY`, `*TOKEN*`, `*PASSWORD*`, ...) are masked in it and in the log shown on the page.

Uploaded figures are stored under plain file names (other characters become `_`, and a second
figure with the same name gets a number), which is how the paper refers to them.

The page follows a run only in the browser session that started it: reloading the tab starts a
new session, and the earlier run keeps going in the background (see below for stopping it).

## How to Kill Unfinished Writing Processes

The frontend launches the pipeline as a background process (`python -m paper_orchestra write ...`)
in its own process group.

### Automatic Cleanup
- The **Stop Run** button interrupts the pipeline and everything it started.
- Stopping the Streamlit server with **Ctrl+C** stops any pipeline it started.
- On Linux, the pipeline is also killed if the Streamlit server process dies without a chance to
  clean up.

### Manual Cleanup
If the server was killed on another platform, or a pipeline is left over for any other reason, you
can stop it manually:

```bash
pkill -f "paper_orchestra write"
```

Or to find the process ID (PID) first:
```bash
pgrep -f "paper_orchestra write"
```
And then kill it:
```bash
kill <PID>
```
