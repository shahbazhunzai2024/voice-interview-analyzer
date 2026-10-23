# Claude Code Instructions

## Project

This is a FastAPI application for live interview audio analysis. Read `README.md` for setup,
API routes, and the current feature status before changing behavior. The backend entry point is
`main.py`; application code lives under `app/`, and the browser UI is in `frontend/`.

## Development

- Use the existing Python modules and interfaces; keep API contracts in `app/schemas/` aligned
  with their consumers and tests.
- Install dependencies with `python -m pip install -r requirements.txt`.
- Run the test suite with `pytest`.
- Start the development server with `uvicorn main:app --reload`.
- Keep changes focused and add or update tests for behavior changes.

## Product and data constraints

- Analysis results are review cues for a human. Do not introduce pass/fail decisions, candidate
  rankings, honesty judgments, or claims that speech was AI-authored.
- Preserve evidence and uncertainty in review flags; avoid presenting heuristic outputs as facts.
- Audio is not retained and sessions are currently held in memory. Do not add audio or session
  persistence without an explicit retention and deletion policy.
- Keep LiveKit and transcription credentials server-side. Do not expose secrets in browser code,
  API responses, or logs.
- `DEV_MODE` is for development and must not be enabled for production deployments.