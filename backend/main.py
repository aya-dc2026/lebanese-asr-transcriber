"""
main.py — FastAPI backend (Phase 2+)

Not implemented yet. Phase 1 is CLI-only (see backend/asr_engine.py).
Once Phase 1 baseline is validated, this file will grow into:

    POST /transcribe          - accepts audio file, returns transcript + record_id
    POST /correct              - accepts record_id + corrected text -> corrections.db
    GET  /corrections/count    - how many new corrections exist
    POST /retrain               - password-protected, triggers train.py
    GET  /retrain/status        - polls training progress

See docs/HLD.pdf sections 7.3-7.6 for the full spec.
"""

# from fastapi import FastAPI
# app = FastAPI()
#
# @app.on_event("startup")
# def startup():
#     from backend.asr_engine import ASREngine
#     import os
#     app.state.engine = ASREngine(
#         model_path=os.environ["MODEL_PATH"],
#         adapter_path=os.environ.get("ADAPTER_PATH") or None,
#     )

if __name__ == "__main__":
    print("Phase 2 not implemented yet - run backend/asr_engine.py directly for now (Phase 1).")
