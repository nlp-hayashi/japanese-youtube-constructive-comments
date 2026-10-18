# Constructive comments demo (Chrome + API)

Scores visible YouTube comments via a local FastAPI server and lets you sort/filter by constructiveness.

## Quick start

1. Copy trained weights to `models/best/` (Hugging Face export with `config.json`, tokenizer files, `model.safetensors`).
2. API:

   ```bash
   cd server && pip install -r requirements.txt
   uvicorn api:app --host 0.0.0.0 --port 8765
   ```

3. Chrome → `chrome://extensions` → Developer mode → Load unpacked → select `extension/`.
4. Open a YouTube video, expand comments, use the on-page panel.

Override model path: `CONSTRUCTIVE_MODEL_DIR` (relative to `server/`).  
For HTTPS deployment, set `API_BASE` in `extension/background.js` and update `manifest.json` host permissions.

Research demo only; respect YouTube Terms of Service.
