from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles  # New import
from pathlib import Path
from torchsig_gui.api.config import router as config_router

app = FastAPI(title="TorchSig Configuration Interface", version="0.1.0")
app.include_router(config_router, prefix="/api/v1")

CURRENT_DIR = Path(__file__).parent

# Mount the static directory so files are accessible at /static/
app.mount("/static", StaticFiles(directory=CURRENT_DIR / "static"), name="static")

TEMPLATE_PATH = CURRENT_DIR / "templates" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    if not TEMPLATE_PATH.exists():
        return "<h3>Configuration interface template file missing.</h3>"
    return TEMPLATE_PATH.read_text()
