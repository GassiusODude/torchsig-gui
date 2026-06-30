from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from torchsig_gui.api.config import router as config_router

app = FastAPI(
    title="TorchSig Configuration Interface",
    version="0.1.0"
)

# Attach API endpoints
app.include_router(config_router, prefix="/api/v1")

# Reference absolute path of your index.html template template file
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    if not TEMPLATE_PATH.exists():
        return "<h3>Configuration interface template file missing.</h3>"
    return TEMPLATE_PATH.read_text()
