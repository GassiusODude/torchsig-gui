from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal, Optional
import yaml

from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST, FAMILY_SHARED_LIST

router = APIRouter()

# Defines your dataset_metadata requirement
class DatasetMetadata(BaseModel):
    name: str = Field(default="My Dataset")
    description: Optional[str] = Field(default="TorchSig data profile")
    extra_attributes: Dict[str, Any] = Field(default_factory=dict)

# Limits choices to 'iq' or 'spectrogram'
class OutputConfig(BaseModel):
    representation: Literal["iq", "spectrogram"] = "iq"

# Limits choices to 'per_signal' or 'per_family'
class SignalSamplingConfig(BaseModel):
    mode: Literal["per_signal", "per_family"] = "per_signal"

# The top-level YAML configuration structure
class FullSignalConfig(BaseModel):
    dataset_metadata: DatasetMetadata
    output: OutputConfig = Field(default_factory=OutputConfig)
    signal_sampling: SignalSamplingConfig = Field(default_factory=SignalSamplingConfig)

# Schema for granular modulation configuration rows
class SignalRowConfig(BaseModel):
    name: str
    included: bool = False
    snr_min: float = -20.0
    snr_max: float = 30.0

class FullDatasetConfigRequest(BaseModel):
    dataset_name: str
    representation: Literal["iq", "spectrogram"]
    sampling_mode: Literal["per_signal", "per_family"]
    signals_grid: List[SignalRowConfig]

@router.post("/save-config")
async def save_config(config: FullSignalConfig):
    try:
        # Here you can process or save the fully validated configurations
        # For example: path.write_text(yaml.safe_dump(config.model_dump()))
        return {
            "status": "success",
            "message": "Configuration validated and saved successfully",
            "received_configuration": config.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")


@router.get("/signal-grid-options")
async def get_signal_grid_options(mode: str = "per_signal"):
    """
    Pulls available structural classes from TorchSig and formats them for the table.
    """
    try:
        if mode == "per_family":
            # Expose standard family wrappers from TorchSig
            items = FAMILY_SHARED_LIST

        else:
            # Fallback directly to the complete flat signal profile array
            items = SIGNALS_SHARED_LIST

        # Standard baseline values typical for TorchSig bounds
        return {
            "signals": [
                {"name": item, "included": True, "snr_min": -10.0, "snr_max": 20.0}
                for item in items
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save-grid-config")
async def save_grid_config(payload: FullDatasetConfigRequest):
    try:
        config_data = payload.model_dump()

        # Filter down the list to export only active signal choices
        config_data["signals_grid"] = [s for s in config_data["signals_grid"] if s["included"]]

        # 1. Generate prettified YAML string
        yaml_content = yaml.safe_dump(
            config_data,
            sort_keys=False,
            default_flow_style=False,
            indent=4
        )

        # 2. Build filename from user input box entry
        safe_filename = payload.dataset_name.strip().lower().replace(" ", "_") + ".yaml"

        # 3. Return a direct Response with file-download headers
        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Access-Control-Expose-Headers": "Content-Disposition"  # Crucial for browser JS to read the filename
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate config: {str(e)}")