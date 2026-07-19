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
class DatasetMetadataIn(BaseModel):
    sample_rate: int = 10000000
    num_iq_samples_dataset: int = 4096
    fft_size: int = 64
    fft_stride: int = 64
    num_signals_min: int = 1
    num_signals_max: int = 1
    cochannel_overlap_probability: float = 0.0
    snr_db_min: float = 0.0
    snr_db_max: float = 50.0
    noise_power_db: float = 0.0
    signal_duration_in_samples_min: int = 3276
    signal_duration_in_samples_max: int = 4096
    bandwidth_min: int = 2500000
    bandwidth_max: int = 3333333
    signal_center_freq_min: int = -2500000
    signal_center_freq_max: int = 2499999
    frequency_min: int = -2500000
    frequency_max: int = 2499999


class FullDatasetConfigRequest(BaseModel):
    # Root Parameters
    schema_version: str = "2.1.1"
    dataset_id: str
    dataset_length: int = 114000
    seed: int = 1234567893
    impairment_level: int = 2

    # Nested Object Configurations
    representation: Literal["iq", "spectrogram"] = "iq"
    sampling_mode: Literal["per_signal", "per_family"] = "per_signal"

    # Incoming layout array from web-table
    dataset_metadata: DatasetMetadataIn
    class_list: List[str] = Field(default_factory=list)
    target_labels: Literal["class_name", "class_index"] = "class_name"

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
        # 3. Restructure payload back to your target flat & nested YAML schema
        final_output = {
            "schema_version": payload.schema_version,
            "dataset_id": payload.dataset_id,
            "dataset_length": payload.dataset_length,
            "seed": payload.seed,
            "impairment_level": payload.impairment_level,
            "output": {
                "representation": payload.representation
            },
            "signal_sampling": {
                "mode": payload.sampling_mode
            },
            "dataset_metadata": payload.dataset_metadata.model_dump(),
            "class_list": payload.class_list,
            "target_labels": payload.target_labels
        }

        # 4. Generate clean, prettified block YAML strings
        yaml_content = yaml.safe_dump(
            final_output,
            sort_keys=False,           # Preserves the explicit sequence order
            default_flow_style=False,  # Enforces block notation (no inline {} or [])
            indent=2                   # Clean 2-space indentation
        )

        safe_filename = f"{payload.dataset_id.strip().lower().replace(' ', '_')}.yaml"

        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": f"attachment; filename={safe_filename}",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate config: {str(e)}")