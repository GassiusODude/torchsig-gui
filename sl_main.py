import streamlit as st
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Dict, Any, List, Literal, Optional
import yaml

# --- Mocking TorchSig Internal Lists ---
# (Replace these placeholders with your actual torchsig imports if running in your ML env)
# from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST, FAMILY_SHARED_LIST
SIGNALS_SHARED_LIST = [
    "ook", "4ask", "8ask", "16ask", "32ask", "64ask",
    "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk", "4gfsk", "4msk", "4gmsk",
    "8fsk", "8gfsk", "8msk", "8gmsk", "16fsk", "16gfsk", "16msk", "16gmsk",
    "bpsk", "qpsk","8psk","16psk","32psk","64psk",
    "16qam", "32qam", "32qam_cross", "64qam", "128qam_cross", "256qam", "512qam_cross", "1024qam",
    "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180", "ofdm-256", "ofdm-300", "ofdm-512",
    "ofdm-600", "ofdm-900", "ofdm-1024", "ofdm-1200", "ofdm-2048",
    "fm",
    "am-dsb-sc", "am-dsb","am-lsb", "am-usb",
    "lfm_data", "lfm_radar","chirpss",
    "tone",
]
FAMILY_SHARED_LIST = [
    "ask", "fsk", "psk", "qam", "ofdm", "fm", "am", "chirp", "tone"]

st.set_page_config(page_title="TorchSig Generator Configurator", layout="wide")
st.title("📡 TorchSig Dataset Generator Configurator")

# --- Embed Your Pydantic Data Structures ---
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

    @model_validator(mode="after")
    def validate_bounds_and_constraints(self) -> "DatasetMetadataIn":
        if self.num_signals_min > self.num_signals_max:
            raise ValueError(f"num_signals_min ({self.num_signals_min}) must be less than or equal to num_signals_max ({self.num_signals_max})")
        if self.snr_db_min > self.snr_db_max:
            raise ValueError(f"snr_db_min ({self.snr_db_min}) must be less than or equal to snr_db_max ({self.snr_db_max})")
        if self.signal_duration_in_samples_min > self.signal_duration_in_samples_max:
            raise ValueError(f"signal_duration_in_samples_min ({self.signal_duration_in_samples_min}) must be less than or equal to signal_duration_in_samples_max ({self.signal_duration_in_samples_max})")
        if self.bandwidth_min > self.bandwidth_max:
            raise ValueError(f"bandwidth_min ({self.bandwidth_min}) must be less than or equal to bandwidth_max ({self.bandwidth_max})")
        if self.signal_center_freq_min > self.signal_center_freq_max:
            raise ValueError(f"signal_center_freq_min ({self.signal_center_freq_min}) must be less than or equal to signal_center_freq_max ({self.signal_center_freq_max})")
        if self.frequency_min > self.frequency_max:
            raise ValueError(f"frequency_min ({self.frequency_min}) must be less than or equal to frequency_max ({self.frequency_max})")

        # Bandwidth constraint: bandwidth must be constrained by sample_rate (e.g. max bandwidth <= sample_rate)
        if self.bandwidth_max > self.sample_rate:
            raise ValueError(f"bandwidth_max ({self.bandwidth_max}) cannot exceed sample_rate ({self.sample_rate})")
        if self.bandwidth_min > self.sample_rate:
            raise ValueError(f"bandwidth_min ({self.bandwidth_min}) cannot exceed sample_rate ({self.sample_rate})")

        # Spectral occupancy constraints based on center frequency bounds and max bandwidth
        nyquist_min = -self.sample_rate / 2.0
        nyquist_max = self.sample_rate / 2.0

        if (self.signal_center_freq_min - self.bandwidth_max / 2.0) <= nyquist_min:
            raise ValueError(
                f"signal_center_freq_min - bandwidth_max / 2 ({self.signal_center_freq_min - self.bandwidth_max / 2.0}) "
                f"must be strictly greater than - sample_rate / 2 ({nyquist_min})"
            )
        if (self.signal_center_freq_max + self.bandwidth_max / 2.0) >= nyquist_max:
            raise ValueError(
                f"signal_center_freq_max + bandwidth_max / 2 ({self.signal_center_freq_max + self.bandwidth_max / 2.0}) "
                f"must be strictly less than sample_rate / 2 ({nyquist_max})"
            )

        return self

class FullDatasetConfigRequest(BaseModel):
    schema_version: str = "2.1.1"
    dataset_id: str
    dataset_length: int = 114000
    seed: int = 1234567893
    impairment_level: int = 2
    representation: Literal["iq", "spectrogram"] = "iq"
    sampling_mode: Literal["per_signal", "per_family"] = "per_signal"
    dataset_metadata: DatasetMetadataIn
    class_list: List[str] = Field(default_factory=list)
    target_labels: Literal["class_name", "class_index"] = "class_name"

    @model_validator(mode="after")
    def validate_class_list_not_empty_and_valid(self) -> "FullDatasetConfigRequest":
        if not self.class_list:
            raise ValueError("class_list cannot be empty. Please include at least one signal or family.")
        
        allowed_list = FAMILY_SHARED_LIST if self.sampling_mode == "per_family" else SIGNALS_SHARED_LIST
        invalid_classes = [c for c in self.class_list if c not in allowed_list]
        if invalid_classes:
            raise ValueError(f"Invalid classes for sampling_mode '{self.sampling_mode}': {invalid_classes}. Allowed list: {allowed_list}")
        
        return self

# --- Sidebar: Core Metadata & Generation Params ---
st.sidebar.header("📁 Root Specifications")
dataset_id = st.sidebar.text_input("Dataset ID", value="torchsig_custom_dataset")
dataset_length = st.sidebar.number_input("Dataset Length (Samples)", value=114000, step=1000)
seed = st.sidebar.number_input("Random Seed", value=1234567893)
impairment_level = st.sidebar.slider("Impairment Level", min_value=0, max_value=3, value=2)

st.sidebar.header("⚙️ Target Output Profiles")
representation = st.sidebar.selectbox("Representation Output", options=["iq", "spectrogram"], index=0)
sampling_mode = st.sidebar.selectbox("Signal Sampling Mode", options=["per_signal", "per_family"], index=0)
target_labels = st.sidebar.selectbox("Target Labels Format", options=["class_name", "class_index"], index=0)

# --- Process Signal Inclusions Based on Sampling Mode ---
# Instead of an API call, we directly read our local variables
items = FAMILY_SHARED_LIST if sampling_mode == "per_family" else SIGNALS_SHARED_LIST
signal_grid_data = [{"name": item, "included": True, "snr_min": -10.0, "snr_max": 20.0} for item in items]

# --- Main Layout: Two Columns ---
col1, col2 = st.columns(2)

with col1:
    st.header("📝 Dataset Metadata Adjustments")

    with st.expander("Physical Layer & FFT Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        sample_rate = c1.number_input("Sample Rate (Hz)", value=10000000)
        num_iq_samples = c2.number_input("IQ Samples Size", value=4096)
        fft_size = c3.number_input("FFT Size", value=64)
        fft_stride = c4.number_input("FFT Stride", value=64)

    with st.expander("Channel & Environment Simulation Bounds", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        num_sig_min = c1.number_input("Min Signals", value=1, min_value=0)
        num_sig_max = c2.number_input("Max Signals", value=1, min_value=0)
        cochannel_prob = c3.number_input("Cochannel Overlap Prob", value=0.0, max_value=1.0, step=0.1)
        snr_min = c4.number_input("Min SNR (dB)", value=0.0)
        snr_max = c5.number_input("Max SNR (dB)", value=50.0)
        noise_power = st.number_input("Noise Power Base (dB)", value=0.0)

    with st.expander("Signal Spectral Boundaries", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        dur_min = c1.number_input("Min Duration (Samples)", value=3276)
        dur_max = c2.number_input("Max Duration (Samples)", value=4096)
        bw_min = c3.number_input("Min Bandwidth (Hz)", value=2500000)
        bw_max = c4.number_input("Max Bandwidth (Hz)", value=3333333)

        c5, c6, c7, c8 = st.columns(4)
        cf_min = c5.number_input("Min Center Freq (Hz)", value=-2500)
        cf_max = c6.number_input("Max Center Freq (Hz)", value=2499)
        freq_min = c7.number_input("Min Freq Boundary (Hz)", value=-2500000)
        freq_max = c8.number_input("Max Freq Boundary (Hz)", value=2499999)

    # --- Replacing Web-Tables with st.data_editor ---
    st.header("🎛️ Dynamic Signal Target Matrix")
    st.write(f"Inclusion filters applied globally via `{sampling_mode}` structures.")

    edited_signals = st.data_editor(
        signal_grid_data,
        key=f"grid_{sampling_mode}", # Force refresh UI when mode alters
        column_config={
            "name": st.column_config.TextColumn("Signal / Family Name", disabled=True),
            "included": st.column_config.CheckboxColumn("Include in Class List", default=True),
            "snr_min": st.column_config.NumberColumn("Default Min SNR", disabled=True),
            "snr_max": st.column_config.NumberColumn("Default Max SNR", disabled=True),
        },
        disabled=["name", "snr_min", "snr_max"],
        hide_index=True,
        width="stretch"
    )

with col2:
    st.header("📄 Export Settings Block")

    # Filter class_list instantly from data editor modifications
    class_list = [row["name"] for row in edited_signals if row.get("included", True)]

    # Assemble validation dictionary payload
    raw_payload = {
        "schema_version": "2.1.1",
        "dataset_id": dataset_id,
        "dataset_length": int(dataset_length),
        "seed": int(seed),
        "impairment_level": int(impairment_level),
        "representation": representation,
        "sampling_mode": sampling_mode,
        "dataset_metadata": {
            "sample_rate": int(sample_rate),
            "num_iq_samples_dataset": int(num_iq_samples),
            "fft_size": int(fft_size),
            "fft_stride": int(fft_stride),
            "num_signals_min": int(num_sig_min),
            "num_signals_max": int(num_sig_max),
            "cochannel_overlap_probability": float(cochannel_prob),
            "snr_db_min": float(snr_min),
            "snr_db_max": float(snr_max),
            "noise_power_db": float(noise_power),
            "signal_duration_in_samples_min": int(dur_min),
            "signal_duration_in_samples_max": int(dur_max),
            "bandwidth_min": int(bw_min),
            "bandwidth_max": int(bw_max),
            "signal_center_freq_min": int(cf_min),
            "signal_center_freq_max": int(cf_max),
            "frequency_min": int(freq_min),
            "frequency_max": int(freq_max)
        },
        "class_list": class_list,
        "target_labels": target_labels
    }

    # --- Live Local Pydantic Validation & Structure Mapping ---
    validation_error = None
    yaml_content = ""

    try:
        # Validate data locally using your exact same Pydantic schema rule loops
        validated_data = FullDatasetConfigRequest(**raw_payload)

        # Restructure precisely into your nested target schema ruleset
        final_output = {
            "schema_version": validated_data.schema_version,
            "dataset_id": validated_data.dataset_id,
            "dataset_length": validated_data.dataset_length,
            "seed": validated_data.seed,
            "impairment_level": validated_data.impairment_level,
            "output": {"representation": validated_data.representation},
            "signal_sampling": {"mode": validated_data.sampling_mode},
            "dataset_metadata": validated_data.dataset_metadata.model_dump(),
            "class_list": validated_data.class_list,
            "target_labels": validated_data.target_labels
        }

        # Clean, human-readable Block YAML processing
        yaml_content = yaml.safe_dump(
            final_output,
            sort_keys=False,
            default_flow_style=False,
            indent=2
        )
    except ValidationError as ve:
        validation_error = ve

    # Display results
    st.subheader("Live Structure Target Preview")
    if validation_error:
        st.error(f"❌ Structural Validation Error: {validation_error}")
        st.download_button(
            label="💾 Download Generated YAML File",
            data="",
            file_name="disabled.yaml",
            mime="application/x-yaml",
            width="stretch",
            disabled=True
        )
    else:
        st.code(yaml_content, language="yaml")

        # Safe filename processor matching your old server layout
        safe_filename = f"{dataset_id.strip().lower().replace(' ', '_')}.yaml"

        # Streamlit Native File Downloader (replaces old file actions)
        st.download_button(
            label="💾 Download Generated YAML File",
            data=yaml_content,
            file_name=safe_filename,
            mime="application/x-yaml",
            width="stretch"
        )
