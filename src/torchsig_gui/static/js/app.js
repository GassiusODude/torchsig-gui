const samplingModeEl = document.getElementById('samplingMode');
const checklistGridEl = document.getElementById('checklistGrid');
const selectAllBtn = document.getElementById('selectAllBtn');
const clearAllBtn = document.getElementById('clearAllBtn');
const matrixForm = document.getElementById('matrixForm');
const statusDiv = document.getElementById('statusMessage');

async function updateDynamicClasses() {
    const currentMode = samplingModeEl.value;
    try {
        const response = await fetch(`/api/v1/signal-grid-options?mode=${currentMode}`);
        const data = await response.json();
        checklistGridEl.innerHTML = '';

        data.signals.forEach(sig => {
            const itemWrapper = document.createElement('label');
            itemWrapper.className = "flex items-center space-x-3 p-2 bg-slate-800 rounded border border-slate-700 hover:border-slate-500 cursor-pointer select-none transition-colors";
            itemWrapper.innerHTML = `
                <input type="checkbox" class="sig-include rounded text-sky-600 border-slate-600 bg-slate-900 focus:ring-sky-500 w-4 h-4" ${sig.included ? 'checked' : ''} data-name="${sig.name}">
                <span class="font-mono text-xs font-medium text-slate-300 uppercase">${sig.name}</span>
            `;
            checklistGridEl.appendChild(itemWrapper);
        });
    } catch (err) {
        console.error("Modulation list tracking error:", err);
    }
}

function toggleAllCheckboxes(checkedState) {
    document.querySelectorAll('.sig-include').forEach(cb => cb.checked = checkedState);
}

samplingModeEl.addEventListener('change', updateDynamicClasses);
window.addEventListener('DOMContentLoaded', updateDynamicClasses);
selectAllBtn.addEventListener('click', () => toggleAllCheckboxes(true));
clearAllBtn.addEventListener('click', () => toggleAllCheckboxes(false));

matrixForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const selectedClasses = [];
    document.querySelectorAll('.sig-include').forEach(checkbox => {
        if (checkbox.checked) {
            selectedClasses.push(checkbox.getAttribute('data-name'));
        }
    });

    const payload = {
        schema_version: "2.1.1",
        dataset_id: document.getElementById('datasetId').value,
        dataset_length: parseInt(document.getElementById('datasetLength').value),
        seed: parseInt(document.getElementById('datasetSeed').value),
        impairment_level: parseInt(document.getElementById('impairmentLevel').value),
        representation: document.getElementById('representation').value,
        sampling_mode: samplingModeEl.value,

        // Dynamic mapping of webpage metadata configuration fields
        dataset_metadata: {
            sample_rate: parseInt(document.getElementById('sampleRate').value),
            num_iq_samples_dataset: parseInt(document.getElementById('numIqSamples').value),
            fft_size: parseInt(document.getElementById('fftSize').value),
            fft_stride: parseInt(document.getElementById('fftStride').value),
            num_signals_min: parseInt(document.getElementById('numSignalsMin').value),
            num_signals_max: parseInt(document.getElementById('numSignalsMax').value),
            cochannel_overlap_probability: parseFloat(document.getElementById('cochannelOverlap').value),
            snr_db_min: parseFloat(document.getElementById('snrMin').value),
            snr_db_max: parseFloat(document.getElementById('snrMax').value),
            noise_power_db: 0.0,
            signal_duration_in_samples_min: parseInt(document.getElementById('durationMin').value),
            signal_duration_in_samples_max: parseInt(document.getElementById('durationMax').value),
            bandwidth_min: parseInt(document.getElementById('bwMin').value),
            bandwidth_max: parseInt(document.getElementById('bwMax').value),
            signal_center_freq_min: -2500000, // Derived automatically based on custom spans
            signal_center_freq_max: 2499999,
            frequency_min: -2500000,
            frequency_max: 2499999,
            classes: selectedClasses
        }
    };

    try {
        const response = await fetch('/api/v1/save-grid-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Server transmission error.");

        const blob = await response.blob();
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'torchsig_config.yaml';
        if (contentDisposition && contentDisposition.includes('filename=')) {
            // 1. Split on 'filename=' to isolate the text on the right
            const parts = contentDisposition.split('filename=');
            if (parts.length > 1) {
                // 2. [1] selects the actual name item, and .replace removes any accidental quotes or spaces
                filename = parts[1].replace(/['"]/g, '').trim();
            }
        }

        const downloadUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();

        document.body.removeChild(link);
        window.URL.revokeObjectURL(downloadUrl);

        statusDiv.classList.remove('hidden', 'bg-red-900/50', 'text-red-200');
        statusDiv.classList.add('bg-emerald-900/50', 'text-emerald-200');
        statusDiv.innerText = `Successfully downloaded: ${filename}`;

    } catch (error) {
        console.error(error);
        statusDiv.classList.remove('hidden', 'bg-emerald-900/50', 'text-emerald-200');
        statusDiv.classList.add('bg-red-900/50', 'text-red-200');
        statusDiv.innerText = "Error compiling file payload metrics.";
    }
});
