# TorchSig GUI

## Config

Based on the example `yaml_dataset_example.ipynb` the main parts of save/load YAML is in `torchsig.utils.yaml` and `torchsig.utils.defaults`

~~~python
    from torchsig.utils.defaults import default_dataset
    from torchsig.utils.yaml import save_dataset_yaml, load_dataset_yaml
~~~

~~~yaml
    "seed": 12345678,
    "target_labels":
        - "2fsk"
        - "2ask"
    "dataset_metadata":
        "fft_size": 256
    "dataset_length": 2000
    "impairment_level": 0
    "output":
        "representation": "iq" # "iq" or "spectrogram"
    "signal_sampling":
        "mode": "per_signal"    # "per_signal" or "per_family"
~~~

# Running

## Launching FastAPI Server - Configure

Launch the server to use the webpage and configure your dataset. Click the `Build & Save Target Dataset Configuration` to save the output YAML.

~~~bash
    uvicorn.exe torchsig_gui.main:app
~~~

## Streamlit

~~~bash
    # Installing Streamlit
    python -m venv venv
    source venv/bin/activate  # Linux
    python -m pip install streamlit

    # Running Streamlit
    streamlit run .\src\torchsig_gui\sl_main.py
~~~

## Generate Data and Train Model

~~~mermaid
classDiagram

class XCiT1d
class XCitClassifier
XCiT1d o-- XCitClassifier
~~~

* XCiTClassifier(input_channels=2, num_classes, ds_method, ds_rate)
  * Number of channels is set to 2 based on the `ComplexTo2D` transform.
  * Number of signals -> number of classes
* XCiT1D(model_name, pretrained, num_classes, in_chans,)

~~~bash
    python .\script_gen_dataset.py family.yaml
        --lr 1e-4 --validate_period 5 --epochs 20
        --ds_type chunk --ds_rate 8 --model model.ckpt
~~~