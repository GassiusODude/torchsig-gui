# TorchSig GUI

From [TorchSig Issue 283](https://github.com/TorchDSP/torchsig/issues/283), there is a desire for a GUI to set up a configuration file.

Example features

* Per Signal
  * [ ] bandwidth on a per signal/signal-class basis
  * [ ] SNR on a per signal
* [x] FFT size
* [x] Number of IQ samples
* [x] Selection of signal or signal classes
* [ ] Impairments
* [ ] Transforms
* [ ] Probability distributions per signal or per signal class
* [ ] Probability distributions per number of signals (wideband case only)
* [x] Narrowband and/or wideband dataset to generate
* [x] Number of samples in the dataset
* [x] Sample Rate
* [x] Impairment Level

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