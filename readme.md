# Diva Dataset

This repository contains data and scripts for exporting, analyzing the presets
of Diva synthesizer.

## Explanation of Files and Directories

Data
- `dataset/dataset.json`: This file contains all the information about preset
  parameters and labels. It comes from
  [dataset.json](https://raw.githubusercontent.com/acids-ircam/flow_synthesizer/75153bf6e2c97752d61ca166631be3882c5517dc/code/dataset.json)
  in [flow synthesizer
  repository](https://github.com/acids-ircam/flow_synthesizer).

Executable python scripts:
- `export_dataset.py`: Exports the dataset to `diva-preset-audio-dataset/`
  directory. You can also uncomment `dump_features()` to dump audio features
  directly to `dataset/features.pkl`.
- `plot_features.py`: Plots the features extracted by `dump_features()`
  function.
- `extract_mel.py`: Uses the data in the `diva-preset-audio-dataset/` directory,
  select all the presets that have at least one timbre label, extracts the
  Mel-spectrogram and the labels, dumps `dataset/mel.pkl` and
  `dataset/labels.pkl`.
- `analyze_dataset`: Analyzes the labels dumped as `dataset/labels.pkl`.
- `model.py`: Defines several machine learning models and uses `dataset/mel.pkl`
  and `dataset/labels.pkl` as training data. Use `python model.py <model name>`
  to run the script. The model will be put into `models/` directory.
- `analyze_latent.py`: Analyzes the latent space of the auto-encoder model.

Utility script:
- `dataset_parser.py`: Provides utility functions to parse `dataset/dataset.json`
  and convert the label format.
- `config.py`: Some constant definitions.

Other scripts
- `osc_handler.py`: Handles the OSC message communication with Ideator plugin
  host.
- `preset_parser.py`: Dedicates to parse the original preset format used by Diva
  synthesizer. The function is not complete for now.

## Configuration
To export the audio, you have to compile RenderMan.

Configuration is for MacOS:
1. Install boost libs: `brew install boost-python3` (My current version is 1.73.0)
2. Install JUCE (the version should be JUCE 5)
3. Make sure have python3.8 installed
4. Get RenderMan from https://github.com/fedden/RenderMan
5. Open `RenderMan-py36.jucer` in the project folder, generate XCode project.
6. Configure the search paths in XCode (This could be different, find the corresponding paths on your machine):
   - Boost dylib path: `/usr/local/Cellar/boost-python3/1.73.0/lib`
   - Python dylib path: `/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/lib`
   - Python header path: `/usr/local/Cellar/python@3.8/3.8.5/Frameworks/Python.framework/Versions/3.8/include/python3.8`
7. Set the link flags: `-shared -lboost_python38 -undefined dynamic_lookup`
7. Build a release build `librenderman.so.dylib`, remove the extra `.dylib` extension.
8. Copy the `librenderman.so` to your python project root directory.
