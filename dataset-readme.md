# Diva Preset Audio Dataset

## Introduction
This is a dataset of thousands of presets of [Diva
Synthesizer](https://u-he.com/products/diva/) with their *Character* tags.

The dataset is based on a [JSON
dataset](https://raw.githubusercontent.com/acids-ircam/flow_synthesizer/75153bf6e2c97752d61ca166631be3882c5517dc/code/dataset.json)
provided by [Flow
Synthesizer](https://github.com/acids-ircam/flow_synthesizer/). We took the
parameter settings from the dataset and rendered the audio using
[RenderMan](https://github.com/fedden/RenderMan).

When processing the data, we removed the presets that don't come with any
character tag.

## Specification
For each audio file:
- audio length: 3 seconds
- note playing: 2 seconds
- sample rate: 44100
- midi note: 60
- midi velocity: 127

## Meta Data
The meta data are in `meta.csv`. The columns are:
```
Bright, Dark, Dynamic, Static, Constant, Moving, Soft, Aggressive, Harmonic, Inharmonic, Phat, Thin, Clean, Dirty, Wide, Narrow, Modern, Vintage, Acoustic, Electric, Natural, Synthetic, audio path
```

The first 22 columns represent *Character* tags. The tags are in pairs and the
selection in a pair is exclusive. The number `1` means it's selected, `0` means
it's not selected.

The following table shows the pairing:
| Bright   | Dark       |
| Dynamic  | Static     |
| Constant | Moving     |
| Soft     | Aggressive |
| Harmonic | Inharmonic |
| Phat     | Thin       |
| Clean    | Dirty      |
| Wide     | Narrow     |
| Modern   | Vintage    |
| Acoustic | Electric   |
| Natural  | Synthetic  |

The last column represents the relative path of the preset audio, which is the
same as the path of the corresponding `h2p` preset file.
