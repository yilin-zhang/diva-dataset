import numpy as np
import pickle

import librenderman as rm
from dataset_parser import get_patch, character_to_binary
# WORKAROUND: This is a hacky solution to get Essentia on macOS
# Essentia only gets installed successfully by homebrew
# https://github.com/MTG/essentia/issues/777
# https://gist.github.com/jarmitage/40e3b7962b0a77a233b2d590d69378d6
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages/')

from essentia import *
from essentia.standard import *


# Important settings. These are good general ones.
sampleRate = 44100
bufferSize = 512
fftSize = 512

# Load the plugin
engine = rm.RenderEngine(sampleRate, bufferSize, fftSize)
path = "/Library/Audio/Plug-Ins/VST/u-he/Diva.vst"
engine.load_plugin(path)

generator = rm.PatchGenerator(engine)

# Note settings
midiNote = 60
midiVelocity = 127
noteLength = 4.0
renderLength = 5.0

# Essentia settings
w = Windowing()
spec = Spectrum()

rms_extractor = RMS()
flux_extractor = Flux()
# centroid_extractor = Centroid(range=22050)
centroid_extractor = Centroid()
rolloff_extractor = RollOff()
decrease_extractor = Decrease()
energy_extractor = Energy()
crest_extractor = Crest()
hfc_extractor = HFC()
strongpeak_extractor = StrongPeak()
zerocrossingrate_extractor = ZeroCrossingRate()
envelope_extractor = Envelope()
logattacktime_extractor = LogAttackTime()


features = []
binary_characters = []

i = 1

# Iterate
for patch, path, character in get_patch("dataset/dataset.json"):
    if not character:
        continue

    binary_character = character_to_binary(character)
    if binary_character == [0]*22:
        continue

    engine.set_patch(patch)
    engine.render_patch(midiNote, midiVelocity, noteLength, renderLength)
    audio = np.array(engine.get_audio_frames(), dtype=np.float32)

    feature_array = []
    for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
        spectrum = spec(w(frame))
        feature_array.append([
            rms_extractor(spectrum),
            flux_extractor(spectrum),
            centroid_extractor(spectrum),
            rolloff_extractor(spectrum),
            decrease_extractor(spectrum),
            energy_extractor(spectrum),
            crest_extractor(spectrum),
            hfc_extractor(spectrum),
            strongpeak_extractor(spectrum),
            zerocrossingrate_extractor(frame),
        ])

    feature_array = np.array(feature_array, dtype=np.float32)
    feature_means = np.mean(feature_array, axis=0)
    feature_stdevs = np.std(feature_array, axis=0)

    # add attack time feature
    feature_means = np.append(feature_means, logattacktime_extractor(envelope_extractor(audio))[0])
    features.append(np.concatenate((feature_means, feature_stdevs)))

    #mean_mfcc = np.mean(np.array(engine.get_mfcc_frames()), axis=0)
    #print(features)
    #mean_mfcc_array.append(mean_mfcc)
    binary_characters.append(binary_character)

    print("Iteration:", i)
    i += 1

features = np.array(features)
binary_characters = np.array(binary_characters)

# Dump the data
with open('dataset/features.pkl', 'wb') as f:
    pickle.dump(features, f)

with open('dataset/binary_characters.pkl', 'wb') as f:
    pickle.dump(binary_characters, f)
