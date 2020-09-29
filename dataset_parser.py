from config import PARAM_DICT, DEFAULT_PARAMS, CHARACTER_TABLE
import json


def get_patch(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)
        for preset in data.values():
            patch = []
            if "Character" in preset["Meta"]:
                character = preset["Meta"]["Character"]
            else:
                character = []

            if "Preset dir" in preset:
                path = preset["Preset dir"]
            else:
                path = ''

            midi = preset["MIDI"]
            for key in PARAM_DICT:
                if PARAM_DICT[key] in midi:
                    patch.append((key, float(midi[PARAM_DICT[key]])))
                else:
                    patch.append((key, DEFAULT_PARAMS[PARAM_DICT[key]]))

            yield patch, path, character


def character_to_binary(character):
    ''' Convert a list of descriptors into a binary list
    Arg:
    - character: a list of character parsed from the dataset
    Return
    - A binray list
    '''
    binary = [0] * (len(CHARACTER_TABLE) * 2)
    for idx, pair in enumerate(CHARACTER_TABLE):
        word_1, word_2 = pair
        if word_1 in character:
            binary[idx*2] = 1
        elif word_2 in character:
            binary[idx*2+1] = 1

    return binary
