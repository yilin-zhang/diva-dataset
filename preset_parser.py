import csv
import re
import pprint

# load lookup table
mapping_table = []
with open("param_mapping.csv") as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        mapping_table.append(row)


###########################################3
# meta data

# /*@Meta

# Author:
# 'Howard Scarr'

# Usage:
# 'mod wheel = shorter\r\npressure = tone'

# Categories:
# 'Seq & Arp:Effects, Seq & Arp:Evolving'

# Features:
# 'Mono, HostSync, Modulated, CrossMod, Percussive'

# Character:
# 'Dirty, Thin, Moving, Natural, Inharmonic, Static'

# */

def get_meta_string(preset_data):
    return re.search(r'/\*@Meta((.|\n|\r)*)\*/', preset_data).group(0)

def parse_meta(preset_data):
    meta_data = get_meta_string(preset_data)
    meta_lines = meta_data.split('\n')
    meta_dict = {}

    try:
        meta_dict['author'] = meta_lines[
            meta_lines.index('Author:') + 1].strip("'")
    except ValueError:
        print("No author")

    try:
        meta_dict['usage'] = meta_lines[
            meta_lines.index('Usage:') + 1].strip("'")
    except ValueError:
        print("No usage")

    try:
        meta_dict['description'] = meta_lines[
            meta_lines.index('Description:') + 1].strip("'")
    except ValueError:
        print("No description")

    try:
        meta_dict['categories'] = meta_lines[
            meta_lines.index('Categories:') + 1].strip("'")
    except ValueError:
        print("No categories")

    try:
        meta_dict['features'] = meta_lines[
            meta_lines.index('Features:') + 1].strip("'")
    except ValueError:
        print("No features")

    try:
        meta_dict['character'] = meta_lines[
            meta_lines.index('Character:') + 1].strip("'")
    except ValueError:
        print("No character")

    return meta_dict


#####################################################
# parameters

# remove the meta data section
def parse_params(preset_data):
    meta_data = get_meta_string(preset_data)
    param_data = ''
    if preset_data.startswith(meta_data):
        param_data = preset_data[len(meta_data):]

    # remove the compressed data section
    for match in re.finditer("// Section for ugly compressed binary Data",
                            param_data):
        param_data = param_data[:match.span()[0]]

    # remove trailing spaces and returns
    param_data = param_data.strip()
    param_lines = param_data.split('\n')

    param_dict = {}

    entry_name = ''
    for line in param_lines:
        match = re.search('#cm=(.*)', line)
        if match:
            entry_name = match.group(1)
            param_dict[entry_name] = {}
            # print(entry_name)
        else:
            if not entry_name == '':
                param_name, param_val = line.split('=')
                if not re.search('[A-Za-z]', param_val):
                    param_val = float(param_val)
                else:
                    param_val = param_val.strip("'")
                param_dict[entry_name][param_name] = param_val

    return param_dict


def parse_preset(preset_path):
    # load preset
    with open(preset_path) as f:
        preset_data = f.read()

    meta_dict = parse_meta(preset_data)
    param_dict = parse_params(preset_data)
    preset_dict = {'meta': meta_dict, 'params': param_dict}

    return preset_dict


def get_mapped_patch(preset_path):
    with open(preset_path) as f:
        preset_data = f.read()
    param_dict = parse_params(preset_data)
    patch = []
    for row in mapping_table:
        group, name, num = row
        num = int(num)
        # check if the parameter exists in the preset
        if (group in param_dict) and (name in param_dict[group]):
            # handle the special case
            if num in (178, 217):
                special_dict = {
                    'Chorus1': 0,
                    'Chorus2': 0,
                    'Phaser1': 1,
                    'Phaser2': 1,
                    'Plate1': 2,
                    'Plate2': 2,
                    'Delay1': 3,
                    'Delay2': 3,
                    'Rotary1': 4,
                    'Rotary2': 4,
                }
                patch.append((num, special_dict[param_dict[group][name]]))
            else:
                patch.append((num, param_dict[group][name]))
        else:
            # set 0 by default
            patch.append((num, 0.0))

    return patch


# pp = pprint.PrettyPrinter(indent=2)
# pp.pprint(parse_preset("HS Guirotron.h2p"))
# pp.pprint(parse_preset("HS Guirotron.h2p"))
# pp.pprint(get_mapped_preset("HS Guirotron.h2p"))
