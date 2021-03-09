import random
from typing import List, Tuple, Any
import torch
import numpy as np

from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc import dispatcher

from audio_feature_extractor import AudioFeatureExtractor
from model import MelDataset
from receive_audio_buffer import UdpBufferReceiver

# TODO: remove this, just for test
from scipy.io.wavfile import write as wavwrite


class OSCHandler:
    def __init__(self, feature_extractor: AudioFeatureExtractor):
        self._num_parameters = 0
        self._library_info = {}
        self._feature_extractor = feature_extractor

    # manage the library data construction
    def add_library_info(self, preset_path: str, buffer: np.array):
        buffer = torch.from_numpy(buffer)
        buffer = buffer.reshape((1, -1))
        preset_feature = self._feature_extractor.encode(buffer)
        self._library_info[preset_path] = preset_feature
        print(f'num info: {len(self._library_info)}')

    def save_library_info_database(self):
        pass


def analyze_library_callback(address: str, args: List[Any], *osc_args: List[Any]) -> None:
    client, osc_handler, udp_buffer_receiver = args
    value = osc_args[0]
    preset_path = osc_args[1]
    print(f'preset_path: {preset_path}, type: {type(preset_path)}')
    print(f'osc value: {value}, value type: {type(value)}')

    # receive an audio buffer
    if value == 1:
        buffer = udp_buffer_receiver.receive()
        # print(f'buffer: {buffer}')
        # wavwrite(f'test_audio/test_{len(osc_handler._library_info)}.wav', 44100, buffer) # TODO: remove this
        osc_handler.add_library_info(preset_path, buffer)
        client.send_message("/Ideator/cpp/analyze_library", 1)

    # data receiving is over, save the library data
    elif value == 2:
        # TODO: save the library data into a file
        pass


if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    client = udp_client.SimpleUDPClient(address="127.0.0.1", port=9001)
    feature_extractor = AudioFeatureExtractor('models/auto-encoder-20201020050003.pt') # NOTE: hard coded here
    udp_buffer_receiver = UdpBufferReceiver(address="127.0.0.1", port=8888)
    osc_handler = OSCHandler(feature_extractor)
    # TODO: maybe put this into the constructor?
    dataset = MelDataset(flat=False)
    feature_extractor.encode_dataset(dataset)

    dispatcher.map("/Ideator/python/analyze_library", analyze_library_callback, client, osc_handler, udp_buffer_receiver)

    # obsolete
    # dispatcher.map("/Ideator/python/num_parameters", set_num_parameters_callback, osc_handler)
    # dispatcher.map("/Ideator/python/get_random_patch", get_random_patch_callback, client, osc_handler)
    # dispatcher.map("/Ideator/python/save_audio/*", render_audio_callback, osc_handler, feature_extractor)

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 7777), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
