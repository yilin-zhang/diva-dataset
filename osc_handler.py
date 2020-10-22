import random
import time
import random
from typing import List, Union, Any

from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc import dispatcher


class OSCHandler:
    def __init__(self):
        self._num_parameters = 0

    @property
    def num_parameters(self) -> int:
        return self._num_parameters

    @num_parameters.setter
    def num_parameters(self, value: int) -> None:
        if value < 0:
            self._num_parameters = 0
        else:
            self._num_parameters = value

    def generate_random_patch(self) -> List[Union[int, float]]:
        patch = []
        for i in range(self._num_parameters):
            patch.append(i)
            patch.append(random.random())
        return patch


def set_num_parameters_callback(address: str, args: List[Any], *osc_args: List[Any]) -> None:
    value = osc_args[0]
    osc_handler = args[0]
    osc_handler.num_parameters = value
    print("num_parameters set")


def get_random_patch_callback(address: str, args: List[Any], *osc_args: List[Any]) -> None:
    client, osc_handler = args
    value = osc_args[0]
    print(f'osc value: {value}, value type: {type(value)}')
    if value == 1:
        print("generate a random patch!")
        random_patch = osc_handler.generate_random_patch()
        client.send_message("/Ideator/cpp/set_parameter", random_patch)


if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    client = udp_client.SimpleUDPClient(address="127.0.0.1", port=9001)
    osc_handler = OSCHandler()

    dispatcher.map("/Ideator/python/num_parameters", set_num_parameters_callback, osc_handler)
    dispatcher.map("/Ideator/python/get_random_patch", get_random_patch_callback, client, osc_handler)

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 7777), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
