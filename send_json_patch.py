from pythonosc import udp_client
from pythonosc import dispatcher
from dataset_parser import get_patch


def send_json_patch(client):
    counter = 0
    for patch, path, character in get_patch("dataset/dataset.json"):
        # only patch is used
        if not character:
            continue
        flatten_patch = [item for pair in patch for item in pair]
        concat_character = ','.join(character)
        client.send_message("/Ideator/cpp/json_patch", [path, concat_character] + flatten_patch)
        counter += 1
        print(f"{counter}: Preset path: {path}")


if __name__ == '__main__':
    # TODO: not sure if dispatcher is necessary
    # dispatcher = dispatcher.Dispatcher()
    # dispatcher.map("/Ideator/python/num_parameters", set_num_parameters_callback, osc_handler)
    client = udp_client.SimpleUDPClient(address="127.0.0.1", port=9001)
    send_json_patch(client)