import torch

from furiosa.torch._C.config import stream as config
from furiosa.torch._C.stream import Stream


def current_stream(device: torch.device | str) -> Stream:
    """Get the current stream for `device`.

    If no stream has been explicitly set, the current stream is the default stream for that
    device. NPU operations are submitted to the current stream of their target device.

    Args:
        device (torch.device or str): Furiosa device to query.

    Returns:
        Stream: The stream currently selected for the device.
    """
    return config.current_stream(device)


def default_stream(device: torch.device | str) -> Stream:
    """Get the default stream for `device`.

    The device keeps this stream alive for as long as it is initialized, so repeated calls return
    handles to the same default stream.

    Args:
        device (torch.device or str): Furiosa device to query.

    Returns:
        Stream: The default stream for the device.
    """
    return config.default_stream(device)


def set_stream(stream: Stream) -> Stream:
    """Set `stream` as the current stream for its device and return the previously current stream.

    Save the returned stream to restore it later.

    Args:
        stream (Stream): Stream to make current for its device.

    Returns:
        Stream: The previously current stream for the same device.
    """
    return config.set_stream(stream)
