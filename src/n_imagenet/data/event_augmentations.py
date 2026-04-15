import numpy as np
import torch

from .imagenet_constants import *


def random_shift_events(event_tensor, max_shift=20, resolution=(IMAGE_H, IMAGE_W)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    event_tensor[:, 0] += x_shift
    event_tensor[:, 1] += y_shift

    valid_events = (
        (event_tensor[:, 0] >= 0)
        & (event_tensor[:, 0] < W)
        & (event_tensor[:, 1] >= 0)
        & (event_tensor[:, 1] < H)
    )
    event_tensor = event_tensor[valid_events]

    return event_tensor


def random_flip_events_along_x(event_tensor, resolution=(IMAGE_H, IMAGE_W), p=0.5):
    H, W = resolution

    if np.random.random() < p:
        event_tensor[:, 0] = W - 1 - event_tensor[:, 0]

    return event_tensor


def random_time_flip(event_tensor, resolution=(IMAGE_H, IMAGE_W), p=0.5):
    if np.random.random() < p:
        event_tensor = torch.flip(event_tensor, [0])
        event_tensor[:, 2] = event_tensor[0, 2] - event_tensor[:, 2]
        event_tensor[:, 3] = -event_tensor[
            :, 3
        ]  # Inversion in time means inversion in polarity
    return event_tensor


def default_event_augmentations(event):
    event = random_time_flip(event)
    event = random_flip_events_along_x(event)
    event = random_shift_events(event)
    return event
