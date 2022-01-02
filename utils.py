import matplotlib.pyplot as plt
import torch

COLORS = [[0,0,0], [255,0,0], [0,0,255]]

MIN_VAL = -512
MAX_VAL = 512

def plot_scores(values, path):
    plt.plot(range(len(values)), values)
    plt.savefig(path)
    plt.clf()


def get_3c_grayscale(volume, hu_min=MIN_VAL, hu_max=MAX_VAL):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = torch.clamp(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = torch.max(volume)
    mnval = torch.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return torch.stack((im_volume, im_volume, im_volume), dim=-1)


def class_to_color(segmentation, colors):
    # initialize output to zeros
    seg_color = torch.zeros(segmentation.shape + (3,), device=segmentation.device)

    # set output to appropriate color at each location
    for i, c in enumerate(colors):
        if i > 0:
            seg_color[segmentation == i] = torch.tensor(c, dtype=seg_color.dtype, device=seg_color.device)
    return seg_color


def overlay(ct_volume, label_volume, alpha=0.3):
    # ct_volume.shape == label_volume.shape ==  (slices, h, w)
    # Get binary array for places where an ROI lives
    ct_volume = get_3c_grayscale(ct_volume)

    label_color_volume = class_to_color(label_volume, COLORS)

    segbin = torch.greater(label_volume, 0)
    repeated_segbin = torch.stack((segbin, segbin, segbin), dim=-1)
    # Weighted sum where there's a value to overlay
    overlayed = torch.where(
        repeated_segbin,
        alpha*label_color_volume + (1-alpha)*ct_volume,
        ct_volume
    )
    overlayed = overlayed.permute(0, 3, 1, 2)
    return overlayed


