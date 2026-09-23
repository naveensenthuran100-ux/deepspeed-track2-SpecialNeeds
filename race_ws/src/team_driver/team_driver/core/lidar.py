"""Turning a raw scan into something it is safe to steer by.

The circuit's boundary is a line of separate 33 cm air ducts, not a wall
(docs/02-simulator.md 2.1). Two consequences shape this whole module:

  * A beam can slip between two ducts and report the far side of the track,
    or nothing at all. The scan therefore contains apparent openings the car
    cannot fit through, and steering at the deepest reading aims at a duct.
  * Going between two ducts is leaving the course, not a shortcut.

`disparity_extend` is the fix for both, and it matters more here than it does
on Track 1 because a duct row produces near/far jumps continuously rather than
only at corners.
"""

import numpy as np

# The devkit's scan, for reference: 1080 beams, -2.356..+2.356 rad, 0.06-10 m.
DEFAULT_MAX_RANGE = 10.0


def clean(ranges, max_range=DEFAULT_MAX_RANGE):
    """Finite, bounded ranges. Never mutates the caller's array.

    The LiDAR reports NaN for a beam that hit nothing and inf past its range,
    and arithmetic on either propagates silently through everything
    downstream - a NaN steering angle is published as a NaN and the car holds
    its last command into a wall.
    """
    array = np.asarray(ranges, dtype=np.float64)
    out = np.nan_to_num(array, nan=0.0, posinf=float(max_range), neginf=0.0)
    return np.clip(out, 0.0, float(max_range))


def beam_angles(angle_min, angle_increment, count):
    """The angle of each beam, in the car's frame. Positive is to the left."""
    return float(angle_min) + np.arange(int(count)) * float(angle_increment)


def disparity_extend(ranges, angles, car_half_width, threshold=0.3):
    """Pull in the far side of every near/far jump by half a car width.

    Where the scan steps from near to far there is an object the LiDAR can see
    past but the car cannot fit past. At the near range `d`, half a car width
    subtends `atan(w / d)`, so the near reading is extended across that many
    beams on the far side of the jump.

    The result is a scan describing where the *car* can go rather than where
    light can go. Ranges only ever decrease.

    Args:
        ranges: cleaned ranges, as from `clean`.
        angles: matching beam angles.
        car_half_width: half the car's width plus whatever margin you want.
        threshold: the step, in metres, that counts as a disparity.
    """
    out = np.array(ranges, dtype=np.float64, copy=True)
    count = out.shape[0]
    if count < 2:
        return out

    increment = abs(float(angles[1]) - float(angles[0]))
    if increment <= 0.0:
        return out

    steps = out[1:] - out[:-1]
    disparities = np.nonzero(np.abs(steps) > float(threshold))[0]

    for index in disparities:
        if out[index] < out[index + 1]:
            near, start, forward = out[index], index + 1, True
        else:
            near, start, forward = out[index + 1], index, False

        # A zero or near-zero reading has no meaningful angular width, and
        # atan would hand us most of the scan.
        if near <= 1e-6:
            continue

        span = int(np.ceil(np.arctan2(float(car_half_width), float(near)) / increment))
        if span <= 0:
            continue

        if forward:
            low, high = start, min(count, start + span + 1)
        else:
            low, high = max(0, start - span), start + 1
        out[low:high] = np.minimum(out[low:high], near)

    return out


def safety_bubble(ranges, angles, radius_m, max_half_angle=0.8):
    """Blank an arc around the closest return.

    The car must never aim at a point it would clip on the way to, and the
    closest return is the one it is most likely to clip.

    `max_half_angle` caps how much of the scan a single very close return may
    erase. Without it, something 0.3 m away blanks the entire scan and leaves
    the gap follower with nowhere to aim - which turns a near miss into a
    stop. The bubble's job is to stop us steering *at* the obstacle, not to
    erase the world.
    """
    out = np.array(ranges, dtype=np.float64, copy=True)
    valid = out > 0.0
    if not np.any(valid):
        return out

    closest = int(np.argmin(np.where(valid, out, np.inf)))
    nearest = out[closest]
    if nearest <= 0.0:
        return out

    half_angle = min(np.arctan2(float(radius_m), float(nearest)), float(max_half_angle))
    out[np.abs(np.asarray(angles) - angles[closest]) <= half_angle] = 0.0
    return out
