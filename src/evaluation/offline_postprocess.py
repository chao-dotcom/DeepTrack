"""Offline tracklet post-processing utilities.

These operations are intentionally separate from online tracking and must be
reported as offline when used for evaluation.
"""
from collections import defaultdict
from typing import Dict, List


def drop_short_tracks(
    tracks: Dict[int, List[List[float]]], minimum_length: int = 5
) -> Dict[int, List[List[float]]]:
    """Remove track IDs observed fewer than ``minimum_length`` frames."""
    return {
        track_id: sorted(observations)
        for track_id, observations in tracks.items()
        if len(observations) >= minimum_length
    }


def interpolate_gaps(
    tracks: Dict[int, List[List[float]]], max_gap: int = 10
) -> Dict[int, List[List[float]]]:
    """Linearly fill missing frame boxes inside each track."""
    result = {}
    for track_id, observations in tracks.items():
        ordered = sorted(observations)
        by_frame = {int(item[0]): item for item in ordered}
        filled = list(ordered)
        for first, second in zip(ordered, ordered[1:]):
            start, end = int(first[0]), int(second[0])
            gap = end - start - 1
            if gap <= 0 or gap > max_gap:
                continue
            for frame in range(start + 1, end):
                ratio = (frame - start) / (end - start)
                box = [
                    first[index] + ratio * (second[index] - first[index])
                    for index in range(1, 5)
                ]
                filled.append([frame, *box, track_id])
        result[track_id] = sorted(filled, key=lambda item: item[0])
    return result


def stitch_tracklets(
    tracks: Dict[int, List[List[float]]],
    max_gap: int = 5,
    max_distance: float = 50.0,
) -> Dict[int, List[List[float]]]:
    """Merge nearby tracklets with a gap using endpoint center distance."""
    merged = {track_id: list(observations) for track_id, observations in tracks.items()}
    changed = True
    while changed:
        changed = False
        ids = sorted(merged)
        for left_id in ids:
            if left_id not in merged:
                continue
            left = sorted(merged[left_id])
            for right_id in ids:
                if right_id == left_id or right_id not in merged:
                    continue
                right = sorted(merged[right_id])
                gap = int(right[0][0]) - int(left[-1][0]) - 1
                if gap < 0 or gap > max_gap:
                    continue
                left_center = ((left[-1][1] + left[-1][3]) / 2,
                               (left[-1][2] + left[-1][4]) / 2)
                right_center = ((right[0][1] + right[0][3]) / 2,
                                (right[0][2] + right[0][4]) / 2)
                distance = (
                    (left_center[0] - right_center[0]) ** 2
                    + (left_center[1] - right_center[1]) ** 2
                ) ** 0.5
                if distance <= max_distance:
                    merged[left_id] = left + right
                    del merged[right_id]
                    changed = True
                    break
            if changed:
                break
    return merged
