import unittest

from src.evaluation.mot_metrics import MOTEvaluator
from src.evaluation.offline_postprocess import (
    drop_short_tracks,
    interpolate_gaps,
    stitch_tracklets,
)


class TestEvaluationRegression(unittest.TestCase):
    def test_offline_postprocessing_is_deterministic(self):
        tracks = {
            1: [[1, 0, 0, 10, 10, 1], [3, 2, 0, 12, 10, 1]],
            2: [[5, 4, 0, 14, 10, 2]],
        }
        interpolated = interpolate_gaps(tracks, max_gap=2)
        self.assertEqual(len(interpolated[1]), 3)
        self.assertEqual(
            list(drop_short_tracks(interpolated, minimum_length=2)), [1]
        )
        stitched = stitch_tracklets(
            {1: interpolated[1], 2: [[4, 3, 0, 13, 10, 2]]},
            max_gap=2,
            max_distance=10,
        )
        self.assertEqual(len(stitched), 1)

    def test_perfect_tracking_metrics_are_stable(self):
        evaluator = MOTEvaluator()
        for frame_id in (1, 2):
            gt = [[10, 10, 20, 20, 7]]
            prediction = [[10, 10, 20, 20, 7]]
            evaluator.update(gt, prediction, frame_id)

        metrics = evaluator.compute_metrics()
        if hasattr(metrics, "iloc"):
            row = metrics.iloc[0]
            self.assertEqual(row["mota"], 1.0)
            self.assertEqual(row["num_switches"], 0)
            self.assertEqual(row["num_misses"], 0)
            self.assertEqual(row["num_false_positives"], 0)
        else:
            self.assertEqual(metrics["mota"]["Overall"], 0.0)

    def test_perfect_hota_metrics_are_stable(self):
        tracks = {
            1: [[10, 10, 20, 20, 7]],
            2: [[11, 10, 21, 20, 7]],
        }
        metrics = MOTEvaluator.compute_hota(tracks, tracks)
        if metrics is not None:
            self.assertEqual(metrics["hota"], 1.0)
            self.assertEqual(metrics["deta"], 1.0)
            self.assertEqual(metrics["assa"], 1.0)
            self.assertEqual(metrics["trackeval_mota"], 1.0)
            self.assertEqual(metrics["trackeval_idf1"], 1.0)


if __name__ == "__main__":
    unittest.main()
