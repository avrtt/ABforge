import unittest
import pandas as pd
import numpy as np
from abforge.stats.metrics import (
    Distribution,
    ConversionDistribution,
    RevenueDistribution,
    ARPUDistribution,
)
from abforge.stats.constants import SAMPLE_SIZE


class DistributionTestCase(unittest.TestCase):
    def test_sample(self):
        dist = Distribution()
        self.assertRaises(NotImplementedError, dist.sample)

    def test_update(self):
        dist = Distribution()
        self.assertRaises(NotImplementedError, dist.update)

    def test_chance_to_beat_winner(self):
        dist = Distribution()
        samplingA = np.array([0.1, 0.1, 0.1])
        samplingB = np.array([0.2, 0.2, 0.2])
        self.assertEqual(dist.chance_to_beat(samplingB, samplingA, 3), 1.0)

    def test_chance_to_beat_loser(self):
        dist = Distribution()
        samplingA = np.array([0.1, 0.1, 0.1])
        samplingB = np.array([0.1, 0.08, 0.09])
        self.assertEqual(dist.chance_to_beat(samplingB, samplingA, 3), 0.0)

    def test_chance_to_beat_almost_winner(self):
        dist = Distribution()
        samplingA = np.array([0.1, 0.1, 0.1])
        samplingB = np.array([0.1, 0.2, 0.2])
        self.assertEqual(dist.chance_to_beat(samplingB, samplingA, 3), 2 / 3)

    def test_expected_loss_winner(self):
        dist = Distribution()
        samplingA = np.array([0.1, 0.3, 0.1])
        samplingB = np.array([0.2, 0.2, 0.2])
        diff = np.array([-0.1, 0.1, -0.1])
        expected_loss = round(sum(diff * (diff > 0)) / 3, 2)

        self.assertEqual(
            round(dist.expected_loss(samplingB, samplingA, 3), 2), expected_loss
        )


if __name__ == "__main__":
    unittest.main(verbosity=1)
