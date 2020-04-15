import unittest
from hyperCosine.sample_factory import SampleCreator

class HyperbolicLawOfCosine2Dim(unittest.TestCase):
    def test_hyberbolic_law_of_cosine_2dim(self):
        sampleCreator = SampleCreator(2, 0.2)
        samples = sampleCreator.create(100)


if __name__ == '__main__':
    unittest.main()