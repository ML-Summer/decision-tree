import pytest
from pytest import approx
from src.data.entropy import entropy

class TestEntropy():
    def testOnThreeToTwoSplit(self):
        labels = [1, 1, 0, 1, 0]
        assert entropy(labels) == approx(0.9710)
    def testOnSingleLabelInList(self):
        labels = [1, 1, 1, 1]
        assert entropy(labels) == approx(-0.0000)
    def testOnEqualSplit(self):
        labels = [1, 1, 0, 1, 0, 0]
        assert entropy(labels) == approx(1.0000)

pytest.main()