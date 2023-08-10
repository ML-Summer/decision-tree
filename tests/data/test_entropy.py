import pytest
from pytest import approx
from data import entropy

class testEntropy():
    def testOnThreeToTwoSplit():
        labels = [1, 1, 0, 1, 0]
        assert entropy(labels) == approx(0.9710)
    def testOnSingleLabelInList():
        labels = [1, 1, 1, 1]
        assert entropy(labels) == approx(-0.0000)
    def testOnEqualSplit():
        labels = [1, 1, 0, 1, 0, 0]
        assert entropy(labels) == approx(1.0000)

pytest.main()