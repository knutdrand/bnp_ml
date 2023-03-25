from bnp_ml.events import RandomVariable, Event, Probability, P
import pytest


@pytest.fixture
def dice():
    return RandomVariable({i: Probability(1/6) for i in range(1, 7)})


@pytest.fixture
def dice_2(dice):
    return Event(dice, 2)


@pytest.fixture
def dice_3(dice):
    return Event(dice, 3)


def test_random_variable(dice):
    assert P(dice == 2).equals(1/6)


def test_event_or(dice_2, dice_3):
    assert P(dice_2 | dice_3)
