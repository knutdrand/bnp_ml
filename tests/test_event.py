from bnp_ml.events import DictRandomVariable, Event, Probability, P
import pytest


@pytest.fixture
def dice():
    return DictRandomVariable({i: Probability(1/6) for i in range(1, 7)})


@pytest.fixture
def coin():
    return DictRandomVariable({i: Probability(1/2) for i in ('H', 'T')})


@pytest.fixture
def dice_2(dice):
    return Event(dice, 2)


@pytest.fixture
def dice_3(dice):
    return Event(dice, 3)


@pytest.fixture
def coin_heads(coin):
    return Event(coin, 'H')


def test_random_variable(dice):
    assert P(dice == 2).equals(1/6)


def test_event_or(dice_2, dice_3):
    assert P(dice_2 | dice_3).equals(2/6)


def test_event_and(dice_2, coin_heads):
    assert P(dice_2 & coin_heads).equals(1/12)


def test_event_not(dice_2):
    assert P(~dice_2).equals(5/6)
