"""
    Rank Discounted Cumulative Gain and Loss algorithm for testing unranked search results against a ranked list.
    Copyright (C) 2023 Healthee

    Yoav.Vollansky (@t) healthee.co

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


"""
This test suite tests the nrdcgl algorithm for different use cases. Some of the tests are relative, in the sense that
they test case where the absolute value of the result is not important, but rather the relative value compared to
previous results. This is not because the algorithm is not deterministic, but because values may vary depending on the
implementation of the algorithm. For example, if we change the calculation of the gain function, the results may be
different, but the relative values should be still comparable as in with the <, >, >=, <=, == operators.

"""

import pytest
from radicugloss.radicugloss import nrdcgl, pnrdcgl, calculate_gain, build_assumed_ranges_mapping, invert_rank


@pytest.fixture
def true_relevance_set():
    """
    This is the expected set / gold set / ground truth / perfect result / pure outcome / holy grail... you name it.
    :return:
    """
    return {
        "Elad": 1,
        "Yoav": 2,
        "Gold": 2,
        "Namer": 2,
        "Uzi": 3,
        "GT": 3
    }


def test_calculate_gain():
    assert calculate_gain(1, 1) == pytest.approx(0.6309297535714575)
    assert calculate_gain(2, 1) == pytest.approx(0.5)


def test_calculate_assumed_rank(true_relevance_set):
    assumed_ranks = build_assumed_ranges_mapping(true_relevance_set)
    assert assumed_ranks == {1: (0, 0), 2: (1, 3), 3: (4, 5)}


def test_invert_ranks():
    assert invert_rank(5, 6) == 2
    assert invert_rank(1, 3) == 3
    assert invert_rank(3, 3) == 1


@pytest.fixture
def perfect_score():
    """
    The perfect score for a search result is 1.0
    :return:
    """
    return 1.0


@pytest.fixture
def search_results_0():
    """
    should be 1.0, it is just like the set.
    """
    return ["Elad",
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            "GT"]


def test_nrdcgl_0(true_relevance_set, search_results_0, perfect_score):
    assert nrdcgl(search_results_0, true_relevance_set) == perfect_score
    assert pnrdcgl(search_results_0, true_relevance_set) == perfect_score
    assert pnrdcgl(search_results_0, true_relevance_set) == perfect_score


@pytest.fixture
def search_results_1a():
    """
    should be 1.0: we replaced entities only inside a single rank
    """
    return ["Elad",
            "Yoav",
            "Namer",  # switched
            "Gold",  # switched
            "Uzi",
            "GT"
            ]


def test_nrdcgl_1a(true_relevance_set, search_results_1a, perfect_score):
    assert nrdcgl(search_results_1a, true_relevance_set) == perfect_score


@pytest.fixture
def search_results_1b():
    """
    should be 1.0, same as above: we replaced entities only inside a single rank (but other ones)
    """
    return ["Elad",
            "Gold",  # switched
            "Yoav",  # switched
            "Namer",
            "Uzi",
            "GT"
            ]


def test_nrdcgl_1b(true_relevance_set, search_results_1b, perfect_score):
    assert nrdcgl(search_results_1b, true_relevance_set) == perfect_score


@pytest.fixture
def search_results_1c():
    """
    should be 1.0, same as above: we replaced entities only inside a single rank. they are of another rank than 1b,
    but it doesn't matter.
    """
    return ["Elad",
            "Gold",
            "Yoav",
            "Namer",
            "GT",  # switched
            "Uzi"  # switched
            ]


def test_nrdcgl_1c(true_relevance_set, search_results_1c, perfect_score):
    assert nrdcgl(search_results_1c, true_relevance_set) == perfect_score


@pytest.fixture
def search_results_2():
    """
    should be 1.0: we replaced entities only inside multiple ranks
    """
    return ["Elad",
            "Namer",  # switched-1
            "Gold",
            "Yoav",  # switched-1
            "GT",  # switched-2
            "Uzi"  # switched-2
            ]


def test_nrdcgl_2(true_relevance_set, search_results_2, perfect_score):
    assert nrdcgl(search_results_2, true_relevance_set) == perfect_score


@pytest.fixture
def search_results_3a():
    """
    should be < 1.0: we made one r2 <-> r3 switch
    """
    return ["Elad",
            "Uzi",  # switched
            "Gold",
            "Namer",
            "Yoav",  # switched
            "GT"
            ]


def test_nrdcgl_3a(true_relevance_set, search_results_3a, perfect_score):
    assert nrdcgl(search_results_3a, true_relevance_set) < perfect_score


@pytest.fixture
def search_results_3b():
    """
    should be < 1.0, same as above: we made one r2 <-> r3 switch, but with a different r3 entity
    """
    return ["Elad",
            "GT",  # switched
            "Gold",
            "Namer",
            "Uzi",
            "Yoav"  # switched
            ]


def test_nrdcgl_3b(true_relevance_set, search_results_3b, search_results_3a, perfect_score):
    assert nrdcgl(search_results_3b, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_3b, true_relevance_set) == nrdcgl(search_results_3a, true_relevance_set)


@pytest.fixture
def search_results_3c():
    """
    should be < 1.0, same as above: we made one r2 <-> r3 switch and one r2 <-> r2 switch
    """
    return ["Elad",
            "Uzi",  # switched-2
            "Namer",  # switched-1
            "Gold",  # switched-1
            "Yoav",  # switched-2
            "GT"
            ]


def test_nrdcgl_3c(true_relevance_set, search_results_3c, search_results_3b, perfect_score):
    assert nrdcgl(search_results_3c, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_3c, true_relevance_set) == nrdcgl(search_results_3b, true_relevance_set)


@pytest.fixture
def search_results_4():
    """
    should be < 1.0, same as above: we made one r2 <-> r3 switch and one r2 <-> r2 switch, but with a different r3
    entity
    """
    return ["Elad",
            "GT",  # switched-2
            "Namer",  # switched-1
            "Gold",  # switched-1
            "Uzi",
            "Yoav"  # switched-2
            ]


def test_nrdcgl_4(true_relevance_set, search_results_4, search_results_3c, perfect_score):
    assert nrdcgl(search_results_4, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_4, true_relevance_set) == nrdcgl(search_results_3c, true_relevance_set)


def test_pnrdcgl_4(true_relevance_set, search_results_4, search_results_3c, perfect_score):
    assert pnrdcgl(search_results_4, true_relevance_set) > 0


@pytest.fixture
def search_results_5a():
    """
    should be less than previous test: we added an irrelevant (false positive)
    """
    return ["Elad",
            "GT",  # switched-2
            "Namer",  # switched-1
            "Gold",  # switched-1
            "Uzi",  # switched-2
            "Yoav",
            "Ron",  # added irrelevant
            ]


def test_nrdcgl_5a(true_relevance_set, search_results_5a, search_results_4, perfect_score):
    assert nrdcgl(search_results_5a, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_5a, true_relevance_set) < nrdcgl(search_results_4, true_relevance_set)


@pytest.fixture
def search_results_5b():
    """
    should be more than 5a: we added an irrelevant (false positive) but the rest is in perfect order
    """
    return ["Elad",
            "Yoav",
            "Gold",  # switched-1
            "Namer",  # switched-1
            "Uzi",
            "GT",
            "Ron",  # added irrelevant
            ]


def test_nrdcgl_5b(true_relevance_set, search_results_5b, search_results_5a, perfect_score):
    assert nrdcgl(search_results_5b, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_5b, true_relevance_set) > nrdcgl(search_results_5a, true_relevance_set)


@pytest.fixture
def search_results_6a():
    """
    should be less than 5b: we added another irrelevant (false positive)
    """
    return ["Elad",
            "GT",  # switched-2
            "Namer",  # switched-1
            "Gold",  # switched-1
            "Uzi",  # switched-2
            "Yoav",
            "Ron",  # added irrelevant
            "Nagar"  # added another irrelevant
            ]


def test_nrdcgl_6a(true_relevance_set, search_results_6a, search_results_5b, perfect_score):
    assert nrdcgl(search_results_6a, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_6a, true_relevance_set) < nrdcgl(search_results_5b, true_relevance_set)


@pytest.fixture
def search_results_6b():
    """
    should be less than 6a: we added another irrelevant (false positive) in a higher index
    """
    return ["Elad",
            "GT",  # switched-2
            "Namer",  # switched-1
            "Gold",  # switched-1
            "Uzi",  # switched-2
            "Nagar"  # added another irrelevant in a higher index
            "Yoav",
            "Ron",  # added irrelevant
            ]


def test_nrdcgl_6b(true_relevance_set, search_results_6b, search_results_6a, perfect_score):
    assert nrdcgl(search_results_6b, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_6b, true_relevance_set) < nrdcgl(search_results_6a, true_relevance_set)


@pytest.fixture
def search_results_7():
    """
    should be less than 6a test: we added an irrelevant (false positive) WITH penalty, but we cut off at k=6
    """
    return ["Elad",
            "GT",  # switched-2
            "Namer",  # switched-1
            "Gold",  # switched-1
            "Uzi",  # switched-2
            "Yoav",
            "Ron",  # added irrelevant
            ]


def test_nrdcgl_7(true_relevance_set, search_results_7, search_results_6a, perfect_score, k=6):
    assert nrdcgl(search_results_7, true_relevance_set, k=k) < perfect_score
    assert nrdcgl(search_results_7, true_relevance_set, k=k) > nrdcgl(search_results_6a, true_relevance_set)


@pytest.fixture
def search_results_8a():
    """
    should be less than 1.0: we removed an entity
    """
    return ["Elad",
            "Yoav",
            # "Gold", a missing entity
            "Namer",
            "Uzi",
            "GT"
            ]


def test_nrdcgl_8a(true_relevance_set, search_results_8a, perfect_score):
    assert nrdcgl(search_results_8a, true_relevance_set) < perfect_score


@pytest.fixture
def search_results_8b():
    """
    should be more than test 8a since the missing entity is of a lower ranks and punish_max=False
    """
    return ["Elad",
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            # "GT", a missing entity of a lower rank
            ]


def test_nrdcgl_8b(true_relevance_set, search_results_8b, search_results_8a, perfect_score):
    assert nrdcgl(search_results_8b, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_8b, true_relevance_set) > nrdcgl(search_results_8a, true_relevance_set)


@pytest.fixture
def search_results_8c():
    """
    should be < 1, but less than 8a: it is the same set but punish_max=True
    """
    return ["Elad",
            "Yoav",
            # "Gold", a missing entity
            "Namer",
            "Uzi",
            "GT"
            ]


def test_nrdcgl_8c(true_relevance_set, search_results_8c, search_results_8a, perfect_score, punish_max=True):
    assert nrdcgl(search_results_8c, true_relevance_set, punish_max=punish_max) < perfect_score
    assert nrdcgl(search_results_8c, true_relevance_set, punish_max=punish_max) < nrdcgl(search_results_8a,
                                                                                         true_relevance_set)


@pytest.fixture
def search_results_8d():
    """
    should be < 1, < 8c because both are punish_max=True and in 8c there's higher rank element that contributes to
    the gain
    """
    return ["Elad",
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            # "GT" # a missing entity of a lower rank
            ]


def test_nrdcgl_8d(true_relevance_set, search_results_8d, search_results_8c, perfect_score, punish_max=True):
    assert nrdcgl(search_results_8d, true_relevance_set, punish_max=punish_max) < perfect_score
    assert nrdcgl(search_results_8d, true_relevance_set, punish_max=punish_max) > nrdcgl(search_results_8c,
                                                                                         true_relevance_set,
                                                                                         punish_max=punish_max)


@pytest.fixture
def search_results_8e():
    """
    should be less than 1.0: we removed the topmost entity
    """
    return [
        # "Elad", a missing entity
        "Yoav",
        "Gold",
        "Namer",
        "Uzi",
        "GT"
    ]


def test_nrdcgl_8e(true_relevance_set, search_results_8e, perfect_score):
    assert nrdcgl(search_results_8e, true_relevance_set) < perfect_score


@pytest.fixture
def search_results_8f():
    """
    identical to 8e, but with punish_max=True. should return the same score.
    """
    return [
        # "Elad", a missing entity
        "Yoav",
        "Gold",
        "Namer",
        "Uzi",
        "GT"
    ]


def test_nrdcgl_8f(true_relevance_set, search_results_8f, perfect_score):
    assert nrdcgl(search_results_8f, true_relevance_set) == nrdcgl(search_results_8f,
                                                                   true_relevance_set,
                                                                   punish_max=True)


@pytest.fixture
def search_results_9():
    """
    should be less than 8a, we removed another entity
    """
    return ["Elad",
            "Yoav",
            # "Gold", a missing entity
            #  "Namer", another missing entity
            "Uzi",
            "GT"
            ]


def test_nrdcgl_9(true_relevance_set, search_results_9, search_results_8a, perfect_score):
    assert nrdcgl(search_results_9, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_9, true_relevance_set) < nrdcgl(search_results_8a, true_relevance_set)


@pytest.fixture
def search_results_10():
    """
    should be the same as 9 test since false positives are not penalized
    """
    return ["Elad",
            "Yoav",
            # "Gold", a missing entity
            #  "Namer", another missing entity
            "Uzi",
            "GT",
            "Ron",  # Added irrelevant
            ]


def test_nrdcgl_10(true_relevance_set, search_results_10, search_results_9, perfect_score):
    assert nrdcgl(search_results_10, true_relevance_set, fp_penalty=0) < perfect_score
    assert nrdcgl(search_results_10, true_relevance_set, fp_penalty=0) == nrdcgl(search_results_9, true_relevance_set)


@pytest.fixture
def search_results_11a():
    """
    Irrelevant in first place "punishing" (not gaining) even without a penalty
    """
    return ["Ron",  # irrelevant
            "Elad",
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            "GT"
            ]


def test_nrdcgl_11a(true_relevance_set, search_results_11a, perfect_score):
    assert nrdcgl(search_results_11a, true_relevance_set, fp_penalty=0) < perfect_score


@pytest.fixture
def search_results_11b():
    """
    Irrelevant in first place with fp_penalty, score should be less than test 11a
    """
    return ["Ron",  # irrelevant
            "Elad",
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            "GT"
            ]


def test_nrdcgl_11b(true_relevance_set, search_results_11b, search_results_11a, perfect_score):
    assert nrdcgl(search_results_11b, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_11b, true_relevance_set) < nrdcgl(search_results_11a, true_relevance_set, fp_penalty=0)


@pytest.fixture
def search_results_11c():
    """
    Irrelevant in first place, missing in last, should be < 11b
    """
    return ["Ron",  # irrelevant
            "Elad",
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            #    "GT" # missing
            ]


def test_nrdcgl_11c(true_relevance_set, search_results_11c, search_results_11b, perfect_score):
    assert nrdcgl(search_results_11c, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_11c, true_relevance_set) < nrdcgl(search_results_11b, true_relevance_set)


@pytest.fixture
def search_results_11d():
    """
    Irrelevant in first place, missing in second: should be  < 11c since missing entity is of a higher rank.
    """
    return ["Ron",  # irrelevant
            # "Elad", # missing
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            "GT"
            ]


def test_nrdcgl_11d(true_relevance_set, search_results_11d, search_results_11c, perfect_score):
    assert nrdcgl(search_results_11d, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_11d, true_relevance_set) < nrdcgl(search_results_11c, true_relevance_set)


@pytest.fixture
def search_results_11e():
    """
    Irrelevant in first place, irrelevant out of ranks: should be < 11b.
    Should also be > 11d since the penalty for a missing entity in ranks should be bigger than fp out of ranks
    """
    return ["Ron",  # irrelevant
            "Elad",
            "Yoav",
            "Gold",
            "Namer",
            "Uzi",
            "GT",
            "Guy"  # irrelevant
            ]


def test_nrdcgl_11e(true_relevance_set, search_results_11e, search_results_11b, search_results_11d, perfect_score):
    assert nrdcgl(search_results_11e, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_11e, true_relevance_set) < nrdcgl(search_results_11b, true_relevance_set)
    assert nrdcgl(search_results_11e, true_relevance_set) > nrdcgl(search_results_11d, true_relevance_set)


@pytest.fixture
def search_results_11f():
    """
    Irrelevant in first place, irrelevant out of ranks: should be > 11b where fp is in a higher rank
    """
    return [
        "Elad",
        "Yoav",
        "Ron",  # irrelevant
        "Gold",
        "Namer",
        "Uzi",
        "GT"
    ]


def test_nrdcgl_11f(true_relevance_set, search_results_11f, search_results_11b, perfect_score):
    assert nrdcgl(search_results_11f, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_11f, true_relevance_set) > nrdcgl(search_results_11b, true_relevance_set)


@pytest.fixture
def search_results_11g():
    """
    Irrelevant in 2nd place: should be < 11f, > 11b
    """
    return [
        "Elad",
        "Ron",  # irrelevant
        "Yoav",
        "Gold",
        "Namer",
        "Uzi",
        "GT"
    ]


# TODO: This test has a problem, fp in 2nd index is punishing less than fp in 3rd index in 11f
def test_nrdcgl_11g(true_relevance_set, search_results_11g, search_results_11f, search_results_11b, perfect_score):
    assert nrdcgl(search_results_11g, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_11g, true_relevance_set) > nrdcgl(search_results_11b, true_relevance_set)
    assert nrdcgl(search_results_11g, true_relevance_set) < nrdcgl(search_results_11f, true_relevance_set)


@pytest.fixture
def search_results_13():
    """
    All entities are irrelevant, should be negative
    """
    return ["Ron",  # irrelevant
            "Idan",  # irrelevant
            "Tzach",  # irrelevant
            "Nagar",  # irrelevant
            ]


def test_nrdcgl_13(true_relevance_set, search_results_13, perfect_score):
    assert nrdcgl(search_results_13, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_13, true_relevance_set) < 0


def test_pnrdcgl_13(true_relevance_set, search_results_13, perfect_score):
    assert pnrdcgl(search_results_13, true_relevance_set) == 0


@pytest.fixture
def search_results_14():
    """
    Should be > 13 since first index is a hit, comparably
    """
    return ["Elad",
            "Idan",  # irrelevant
            "Tzach",  # irrelevant
            "Nagar"  # irrelevant
            ]


def test_nrdcgl_14(true_relevance_set, search_results_14, search_results_13, perfect_score):
    assert nrdcgl(search_results_14, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_14, true_relevance_set) > nrdcgl(search_results_13, true_relevance_set)


@pytest.fixture
def search_results_15():
    """
    Should be > 14 since both first and second index are a hit, comparably
    """
    return ["Elad",
            "Yoav",
            "Tzach",  # irrelevant
            "Nagar"  # irrelevant
            ]


def test_nrdcgl_15(true_relevance_set, search_results_15, search_results_14, perfect_score):
    assert nrdcgl(search_results_15, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_15, true_relevance_set) > nrdcgl(search_results_14, true_relevance_set)


@pytest.fixture
def search_results_16():
    """
    Should be < 15 since we are missing the second hit, comparably
    Should be > 14 since we have less fp, comparably
    """
    return ["Elad",
            "Tzach",  # irrelevant
            "Nagar"  # irrelevant
            ]


def test_nrdcgl_16(true_relevance_set, search_results_16, search_results_15, search_results_14, perfect_score):
    assert nrdcgl(search_results_16, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_16, true_relevance_set) < nrdcgl(search_results_15, true_relevance_set)
    assert nrdcgl(search_results_16, true_relevance_set) > nrdcgl(search_results_14, true_relevance_set)


@pytest.fixture
def search_results_17():
    """
    Should be < 0: nothing gives gain but we lose for all the fn.
    If we set fn_penalty=0, then we should get 0 score.
    """
    return []


def test_nrdcgl_17(true_relevance_set, search_results_17, perfect_score):
    assert nrdcgl(search_results_17, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_17, true_relevance_set) < 0
    assert nrdcgl(search_results_17, true_relevance_set, fn_penalty=0) == 0
    assert nrdcgl(search_results_17, true_relevance_set, fn_penalty=0, fp_penalty=0) == 0


def test_pnrdcgl_17(true_relevance_set, search_results_17, perfect_score):
    assert pnrdcgl(search_results_17, true_relevance_set) < perfect_score
    assert pnrdcgl(search_results_17, true_relevance_set) == 0
    assert pnrdcgl(search_results_17, true_relevance_set, fn_penalty=0) == 0
    assert pnrdcgl(search_results_17, true_relevance_set, fn_penalty=0, fp_penalty=0) == 0


@pytest.fixture
def search_results_18a():
    """
    Should be < 1.
    If punish_max=True, should be the same score since this is a first rank entity
    """
    return [
        # "Elad",  # missing
        "Yoav",
        "Gold",
        "Namer",
        "Uzi",
        "GT"
    ]


def test_nrdcgl_18a(true_relevance_set, search_results_18a, perfect_score):
    assert nrdcgl(search_results_18a, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_18a, true_relevance_set) == nrdcgl(search_results_18a,
                                                                    true_relevance_set,
                                                                    punish_max=True)


@pytest.fixture
def search_results_18c():
    """
    should be > 18b: punish_max=True indeed so we have the same loss, but Elad's gain > Yoav's gain due to ranking
    """
    return [
        "Elad",
        # "Yoav", # missing
        "Gold",
        "Namer",
        "Uzi",
        "GT"
    ]


def test_nrdcgl_18c(true_relevance_set, search_results_18c, search_results_18a, perfect_score):
    assert nrdcgl(search_results_18c, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_18c, true_relevance_set) > nrdcgl(search_results_18a,
                                                                   true_relevance_set,
                                                                   punish_max=True)


@pytest.fixture
def search_results_18d():
    """
    should be = 18c: punish_max=True so losses are equal, and Gold's gain == Yoav's gain
    """
    return [
        "Elad",
        "Yoav",
        # "Gold", # missing
        "Namer",
        "Uzi",
        "GT"
    ]


def test_nrdcgl_18d(true_relevance_set, search_results_18d, search_results_18c, perfect_score):
    assert nrdcgl(search_results_18d, true_relevance_set) < perfect_score
    assert nrdcgl(search_results_18d, true_relevance_set, punish_max=True) == nrdcgl(search_results_18c,
                                                                                     true_relevance_set,
                                                                                     punish_max=True)
