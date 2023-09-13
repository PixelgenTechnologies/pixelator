"""
Tests for the cell calling module

Copyright (c) 2023 Pixelgen Technologies AB.
"""

import numpy as np

from pixelator.annotate.cell_calling import (
    find_component_size_limits,
)


def test_find_component_min_size_limits_signal_and_noise():
    # Generate two distributions, one that's noise (poisson distribution),
    #  one that is signal (normal distribution)
    random_state = np.random.default_rng(seed=1)
    signal = random_state.normal(1500, 500, 1000)
    noise = random_state.poisson(100, 500)
    test_data = np.concatenate((signal, noise), axis=None)

    min_bound = find_component_size_limits(np.absolute(test_data), direction="lower")
    assert min_bound > 200
    assert min_bound < 500


def test_find_component_min_size_limits_only_signal():
    # Generate only a single distributions, that resembles signal (normal distribution)
    random_state = np.random.default_rng(seed=1)
    test_data = random_state.normal(1500, 500, 500)

    min_bound = find_component_size_limits(np.absolute(test_data), direction="lower")
    assert min_bound > 100
    assert min_bound < 200


def test_find_component_min_size_limits_only_noise():
    # Generate only a single distributions, that resembles noise (poisson distribution)
    random_state = np.random.default_rng(seed=1)
    test_data = random_state.poisson(100, 500)

    min_bound = find_component_size_limits(np.absolute(test_data), direction="lower")
    assert min_bound < 100


def test_find_component_min_size_limits_signal_and_many_doublets():
    random_state = np.random.default_rng(seed=1)
    signal = random_state.normal(1500, 500, 500)
    noise = random_state.poisson(100, 500)
    doublets = random_state.normal(6000, 500, 500)
    test_data = np.concatenate((signal, noise, doublets), axis=None)

    min_bound = find_component_size_limits(np.absolute(test_data), direction="lower")
    assert min_bound > 200
    assert min_bound < 600


def test_find_component_max_size():
    random_state = np.random.default_rng(seed=1)
    signal = random_state.normal(1500, 100, 500)
    noise = random_state.normal(3000, 200, 50)
    test_data = np.concatenate((signal, noise), axis=None)

    max_bound = find_component_size_limits(np.absolute(test_data), direction="upper")
    assert max_bound > 1500
