"""Tests for bus_tt.utils (seed, logging, paths)."""

import logging
from pathlib import Path

import numpy as np
import torch

from bus_tt.utils.seed import set_seed
from bus_tt.utils.logging import get_logger
from bus_tt.utils.paths import PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, CONFIG_DIR


class TestSetSeed:
    def test_reproducible_numpy(self):
        set_seed(0)
        a = np.random.rand(5)
        set_seed(0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_reproducible_torch(self):
        set_seed(0)
        a = torch.rand(5)
        set_seed(0)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = np.random.rand(5)
        set_seed(1)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)


class TestGetLogger:
    def test_returns_logger(self):
        log = get_logger("test_logger")
        assert isinstance(log, logging.Logger)

    def test_has_handler(self):
        log = get_logger("test_handler")
        assert len(log.handlers) >= 1

    def test_no_duplicate_handlers(self):
        log1 = get_logger("test_dup")
        n1 = len(log1.handlers)
        log2 = get_logger("test_dup")
        assert len(log2.handlers) == n1


class TestPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_data_dir(self):
        assert DATA_DIR == PROJECT_ROOT / "data"

    def test_output_dir(self):
        assert OUTPUT_DIR == PROJECT_ROOT / "outputs"

    def test_config_dir(self):
        assert CONFIG_DIR == PROJECT_ROOT / "configs"
