"""Tests for bus_tt.data.datasets."""

import numpy as np
import torch

from bus_tt.data.datasets import SeqCtxDataset, TabularDataset


class TestTabularDataset:
    def test_len(self):
        x = np.random.randn(20, 6).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)
        ds = TabularDataset(x, y)
        assert len(ds) == 20

    def test_getitem_shapes(self):
        x = np.random.randn(10, 6).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        ds = TabularDataset(x, y)
        xi, yi = ds[0]
        assert xi.shape == (6,)
        assert yi.shape == (1,)

    def test_dtype(self):
        x = np.random.randn(5, 6).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        ds = TabularDataset(x, y)
        xi, yi = ds[0]
        assert xi.dtype == torch.float32
        assert yi.dtype == torch.float32


class TestSeqCtxDataset:
    def test_len(self):
        x_seq = np.random.randn(15, 2, 1).astype(np.float32)
        x_ctx = np.random.randn(15, 4).astype(np.float32)
        y = np.random.randn(15).astype(np.float32)
        ds = SeqCtxDataset(x_seq, x_ctx, y)
        assert len(ds) == 15

    def test_getitem_shapes(self):
        x_seq = np.random.randn(10, 2, 1).astype(np.float32)
        x_ctx = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        ds = SeqCtxDataset(x_seq, x_ctx, y)
        s, c, yi = ds[0]
        assert s.shape == (2, 1)
        assert c.shape == (4,)
        assert yi.shape == (1,)

    def test_dtype(self):
        x_seq = np.random.randn(5, 2, 1).astype(np.float32)
        x_ctx = np.random.randn(5, 4).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        ds = SeqCtxDataset(x_seq, x_ctx, y)
        s, c, yi = ds[0]
        assert s.dtype == torch.float32
        assert c.dtype == torch.float32
        assert yi.dtype == torch.float32
