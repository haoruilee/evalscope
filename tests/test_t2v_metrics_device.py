# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for the FGA-BLIP2 / t2v-metrics device-agnostic fix.

See https://github.com/modelscope/evalscope/issues/1331

Pre-fix, the FGA-BLIP2 score model (and its BLIP2-ITM / CLIP-T5 siblings)
hard-coded ``device='cuda'`` and decorated their ``forward`` methods with
``@torch.autocast(device_type='cuda', dtype=torch.float16)``. On a non-CUDA
host this caused two distinct failure modes:

1. The autocast decorator emitted a ``UserWarning: CUDA is not available...``
   at *module-import time*, polluting logs even when no metric ran.
2. The ``self.device='cuda'`` default was passed straight to ``model.to('cuda')``
   (and many ``tensor.to(self.device)`` calls), raising ``RuntimeError`` on
   any host without CUDA — Ascend NPU, Apple MPS, plain CPU, etc.

The fix introduces ``resolve_device`` and a ``maybe_autocast`` helper on
``ScoreModel``. These tests pin the new behavior and guard against
regressions.
"""

import contextlib
import unittest
from unittest import mock

import torch

from evalscope.metrics.t2v_metrics.models.model import ScoreModel, resolve_device


class _DummyScoreModel(ScoreModel):
    """Minimal concrete ScoreModel for testing the base class behavior.

    Avoids touching the network or the model cache.
    """

    def load_model(self):
        self.model = None

    def load_images(self, image):
        return None

    def forward(self, images, texts, **kwargs):
        return torch.tensor([0.0])


class TestResolveDevice(unittest.TestCase):

    def test_none_returns_autodetected(self):
        # On a CPU-only host the helper should yield 'cpu'.
        if torch.cuda.is_available():
            self.skipTest('host has CUDA; cannot pin CPU fallback')
        self.assertEqual(resolve_device(None), 'cpu')

    def test_cuda_falls_back_when_unavailable(self):
        # The original bug: a hard-coded 'cuda' default on a non-CUDA host.
        # resolve_device must gracefully fall back rather than propagating.
        if torch.cuda.is_available():
            self.skipTest('host has CUDA; cannot pin CPU fallback')
        resolved = resolve_device('cuda')
        self.assertNotEqual(resolved, 'cuda')

    def test_cpu_passthrough(self):
        self.assertEqual(resolve_device('cpu'), 'cpu')

    def test_npu_falls_back_when_unavailable(self):
        # No torch_npu installed in CI ⇒ resolve_device must NOT silently
        # trust the user. It should fall back to autodetect.
        try:
            from transformers.utils import is_torch_npu_available
            if is_torch_npu_available():
                self.skipTest('torch_npu is installed; cannot pin fallback')
        except ImportError:
            pass
        resolved = resolve_device('npu')
        self.assertNotEqual(resolved, 'npu')

    def test_mps_falls_back_when_unavailable(self):
        try:
            from transformers.utils import is_torch_mps_available
            if is_torch_mps_available():
                self.skipTest('host has MPS; cannot pin fallback')
        except ImportError:
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                self.skipTest('host has MPS; cannot pin fallback')
        resolved = resolve_device('mps')
        self.assertNotEqual(resolved, 'mps')

    def test_cuda_with_index_passthrough(self):
        # Indexed device strings (e.g. 'cuda:1') must pass through unchanged
        # when the device family is available.
        with mock.patch.object(torch.cuda, 'is_available', return_value=True):
            self.assertEqual(resolve_device('cuda:1'), 'cuda:1')


class TestScoreModelDeviceResolution(unittest.TestCase):

    def test_init_resolves_cuda_to_cpu_on_cpu_host(self):
        if torch.cuda.is_available():
            self.skipTest('host has CUDA; cannot pin CPU fallback')
        m = _DummyScoreModel(model_name='dummy', device='cuda', cache_dir='/tmp/_evalscope_test_cache')
        self.assertEqual(m.device, 'cpu')

    def test_init_with_none_resolves(self):
        if torch.cuda.is_available():
            self.skipTest('host has CUDA; cannot pin CPU fallback')
        m = _DummyScoreModel(model_name='dummy', device=None, cache_dir='/tmp/_evalscope_test_cache')
        self.assertEqual(m.device, 'cpu')

    def test_maybe_autocast_returns_nullcontext_on_cpu(self):
        m = _DummyScoreModel(model_name='dummy', device='cpu', cache_dir='/tmp/_evalscope_test_cache')
        ctx = m.maybe_autocast(dtype=torch.float16)
        self.assertIsInstance(ctx, contextlib.nullcontext)

    def test_maybe_autocast_is_safe_to_use_on_cpu(self):
        m = _DummyScoreModel(model_name='dummy', device='cpu', cache_dir='/tmp/_evalscope_test_cache')
        with m.maybe_autocast(dtype=torch.float16):
            x = torch.zeros(2)
        self.assertEqual(x.shape, (2,))


class TestRegressionVsOriginalBehavior(unittest.TestCase):
    """Pin the failure modes of the pre-fix code.

    These tests ensure that the patterns we removed actually were broken on
    non-CUDA hosts, so a regression would re-introduce the issue.
    """

    def test_old_decorator_warns_at_construction_on_cpu(self):
        # ``@torch.autocast(device_type='cuda', ...)`` emits a UserWarning at
        # decorator-construction time (i.e. at module import) on a non-CUDA
        # host. Removing the decorator silences it.
        if torch.cuda.is_available():
            self.skipTest('host has CUDA; pre-fix decorator would not warn')
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            torch.autocast(device_type='cuda', dtype=torch.float16)
        self.assertTrue(
            any('CUDA is not available' in str(w.message) for w in caught),
            f'expected CUDA-unavailable warning at decorator construction; got {[str(w.message) for w in caught]}',
        )

    def test_real_bug_tensor_to_cuda_raises_on_cpu_host(self):
        # The actual user-visible failure path: any ``.to('cuda')`` call
        # raises on a non-CUDA host. The fix re-routes ``self.device`` away
        # from a hard-coded 'cuda' so these calls never get made on the
        # wrong device.
        if torch.cuda.is_available():
            self.skipTest('host has CUDA; .to(cuda) would not raise')
        with self.assertRaises((RuntimeError, AssertionError)):
            torch.zeros(1).to('cuda')

    def test_new_pattern_works_on_cpu_host(self):
        if torch.cuda.is_available():
            self.skipTest('host has CUDA; cannot pin CPU fallback')
        m = _DummyScoreModel(model_name='dummy', device='cuda', cache_dir='/tmp/_evalscope_test_cache')
        self.assertEqual(m.device, 'cpu')

        @torch.no_grad()
        def new_forward(x):
            with m.maybe_autocast(dtype=torch.float16):
                y = x + 1
            return y.to(m.device)

        result = new_forward(torch.zeros(1))
        self.assertEqual(result.item(), 1.0)
        self.assertEqual(result.device.type, 'cpu')


class TestSourceLevelInvariants(unittest.TestCase):
    """Static guards: the offending decorator must not creep back in."""

    _PATCHED_FILES = [
        'evalscope/metrics/t2v_metrics/models/itmscore_models/fga_blip2_model.py',
        'evalscope/metrics/t2v_metrics/models/itmscore_models/blip2_itm_model.py',
        'evalscope/metrics/t2v_metrics/models/vqascore_models/clip_t5_model.py',
    ]

    def test_no_cuda_bound_autocast_decorator(self):
        import os

        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        for rel in self._PATCHED_FILES:
            path = os.path.join(repo_root, rel)
            with open(path, encoding='utf-8') as f:
                src = f.read()
            self.assertNotIn(
                "@torch.autocast(device_type='cuda'", src,
                msg=f'{rel}: cuda-bound autocast decorator must stay removed (issue #1331).',
            )
            self.assertIn(
                'with self.maybe_autocast', src,
                msg=f'{rel}: maybe_autocast helper must be used in place of the cuda-bound decorator.',
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
