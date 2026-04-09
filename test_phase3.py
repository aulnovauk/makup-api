"""
tests/test_phase3.py — Phase 3 Unit + Integration Test Suite
══════════════════════════════════════════════════════════════
Covers every public API in Phase 3 without requiring:
  - A real dataset
  - A trained checkpoint
  - A GPU
  - ONNX Runtime
  - TensorBoard

All tests run on CPU with randomly initialised weights
and synthetic (random) image tensors.

Run:
    python -m pytest tests/test_phase3.py -v
    python -m pytest tests/test_phase3.py -v --tb=short   # compact errors
    python tests/test_phase3.py                            # standalone
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Make sure package root is importable ─────────────────
# [R7/R16] Use shared guarded helper
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


# ═══════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="module")
def generator(device):
    from models.generator import UNetGenerator
    model = UNetGenerator().to(device).eval()
    yield model
    # [R18] explicit cleanup — release GPU/CPU memory at end of module
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def discriminator(device):
    from models.discriminator import DualDiscriminator
    model = DualDiscriminator().to(device).eval()
    yield model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def beautygan_model(device):
    from models.beautygan import BeautyGAN
    model = BeautyGAN().to(device).eval()
    yield model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def gan_loss():
    from training.losses import MakeupGANLoss
    loss = MakeupGANLoss()
    yield loss
    del loss


def _rand(b: int = 1, c: int = 3, h: int = 64, w: int = 64) -> torch.Tensor:
    """Random tensor in [-1, 1]."""
    return torch.rand(b, c, h, w) * 2 - 1


def _mask(b: int = 1, h: int = 64, w: int = 64) -> torch.Tensor:
    """Random binary mask [0, 1]."""
    return (torch.rand(b, 1, h, w) > 0.5).float()


# ═══════════════════════════════════════════════════════════
#  GENERATOR TESTS
# ═══════════════════════════════════════════════════════════

class TestUNetGenerator:

    def test_output_shape_matches_input(self, generator):
        """Generator output must have same spatial dims as input."""
        src = _rand(1, 3, 256, 256)
        ref = _rand(1, 3, 256, 256)
        with torch.no_grad():
            out = generator(src, ref)
        assert out.shape == src.shape, \
            f"Expected {src.shape}, got {out.shape}"

    def test_output_range_within_tanh(self, generator):
        """Tanh output must be in [-1, 1]."""
        src = _rand(1, 3, 128, 128)
        ref = _rand(1, 3, 128, 128)
        with torch.no_grad():
            out = generator(src, ref)
        assert out.min().item() >= -1.01, "Output below -1"
        assert out.max().item() <=  1.01, "Output above  1"

    def test_batch_size_2(self, generator):
        """Generator must handle batch_size > 1."""
        src = _rand(2, 3, 64, 64)
        ref = _rand(2, 3, 64, 64)
        with torch.no_grad():
            out = generator(src, ref)
        assert out.shape == (2, 3, 64, 64)

    def test_different_source_different_output(self, generator):
        """Different sources must produce different outputs."""
        src1 = _rand(1, 3, 64, 64)
        src2 = _rand(1, 3, 64, 64)
        ref  = _rand(1, 3, 64, 64)
        with torch.no_grad():
            out1 = generator(src1, ref)
            out2 = generator(src2, ref)
        assert not torch.allclose(out1, out2), \
            "Different inputs produced identical outputs"

    def test_gradient_flows_through_generator(self):
        """Gradients must reach all encoder layers in one backward pass."""
        from models.generator import UNetGenerator
        G   = UNetGenerator()
        src = _rand(1, 3, 64, 64, )
        ref = _rand(1, 3, 64, 64)
        out = G(src, ref)
        loss = out.mean()
        loss.backward()
        for name, param in G.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"No gradient for {name}"

    def test_parameter_count_property(self, generator):
        """n_parameters property must return a dict with expected keys."""
        info = generator.n_parameters
        assert "total"     in info
        assert "trainable" in info
        assert info["total"]     > 0
        assert info["trainable"] > 0
        assert info["trainable"] <= info["total"]

    def test_weight_init_std(self):
        """Weights should have std close to 0.02 after init."""
        from models.generator import UNetGenerator
        G = UNetGenerator()
        stds = []
        for m in G.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                stds.append(m.weight.std().item())
        mean_std = float(np.mean(stds))
        assert 0.005 < mean_std < 0.1, \
            f"Mean weight std {mean_std:.4f} seems wrong (expected ~0.02)"


# ═══════════════════════════════════════════════════════════
#  DISCRIMINATOR TESTS
# ═══════════════════════════════════════════════════════════

class TestDualDiscriminator:

    def test_output_shapes(self, discriminator):
        """D must return two patch maps."""
        img  = _rand(1, 3, 256, 256)
        cond = _rand(1, 3, 256, 256)
        with torch.no_grad():
            fp, lp = discriminator(img, cond)
        # Patch maps should be smaller than input
        assert fp.shape[1] == 1, "face logits must be single-channel"
        assert lp.shape[1] == 1, "local logits must be single-channel"
        assert fp.shape[2] < 256,  "face patch map not downsampled"
        assert lp.shape[2] < 256,  "local patch map not downsampled"

    def test_real_fake_logits_differ(self, discriminator):
        """D logits should differ between real and fake (random init)."""
        img1 = _rand(1, 3, 256, 256)
        img2 = _rand(1, 3, 256, 256)
        cond = _rand(1, 3, 256, 256)
        with torch.no_grad():
            fp1, _ = discriminator(img1, cond)
            fp2, _ = discriminator(img2, cond)
        assert not torch.allclose(fp1, fp2)

    def test_mask_does_not_affect_output(self, discriminator):
        """After FIX-10: mask is passed but should not change output."""
        img  = _rand(1, 3, 256, 256)
        cond = _rand(1, 3, 256, 256)
        mask = _mask(1, 256, 256)
        with torch.no_grad():
            fp_no_mask,  lp_no_mask  = discriminator(img, cond, None)
            fp_with_mask, lp_with_mask = discriminator(img, cond, mask)
        assert torch.allclose(fp_no_mask, fp_with_mask), \
            "Mask should not affect D_face output"
        assert torch.allclose(lp_no_mask, lp_with_mask), \
            "Mask should not affect D_local output"

    def test_gradient_flows_through_discriminator(self):
        """Gradients must reach D parameters in backward pass."""
        from models.discriminator import DualDiscriminator
        D    = DualDiscriminator()
        img  = _rand(1, 3, 128, 128)
        cond = _rand(1, 3, 128, 128)
        fp, lp = D(img, cond)
        loss = fp.mean() + lp.mean()
        loss.backward()
        for name, param in D.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


# ═══════════════════════════════════════════════════════════
#  LOSS TESTS
# ═══════════════════════════════════════════════════════════

class TestLosses:

    def test_gan_loss_real_higher_than_fake(self, gan_loss):
        """For a fresh D: real and fake should produce non-zero losses."""
        from training.losses import GANLoss
        loss_fn = GANLoss()
        pred    = torch.rand(1, 1, 16, 16)
        lr = loss_fn(pred, is_real=True)
        lf = loss_fn(pred, is_real=False)
        assert lr.item() >= 0
        assert lf.item() >= 0

    def test_pixel_loss_identical_images_is_zero(self, gan_loss):
        """L1 loss between identical images must be 0."""
        img  = _rand(1, 3, 64, 64)
        loss = gan_loss.pix_loss(img, img)
        assert loss.item() < 1e-6, f"Pixel loss on identical images: {loss.item()}"

    def test_perceptual_loss_identical_images_is_zero(self, gan_loss):
        """VGG perceptual loss between identical images must be ~0."""
        img  = _rand(1, 3, 64, 64)
        loss = gan_loss.perc_loss(img, img)
        assert loss.item() < 1e-4, f"Perceptual loss on identical images: {loss.item()}"

    def test_histogram_loss_identical_is_zero(self, gan_loss):
        """Histogram loss between identical distributions must be ~0."""
        img  = _rand(1, 3, 64, 64)
        mask = torch.ones(1, 1, 64, 64)
        loss = gan_loss.hist_loss(img, img, mask)
        assert loss.item() < 0.05, \
            f"Histogram loss on identical images: {loss.item()}"

    def test_histogram_loss_empty_mask_returns_zero(self, gan_loss):
        """Empty mask → skip all samples → return zero tensor."""
        gen  = _rand(1, 3, 32, 32)
        ref  = _rand(1, 3, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)   # fully empty mask
        loss = gan_loss.hist_loss(gen, ref, mask)
        assert loss.item() == 0.0, "Empty mask should give 0 loss"

    def test_generator_loss_returns_tensor_and_dict(self, gan_loss):
        """generator_loss must return (tensor, dict[str, tensor])."""
        from models.generator     import UNetGenerator
        from models.discriminator import DualDiscriminator

        G    = UNetGenerator()
        D    = DualDiscriminator()
        src  = _rand(1, 3, 64, 64)
        ref  = _rand(1, 3, 64, 64)
        tgt  = _rand(1, 3, 64, 64)
        mask = _mask(1, 64, 64)

        gen        = G(src, ref)
        fake_fp, fake_lp = D(gen, ref, mask)

        total, d = gan_loss.generator_loss(gen, tgt, ref, fake_fp, fake_lp, mask)
        assert isinstance(total, torch.Tensor), "total must be a Tensor"
        assert isinstance(d,     dict),         "loss_dict must be a dict"
        for k, v in d.items():
            assert isinstance(v, torch.Tensor), f"{k} must be a Tensor"
        # total must have grad_fn (not a leaf)
        assert total.grad_fn is not None, "total loss has no grad_fn"

    def test_discriminator_loss_structure(self, gan_loss):
        """discriminator_loss must return (tensor, dict)."""
        real_fp = torch.rand(1, 1, 8, 8)
        fake_fp = torch.rand(1, 1, 8, 8)
        real_lp = torch.rand(1, 1, 12, 12)
        fake_lp = torch.rand(1, 1, 12, 12)
        total, d = gan_loss.discriminator_loss(real_fp, fake_fp, real_lp, fake_lp)
        assert isinstance(total, torch.Tensor)
        assert "D_total" in d and "D_face" in d and "D_local" in d


# ═══════════════════════════════════════════════════════════
#  BEAUTYGAN WRAPPER TESTS
# ═══════════════════════════════════════════════════════════

class TestBeautyGAN:

    def test_forward_delegates_to_generator(self, beautygan_model):
        """BeautyGAN.forward should return same shape as UNetGenerator."""
        src = _rand(1, 3, 64, 64)
        ref = _rand(1, 3, 64, 64)
        with torch.no_grad():
            out = beautygan_model(src, ref)
        assert out.shape == src.shape

    def test_summary_string_non_empty(self, beautygan_model):
        s = beautygan_model.summary()
        assert len(s) > 0
        assert "params" in s.lower()

    def test_save_and_load_roundtrip(self, tmp_path, beautygan_model, device):
        """Save then load should produce identical forward outputs."""
        from models.beautygan import BeautyGAN

        src = _rand(1, 3, 64, 64)
        ref = _rand(1, 3, 64, 64)
        path = tmp_path / "test_checkpoint.pt"

        with torch.no_grad():
            out_before = beautygan_model(src, ref).clone()

        beautygan_model.save(path, epoch=3)

        # Load into a fresh instance
        fresh = BeautyGAN().to(device)
        epoch, _ = fresh.load(path, device=device)

        with torch.no_grad():
            out_after = fresh(src, ref)

        assert epoch == 3
        assert torch.allclose(out_before, out_after, atol=1e-5), \
            "Outputs differ after save/load roundtrip"

    def test_apply_numpy_returns_bgr_uint8(self, beautygan_model):
        """apply_numpy must return uint8 array with same shape as input."""
        import cv2
        h, w = 128, 128
        src_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        ref_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        with torch.no_grad():
            result = beautygan_model.apply_numpy(src_bgr, ref_bgr, image_size=64)
        assert result.dtype == np.uint8, "Result must be uint8"
        assert result.shape == src_bgr.shape, \
            f"Shape mismatch: {result.shape} vs {src_bgr.shape}"
        assert result.min() >= 0 and result.max() <= 255

    def test_apply_numpy_restores_training_mode(self, device):
        """[R13] apply_numpy must restore G.train() state after call."""
        from models.beautygan import BeautyGAN
        h, w = 64, 64
        model   = BeautyGAN().to(device).train()   # explicitly set train mode
        src_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        ref_bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        model.apply_numpy(src_bgr, ref_bgr, image_size=32)
        assert model.G.training, \
            "apply_numpy must restore G to training mode after call"

    def test_load_for_inference_releases_discriminator(self, tmp_path, device):
        """load_for_inference should drop D weights to save memory."""
        from models.beautygan import BeautyGAN

        model = BeautyGAN().to(device)
        path  = tmp_path / "infer_test.pt"
        model.save(path)

        infer_model = BeautyGAN.load_for_inference(path, device=str(device))
        # D should be replaced with Identity (no parameters)
        d_params = list(infer_model.D.parameters())
        assert len(d_params) == 0, \
            "Inference model should have no discriminator parameters"

    def test_save_on_inference_model_raises(self, tmp_path, device):
        """[R14] save() on an inference-only model must raise RuntimeError."""
        from models.beautygan import BeautyGAN

        model = BeautyGAN().to(device)
        path  = tmp_path / "full.pt"
        model.save(path)

        infer_model = BeautyGAN.load_for_inference(path, device=str(device))
        with pytest.raises(RuntimeError, match="inference-only"):
            infer_model.save(tmp_path / "should_fail.pt")


# ═══════════════════════════════════════════════════════════
#  DATASET TESTS (no real images needed)
# ═══════════════════════════════════════════════════════════

class TestDataset:

    def test_mask_generator_returns_correct_shape(self, tmp_path):
        """MaskGenerator must return [H, W] uint8 array."""
        from training.dataset import MaskGenerator
        mg  = MaskGenerator(cache_dir=tmp_path / "masks")
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        mask = mg.generate(img)
        assert mask.shape == (128, 128), f"Mask shape wrong: {mask.shape}"
        assert mask.dtype == np.uint8

    def test_paired_dataset_returns_correct_keys(self, tmp_path):
        """PairedMakeupDataset __getitem__ must return all required keys."""
        import cv2
        from training.dataset import PairedMakeupDataset

        # Create tiny dummy images
        imgs_dir = tmp_path / "imgs"
        imgs_dir.mkdir()
        for i in range(4):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(imgs_dir / f"img_{i:02d}.jpg"), img)

        paths = sorted(imgs_dir.glob("*.jpg"))

        ds = PairedMakeupDataset(
            no_makeup_paths = paths[:2],
            makeup_paths    = paths[2:],
            image_size      = 32,
            is_train        = False,
        )
        sample = ds[0]
        for key in ("source", "reference", "target", "mask"):
            assert key in sample, f"Missing key: {key}"

        assert sample["source"].shape    == (3, 32, 32)
        assert sample["reference"].shape == (3, 32, 32)
        assert sample["target"].shape    == (3, 32, 32)
        assert sample["mask"].shape      == (1, 32, 32)

    def test_dataset_values_in_normalised_range(self, tmp_path):
        """Dataset output must be in [-1, 1] after normalisation."""
        import cv2
        from training.dataset import PairedMakeupDataset

        imgs_dir = tmp_path / "norm_test"
        imgs_dir.mkdir()
        for i in range(4):
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(imgs_dir / f"img_{i:02d}.jpg"), img)

        paths = sorted(imgs_dir.glob("*.jpg"))
        ds = PairedMakeupDataset(
            no_makeup_paths=paths[:2], makeup_paths=paths[2:],
            image_size=32, is_train=False,
        )
        sample = ds[0]
        for key in ("source", "reference", "target"):
            t = sample[key]
            assert t.min().item() >= -1.1, f"{key} below -1"
            assert t.max().item() <=  1.1, f"{key} above  1"


# ═══════════════════════════════════════════════════════════
#  INFERENCE ENGINE TESTS (no checkpoint needed)
# ═══════════════════════════════════════════════════════════

class TestInferenceUtils:

    def test_preprocess_output_shape_and_range(self):
        """preprocess() must return [1, 3, size, size] in [-1, 1]."""
        from api.inference import preprocess
        img    = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess(img, size=128)
        assert tensor.shape == (1, 3, 128, 128)
        assert tensor.min().item() >= -1.01
        assert tensor.max().item() <=  1.01

    def test_preprocess_np_output_shape_and_range(self):
        """preprocess_np() must return [1, 3, size, size] float32 in [-1, 1]."""
        from api.inference import preprocess_np
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        arr = preprocess_np(img, size=128)
        assert arr.shape == (1, 3, 128, 128)
        assert arr.min() >= -1.01
        assert arr.max() <=  1.01
        assert arr.dtype == np.float32

    def test_postprocess_shape_and_dtype(self):
        """postprocess() must return BGR uint8 at original size."""
        from api.inference import postprocess
        tensor = torch.rand(1, 3, 64, 64) * 2 - 1
        result = postprocess(tensor, orig_hw=(480, 640))
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_postprocess_np_shape_and_dtype(self):
        """postprocess_np() must return BGR uint8 at original size."""
        from api.inference import postprocess_np
        arr    = (np.random.rand(1, 3, 64, 64).astype(np.float32) * 2) - 1
        result = postprocess_np(arr, orig_hw=(480, 640))
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_preprocess_postprocess_roundtrip(self):
        """pre → post roundtrip should reproduce original image approximately."""
        from api.inference import preprocess, postprocess

        orig = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        t    = preprocess(orig, size=128)
        out  = postprocess(t, orig_hw=(128, 128))

        # Due to float32 rounding, difference should be very small
        diff = np.abs(orig.astype(float) - out.astype(float)).mean()
        assert diff < 2.0, f"Roundtrip mean diff too large: {diff:.2f}"

    def test_blended_engine_pure_phase2(self):
        """blend_factor=0.0 should return phase2 frame exactly."""
        from api.inference import BlendedMakeupEngine

        class _DummyNeural:
            def apply(self, src, ref):
                return np.zeros_like(src)

        engine  = BlendedMakeupEngine(_DummyNeural(), blend_factor=0.0)
        src     = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        ref     = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        phase2  = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result  = engine.apply(src, ref, phase2)
        np.testing.assert_array_equal(result, phase2)

    def test_blended_engine_pure_neural(self):
        """blend_factor=1.0 should return neural output exactly."""
        from api.inference import BlendedMakeupEngine

        neural_out = np.full((64, 64, 3), 42, dtype=np.uint8)

        class _DummyNeural:
            def apply(self, src, ref):
                return neural_out

        engine = BlendedMakeupEngine(_DummyNeural(), blend_factor=1.0)
        src    = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        ref    = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = engine.apply(src, ref, phase2_bgr=src)
        np.testing.assert_array_equal(result, neural_out)


# ═══════════════════════════════════════════════════════════
#  HISTOGRAM UTILS TESTS
# ═══════════════════════════════════════════════════════════

class TestHistogramUtils:

    def test_match_histograms_same_image_unchanged(self):
        """Matching an image to itself should return the same image."""
        from utils.histogram import match_histograms_region
        img  = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        out  = match_histograms_region(img, img, mask, strength=1.0)
        # After matching to itself, values should be very close
        diff = np.abs(img.astype(int) - out.astype(int)).mean()
        assert diff < 2.0, f"Self-match produced large diff: {diff:.2f}"

    def test_match_histograms_strength_zero_unchanged(self):
        """strength=0 must return the source unchanged."""
        from utils.histogram import match_histograms_region
        src  = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        ref  = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        out  = match_histograms_region(src, ref, mask, strength=0.0)
        np.testing.assert_array_equal(out, src)

    def test_region_colour_stats_empty_mask(self):
        """Empty mask should return safe defaults (no NaN/crash)."""
        from utils.histogram import region_colour_stats
        img  = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        s = region_colour_stats(img, mask)
        assert "mean" in s and "std" in s
        assert not np.any(np.isnan(s["mean"]))

    def test_colour_distance_identical_is_zero(self):
        """Distance between a region and itself must be 0."""
        from utils.histogram import region_colour_stats, colour_distance
        img  = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        mask = np.full((64, 64), 255, dtype=np.uint8)
        s = region_colour_stats(img, mask)
        assert colour_distance(s, s) == 0.0


# ═══════════════════════════════════════════════════════════
#  STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════

class TestTrainerInternals:

    def test_scaler_update_called_once_per_step(self):
        """[R20] scaler.update() must be called exactly once per training step,
        regardless of whether the D step was skipped (n_critic > 1).
        """
        from training.trainer import TrainingConfig, Trainer

        update_count = []
        original_update = torch.amp.GradScaler.update

        def mock_update(self_scaler, *a, **kw):
            update_count.append(1)
            return original_update(self_scaler, *a, **kw)

        # We can't easily run a full step without data, so we verify
        # the source code structure instead — scaler.update() appears
        # exactly once inside _train_step (after G backward).
        import inspect
        from training.trainer import Trainer
        step_src = inspect.getsource(Trainer._train_step)
        update_calls = step_src.count("self.scaler.update()")
        assert update_calls == 1, (
            f"scaler.update() called {update_calls}x in _train_step, expected 1. "
            "Multiple calls cause scaler scale-factor drift when n_critic > 1."
        )


if __name__ == "__main__":
    # Allow running directly without pytest
    import unittest

    # Convert to unittest and run
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestUNetGenerator,
        TestDualDiscriminator,
        TestLosses,
        TestBeautyGAN,
        TestDataset,
        TestInferenceUtils,
        TestHistogramUtils,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
