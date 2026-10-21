import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")

from app import check_gt  # noqa: E402


def image(height=40, width=60):
    generator = torch.Generator().manual_seed(0)
    return torch.rand((height, width, 3), generator=generator)


def test_same_camera_gives_the_same_image():
    original = image()
    camera = (50.0, 50.0, 30.0, 20.0)
    warped, mask = check_gt.warp(original, camera, camera, (60, 40))
    assert torch.allclose(warped[mask], original[mask], atol=1e-5)
    assert not mask[:, :check_gt.BORDER].any() and mask[10, 30]  # bordo escluso, centro incluso


def test_principal_point_shift_is_a_pixel_translation():
    original = image()
    # Stessa focale, centro ottico spostato di 3 pixel: il pixel (r, c) della nuova camera e' (r, c + 3) dell'altra.
    warped, mask = check_gt.warp(original, (50.0, 50.0, 30.0, 20.0), (50.0, 50.0, 27.0, 20.0), (50, 40))
    assert torch.allclose(warped[5:35, 5:40], original[5:35, 8:43], atol=1e-5)
    assert mask[5:35, 5:40].all()


def test_comparison_on_the_common_area():
    a = image()
    same = check_gt.compare(a, a.clone(), torch.ones(40, 60, dtype=torch.bool))
    assert same["psnr"] == float("inf") and same["ssim"] == pytest.approx(1.0)
    noisy = (a + 0.01).clamp(0, 1)
    mask = torch.zeros(40, 60, dtype=torch.bool)
    mask[5:30, 10:50] = True
    result = check_gt.compare(a, noisy, mask)
    assert 38 < result["psnr"] < 41 and result["common_fraction"] == pytest.approx(25 * 40 / (40 * 60))
