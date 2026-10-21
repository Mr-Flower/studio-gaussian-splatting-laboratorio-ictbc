import pytest

np = pytest.importorskip("numpy")

from app import scala  # noqa: E402


def test_similarity_2d_recovers_scale_rotation_and_shift():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(50, 2))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    target = (2.5 * (rotation @ source.T)).T + np.array([10.0, -4.0])
    scale, found, shift = scala.similarity_2d(source, target)
    assert scale == pytest.approx(2.5) and np.allclose(found, rotation) and np.allclose(shift, [10, -4])


def test_aligner_output_is_read():
    log = ("Using the first GPS coordinate as the ENU origin: 41.1325  14.779 139.055\n"
           "=> Using 838 reference images\n=> Alignment error: 9.402432 (mean), 6.778249 (median)\n"
           "=> Alignment succeeded\n")
    values = scala.parse_log(log)
    assert values == {"enu_origin_lat_lon_alt": [41.1325, 14.779, 139.055], "reference_images": 838,
                      "error_mean_m": 9.402432, "error_median_m": 6.778249, "succeeded": True}
    transform = scala.parse_transform("2.49 -0.3 0.17 -0.62 0.70 13.6 9.48 7.12")
    assert transform["scale"] == 2.49 and len(transform["rotation_wxyz"]) == 4 and transform["translation"][0] == 13.6
    with pytest.raises(ValueError):
        scala.parse_transform("1 2 3")
