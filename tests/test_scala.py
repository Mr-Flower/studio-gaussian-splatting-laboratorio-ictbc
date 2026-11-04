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


def _ring(width, depth, angle, centre, step=0.01):
    """Punti sul contorno di un rettangolo ruotato, come la sezione di un pilone nella nuvola densa."""
    u = np.arange(0, width, step)
    v = np.arange(0, depth, step)
    edge = np.concatenate([np.c_[u, 0 * u], np.c_[u, 0 * u + depth], np.c_[0 * v, v], np.c_[0 * v + width, v]])
    edge -= [width / 2, depth / 2]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return edge @ rotation.T + centre


def test_min_area_rectangle_of_rotated_pier():
    pytest.importorskip("scipy")
    sides, angle, centre = scala.min_area_rectangle(_ring(1.6, 2.2, np.radians(25), [3.0, -1.0]))
    assert sides == pytest.approx([1.6, 2.2], abs=0.011) and centre == pytest.approx([3.0, -1.0], abs=0.01)
    assert angle == pytest.approx(115, abs=0.5)  # orientamento del lato lungo


def test_two_piers_are_separated_and_give_the_scale():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    clutter = rng.uniform(-6, 6, size=(300, 2))  # punti sparsi: non formano gruppi
    piers = np.vstack([_ring(1.6, 2.2, 0.3, [-2, 0], 0.004), _ring(1.6, 2.2, 0.3, [2, 1], 0.004), clutter])
    blobs = scala.largest_blobs(piers, cell=0.02)
    assert len(blobs) == 2
    for blob in blobs:
        sides, _, _ = scala.min_area_rectangle(blob)
        assert sides == pytest.approx([1.6, 2.2], abs=0.02)
    # lati in unita' proporzionali a quelli in sito: la scala e' esatta
    assert scala.pier_scale(np.array(scala.PIER_SIDES_M) / 2.75) == pytest.approx(2.75)
