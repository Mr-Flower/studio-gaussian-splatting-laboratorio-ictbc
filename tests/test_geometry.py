import struct

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from app import geometry  # noqa: E402


def test_sparse_points_are_read(tmp_path):
    with open(tmp_path / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 2))
        for point_id, xyz, track in ((1, (1.0, 2.0, 3.0), 2), (7, (-1.0, 0.5, 4.0), 3)):
            f.write(struct.pack("<Q3d", point_id, *xyz))
            f.write(struct.pack("<3Bd", 10, 20, 30, 0.5))
            f.write(struct.pack("<Q", track))
            f.write(b"\x00" * 8 * track)
    assert geometry.read_points3d(tmp_path).tolist() == [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]


def test_distance_statistics_in_metres():
    distances = np.array([0.01, 0.02, 0.03, 0.04, 0.2])  # in unita' del modello
    stats = geometry.distance_stats(distances, metres=2.0)  # 2 m per unita'
    assert stats["median"] == pytest.approx(0.06)
    assert stats["within_5cm"] == pytest.approx(0.4) and stats["within_10cm"] == pytest.approx(0.8)
    assert stats["within_20cm"] == pytest.approx(0.8)


def test_rank_correlation_between_quality_and_geometry():
    results = {"_riferimento": {}, **{k: {"accuratezza": {"median": d}} for k, d in
                                      (("a", 0.01), ("b", 0.02), ("c", 0.03), ("d", 0.05))}}
    rho, _ = geometry.spearman_quality_geometry(results, {"a": 22.0, "b": 21.0, "c": 20.0, "d": 19.0})
    assert rho == pytest.approx(-1.0)  # piu' qualita' d'immagine, distanza minore
    assert geometry.spearman_quality_geometry(results, {"a": 1.0}) is None
