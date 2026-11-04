import pytest

np = pytest.importorskip("numpy")

from app import surface  # noqa: E402

# Camera sintetica: guarda un piano noto da una posizione e con un orientamento qualsiasi.
FX, FY, CX, CY, WIDTH, HEIGHT = 80.0, 75.0, 33.0, 22.5, 64, 48


def _camera():
    angle = 0.4
    tilt = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    turn = np.array([[np.cos(0.3), 0, np.sin(0.3)], [0, 1, 0], [-np.sin(0.3), 0, np.cos(0.3)]])
    return np.column_stack([turn @ tilt, [0.5, -1.0, -6.0]])


def _plane_depths(origins, directions, forward):
    """Distanza lungo il raggio e z in camera dei punti in cui i raggi incontrano il piano z = 2."""
    t = (2.0 - origins[..., 2]) / directions[..., 2]
    return t, t * (directions @ forward)


@pytest.mark.parametrize("convention", ["ray", "z"])
def test_backprojected_points_lie_on_the_plane_and_reproject_to_their_pixel(convention):
    camera = _camera()
    origins, directions, forward = surface.pinhole_rays(WIDTH, HEIGHT, FX, FY, CX, CY, camera)
    along_ray, along_z = _plane_depths(origins, directions, forward)
    depth = along_ray if convention == "ray" else along_z
    points = surface.backproject(origins, directions, depth, forward, convention)
    assert np.allclose(points[..., 2], 2.0)
    pixels = surface.project(points, FX, FY, CX, CY, camera)
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    assert np.allclose(pixels[..., 0], u) and np.allclose(pixels[..., 1], v)


def test_wrong_convention_moves_the_points_off_the_plane():
    camera = _camera()
    origins, directions, forward = surface.pinhole_rays(WIDTH, HEIGHT, FX, FY, CX, CY, camera)
    _, along_z = _plane_depths(origins, directions, forward)
    wrong = surface.backproject(origins, directions, along_z, forward, "ray")  # z letta come distanza sul raggio
    assert np.abs(wrong[..., 2] - 2.0).max() > 0.05
    with pytest.raises(ValueError):
        surface.backproject(origins, directions, along_z, forward, "altro")


def test_statistical_filter_drops_only_the_outliers():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60)), -1).reshape(-1, 2)
    plane = np.column_stack([grid, rng.normal(0, 0.002, len(grid))])
    outliers = np.array([[5.0, 5.0, 5.0], [-4.0, 0.5, 3.0], [0.5, 9.0, -6.0]])
    keep = surface.statistical_filter(np.vstack([plane, outliers]), neighbours=20, std_ratio=10.0)
    assert keep[:len(plane)].all() and not keep[len(plane):].any()


def test_same_views_for_every_method():
    split = {"train": [f"jpg/foto_{i:03d}.JPG" for i in range(40, 0, -1)]}
    views = surface.surface_views(split)
    assert views == [f"jpg/foto_{i:03d}.JPG" for i in range(1, 41, surface.VIEW_EVERY)]


def test_surface_ply_round_trip(tmp_path):
    pytest.importorskip("plyfile")
    points = np.random.default_rng(1).normal(size=(100, 3)).astype(np.float32)
    surface.write_ply(tmp_path / "s.ply", points)
    assert np.allclose(surface.read_ply(tmp_path / "s.ply"), points)
