"""I parser sono verificati su righe copiate dai log reali di COLMAP 4.2.1 e nerfstudio 1.1.5."""
import pytest

from app import progress


def test_feature_extraction_ratio():
    parse = progress.ratio(r"Processed file \[(\d+)/(\d+)\]")
    assert parse("I20261002 11:52:06.443887 12776 feature_extraction.cc:270] Processed file [453/906]") == 0.5
    assert parse("I20261002 11:50:42.365088 12776 feature_extraction.cc:294]   Features:        11265 (SIFT)") is None


def test_matching_blocks():
    line = "I20261002 12:03:03.374410 20092 pairing.cc:214] Processing block [8/19, 2/19]"
    assert progress.matching(line) == pytest.approx((7 * 19 + 2) / 361)
    assert progress.matching("pairing.cc:214] Processing block [19/19, 19/19]") == 1.0


def test_mapper_counts_registered_frames():
    parse = progress.mapper(906)
    line = "I20261002 12:59:53.627285 18304 incremental_pipeline.cc:620] Registering image #16 (num_reg_frames=453)"
    assert parse(line) == 0.5
    assert parse("incremental_pipeline.cc:88] Retriangulation and Global bundle adjustment") is None


@pytest.mark.parametrize("line, fraction, detail", [
    ("2760 (9.20%)        14.956 ms            6 m, 47 s            122.44 M",
     0.092, "iterazione 2761 di 30000, restano circa 6 m, 47 s"),
    # nerfacto, prima riga: manca la colonna dei raggi al secondo
    ("0 (0.00%)           553.094 ms           3 m, 41 s", 0.0, "iterazione 1 di 30000, restano circa 3 m, 41 s"),
    # iterazioni piu' lunghe di un secondo: il tempo per iterazione contiene spazi
    ("260 (86.67%)        1 s, 900.930 ms      1 m, 16 s            4.86 K",
     0.8667, "iterazione 261 di 30000, restano circa 1 m, 16 s"),
    ("270 (90.00%)        1 s, 906.870 ms      57 s, 206.101 ms     4.85 K",
     0.9, "iterazione 271 di 30000, restano circa 57 s, 206.101 ms"),
    ("29999 (100.00%)", None, None),
])
def test_training_rows(line, fraction, detail):
    assert progress.train(line) == (pytest.approx(fraction) if fraction is not None else None)
    assert progress.train_detail(30000)(line) == detail


def test_training_table_is_hidden_from_the_on_screen_log():
    assert progress.is_train_table("2760 (9.20%)        14.956 ms            6 m, 47 s            122.44 M")
    assert progress.is_train_table("Step (% Done)       Train Iter (time)    ETA (time)           Train Rays / Sec")
    assert progress.is_train_table("-----------------------------------------------------------------------------------")
    assert not progress.is_train_table("Viewer running locally at: http://localhost:7007 (listening on 0.0.0.0)")


def test_patch_match_runs_two_passes_per_view():
    parse = progress.patch_match()
    line = "I20261002 11:44:53.480935 16016 patch_match.cc:419] === Processing view 28 / 40 for P1180168.JPG ==="
    values = [parse(line) for _ in range(80)]
    assert values[0] == pytest.approx(1 / 80)
    assert values[39] == 0.5
    assert values[-1] == 1.0


def test_viewer_url_uses_the_port_actually_chosen():
    # Se la 7007 e' occupata nerfstudio ne sceglie un'altra: l'indirizzo va letto dal log.
    assert progress.viewer_url("Viewer running locally at: http://localhost:51274 (listening on 0.0.0.0)") == "http://localhost:51274"
    # ns-viewer stampa solo il riquadro di viser
    assert progress.viewer_url("│   HTTP      │ http://0.0.0.0:56050   │") == "http://localhost:56050"
    assert progress.viewer_url("│   Websocket │ ws://0.0.0.0:56050     │") is None
    assert progress.viewer_url("Loading latest checkpoint from load_dir") is None


def test_alignment_stats():
    stats = progress.AlignmentStats()
    for line in (
        "I20261002 13:02:55.749840 10268 model.cc:437] Registered frames: 906",
        "I20261002 13:02:55.749852 10268 model.cc:440] Registered images: 904",
        "I20261002 13:02:55.749858 10268 model.cc:442] Points: 694759",
        "I20261002 11:21:59.659122 14928 model.cc:445] Mean track length: 4.394317",
        "I20261002 13:02:55.749976 10268 model.cc:450] Mean reprojection error: 1.061005px",
    ):
        stats.feed(line)
    assert stats.values == {"registered_images": 904, "points": 694759, "mean_track_length": 4.394317,
                            "mean_reprojection_error_px": 1.061005}
