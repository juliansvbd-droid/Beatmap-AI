from beatmap_ai.dataset import iter_beatmap_texts


def test_ranked_dataset_scan_excludes_generated_output(tmp_path):
    root = tmp_path / "BeatMap-AI-Dataset"
    for name in ("123456", "day", "rhythm-v5-2026-09-29"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "map.osu").write_text("AudioFilename: audio.mp3\n", encoding="utf-8")
        (folder / "audio.mp3").write_bytes(b"placeholder")

    names = [name for name, *_ in iter_beatmap_texts(root)]
    assert names == ["123456\\map.osu"] or names == ["123456/map.osu"]
