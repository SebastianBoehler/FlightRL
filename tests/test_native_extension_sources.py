from pathlib import Path
import tomllib


def test_native_extension_links_door_airframe_mask_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    setup_source = (root / "setup.py").read_text()

    assert '"src/flightrl/native/native_door_self_mask.c"' in setup_source


def test_native_extension_does_not_disable_nonfinite_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    setup_source = (root / "setup.py").read_text()

    assert "-ffast-math" not in setup_source


def test_native_sources_and_includes_ship_in_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())

    assert project["tool"]["setuptools"]["package-data"]["flightrl"] == [
        "native/*.c",
        "native/*.h",
        "native/*.inc",
    ]
    assert "recursive-include src/flightrl/native *.c *.h *.inc" in (
        root / "MANIFEST.in"
    ).read_text()
