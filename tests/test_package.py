from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_package_imports() -> None:
    import dream_handoff

    assert dream_handoff.__name__ == "dream_handoff"


def test_built_distributions_contain_project_license_and_third_party_notices(
    tmp_path: Path,
) -> None:
    output = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel,) = output.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        license_name = next(name for name in names if name.endswith("/licenses/LICENSE"))
        notices_name = next(
            name for name in names if name.endswith("/licenses/THIRD_PARTY_NOTICES.md")
        )
        optimizer_name = next(name for name in names if name.endswith("/r2dreamer/optimization.py"))
        metadata = archive.read(metadata_name).decode()
        license_text = archive.read(license_name).decode()
        notices_text = archive.read(notices_name).decode()
        optimizer_text = archive.read(optimizer_name).decode()

    assert not any(name.endswith("/NOTICE") for name in names)
    assert "Author: Paul Dähn\n" in metadata
    assert "License-Expression: Apache-2.0\n" in metadata
    assert {line for line in metadata.splitlines() if line.startswith("License-File: ")} == {
        "License-File: LICENSE",
        "License-File: THIRD_PARTY_NOTICES.md",
    }
    assert license_text == (ROOT / "LICENSE").read_text()
    assert notices_text == (ROOT / "THIRD_PARTY_NOTICES.md").read_text()

    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert "NM512/r2dreamer" in notices_text
    assert "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f" in notices_text
    assert "MIT License" in notices_text
    assert "Copyright (c) 2026 Naoki Morihira" in notices_text
    assert "Permission is hereby granted, free of charge" in notices_text
    assert "The above copyright notice and this permission notice shall be included" in notices_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in notices_text
    assert "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS" in notices_text
    assert "Copyright (c) 2020 Wang, T. Zhikang" in optimizer_text
    assert "Permission is hereby granted, free of charge" in optimizer_text
    assert (
        "The above copyright notice and this permission notice shall be included" in optimizer_text
    )
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in optimizer_text

    (sdist,) = output.glob("*.tar.gz")
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_names = archive.getnames()
        sdist_license = next(name for name in sdist_names if name.endswith("/LICENSE"))
        sdist_notices = next(
            name for name in sdist_names if name.endswith("/THIRD_PARTY_NOTICES.md")
        )
        sdist_metadata = next(name for name in sdist_names if name.endswith("/PKG-INFO"))
        assert archive.extractfile(sdist_license).read().decode() == license_text
        assert archive.extractfile(sdist_notices).read().decode() == notices_text
        assert archive.extractfile(sdist_metadata).read().decode() == metadata
    assert not any(name.endswith("/NOTICE") for name in sdist_names)
