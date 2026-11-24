"""Locate and download meshed LDraw parts."""

import hashlib
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import platformdirs

GLB_ARCHIVE_URL = "https://keeper.mpdl.mpg.de/seafhttp/f/737aee1bec9849849791/"
GLB_ARCHIVE_SHA256 = "4bbc99fec48914e24fa06547a4586cc9a1f479280f8c1f5822385d24337c1dd0"


def data_dir() -> Path:
    """Return the shared BrickNet data directory."""

    configured = os.environ.get("BRICKNET_DATA")
    return Path(configured) if configured else Path(platformdirs.user_data_dir("bricknet"))


def default_glb_dir() -> Path:
    """Return the default directory containing part GLBs."""

    return data_dir() / "glb"


def _download(url: str, path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError("%s SHA-256 is %s; expected %s" % (path.name, actual, expected_sha256))


def fetch_glbs(destination: str | Path | None = None) -> Path:
    """Download and install the part GLBs, returning their directory."""

    destination = default_glb_dir() if destination is None else Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bricknet-glb-", dir=destination.parent) as temp:
        temporary = Path(temp)
        archive = temporary / "glb.tar.xz"
        _download(GLB_ARCHIVE_URL, archive, GLB_ARCHIVE_SHA256)
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(destination, filter="data")
    return destination
