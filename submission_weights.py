"""Download and cache the checkpoint bundle used by the submitted classifier."""

import hashlib
import shutil
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


RELEASE_URL = (
    "https://github.com/MotoiHAGIHARA/AI-FDE-ResNet18-Submission/releases/download/"
    "submission-weights-v1/resnet18-weights.tar.gz"
)
RELEASE_SHA256 = "cbdfad89298909c954d445af503be8c7a74ae8ea2488ecae0abe6c742fdbd280"


def available(folder, filenames):
    return all((folder / name).is_file() for name in filenames)


def fetch(folder, filenames):
    if available(folder, filenames):
        return folder
    if folder.exists():
        raise ValueError(f"Incomplete weights in {folder}. Move that folder aside and retry.")

    print("Downloading the submitted model weights (625 MB)...", file=sys.stderr)
    with TemporaryDirectory(prefix=".download-weights-", dir=folder.parent) as work:
        archive_path = Path(work) / "weights.tar.gz"
        checksum = hashlib.sha256()
        with urlopen(RELEASE_URL, timeout=120) as response, archive_path.open("wb") as archive:
            while block := response.read(1024 * 1024):
                checksum.update(block)
                archive.write(block)
        if checksum.hexdigest() != RELEASE_SHA256:
            raise ValueError("Downloaded model weights failed the SHA-256 check")

        unpacked = Path(work) / "checkpoints"
        with tarfile.open(archive_path, "r:gz") as archive:
            for name in filenames:
                member = archive.getmember(f"submission-weights/{name}")
                if not member.isfile():
                    raise ValueError(f"Expected a regular file in the weights archive: {name}")
                dest = unpacked / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, dest.open("wb") as output:
                    shutil.copyfileobj(source, output)
        try:
            unpacked.rename(folder)
        except OSError:
            if not available(folder, filenames):
                raise
    return folder
