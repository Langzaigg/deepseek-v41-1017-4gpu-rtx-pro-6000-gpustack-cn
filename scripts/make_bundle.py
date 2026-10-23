#!/usr/bin/env python3
"""Build a deterministic standalone source deployment ZIP from an explicit allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "deepseek-v41-1017-4gpu-docker-cn-1.1.0"
FILES = [".dockerignore", "Dockerfile", "launch.py", "LICENSE", "PROVENANCE.txt", "vendor-lock.json",
         "deploy-docker.sh", "download-model.sh", "scripts/deploy.py", "scripts/standalone.py",
         "profiles/ram-4gpu.json"]


def payloads(root=ROOT):
    files = {name: (root / name).read_bytes() for name in FILES}
    for name, expected in json.loads(files["vendor-lock.json"])["files"].items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Vendor hash mismatch: {name}")
        files[name] = data
    files["README.md"] = (root / "standalone/README.md").read_bytes()
    files[".gitignore"] = b".state/\n.venv*/\n__pycache__/\n*.pyc\n*.key\n.env*\nmodels/\ncache/\n"
    # Normalize only authored text; retain byte-identical third-party sources/licenses.
    for name in files:
        if not name.startswith(("adapter/", "runtime/", "licenses/")):
            files[name] = files[name].replace(b"\r\n", b"\n")
    files["SHA256SUMS.txt"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())
    ).encode()
    return files


def build(output, root=ROOT):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads(root).items()):
            entry = zipfile.ZipInfo(f"{NAME}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    output = ROOT / "packages" / f"{NAME}.zip"
    digest = build(output)
    (output.parent / "SHA256SUMS.txt").write_text(f"{digest}  {output.name}\n", encoding="utf-8", newline="\n")
    print(f"{output.name}\nSHA256 {digest}")
