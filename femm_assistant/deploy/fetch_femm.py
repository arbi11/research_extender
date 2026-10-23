"""Fetch the exact upstream source, verify it, and locate its build root."""
import hashlib
from pathlib import Path
import ssl
import urllib.request
import zipfile

URL = "https://femm.info/doku/lib/exe/fetch.php?media=femm-qt%3Adist%3Afemm-qt-py-src_06oct2026_bd03d0b.zip"
SHA256 = "0ae785ab5f12b32f9529f479e9f1cbafea18aa23b32976d8ea559a08e016450d"
certificate = Path("/run/secrets/proxy_ca")
context = ssl.create_default_context(cafile=str(certificate) if certificate.exists() else None)
archive = Path("/tmp/femm5-source.zip")
request = urllib.request.Request(URL, headers={
    "User-Agent": "Mozilla/5.0 (compatible; research-extender FEMM build)",
    "Accept": "application/zip,application/octet-stream;q=0.9,*/*;q=0.8",
})
with urllib.request.urlopen(request, context=context, timeout=120) as response, archive.open("wb") as output:
    while chunk := response.read(1024 * 1024):
        output.write(chunk)
with archive.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
        raise SystemExit("FEMM source checksum mismatch. Refusing to build.")
destination = Path("/tmp/femm-source")
with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
        path = (destination / member.filename).resolve()
        if not path.is_relative_to(destination):
            raise SystemExit("Unsafe path in source archive.")
    source.extractall(destination)
    for member in source.infolist():
        mode = (member.external_attr >> 16) & 0o777
        if mode:
            (destination / member.filename).chmod(mode)
roots = list(destination.rglob("linux_build.sh"))
if len(roots) != 1:
    raise SystemExit("Expected exactly one FEMM build root.")
roots[0].parent.rename("/opt/femm-source")
