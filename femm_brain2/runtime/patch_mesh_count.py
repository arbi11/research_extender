"""Backport FEMM's mesh node-count return to the retained, hash-pinned source."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ARCHIVE_SHA256 = "0ae785ab5f12b32f9529f479e9f1cbafea18aa23b32976d8ea559a08e016450d"
EXPORT_PATCH_SHA256 = "1f884ec852d23936dc4d15c77e44890ef71d4d91048a44ba8704e9a11be5dd81"
INPUT_SHA256 = {
    "libfemm/py_editor_common.h": "7c179f9f6522858598cbf0895cf432bfd1a603d525777b895eb0140a6bd3a82b",
    "libfemm/script_callbacks.cpp": "491d6df41af7f7acce529f9a35d5701c543bfe6754280dba68773e6d8d59bb4c",
    "libfemm/script_callbacks.h": "a2a19bfc609288b0beafe6db8c05139f4830f3b62fd08676624e392ccf9db2a8",
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Unexpected FEMM source: mesh-count patch anchor mismatch")
    return text.replace(before, after)


def patch(source, archive, export_patch, manifest):
    if digest(archive.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("Unexpected original FEMM archive")
    if digest(export_patch.read_bytes()) != EXPORT_PATCH_SHA256:
        raise ValueError("Unexpected original mesh-export patch")
    # Restore the exact retained source, reusing unchanged build inputs and objects.
    # Never take source changes from the cached compiler image on trust.
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            parts = Path(entry.filename).parts
            if not parts or parts[0] != "femm-qt" or ".." in parts:
                raise ValueError("Unexpected FEMM archive path")
            if entry.is_dir():
                continue
            target = source.joinpath(*parts[1:])
            content = package.read(entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.read_bytes() != content:
                target.write_bytes(content)
    spec = importlib.util.spec_from_file_location("retained_mesh_export", export_patch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_source(source)
    for name, expected in INPUT_SHA256.items():
        if digest((source / name).read_bytes()) != expected:
            raise ValueError("Unexpected pre-patch FEMM source: " + name)
    texts = {name: (source / name).read_text() for name in INPUT_SHA256}
    name = "libfemm/py_editor_common.h"
    texts[name] = replace_once(texts[name], '''    m.def(P + "_createmesh", [&s_doc, errmsg]() {
        CHK();
        char msg[256] = {};
        int ret = createMeshForDocument(s_doc, msg, sizeof(msg));
        if (ret != 0) throw std::runtime_error(msg);
    });''', '''    m.def(P + "_createmesh", [&s_doc, errmsg]() -> int {
        CHK();
        char msg[256] = {};
        int nodes = 0;
        int ret = createMeshForDocument(s_doc, msg, sizeof(msg), &nodes);
        if (ret != 0) throw std::runtime_error(msg);
        return nodes;
    });''')
    name = "libfemm/script_callbacks.h"
    texts[name] = replace_once(texts[name],
        "int createMeshForDocument(DocumentBase *doc, char *errmsg = nullptr, int errmsg_size = 0);",
        "int createMeshForDocument(DocumentBase *doc, char *errmsg = nullptr, int errmsg_size = 0, int *node_count = nullptr);")
    name = "libfemm/script_callbacks.cpp"
    texts[name] = replace_once(texts[name],
        "int createMeshForDocument(DocumentBase *doc, char *errmsg, int errmsg_size)",
        "int createMeshForDocument(DocumentBase *doc, char *errmsg, int errmsg_size, int *node_count)")
    texts[name] = replace_once(texts[name],
        "    // research_extender: opt-in diagnostic mesh export.",
        "    if (node_count) *node_count = static_cast<int>(tmesh.vertices.size());\n\n"
        "    // research_extender: opt-in diagnostic mesh export.")
    for name, text in texts.items():
        (source / name).write_text(text)
    manifest.write_text(json.dumps({
        "purpose": "Return the actual pre-solve mesh vertex count; meshing and solving algorithms are unchanged.",
        "source_archive_sha256": ARCHIVE_SHA256,
        "retained_export_patch_sha256": EXPORT_PATCH_SHA256,
        "mesh_count_patch_sha256": digest(Path(__file__).read_bytes()),
        "source_files_before": INPUT_SHA256,
        "source_files_after": {name: digest((source / name).read_bytes()) for name in texts},
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--export-patch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    patch(args.source, args.archive, args.export_patch, args.manifest)
