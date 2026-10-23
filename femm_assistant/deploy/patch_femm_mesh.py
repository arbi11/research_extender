"""Add an opt-in diagnostic mesh export to the pinned upstream FEMM source.

FEMM 5 keeps pre-solve mesh triangles in memory. This patch copies that exact
mesh to Triangle-compatible ASCII files; it does not change meshing or solving.
The original archive and this patcher are retained with the Docker image.
"""
import hashlib
from pathlib import Path
import sys


SOURCE_SHA256 = "401591a90a089694091cb25ad2f8fe4892b4aa763ffcbf2e5cc4dfb021ae2fd6"
MARKER = "// research_extender: opt-in diagnostic mesh export."
ANCHOR = '''    Mesh tmesh;
    int ret = tangle_mesh_fem(full, tmesh);
    if (ret != 0) {
        if (errmsg) snprintf(errmsg, errmsg_size, "Mesher failed (error %d)", ret);
        return ret;
    }

    auto meshCb = get_mesh_loaded_callback();'''

EXPORT_BLOCK = r'''    // research_extender: opt-in diagnostic mesh export.
    // Copy the actual pre-solve mesh; do not rerun or alter the mesher.
    const char *meshExport = std::getenv("FEMM_MESH_EXPORT");
    if (meshExport && std::strcmp(meshExport, "1") == 0) {
        fs::path nodePath = p, elePath = p;
        nodePath.replace_extension(".node");
        elePath.replace_extension(".ele");
        std::ofstream nodes(nodePath, std::ios::out | std::ios::trunc);
        std::ofstream elements(elePath, std::ios::out | std::ios::trunc);
        nodes.imbue(std::locale::classic());
        elements.imbue(std::locale::classic());
        nodes << std::setprecision(17);
        elements << std::setprecision(17);
        if (!nodes || !elements) {
            if (errmsg) snprintf(errmsg, errmsg_size, "Could not open diagnostic mesh files");
            return -1;
        }
        nodes << tmesh.vertices.size() << " 2 0 0\n";
        for (size_t i = 0; i < tmesh.vertices.size(); ++i) {
            const auto &v = tmesh.vertices[i];
            nodes << i << ' ' << v.x << ' ' << v.y << '\n';
        }
        elements << tmesh.triangles.size() << " 3 1\n";
        for (size_t i = 0; i < tmesh.triangles.size(); ++i) {
            const auto &t = tmesh.triangles[i];
            elements << i << ' ' << t.v[0] << ' ' << t.v[1] << ' '
                     << t.v[2] << ' ' << t.region_attrib << '\n';
        }
        nodes.close();
        elements.close();
        if (!nodes || !elements) {
            if (errmsg) snprintf(errmsg, errmsg_size, "Could not write complete diagnostic mesh files");
            return -1;
        }
    }

'''


def patch_text(source: str) -> str:
    """Transform the one audited mesh callback, rejecting ambiguous matches."""
    if MARKER in source:
        raise ValueError("FEMM diagnostic mesh patch is already applied.")
    if source.count(ANCHOR) != 1 or source.count("#include <cstdio>\n") != 1:
        raise ValueError("Unexpected FEMM source: diagnostic mesh patch anchor mismatch.")
    source = source.replace("#include <cstdio>\n", "#include <cstdio>\n#include <cstdlib>\n#include <fstream>\n#include <iomanip>\n#include <locale>\n", 1)
    return source.replace(ANCHOR, ANCHOR.replace("    auto meshCb", EXPORT_BLOCK + "    auto meshCb"), 1)


def patch_source(root: Path) -> None:
    path = root / "libfemm" / "script_callbacks.cpp"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected FEMM source checksum; refusing diagnostic mesh patch.")
    source = raw.decode("utf-8").replace("\r\n", "\n")
    path.write_text(patch_text(source), encoding="utf-8")


if __name__ == "__main__":
    patch_source(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/femm-source"))
