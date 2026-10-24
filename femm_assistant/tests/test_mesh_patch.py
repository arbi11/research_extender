"""Exercise the native diagnostic export without a complete FEMM build."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PATCHER = Path(__file__).resolve().parents[1] / "deploy" / "patch_femm_mesh.py"
if not PATCHER.is_file():
    PATCHER = Path("/opt/femm/patch_femm_mesh.py")
spec = importlib.util.spec_from_file_location("patch_femm_mesh", PATCHER)
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class MeshPatchTests(unittest.TestCase):
    def test_patch_rejects_changed_or_repeated_anchors(self):
        source = "#include <cstdio>\n" + patcher.ANCHOR
        patched = patcher.patch_text(source)
        self.assertEqual(patched.count(patcher.MARKER), 1)
        self.assertLess(patched.index(patcher.MARKER), patched.index("auto meshCb"))
        for invalid in ("wrong source", source + patcher.ANCHOR, patched):
            with self.assertRaises(ValueError):
                patcher.patch_text(invalid)

    def test_build_patcher_requires_pinned_source_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "libfemm").mkdir()
            path = root / "libfemm" / "script_callbacks.cpp"
            path.write_text("#include <cstdio>\n" + patcher.ANCHOR)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "checksum"):
                patcher.patch_source(root)
            self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(shutil.which("g++"), "A C++ compiler is required for native export checks")
    def test_native_export_is_opt_in_exact_and_reports_write_failure(self):
        source = r'''
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <vector>
struct Vertex { double x, y; };
struct Triangle { std::array<int,3> v; double region_attrib; };
struct Mesh { std::vector<Vertex> vertices; std::vector<Triangle> triangles; };
int export_mesh(const std::filesystem::path &p) {
    namespace fs = std::filesystem;
    char error[256] = {};
    char *errmsg = error;
    int errmsg_size = sizeof(error);
    Mesh tmesh{{{0,0},{0.12345678901234566,0},{0,1}}, {{{0,1,2},2}}};
''' + patcher.EXPORT_BLOCK + r'''
    return 0;
}
int main(int argc, char **argv) { return export_mesh(argv[1]) == 0 ? 0 : 1; }
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpp, binary = root / "mesh.cpp", root / "mesh"
            cpp.write_text(source)
            subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(binary)], check=True, capture_output=True, timeout=30)
            env = {key: value for key, value in os.environ.items() if key != "FEMM_MESH_EXPORT"}
            model = root / "model.fem"
            subprocess.run([str(binary), str(model)], env=env, check=True, timeout=5)
            self.assertFalse(model.with_suffix(".node").exists())
            env["FEMM_MESH_EXPORT"] = "1"
            subprocess.run([str(binary), str(model)], env=env, check=True, timeout=5)
            nodes = model.with_suffix(".node").read_text().splitlines()
            self.assertEqual(nodes[0], "3 2 0 0")
            self.assertEqual(nodes[1], "0 0 0")
            self.assertEqual(float(nodes[2].split()[1]), 0.12345678901234566)
            self.assertEqual(model.with_suffix(".ele").read_text(), "1 3 1\n0 0 1 2 2\n")
            failed = subprocess.run([str(binary), str(root / "missing" / "model.fem")], env=env, timeout=5)
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
