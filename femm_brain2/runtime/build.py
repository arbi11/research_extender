"""Build the isolated FEMM fix from checked local images without pulling sources."""
import argparse
from pathlib import Path
import subprocess

IMAGES = {
    "femm-brain2-build-base:85f07dce379b": "sha256:85f07dce379bff58dc539f8ed49ffba53a94eba332fbf81d4c9fd6c3df94725d",
    "femm-brain2-compiler:77ff8c6affdd": "sha256:77ff8c6affdd6a6fe905076c8849620de31f715911016f28c7fb02b70b0b59f6",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="femm-brain2:mesh-count-20261125")
    args = parser.parse_args()
    for alias, expected in IMAGES.items():
        result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", expected],
                                capture_output=True, text=True, check=True)
        if result.stdout.strip() != expected:
            raise ValueError("Required local Docker image differs: " + expected)
        # BuildKit needs named references, rather than bare local config IDs.
        subprocess.run(["docker", "tag", expected, alias], check=True)
    root = Path(__file__).resolve().parent
    subprocess.run(["docker", "build", "--network", "none", "-f", str(root / "Dockerfile.mesh-count"),
                    "-t", args.tag, str(root)], check=True)
    subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", args.tag], check=True)


if __name__ == "__main__":
    main()
