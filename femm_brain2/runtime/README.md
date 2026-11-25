# FEMM mesh-count return fix

The retained source in the original runtime creates a mesh but exposes
`*_createmesh` through a void C++ lambda. Lua therefore receives nil.
The patch returns the real `tmesh.vertices.size()` through the shared helper
and the binding. It follows the node-count contract in the pinned API source.

Build a separate image using locally available, digest-pinned images:

```bash
docker build --network none -f femm_brain2/runtime/Dockerfile.mesh-count \
  -t femm-brain2:mesh-count-20261125 femm_brain2/runtime
```

The compiler image supplies an existing toolchain and build cache. Source is
restored from the original runtime's hash-checked archive; its existing mesh
export patch is reapplied, then three hash-checked files receive the return fix.
Only the rebuilt headless executable is copied into a child of the original
runtime. Existing images, solver executables and running services are retained.
The new image includes both patch scripts, the original source archive and
`/opt/femm/mesh-count-manifest.json` with source and binary fingerprints.

Physical inputs, analytical targets and tolerances are unchanged. Native checks
must compare the returned count with the generated mesh and the solved `.ans`
mesh; the value is never substituted in the Lua client or checker.
