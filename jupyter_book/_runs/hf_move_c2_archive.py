"""Move the C2 files archived by _runs/c2_lean/archive_unreferenced.sh to their
archive path on the Hugging Face dataset repo, so a download mirrors the lean C2.

  jupyter_book/Chp2/content/<path>  ->  jupyter_book/_archive/C2_removed_2026-10-10/<path>

Safety:
  * dry run unless --apply is given;
  * aborts if any HF file under Chp2/content that is no longer in local C2 has no
    byte-identical copy in the local archive;
  * step 1 copies/adds everything to the archive path (large files are copied
    server-side, nothing is re-uploaded), step 2 checks every new path exists
    with the right size, and only then step 3 deletes the old paths.

Usage (from anywhere):
  uv run --no-project --with huggingface_hub python ~/code/compem_book/_runs/hf_move_c2_archive.py           # dry run
  uv run --no-project --with huggingface_hub python ~/code/compem_book/_runs/hf_move_c2_archive.py --apply
"""
import hashlib
import os
import sys

from huggingface_hub import (HfApi, CommitOperationAdd, CommitOperationCopy,
                             CommitOperationDelete)

REPO = 'rb1189/research_extender-artifacts'
W = os.path.expanduser('~/code/compem_book')
ARCH = os.path.join(W, '_archive/C2_removed_2026-10-10')
OLD = 'jupyter_book/Chp2/content'
NEW = 'jupyter_book/_archive/C2_removed_2026-10-10'
# quick to regenerate / duplicate: not backed up anywhere on the Hub
SKIP = ('__pycache__/', 'dataset/grid_cache/', 'dataset/coil_cnn_dataset_checkpoint.pkl')
BATCH = 80
APPLY = '--apply' in sys.argv
api = HfApi()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def git_sha1(path):
    d = open(path, 'rb').read()
    return hashlib.sha1(b'blob %d\0' % len(d) + d).hexdigest()


def same(local, entry):
    if os.path.getsize(local) != entry.size:
        return False
    if getattr(entry, 'lfs', None):
        return sha256(local) == entry.lfs.sha256
    return git_sha1(local) == entry.blob_id


def commit(ops, msg):
    for i in range(0, len(ops), BATCH):
        part = ops[i:i + BATCH]
        api.create_commit(REPO, repo_type='dataset', operations=part,
                          commit_message=f'{msg} ({i // BATCH + 1}/{(len(ops) + BATCH - 1) // BATCH})')
        print(f'  committed {i + len(part)}/{len(ops)}', flush=True)


def remote_files():
    return {e.path: e for e in api.list_repo_tree(REPO, repo_type='dataset', recursive=True, expand=True)
            if hasattr(e, 'size')}


remote = remote_files()

# Local archive files to back up
archive = {}
for root, _, files in os.walk(ARCH):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), ARCH)
        if not rel.startswith(SKIP):
            archive[rel] = os.path.join(root, f)

# HF files under Chp2 that local C2 no longer has (what the lean-up moved)
stale = [p for p in remote if p.startswith(OLD + '/')
         and not os.path.exists(os.path.join(W, 'C2', p[len(OLD) + 1:]))]

def is_debug_dump(p):
    # femm_solver(debug_mode=True) rewrites these on every run, and the transformer
    # geometry is random per run, so a newer local copy legitimately differs
    return p.endswith(('_debug.fem', '_debug.ans'))


problems, superseded = [], []
for p in stale:
    rel = p[len(OLD) + 1:]
    if rel in archive and same(archive[rel], remote[p]):
        continue
    if rel in archive and is_debug_dump(rel):
        superseded.append(p)
    else:
        problems.append(p)
for p in superseded:
    print(f'note: {p} differs from the newer local archive copy (FEMM debug dump); the local copy is kept')
if problems:
    print(f'ABORT: {len(problems)} HF file(s) under {OLD} have no identical local archive copy:')
    for p in problems[:20]:
        print('  ', p)
    sys.exit(1)

copy_ops, add_ops, done = [], [], 0
for rel, local in sorted(archive.items()):
    new, old = f'{NEW}/{rel}', f'{OLD}/{rel}'
    if new in remote and same(local, remote[new]):
        done += 1
    elif old in remote and getattr(remote[old], 'lfs', None) and old not in superseded:
        copy_ops.append(CommitOperationCopy(src_path_in_repo=old, path_in_repo=new))
    else:
        add_ops.append(CommitOperationAdd(path_in_repo=new, path_or_fileobj=local))
del_ops = [CommitOperationDelete(path_in_repo=p) for p in sorted(stale)]

# The lean-up also renamed 01_Intro -> 01_intro; its executed copy was re-uploaded
# under the new name, so drop the old one once the new one is confirmed on the Hub
RENAMED = {'jupyter_book/_runs/executed/C2__01_Intro.ipynb':
           ('jupyter_book/_runs/executed/C2__01_intro.ipynb', '_runs/executed/C2__01_intro.ipynb')}
for old_p, (new_p, local_rel) in RENAMED.items():
    local = os.path.join(W, local_rel)
    if old_p in remote and new_p in remote and os.path.exists(local) and same(local, remote[new_p]):
        del_ops.append(CommitOperationDelete(path_in_repo=old_p))
        print(f'note: {old_p} replaced by {new_p} (renamed); the old name will be deleted')

add_mb = sum(os.path.getsize(op.path_or_fileobj) for op in add_ops) / 1e6
print(f'archive files to back up : {len(archive)}  (already in place: {done})')
print(f'  server-side copies     : {len(copy_ops)}')
print(f'  uploads from local     : {len(add_ops)}  ({add_mb:.1f} MB)')
print(f'old paths to delete      : {len(del_ops)}  ({len(stale)} C2 files verified against the local '
      f'archive + {len(del_ops) - len(stale)} renamed)')
if not APPLY:
    print('\nDry run only. Re-run with --apply to make these changes.')
    sys.exit(0)

print('\nStep 1: copy/add to', NEW)
if copy_ops or add_ops:
    commit(copy_ops + add_ops, 'Archive C2 files moved by the lean-up')

print('Step 2: verify')
remote = remote_files()
missing = [rel for rel, local in archive.items()
           if f'{NEW}/{rel}' not in remote or remote[f'{NEW}/{rel}'].size != os.path.getsize(local)]
if missing:
    print(f'ABORT before deleting anything: {len(missing)} archive file(s) not in place, e.g. {missing[:5]}')
    sys.exit(1)
print(f'  all {len(archive)} archive files present with correct sizes')

print('Step 3: delete old C2 paths')
if del_ops:
    commit(del_ops, 'Remove C2 files moved to _archive by the lean-up')
print('Done.')
