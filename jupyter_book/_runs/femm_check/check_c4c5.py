"""Physics sanity checks for the C4/C5 FEMM environments (book code imported unchanged).

Run from _runs/femm_check/:  ../../.venv/bin/python ../femm_port.py 48021 check_c4c5

C-core (C4 seqTO_env, C5 seqTO_v2_env):
  * book force  = mo_blockintegral(19) on armature group x (-10)
  * check 1     = raw block integral (already N for the 40 mm deep half model)
  * check 2     = Maxwell-stress line integral on an air contour around the armature
  * check 3     = coil flux linkage -> inductance, vs. a magnetic-circuit estimate
  * C5 vs C4    = same masks must give identical force / reward
SynRM (C4 seqTO_SynRM_env), torque vs rotor angle, three independent methods:
  * book        = mo_blockintegral(22) on rotor design-cell group
  * gap         = mo_gapintegral('AirGap', 0)   (air-gap element torque)
  * virtual work= dW'/dtheta from total co-energy, constant current
Field maps (|B| + flux lines) are written as PNGs to plots/.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import femm

HERE = os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(PLOTS, exist_ok=True)
MU0 = 4e-7 * np.pi
RESULTS = {}

# No display on this box: always launch femm_headless.
_open = femm.openfemm
femm.openfemm = lambda *a: _open(1)


def sample_field(xs, ys):
    """|B| and A on a grid of points (current solution must be loaded)."""
    B = np.full((len(ys), len(xs)), np.nan)
    A = np.full((len(ys), len(xs)), np.nan)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            try:
                bx, by = femm.mo_getb(x, y)
                B[iy, ix] = np.hypot(bx, by)
                A[iy, ix] = femm.mo_geta(x, y)
            except Exception:
                pass
    return B, A


# ════════════════════════════════════════════════════════════════════════
# C-core
# ════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(ROOT, 'C4'))
import seqTO_env as c4

CCORE = {}          # tag -> dict filled by the closefemm hook
_tag = [None]
def _real_close():
    # Keep one femm-qt process: book code calls closefemm/openfemm per solve, and femm-qt's
    # quit + immediate relaunch on the same port races (connection reset). Close documents only.
    for f in (femm.mo_close, femm.mi_close):
        try: f()
        except Exception: pass
femm.closefemm = _real_close


def _ccore_hook():
    tag = _tag[0]
    if tag is not None:
        d = {}
        femm.mo_clearblock()
        femm.mo_groupselectblock(5)
        d['raw_fy'] = float(femm.mo_blockintegral(19))
        d['raw_fx'] = float(femm.mo_blockintegral(18))
        femm.mo_clearblock()
        # Stress-tensor line integral on air contour around the armature
        # (open along the x=0 symmetry plane, where Bx = 0 so it adds no y-force).
        femm.mo_clearcontour()
        for x, y in [(0, 26.5), (15.5, 26.5), (15.5, 33.5), (0, 33.5)]:
            femm.mo_addcontour(x, y)
        fl = femm.mo_lineintegral(3)
        femm.mo_clearcontour()
        d['line_fx'], d['line_fy'] = float(fl[0]), float(fl[1])
        i, v, lam = femm.mo_getcircuitproperties('icoil')
        d['coil_I'], d['coil_lambda'] = float(np.real(i)), float(np.real(lam))
        # B in the 1 mm gap between design window (y<=26) and armature (y>=27)
        d['B_gap'] = [float(np.hypot(*femm.mo_getb(x, 26.5))) for x in np.arange(0.5, 15, 1.0)]
        xs = np.arange(0.1, 30, 0.25)
        ys = np.arange(-8, 45, 0.25)
        d['_field'] = (xs, ys) + sample_field(xs, ys)
        CCORE[tag] = d
    _real_close()


def plot_ccore(tag, mask, title):
    xs, ys, B, A = CCORE[tag]['_field']
    # mirror about the x=0 symmetry plane to show the whole device
    xf = np.concatenate([-xs[::-1], xs])
    Bf = np.concatenate([B[:, ::-1], B], axis=1)
    Af = np.concatenate([A[:, ::-1], A], axis=1)   # A is even about x=0 (Neumann plane)
    fig, ax = plt.subplots(figsize=(7, 7))
    pc = ax.pcolormesh(xf, ys, Bf, cmap='magma', vmin=0, vmax=max(1.0, np.nanpercentile(Bf, 99.5)), shading='auto')
    ax.contour(xf, ys, Af, levels=30, colors='w', linewidths=0.5, alpha=0.7)
    def box(x0, x1, y0, y1, c, lab=None):
        for s in (1, -1):
            ax.add_patch(plt.Rectangle((min(s*x0, s*x1), y0), abs(x1-x0), y1-y0, fill=False, ec=c, lw=1.2,
                                       label=lab if s == 1 else None))
    box(0, 9, 2, 5, 'cyan', 'coil (-)')
    box(0, 9, 11, 14, 'lime', 'coil (+)')
    box(0, 15, 27, 33, 'deepskyblue', 'armature')
    for i, j in np.argwhere(mask == 1):
        for s in (1, -1):
            ax.add_patch(plt.Rectangle((s*i if s == 1 else -i-1, j), 1, 1, fill=False, ec='0.6', lw=0.3))
    ax.axvline(0, color='0.5', ls=':', lw=0.8)
    ax.set_aspect('equal'); ax.set_xlim(-25, 25); ax.set_ylim(-6, 42)
    ax.set_xlabel('x [mm]  (x=0 symmetry plane; left half mirrored)'); ax.set_ylabel('y [mm]')
    ax.set_title(title, fontsize=9); ax.legend(loc='lower right', fontsize=7)
    fig.colorbar(pc, ax=ax, label='|B| [T]', shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, f'ccore_{tag}.png'), dpi=130); plt.close(fig)


def ccore_masks():
    R, C = c4.ROWS, c4.COLS
    empty = np.zeros((R, C), int)
    full = np.array([[1 if c4.is_design_cell(i, j) else 0 for j in range(C)] for i in range(R)])
    # conventional: same construction as c4.test_conventional_topology()
    conv = np.zeros((R, C), int)
    for i in range(*c4.DW_ROWS):
        for j in list(range(5, 10)) + list(range(20, 26)):
            if c4.is_design_cell(i, j): conv[i, j] = 1
    for i in range(c4.DW_ROWS[1]-3, c4.DW_ROWS[1]):
        for j in range(*c4.DW_COLS):
            if c4.is_design_cell(i, j): conv[i, j] = 1
    if conv.sum() > c4.K2:
        for idx in np.argwhere(conv == 1)[c4.K2:]:
            conv[idx[0], idx[1]] = 0
    # a closed C around the right coil: core leg y 5-10, back bar, pole leg y 20-25
    ccl = np.zeros((R, C), int)
    for i in range(0, 15):
        for j in range(*c4.DW_COLS):
            if c4.is_design_cell(i, j) and (j < 11 or j >= 20 or i >= 10):
                ccl[i, j] = 1
    return {'empty': empty, 'conventional': conv, 'closed_C': ccl, 'full_iron': full}


def run_ccore():
    print('\n=== C4 C-core (seqTO_env) ===', flush=True)
    t0 = time.time()
    c4.build_base_geometry()
    print(f'  geometry built in {time.time()-t0:.1f}s', flush=True)
    femm.closefemm = _ccore_hook
    masks = ccore_masks()
    for tag, m in masks.items():
        _tag[0] = tag
        t0 = time.time()
        reward, force, raw = c4.calculate_reward(m)
        d = CCORE[tag]
        d.update(book_reward=reward, book_force=force, book_raw=raw, n_iron=int(m.sum()),
                 solve_s=time.time()-t0)
        print(f"  {tag:13s} iron={d['n_iron']:3d}  book F={force:8.3f} N (raw {raw:+.4f})  "
              f"line Fy={d['line_fy']:+.4f}  Fx_blk={d['raw_fx']:+.4f}  "
              f"lambda={d['coil_lambda']:.3e} Wb  B_gap max={max(d['B_gap']):.3f} T  "
              f"reward={reward:.1f}  ({d['solve_s']:.1f}s)", flush=True)
        plot_ccore(tag, m, f"C4 C-core '{tag}' ({d['n_iron']} iron cells): book force {force:.2f} N, "
                           f"raw Maxwell {raw:+.3f} N (half model)")
    _tag[0] = None
    femm.closefemm = _real_close

    # Run the book's own MDP episode test to be sure the env loop works on femm-qt.
    np.random.seed(0)
    c4.test_mdp_episode(n_steps=5)

    # ── C5 v2 env on identical masks ──
    print('\n=== C5 C-core (seqTO_v2_env) ===', flush=True)
    sys.path.insert(0, os.path.join(ROOT, 'C5'))
    import seqTO_v2_env as c5
    c5.build_base_geometry('ccore_seqto_c5check.fem')
    for tag, m in masks.items():
        mb, fN, flux = c5._solve_and_extract(m, 'ccore_seqto_c5check.fem', '_tmp_c5check.fem')
        CCORE[tag]['c5_force'], CCORE[tag]['c5_reward'] = fN, mb
        CCORE[tag]['c5_flux'] = flux
        print(f"  {tag:13s} C5 F={fN:8.3f} N  reward={mb:.1f}   (C4: {CCORE[tag]['book_force']:.3f} N, "
              f"{CCORE[tag]['book_reward']:.1f})", flush=True)
    fig, axs = plt.subplots(1, len(masks), figsize=(14, 4))
    for ax, (tag, m) in zip(axs, masks.items()):
        im = ax.imshow(CCORE[tag]['c5_flux'].T * 2.0, origin='lower', cmap='magma', vmin=0, vmax=1.5,
                       extent=[0, 18, 0, 35])
        ax.contour(np.arange(18) + .5, np.arange(35) + .5, m.T, levels=[.5], colors='c', linewidths=.6)
        ax.set_title(f'C5 flux channel: {tag}', fontsize=8); ax.set_xlabel('x [mm]')
    fig.colorbar(im, ax=axs, label='|B| [T] (channel x flux_norm)', shrink=.8)
    fig.savefig(os.path.join(PLOTS, 'c5_flux_channel.png'), dpi=120); plt.close(fig)

    for tag in masks:
        RESULTS[f'ccore_{tag}'] = {k: v for k, v in CCORE[tag].items() if not k.startswith('_') and k != 'c5_flux'}


# ════════════════════════════════════════════════════════════════════════
# SynRM
# ════════════════════════════════════════════════════════════════════════
def run_synrm():
    print('\n=== C4 SynRM (seqTO_SynRM_env) ===', flush=True)
    import seqTO_SynRM_env as sy
    t0 = time.time()
    sy.build_synrm_geometry()
    print(f'  geometry built in {time.time()-t0:.1f}s; I_A,I_B,I_C = {sy.IA:.2f}, {sy.IB:.2f}, {sy.IC:.2f} A', flush=True)

    rings = np.zeros((sy.DD_ROWS, sy.DD_COLS), int); rings[[0, 2, 4], :] = 1
    rib = np.zeros_like(rings); rib[:, 2] = 1
    # book's controller pattern end state (RIGHT RIGHT UP*4 LEFT LEFT DOWN DOWN from (0,0))
    ctrl = np.zeros_like(rings)
    for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (4, 1), (4, 0), (3, 0), (2, 0)]:
        ctrl[i, j] = 1
    # q-axis barriers: iron everywhere except the two radial bands i=1,3 at j!=2 (flux along d at 45 deg)
    barrier = np.ones_like(rings); barrier[1, [0, 1, 3, 4]] = 0; barrier[3, [0, 1, 3, 4]] = 0
    masks = {'empty': np.zeros_like(rings), 'rings(book conventional)': rings, 'full_iron': np.ones_like(rings),
             'controller_pattern': ctrl, 'rib_45deg': rib, 'd-axis_barriers': barrier}

    # Book reward on each mask (book's own calculate_reward, 6 angles)
    for tag, m in masks.items():
        t0 = time.time()
        r, tq = sy.calculate_reward(m)
        RESULTS[f'synrm_{tag}'] = {'book_reward_max_abs_T': r, 'book_per_angle': tq, 'solve_s': time.time() - t0}
        print(f"  {tag:26s} book reward={r:.4f}  per-angle={[round(t, 4) for t in tq]}  ({time.time()-t0:.1f}s)", flush=True)

    # Fine sweep with three torque methods (same geometry & helpers, our loop)
    angles = np.arange(0, 45.01, 3.0)
    femm.openfemm(1)
    femm.opendocument(os.path.abspath(sy.BASE_FEM_FILE))
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, (tag, m) in zip(axs.flat, masks.items()):
        sy._reset_design_cells_to_air()
        for i, j in np.argwhere(m == 1):
            femm.mi_selectlabel(*sy._cell_centroid(i, j))
        if m.any():
            femm.mi_setblockprop('M-19 Steel', 1, 0, '<None>', 0, sy.ROTOR_GROUP, 0)
        femm.mi_clearselected()
        Tb, Tg, W, lam = [], [], [], []
        for k, a in enumerate(angles):
            femm.mi_modifyboundprop('AirGap', 10, a)
            femm.mi_saveas(os.path.abspath('_tmp_synrm_check.fem'))
            femm.mi_analyse(0); femm.mi_loadsolution()
            femm.mo_clearblock(); femm.mo_groupselectblock(sy.ROTOR_GROUP); Tb.append(float(femm.mo_blockintegral(22))); femm.mo_clearblock()
            Tg.append(float(np.real(femm.mo_gapintegral('AirGap', 0))))
            femm.mo_groupselectblock(); W.append(float(femm.mo_blockintegral(17))); femm.mo_clearblock()
            lam.append([float(np.real(femm.mo_getcircuitproperties(p)[2])) for p in ('PhaseA', 'PhaseB', 'PhaseC')])
            if k == 0 and tag in ('rings(book conventional)', 'controller_pattern', 'd-axis_barriers'):
                rr = np.linspace(9, 59.5, 140); th = np.radians(np.linspace(0.25, 89.75, 120))
                Bp = np.full((len(rr), len(th)), np.nan); Ap = Bp.copy()
                for a_i, r_ in enumerate(rr):
                    for t_i, t_ in enumerate(th):
                        try:
                            bx, by = femm.mo_getb(r_*np.cos(t_), r_*np.sin(t_))
                            Bp[a_i, t_i] = np.hypot(bx, by); Ap[a_i, t_i] = femm.mo_geta(r_*np.cos(t_), r_*np.sin(t_))
                        except Exception:
                            pass
                RR, TT = np.meshgrid(rr, th, indexing='ij')
                f2, a2 = plt.subplots(figsize=(6.5, 6))
                pc = a2.pcolormesh(RR*np.cos(TT), RR*np.sin(TT), Bp, cmap='magma', vmin=0, vmax=2.0, shading='auto')
                a2.contour(RR*np.cos(TT), RR*np.sin(TT), Ap, levels=25, colors='w', linewidths=.5, alpha=.7)
                for i, j in np.argwhere(m == 1):
                    r0 = sy.R_HUB + i*sy.DR; t0_, t1_ = np.radians(j*sy.DTHETA), np.radians((j+1)*sy.DTHETA)
                    tt = np.linspace(t0_, t1_, 10)
                    xx = np.r_[r0*np.cos(tt), (r0+sy.DR)*np.cos(tt[::-1]), r0*np.cos(t0_)]
                    yy = np.r_[r0*np.sin(tt), (r0+sy.DR)*np.sin(tt[::-1]), r0*np.sin(t0_)]
                    a2.plot(xx, yy, color='c', lw=.6)
                a2.set_aspect('equal'); a2.set_title(f'SynRM quarter, rotor 0 deg: {tag}\n(cyan = iron design cells)', fontsize=9)
                f2.colorbar(pc, ax=a2, label='|B| [T]', shrink=.8); f2.tight_layout()
                f2.savefig(os.path.join(PLOTS, f"synrm_{tag.split('(')[0]}.png"), dpi=130); plt.close(f2)
            femm.mo_close()
        W = np.array(W); th_rad = np.radians(angles)
        Tvw = np.gradient(W, th_rad)               # dW'/dtheta at constant current
        lam = np.array(lam)
        # stator-frame torque from flux linkages (amplitude-invariant Clarke), quarter model
        ia, ib, ic = sy.IA, sy.IB, sy.IC
        i_al, i_be = (2*ia - ib - ic)/3, (ib - ic)/np.sqrt(3)
        l_al = (2*lam[:, 0] - lam[:, 1] - lam[:, 2])/3; l_be = (lam[:, 1] - lam[:, 2])/np.sqrt(3)
        T_lam = 1.5 * sy.POLE_PAIRS * (l_al*i_be - l_be*i_al) / 4.0   # /4: phase lambda here is one quarter's share
        RESULTS[f'synrm_{tag}'].update(angles_mech=angles.tolist(), T_block=Tb, T_gap=Tg, T_virtual_work=Tvw.tolist(),
                                       coenergy=W.tolist(), lambdas=lam.tolist())
        print(f"  {tag:26s} |T| max  block={np.max(np.abs(Tb)):.4f}  gap={np.max(np.abs(Tg)):.4f}  "
              f"virtual-work={np.max(np.abs(Tvw)):.4f}  flux-linkage(approx)={np.max(np.abs(T_lam)):.4f} N·m", flush=True)
        ax.plot(angles, Tb, 'o-', label='book: blockintegral(22) rotor cells')
        ax.plot(angles, Tg, 's--', label="gapintegral('AirGap')")
        ax.plot(angles, Tvw, '^:', label="virtual work dW'/dθ")
        ax.plot(angles, T_lam, 'x-.', alpha=.6, label='flux-linkage dq (approx.)')
        ax.axhline(0, color='0.6', lw=.5); ax.set_title(tag, fontsize=9)
        for a in (angles[0], angles[-1]): pass
    for ax in axs[-1]: ax.set_xlabel('rotor angle [deg mech]')
    for ax in axs[:, 0]: ax.set_ylabel('torque [N·m]')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle('SynRM torque vs rotor angle (signed), three methods', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, 'synrm_torque_methods.png'), dpi=120); plt.close(fig)
    femm.closefemm()


if __name__ == '__main__':
    which = sys.argv[1:] or ['ccore', 'synrm']
    if 'ccore' in which: run_ccore()
    if 'synrm' in which: run_synrm()
    with open('results.json', 'w') as f:
        json.dump(RESULTS, f, indent=1, default=float)
    print('\nCHECK_DONE')
