# Test harness for femm_brain_daro_v2, put in front of every generated batch by run_femm.py.
# Ordinary Python 3 (in the FEMM image) drives FEMM through femm-qt's pyFEMM (femm.py), over TCP,
# the way a user script does: from femm import *, then openfemm(1).
import math
import os
import re
import sys
import textwrap

sys.path.insert(0, "/opt/femm/share/femm/pyfemm")
import femm  # noqa: E402
from femm import *  # noqa: E402,F401,F403


def check(cond, message=""):
    """Fail the test with `message` unless cond is true."""
    if not cond:
        raise AssertionError("check failed: " + str(message))


def near(a, b, rel=1e-6):
    """True when a is within rel (relative) of b."""
    return abs(a - b) <= rel * max(abs(b), 1e-30)


def show(*values):
    """Join any number of values into one observation string."""
    return ", ".join(str(v) for v in values)


def fails(call, *args):
    """Run call(*args); return the text of the error it raises, or None when it raises nothing."""
    try:
        call(*args)
    except Exception as error:
        return " ".join(str(error).split())
    return None


def alive():
    """True when pyFEMM is connected to a FEMM that still answers."""
    if femm._default is None:
        return False
    try:
        return femm._default.callfemm("0") == 0
    except Exception:
        return False


def run(name, source):
    """Run one test: `source` is the body of a function that raises to fail and returns an
    observation (anything but None) to pass. Print a RESULT line for run_femm.py."""
    try:
        scope = {}
        code = compile("def _test():\n" + textwrap.indent(source, "    "), "<" + name + ">", "exec")
        exec(code, globals(), scope)
        observed = scope["_test"]()
        status, detail = ("pass", observed) if observed is not None else ("fail", "test returned nothing")
    except Exception as error:
        status, detail = "fail", type(error).__name__ + ": " + str(error)
    print("RESULT\t" + name + "\t" + status + "\t" + " ".join(str(detail).split()), flush=True)
    # Close whatever the test left open, and relaunch FEMM if the test closed it or it died.
    for close in (mo_close, mi_close):
        try:
            close()
        except Exception:
            pass
    if not alive():
        femm._default = None
        openfemm(1)

# Fixtures shared by the tests. Lengths in millimetres, planar, depth 10 mm.
#
#   outer circle r = 50 at the origin ............ boundary "A0" (A = 0)
#   wire: circle r = 1 at the origin .............. material "Copper", circuit "Iw" = 100 A, group 1
#   air-gap band between circles r = 10 and r = 11  boundary "AGap" (periodic air gap), label "<No Mesh>"
#   block labels: wire (0,0); inner air (5,0); band (0,10.5); outer air (30,0)
#   fixture_pre() only: "Iron" square, corners (23,-2) and (27,2), label (25,0), group 2
#
# Every circle is two 180-degree arcs: upper half from (r,0) to (-r,0), lower half back.


def circle(r, maxseg):
    mi_drawarc(r, 0, -r, 0, 180, maxseg)
    mi_drawarc(-r, 0, r, 0, 180, maxseg)


def label(x, y, material, circuit=None, turns=0, group=0, meshsize=None):
    """Add a block label and set its properties; without meshsize the mesher chooses (automesh)."""
    mi_addblocklabel(x, y)
    mi_clearselected()
    mi_selectlabel(x, y)
    mi_setblockprop(material, 1 if meshsize is None else 0, meshsize or 0, circuit or "<None>", 0,
                    group or 0, turns or 0)
    mi_clearselected()


def arcprop(x, y, maxseg, boundary):
    mi_clearselected()
    mi_selectarcsegment(x, y)
    mi_setarcsegmentprop(maxseg, boundary, 0, 0)
    mi_clearselected()


def fixture_geometry():
    newdocument(0)
    mi_probdef(0, "millimeters", "planar", 1e-8, 10, 30, 0)
    mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
    mi_addmaterial("Copper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0, 1, 0)
    mi_addmaterial("Iron", 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
    mi_addcircprop("Iw", 100, 1)
    mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    mi_addboundprop("AGap", 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0)
    circle(1, 5)
    circle(10, 1)
    circle(11, 1)
    circle(50, 5)
    for y in (10, -10, 11, -11):
        arcprop(0, y, 1, "AGap")
    arcprop(0, 50, 5, "A0")
    arcprop(0, -50, 5, "A0")
    label(0, 0, "Copper", "Iw", 1, 1, 0.2)
    label(5, 0, "Air", None, 0, 0, 0.4)
    label(0, 10.5, "<No Mesh>")  # the air-gap band must be a no-mesh hole or the solver fails
    label(30, 0, "Air")


def blank(depth=10):
    """A planar millimetre document with the "Air" material and boundary "A0" only."""
    newdocument(0)
    mi_probdef(0, "millimeters", "planar", 1e-8, depth, 30, 0)
    mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)
    mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


def solve(name):
    """Save the open document as /tmp/<name>.fem, solve it and load the solution."""
    mi_saveas("/tmp/" + name + ".fem")
    mi_analyze(1)
    mi_loadsolution()


def fixture_pre():
    fixture_geometry()
    mi_drawrectangle(23, -2, 27, 2)
    label(25, 0, "Iron", None, 0, 2)


def fixture_post():
    """Build, solve and load the fixture; post-processor tests then read it."""
    fixture_geometry()
    solve("post")


openfemm(1)
run('mo_clearblock', 'fixture_post()\nmo_selectblock(0, 0)\nwire = mo_blockintegral(5)\nr = mo_clearblock()\nmo_selectblock(5, 0)\nair = mo_blockintegral(5)\nmo_clearblock()\nnone = fails(mo_blockintegral, 5)\nannulus = (180 * math.sin(math.pi / 180) * 100 - 36 * math.sin(5 * math.pi / 180)) * 1e-6\nobs = f\'returned {r}; wire {wire:.6g} m^2; after mo_clearblock the inner air alone {air:.6g} m^2 (annulus {annulus:.6g}); with nothing selected mo_blockintegral(5) raised "{none}"\'\ncheck(near(air, annulus, 1e-4), obs)\ncheck(none is not None and \'No area selected for integration\' in none, obs)\nreturn obs')
run('mo_groupselectblock', 'fixture_post()\nmo_groupselectblock(1)\nwire = mo_blockintegral(5)\nmo_clearblock()\nmo_groupselectblock()\nwhole = mo_blockintegral(5)\nmo_clearblock()\nmo_groupselectblock(None)\nwhole_none = mo_blockintegral(5)\nmo_clearblock()\nmo_groupselectblock(1)\nmo_groupselectblock(1)\ntwice = fails(mo_blockintegral, 5)\nmo_clearblock()\nmo_groupselectblock(99)\nnogroup = fails(mo_blockintegral, 5)\nmo_clearblock()\nmeshed = (36 * math.sin(5 * math.pi / 180) * 2500 - 180 * math.sin(math.pi / 180) * 21) * 1e-6\nobs = f\'group 1: {wire:.6g} m^2; no argument: {whole:.6g} m^2, None: {whole_none:.6g} m^2 (meshed area {meshed:.6g}); group 1 selected twice, then mo_blockintegral(5) raised "{twice}"; group 99 (no blocks): "{nogroup}"\'\ncheck(near(wire, 36 * math.sin(5 * math.pi / 180) * 1e-6, 1e-4), obs)\ncheck(near(whole, meshed, 1e-4) and near(whole_none, whole, 1e-12), obs)\ncheck(twice is not None and \'No area selected\' in twice, obs)\ncheck(nogroup is not None and \'No area selected\' in nogroup, obs)\nreturn obs')
run('mo_selectblock', 'fixture_post()\nmo_selectblock(0, 0)\nwire = mo_blockintegral(5)\nmo_selectblock(5, 0)\nboth = mo_blockintegral(5)\nmo_selectblock(5, 0)\nagain = mo_blockintegral(5)\nmo_clearblock()\nmo_selectblock(100, 100)\nmo_selectblock(0, 10.5)\nempty = fails(mo_blockintegral, 5)\nmo_clearblock()\nobs = f\'wire {wire:.6g} m^2; plus the inner air {both:.6g} m^2; selecting (5,0) a second time leaves {again:.6g} m^2; points outside the model and in the <No Mesh> band selected nothing: "{empty}"\'\ncheck(near(wire, 36 * math.sin(5 * math.pi / 180) * 1e-6, 1e-4), obs)\ncheck(near(both, 180 * math.sin(math.pi / 180) * 100e-6, 1e-4), obs)\ncheck(near(again, wire, 1e-9), obs)\ncheck(empty is not None and \'No area selected\' in empty, obs)\nreturn obs')
run('mo_seteditmode', 'fixture_post()\nmo_addcontour(2, 0)\nmo_addcontour(2, 5)\nmo_seteditmode(\'area\')\nlength = mo_lineintegral(2)[0]\nmo_selectblock(0, 0)\nmo_seteditmode(\'contour\')\narea = mo_blockintegral(5)\nr = mo_seteditmode(\'point\')\nbad = fails(mo_seteditmode, \'nonsense\')\nmo_clearcontour()\nmo_clearblock()\nobs = f\'after "area" the contour still measures {length:.6g} m; after "contour" the wire is still selected ({area:.6g} m^2); mo_seteditmode returned {r}; mode "nonsense" raised {bad}\'\ncheck(near(length, 0.005, 1e-6) and near(area, 36 * math.sin(5 * math.pi / 180) * 1e-6, 1e-4), obs)\ncheck(bad is None, obs)\nreturn obs')
run('mo_getcircuitproperties', "fixture_post()\ni, v, flux = mo_getcircuitproperties('Iw')\nunknown = fails(mo_getcircuitproperties, 'NoSuch')\nmo_close()\nmi_close()\nmu0 = 4e-7 * math.pi\nresistance = 0.010 / (58e6 * 36 * math.sin(5 * math.pi / 180) * 1e-6)\nlinkage = 0.010 * 100 * mu0 / (2 * math.pi) * (math.log(50) + 0.25)\ncheck(abs(i - 100) < 1e-6, 'current ' + str(i))\ncheck(abs(v / (100 * resistance) - 1) < 0.01, 'volts ' + str(v) + ', expected ' + str(100 * resistance))\ncheck(abs(flux / linkage - 1) < 0.01, 'flux linkage ' + str(flux) + ', expected ' + str(linkage))\ncheck(unknown is not None and 'Unknown circuit' in unknown, 'unknown circuit: ' + str(unknown))\nfixture_geometry()\nmi_probdef(50)\nsolve('circuit_ac')\nia, va, fa = mo_getcircuitproperties('Iw')\nmo_close()\nomega = 2 * math.pi * 50\ncheck(isinstance(va, complex) and isinstance(fa, complex) and ia == 100, 'AC types ' + show(repr(ia), repr(va), repr(fa)))\ncheck(near(va.imag, omega * fa.real, 1e-3) and near(va.real, v, 1e-3), 'AC volts ' + str(va) + ', omega x flux ' + str(omega * fa))\nreturn 'current %.6g A; volts %.5g V (I R = %.5g V); flux linkage %.5g Wb (analytic %.5g Wb); unknown circuit: %s; at 50 Hz: current %r (%s), volts %.5g%+.4gj V (omega x flux linkage = %.4g), flux %.5g%+.2gj Wb' % (i, v, 100 * resistance, flux, linkage, unknown, ia, type(ia).__name__, va.real, va.imag, omega * fa.real, fa.real, fa.imag)")
run('mo_getprobleminfo', "fixture_post()\ninfo = mo_getprobleminfo()\ncheck(isinstance(info, list) and len(info) == 4, 'mo_getprobleminfo() -> ' + repr(info))\nptype, freq, depth, unit = info\ncheck(ptype == 0 and freq == 0, 'type, frequency ' + show(ptype, freq))\ncheck(near(depth, 0.01, 1e-9) and near(unit, 0.001, 1e-9), 'depth, unit ' + show(depth, unit))\nreturn 'mo_getprobleminfo() -> %s for a planar mm model, 10 mm deep, at 0 Hz' % info")
run('mo_close', 'fixture_geometry()\nsolve(\'b_close\')\nby = mo_getb(5, 0)[1]\nmo_close()\nafter = fails(mo_numnodes)\nagain = fails(mo_close)\nxy = mi_selectlabel(5, 0)\nmi_clearselected()\ncallfemm(\'mo_setfocus("b_close")\')\nback = mo_getb(5, 0)[1]\ncheck(after is not None and \'No magnetics output in focus\' in after and again is not None, show(after, again))\ncheck(near(xy[0], 5) and abs(xy[1]) < 1e-9, \'pre-processor label \' + show(xy))\ncheck(back == by, \'the closed output came back with \' + show(back, by))\np = femm._default._proc\ncheck(p is not None, \'FEMM was not launched by this session, so its memory cannot be read\')\ndef mb():\n    with open(\'/proc/%d/status\' % p.pid) as f:\n        return int(re.search(r\'VmRSS:\\s*(\\d+)\', f.read()).group(1)) / 1024\nfor k in range(4):\n    mi_analyze(1)\n    mi_loadsolution()\nm0 = mb()\nfor k in range(3):\n    mi_analyze(1)\n    mi_loadsolution()\nm1 = mb()\nfor k in range(5):\n    mi_analyze(1)\n    mi_loadsolution()\n    mo_close()\nm2 = mb()\ncheck(m1 - m0 < 2 and m2 - m1 > 10, f\'memory {m0:.1f} -> {m1:.1f} -> {m2:.1f} MB\')\nreturn f\'after mo_close, mo_numnodes raises "{after}"; mi_selectlabel(5, 0) still finds \' + show(*xy) + f\'; mo_setfocus("b_close") brought the closed output back (By {back:.5g} T); FEMM memory: +{m1 - m0:.1f} MB over 3 more reloads without mo_close (after 4 to warm up), +{m2 - m1:.1f} MB over 5 load-and-close\'')
run('mo_gettitle', "fixture_post()\nt = mo_gettitle()\nmi_saveas('/tmp/b_Iron_title.fem')\nmi_analyze(1)\nmi_loadsolution()\ng = mo_gettitle()\nreal_text = ''.join(map(chr, callfemm('[ord(c) for c in mo_gettitle()]')))\nmo_close()\nclosed = fails(mo_gettitle)\ncheck(t == '/tmp/post.ans', show(t))\ncheck(g == '/tmp/b_1jron_t1jtle.ans' and real_text == '/tmp/b_Iron_title.ans', show(g, real_text))\ncheck(closed is not None and 'No magnetics output in focus' in closed, show(closed))\nreturn 'for /tmp/post.ans mo_gettitle() -> ' + t + '; for /tmp/b_Iron_title.ans -> ' + g + ' (real text ' + real_text + '); after mo_close: ' + closed")
run('mo_savebitmap', 'import time\nfixture_post()\npaths = [\'/tmp/mo_view.png\', \'/tmp/mo_view.bmp\', \'/tmp/mo_view2.png\', \'/tmp/mo_view.png.bmp\', \'/tmp/mo_view.png.png\']\nfor path in paths:\n    if os.path.exists(path):\n        os.remove(path)\nr1 = mo_savebitmap(paths[0])\nr2 = mo_savebitmap(paths[1])\nmo_showdensityplot(1, 0, 0.01, 0, \'bmag\')\nmo_zoomnatural()\nmo_refreshview()\nr3 = callfemm(\'mo_savebitmap("\' + paths[2] + \'")\')\ntime.sleep(1)\nfound = [path for path in paths if os.path.exists(path)]\ncheck(r1 is None and r2 is None and r3 == [], show(r1, r2, r3))\ncheck(found == [], \'files written: \' + show(*found))\nreturn \'returned \' + show(r1, r2) + \' (FEMM replied \' + show(r3) + \'); no file for .png, .bmp, or .png after a density plot, 1 s later\'')
run('mo_savemetafile', 'fixture_post()\nif os.path.exists(\'/tmp/mo_view.emf\'):\n    os.remove(\'/tmp/mo_view.emf\')\nr = mo_savemetafile(\'/tmp/mo_view.emf\')\nraw = callfemm(\'mo_savemetafile("/tmp/mo_view.emf")\')\nwritten = os.path.exists(\'/tmp/mo_view.emf\')\ncheck(r is None and raw == [] and not written, show(r, raw, written))\nreturn \'ran headless, returned None (FEMM replied \' + show(raw) + \'); no file was written\'')
run('mo_setfocus', 'missing = fails(lambda: mo_setfocus(\'x\'))\nfixture_geometry()\nsolve(\'b_focus1\')\nmo_close()\nmi_saveas(\'/tmp/b_focus2.fem\')\nmi_modifycircprop(\'Iw\', 1, 200)\nmi_analyze(1)\nmi_loadsolution()\nt2, by2 = mo_gettitle(), mo_getb(5, 0)[1]\nr = callfemm(\'mo_setfocus("b_focus1")\')\nt1, by1 = mo_gettitle(), mo_getb(5, 0)[1]\ncallfemm(\'mo_setfocus("b_focus2.ans")\')\nt3, by3 = mo_gettitle(), mo_getb(5, 0)[1]\ncallfemm(\'mo_setfocus("nosuch")\')\nt4 = mo_gettitle()\ncallfemm(\'mo_setfocus("b_focus1")\')\nmi_saveas(\'/tmp/b_focus3.fem\')\nmi_analyze(1)\nmi_loadsolution()\nt5 = mo_gettitle()\ncallfemm(\'mo_setfocus("b_focus2")\')\nt6 = mo_gettitle()\ncallfemm(\'mo_setfocus("b_focus1")\')\nt7 = mo_gettitle()\nmo_close()\nmi_saveas(\'/tmp/b_focus1.fem\')\nmi_modifycircprop(\'Iw\', 1, 300)\nmi_analyze(1)\nmi_loadsolution()\nby_new = mo_getb(5, 0)[1]\ncallfemm(\'mo_setfocus("b_focus1")\')\nby_pick = mo_getb(5, 0)[1]\ncheck(missing == "name \'mo_setfocus\' is not defined", show(missing))\ncheck(r == [] and [t1, t2, t3, t4] == [\'/tmp/b_focus1.ans\', \'/tmp/b_focus2.ans\', \'/tmp/b_focus2.ans\', \'/tmp/b_focus2.ans\'], show(r, t1, t2, t3, t4))\ncheck(near(by2, 2 * by1, 1e-3) and by3 == by2, show(by1, by2, by3))\ncheck([t5, t6, t7] == [\'/tmp/b_focus3.ans\', \'/tmp/b_focus3.ans\', \'/tmp/b_focus1.ans\'], \'with the focus on b_focus1, loading b_focus3 left: \' + show(t5, t6, t7))\ncheck(near(by_new, 3 * by1, 1e-3) and by_pick == by1, show(by_new, by_pick))\nreturn f\'pyFEMM: {missing}; titles {t2} | {t1} | {t3} | {t4} (nosuch); By at (5,0): b_focus2 {by2:.5g}, after mo_setfocus("b_focus1") {by1:.5g}, after mo_setfocus("b_focus2.ans") {by3:.5g}; with the focus moved to b_focus1, loading b_focus3 replaced b_focus2, the last loaded (mo_setfocus("b_focus2") stayed on {t6}), and kept b_focus1 ({t7}); b_focus1 re-solved at 300 A and loaded again ({by_new:.5g} T), then mo_setfocus("b_focus1") gave the old {by_pick:.5g} T\'')
run('mo_addcontour', "fixture_post()\nmo_clearcontour()\nfirst = mo_addcontour(2, 0)\nmo_addcontour(2, 5)\nlength, area = mo_lineintegral(2)\nmo_addcontour(7, 5)\nlonger = mo_lineintegral(2)[0]\nmo_clearcontour()\nobs = ('contour (2,0) -> (2,5) mm: mo_lineintegral(2) = ' + show(length, area) + '; adding (7,5) extends it to '\n       + format(longer, '.6g') + ' m; mo_addcontour returned ' + repr(first))\ncheck(first is None, obs)\ncheck(near(length, 0.005) and near(area, 0.005 * 0.010), obs)\ncheck(near(longer, 0.010), obs)\nreturn obs + ' (length in m; length x depth in m^2)'")
run('mo_bendcontour', "fixture_post()\nmo_clearcontour()\nmo_addcontour(5, 0)\nmo_addcontour(-5, 0)\nstraight = mo_lineintegral(2)[0]\nmo_bendcontour(180, 1)\narc = mo_lineintegral(2)[0]\nht = mo_lineintegral(1)[0]\nmo_clearcontour()\nmo_addcontour(5, 0)\nmo_addcontour(-5, 0)\nmo_bendcontour(-180, 1)\nhtneg = mo_lineintegral(1)[0]\nmo_clearcontour()\nmo_addcontour(5, 0)\nmo_addcontour(-5, 0)\nmo_bendcontour(270, 1)\nover = mo_lineintegral(2)[0]\nmo_clearcontour()\nchords = 180 * 2 * 0.005 * math.sin(math.radians(0.5))\nobs = ('straight %.6g m; bent +180 by 1 deg steps %.7g m (180 chords %.7g m), H.t %.5g A; '\n       'bent -180: H.t %.5g A; bent 270: %.6g m' % (straight, arc, chords, ht, htneg, over))\ncheck(near(straight, 0.010), obs)\ncheck(near(arc, chords), obs)\ncheck(abs(ht - 50) < 0.5 and abs(htneg + 50) < 0.5, obs)\ncheck(near(over, 0.010), obs)\nreturn obs")
run('mo_blockintegral', "# Results are SI whatever the drawing unit, and include the depth (10 mm here).\nfixture_post()\nmo_clearblock()\nselected = mo_selectblock(0, 0)\narea = mo_blockintegral(5)\ncurrent = mo_blockintegral(7)\nvolume = mo_blockintegral(10)\nloss = mo_blockintegral(4)\naj = mo_blockintegral(0)\nunknown = mo_blockintegral(31)\nmo_selectblock(0, 0)\ntwice = fails(mo_blockintegral, 5)\ngrouped = mo_groupselectblock(1)\ngroup = mo_blockintegral(5)\nmo_groupselectblock(1)\ngroup_twice = fails(mo_blockintegral, 5)\nmo_clearblock()\nflux = mo_getcircuitproperties('Iw')[2]\nmo_close()\nmi_close()\n# DC resistive loss I^2 R with R = depth / (sigma A).\ni2r = 100 * 100 * 0.010 / (58e6 * area)\npolygon = 36 * math.sin(math.radians(5)) * 1e-6\nobs = ('wire: area %.6g m^2 (72-gon, r = 1 mm), current %.6g A, volume %.6g m^3 = area x depth; '\n       'resistive loss %.5g W (I^2 R = %.5g W); A.J / I = %.5g Wb (flux linkage %.5g Wb); type 31 -> %r; '\n       'block selected twice: %s; group 1 %.6g m^2, selected twice: %s; the select wrappers returned %r, %r'\n       % (area, current, volume, loss, i2r, aj / 100, flux, unknown, twice, group, group_twice, selected, grouped))\ncheck(abs(area / polygon - 1) < 1e-3, obs)\ncheck(abs(current - 100) < 0.1, obs)\ncheck(near(volume, area * 0.010), obs)\ncheck(abs(loss / i2r - 1) < 0.01, obs)\ncheck(near(aj / 100, flux, 1e-6), obs)\ncheck(unknown == 0, obs)\ncheck(twice is not None and 'No area selected' in twice, obs)\ncheck(near(group, area) and group_twice is not None and 'No area selected' in group_twice, obs)\ncheck(selected is None and grouped is None, obs)\n# 50 Hz: phasor quantities come back complex, losses and energy real, as time averages for peak amplitudes.\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('blockac')\nmo_selectblock(0, 0)\niac = mo_blockintegral(7)\najac = mo_blockintegral(0)\nlossac = mo_blockintegral(4)\nenergyac = mo_blockintegral(2)\nareaac = mo_blockintegral(5)\nmo_clearblock()\nobs += ('; at 50 Hz: current %r, A.J %r, loss %.5g W (I^2 R / 2 = %.5g W), energy %r, area %r'\n        % (iac, ajac, lossac, i2r / 2, energyac, areaac))\ncheck(isinstance(iac, complex) and abs(iac - 100) < 0.1 and isinstance(ajac, complex), obs)\ncheck(isinstance(lossac, float) and abs(lossac / (i2r / 2) - 1) < 0.01, obs)\ncheck(isinstance(energyac, float) and isinstance(areaac, float) and near(areaac, area), obs)\nreturn obs")
run('mo_clearcontour', "fixture_post()\nmo_addcontour(2, 0)\nmo_addcontour(2, 5)\nr = mo_clearcontour()\nempty = mo_lineintegral(2)\nplot = fails(mo_makeplot, 1, 5, '/tmp/mo_noplot.csv')\nmo_addcontour(3, 0)\nmo_addcontour(3, 2)\nlength = mo_lineintegral(2)[0]\nmo_clearcontour()\nobs = ('a new 2 mm contour after clearing measures %.6g m; on the cleared contour mo_lineintegral(2) returned %r '\n       'and mo_makeplot raised %r; mo_clearcontour returned %r' % (length, empty, plot, r))\ncheck(near(length, 0.002), obs)\ncheck(empty == [0, 0], obs)\ncheck(plot is not None and 'No contour defined' in plot, obs)\ncheck(r is None, obs)\nreturn obs")
run('mo_getalla', "fixture_post()\nne = mo_numelements()\na = mo_getalla()\nraw = callfemm('mo_getalla()')\nk = ne // 2\ne1, ek = mo_getelement(1), mo_getelement(k)\na1 = mo_getpointvalues(e1[3], e1[4])[0]\nak = mo_getpointvalues(ek[3], ek[4])[0]\nmo_close()\nmi_close()\nobs = ('%d values for %d elements (raw callfemm %d); A at the centroids of elements 1 and %d: %r, %r against list items %r, %r'\n       % (len(a), ne, len(raw), k, a1, ak, a[0], a[k - 1]))\ncheck(isinstance(a, list) and len(a) == ne and len(raw) == ne, obs)\ncheck(near(a[0], a1, 1e-9) and near(a[k - 1], ak, 1e-9), obs)\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('allaac')\naac = mo_getalla()\ncheck(len(aac) == mo_numelements() and isinstance(aac[0], complex), repr(aac[:2]))\nreturn obs + '; at 50 Hz complex, first ' + repr(aac[0])")
run('mo_getallb', "fixture_post()\nne = mo_numelements()\nb = mo_getallb()\nk = ne // 2\ne1, ek = mo_getelement(1), mo_getelement(k)\nsmoothed = mo_getpointvalues(e1[3], e1[4])[1:3]\nmo_smooth('off')\nown1 = mo_getpointvalues(e1[3], e1[4])[1:3]\nownk = mo_getpointvalues(ek[3], ek[4])[1:3]\nmo_smooth('on')\nmo_close()\nmi_close()\nobs = ('%d pairs for %d elements; pair 1 %r; unsmoothed B at the centroid of element 1 %r, smoothed %r; '\n       'pair %d %r, unsmoothed %r' % (len(b), ne, b[0], own1, smoothed, k, b[k - 1], ownk))\ncheck(len(b) == ne and all(len(p) == 2 for p in b), obs)\ncheck(near(b[0][0], own1[0], 1e-9) and near(b[0][1], own1[1], 1e-9), obs)\ncheck(near(b[k - 1][0], ownk[0], 1e-9) and near(b[k - 1][1], ownk[1], 1e-9), obs)\ncheck(not near(b[0][1], smoothed[1], 1e-3), obs)\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('allbac')\nbac = mo_getallb()\ncheck(len(bac) == mo_numelements() and isinstance(bac[0][1], complex), repr(bac[:1]))\nreturn obs + '; at 50 Hz complex, pair 1 ' + repr(bac[0])")
run('mo_getallelements', "import time\nfixture_post()\nne, nn = mo_numelements(), mo_numnodes()\nt = time.time()\nrows = mo_getallelements()\nbulk = time.time() - t\nt = time.time()\nloop = [mo_getelement(k) for k in range(1, ne + 1)]\nlooped = time.time() - t\nraw = callfemm('mo_getallelements()')\nlo = min(min(r[:3]) for r in rows)\nhi = max(max(r[:3]) for r in rows)\nobs = ('%d rows of %r values for %d elements (raw callfemm %d values); identical to the mo_getelement loop: %s; '\n       'node indices %d..%d (mo_numnodes %d); first row %r; %.2f s against %.2f s for the loop'\n       % (len(rows), sorted(set(len(r) for r in rows)), ne, len(raw), rows == loop, lo, hi, nn, rows[0], bulk, looped))\ncheck(len(rows) == ne and all(len(r) == 7 for r in rows) and len(raw) == 7 * ne, obs)\ncheck(rows == loop, obs)\ncheck(lo == 1 and hi == nn, obs)\nreturn obs")
run('mo_getallgapa', "fixture_post()\nfull = mo_getallgapa('AGap')\neight = mo_getallgapa('AGap', 8)\nmany = mo_getallgapa('AGap', 1000)\none = mo_getallgapa('AGap', 1)\nzero = mo_getallgapa('AGap', 0)\nwrong = mo_getallgapa('Agap')\na0, a1, a45 = mo_getgapa('AGap', 0), mo_getgapa('AGap', 1), mo_getgapa('AGap', 45)\nobs = ('default %d values, first two %r, %r (mo_getgapa at 0 and 1 deg: %r, %r); npts 8: %d values, item 1 %r '\n       '(mo_getgapa at 45 deg %r); npts 1000: %d values; npts 1: %r; npts 0: %r; misspelt name: %r'\n       % (len(full), full[0], full[1], a0, a1, len(eight), eight[1], a45, len(many), one, zero, wrong))\ncheck(len(full) == 360 and len(eight) == 8 and len(many) == 1000, obs)\ncheck(near(full[0], a0, 1e-9) and near(full[1], a1, 1e-9) and near(eight[1], a45, 1e-9), obs)\ncheck(len(one) == 1 and near(one[0], a0, 1e-9) and zero == one and wrong == [], obs)\nreturn obs")
run('mo_getallgapb', "fixture_post()\npairs = mo_getallgapb('AGap')\nseven = mo_getallgapb('AGap', 7)\nwrong = mo_getallgapb('Agap')\ng0, g90 = mo_getgapb('AGap', 0), mo_getgapb('AGap', 90)\ng3 = mo_getgapb('AGap', 3 * 360 / 7)\nmean = sum(p[1] for p in pairs) / len(pairs)\nexpected = 2e-7 * 100 / 0.0105\nobs = ('default %d pairs, pair 0 %r and pair 90 %r (mo_getgapb at 0 and 90 deg %r, %r); npts 7: pair 3 %r '\n       '(mo_getgapb at %.4g deg %r); mean Bt %.6g T (mu0 I / (2 pi r) = %.6g); misspelt name %r'\n       % (len(pairs), pairs[0], pairs[90], g0, g90, seven[3], 3 * 360 / 7, g3, mean, expected, wrong))\ncheck(len(pairs) == 360 and all(len(p) == 2 for p in pairs) and len(seven) == 7, obs)\ncheck(all(near(pairs[0][i], g0[i], 1e-9) and near(pairs[90][i], g90[i], 1e-9) and near(seven[3][i], g3[i], 1e-9)\n          for i in (0, 1)), obs)\ncheck(near(mean, expected, 1e-3), obs)\ncheck(wrong == [], obs)\nreturn obs")
run('mo_getallgapharmonics', "fixture_post()\nrows = mo_getallgapharmonics('AGap')\nnmax = mo_getgapharmonics('AGap')\nh0 = mo_getgapharmonics('AGap', 0)\nh1 = mo_getgapharmonics('AGap', 1)\nh2 = mo_getgapharmonics('AGap', 2)\nth = math.radians(30)\na30 = sum(r[0] * math.cos(n * th) + r[1] * math.sin(n * th) for n, r in enumerate(rows))\nbr30 = sum(r[2] * math.cos(n * th) + r[3] * math.sin(n * th) for n, r in enumerate(rows))\nbt30 = sum(r[4] * math.cos(n * th) + r[5] * math.sin(n * th) for n, r in enumerate(rows))\nga30 = mo_getgapa('AGap', 30)\ngbr30, gbt30 = mo_getgapb('AGap', 30)\nextra = fails(mo_getallgapharmonics, 'AGap', 5)\nwrong = mo_getallgapharmonics('Agap')\nobs = ('%d rows of 6 (highest harmonic %d); row 0 %r against mo_getgapharmonics(name, 0) %r; at 30 deg the rows sum to '\n       'A %.8g, Br %.4g, Bt %.8g against mo_getgapa %.8g and mo_getgapb %.4g, %.8g; a second argument: %s; misspelt name %r'\n       % (len(rows), nmax, rows[0], h0, a30, br30, bt30, ga30, gbr30, gbt30, extra, wrong))\ncheck(len(rows) == nmax + 1 and all(len(r) == 6 for r in rows), obs)\ncheck(near(rows[0][0], h0[0], 1e-9) and near(rows[0][4], 2e-7 * 100 / 0.0105, 1e-3) and h0[4] == 0, obs)\ncheck(all(near(rows[1][i], h1[i], 1e-9) and near(rows[2][i], h2[i], 1e-9) for i in range(6)), obs)\ncheck(near(a30, ga30, 1e-9) and near(bt30, gbt30, 1e-9) and abs(br30 - gbr30) < 1e-12, obs)\ncheck(extra is not None and 'positional argument' in extra and wrong == [], obs)\nreturn obs")
run('mo_getallnodes', "import time\nfixture_post()\nnn = mo_numnodes()\nt = time.time()\nrows = mo_getallnodes()\nbulk = time.time() - t\nt = time.time()\nloop = [mo_getnode(k) for k in range(1, nn + 1)]\nlooped = time.time() - t\na, b, c = mo_getelement(1)[:3]\ncorners = [rows[a - 1], rows[b - 1], rows[c - 1]]\nobs = ('%d pairs for %d nodes; identical to the mo_getnode loop: %s; first %r; element 1 uses nodes %d, %d, %d = items %d, %d, %d; '\n       '%.2f s against %.2f s for the loop' % (len(rows), nn, rows == loop, rows[0], a, b, c, a - 1, b - 1, c - 1, bulk, looped))\ncheck(len(rows) == nn and all(len(r) == 2 for r in rows), obs)\ncheck(rows == loop, obs)\ncheck(corners == [mo_getnode(a), mo_getnode(b), mo_getnode(c)], obs)\nreturn obs")
run('mo_getelement', "fixture_post()\nne, nn = mo_numelements(), mo_numnodes()\nlo, hi, total, wire = nn, 0, 0.0, 0.0\nfor k in range(1, ne + 1):\n    a, b, c, xc, yc, area, group = mo_getelement(k)\n    lo = min(lo, a, b, c)\n    hi = max(hi, a, b, c)\n    total += area\n    if group == 1:\n        wire += area\na, b, c, xc, yc, area, group = mo_getelement(1)\nax, ay = mo_getnode(a)\nbx, by = mo_getnode(b)\ncx, cy = mo_getnode(c)\ntri = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2\nzero, negative, beyond = mo_getelement(0), mo_getelement(-1), mo_getelement(ne + 1)\nhalf = fails(mo_getelement, 1.5)\npolygon = 36 * math.sin(math.radians(5))\nmeshed = polygon * 2500 - 180 * math.sin(math.radians(1)) * 21\nobs = ('%d elements use node indices %d..%d (mo_numnodes %d); element 1 = %s, node triangle %.6g; areas sum to %.6g '\n       '(meshed %.6g mm^2), group 1 to %.6g (wire %.6g); mo_getelement(0), (-1), (%d) -> %r, %r, %r; mo_getelement(1.5): %s'\n       % (ne, lo, hi, nn, show(a, b, c, xc, yc, area, group), tri, total, meshed, wire, polygon, ne + 1, zero, negative, beyond, half))\ncheck(lo == 1 and hi == nn and zero == [] and negative == [] and beyond == [], obs)\ncheck(near(tri, area, 1e-4), obs)\ncheck(near(total, meshed, 1e-4) and near(wire, polygon, 1e-4), obs)\ncheck(half is not None and 'non-integer' in half, obs)\nreturn obs")
run('mo_getgapa', "fixture_post()\na0 = mo_getgapa('AGap', 0)\na90 = mo_getgapa('AGap', 90)\na360 = mo_getgapa('AGap', 360)\nam90, a270 = mo_getgapa('AGap', -90), mo_getgapa('AGap', 270)\nwrong = mo_getgapa('Agap', 0)\nmo_close()\nmi_close()\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('gapaac')\naac = mo_getgapa('AGap', 0)\nexpected = 2e-7 * 100 * math.log(50 / 10.5)\nobs = ('A on the centre line (r = 10.5 mm): %r Wb/m at 0 deg, %.6g at 90 deg; analytic %.6g; 360 deg %r; '\n       '-90 deg %r and 270 deg %r; misspelt name %r; at 50 Hz %r' % (a0, a90, expected, a360, am90, a270, wrong, aac))\ncheck(isinstance(a0, float), obs)\ncheck(near(a0, expected, 2e-3) and near(a90, a0, 1e-3), obs)\ncheck(near(a360, a0, 1e-12) and near(am90, a270, 1e-12), obs)\ncheck(wrong == 0, obs)\ncheck(isinstance(aac, complex) and near(aac, a0, 1e-3), obs)\nreturn obs")
run('mo_getgapb', "# Br, Bt on the centre line of the air-gap band (r = 10.5 mm) round a +100 A wire.\nfixture_post()\nbr, bt = mo_getgapb('AGap', 0)\nwrong = mo_getgapb('Agap', 0)\nmo_close()\nmi_close()\nexpected = 2e-7 * 100 / 0.0105\ncheck(abs(br) < 0.02 * expected, 'Br = ' + repr(br))\ncheck(abs(bt / expected - 1) < 0.02, 'Bt = ' + repr(bt) + ', expected +' + repr(expected))\ncheck(wrong == [0, 0], 'misspelt name: ' + repr(wrong))\n# Angle reference: a uniform By = 1 T (prescribed A = -0.001 x on the outer circle).\n# From +x counter-clockwise: Br = 0, Bt = +1 at 0 deg; Br = +1, Bt = 0 at 90 deg.\nfixture_geometry()\nmi_addboundprop('Uniform', 0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0)\narcprop(0, 50, 5, 'Uniform')\narcprop(0, -50, 5, 'Uniform')\nsolve('gapangle')\nr0, t0 = mo_getgapb('AGap', 0)\nr90, t90 = mo_getgapb('AGap', 90)\nmo_close()\nmi_close()\ncheck(abs(r0) < 0.05 and abs(t0 - 1) < 0.05, 'uniform By, 0 deg: ' + show(r0, t0))\ncheck(abs(r90 - 1) < 0.05 and abs(t90) < 0.05, 'uniform By, 90 deg: ' + show(r90, t90))\n# The band: no label fails, an Air label halves Bt, anti-periodic gives 0.\nfixture_geometry()\nmi_selectlabel(0, 10.5)\nmi_deleteselectedlabels()\nmi_clearselected()\nnolabel = fails(solve, 'gapnolabel')\nmi_close()\nfixture_geometry()\nmi_selectlabel(0, 10.5)\nmi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)\nmi_clearselected()\nsolve('gapair')\nbtair = mo_getgapb('AGap', 0)[1]\nmo_close()\nmi_close()\nfixture_geometry()\nmi_modifyboundprop('AGap', 9, 7)\nsolve('gapanti')\nbtanti = mo_getgapb('AGap', 0)[1]\nmo_close()\nmi_close()\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('gapbac')\nac = mo_getgapb('AGap', 0)\nobs = ('wire: Br = %.2g T, Bt = %.5g T (mu0 I/(2 pi r) = %.5g T); uniform By = 1 T: (Br, Bt) = (%.3f, %.3f) at 0 deg, '\n       '(%.3f, %.3f) at 90 deg; no band label: %s; Air in the band: Bt = %.5g T; anti-periodic: Bt = %.2g T; '\n       'misspelt name %r; at 50 Hz %r' % (br, bt, expected, r0, t0, r90, t90, nolabel, btair, btanti, wrong, ac))\ncheck(nolabel is not None and '2560' in nolabel, obs)\ncheck(abs(btair / bt - 0.5) < 0.02, obs)\ncheck(abs(btanti) < 1e-9, obs)\ncheck(isinstance(ac[1], complex) and near(ac[1], bt, 1e-3), obs)\nreturn obs")
print("BATCH_DONE", flush=True)
closefemm()
