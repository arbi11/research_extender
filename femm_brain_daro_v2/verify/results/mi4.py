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
run('mi_gridsnap', 'fixture_pre()\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = [mi_gridsnap(f) for f in (\'on\', \'off\', \'On\', \'OFF\')]\nfinally:\n    del femm._default.callfemm\ncheck(r == [None] * 4, \'returned \' + show(r))\ncheck(sent == [\'mi_gridsnap("on")\', \'mi_gridsnap("off")\', \'mi_gridsnap("On")\', \'mi_gridsnap("OFF")\'], \'sent \' + show(sent))\ne_int = fails(mi_gridsnap, 1)\ne_word = fails(mi_gridsnap, \'yes\')\ne_none = fails(mi_gridsnap)\ncheck(e_int is not None and "can\'t convert \'int\'" in e_int, show(e_int))\ncheck(e_word is not None and \'Unknown option for grid snap\' in e_word, show(e_word))\ncheck(e_none is not None and \'positional argument\' in e_none, show(e_none))\nblank()\nmi_setgrid(1, \'cart\')\nmi_gridsnap(\'off\')\nmi_saveas(\'/tmp/v5_snap_a.fem\')\nmi_gridsnap(\'on\')\nmi_saveas(\'/tmp/v5_snap_b.fem\')\ntexts = []\nfor p in (\'/tmp/v5_snap_a.fem\', \'/tmp/v5_snap_b.fem\'):\n    with open(p) as fh:\n        texts.append(fh.read())\ncheck(texts[0] == texts[1], \'the saved file changed with snapping\')\nmi_addnode(0.3, 0.3)\nx, y = mi_selectnode(0.3, 0.3)\ncheck(near(x, 0.3) and near(y, 0.3), \'node at \' + show(x, y))\nreturn \'sent \' + \' \'.join(sent) + \', no error; 1: \' + e_int + \'; "yes": \' + e_word + \'; snapping on, 1 mm grid: node stayed at \' + show(x, y) + \'; .fem identical with snapping on and off\'')
run('mi_hidegrid', "fixture_pre()\nr1 = mi_hidegrid()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_hidegrid())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_hidenames', "fixture_pre()\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = mi_hidenames()\nfinally:\n    del femm._default.callfemm\nbox = mi_getboundingbox()\ncheck(r is None and sent == ['mi_shownames(0)'], show(r, sent))\ncheck(near(box[0], -50) and near(box[1], 50), 'box ' + show(*box))\nmi_close()\ne = fails(mi_hidenames)\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'mi_hidenames() sent ' + sent[0] + ', returned None, drawing unchanged, and no error with no document open'")
run('mi_maximize', "fixture_pre()\nr1 = mi_maximize()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_maximize())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_minimize', "fixture_pre()\nr1 = mi_minimize()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_minimize())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_refreshview', "fixture_pre()\nr1 = mi_refreshview()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_refreshview())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_resize', "fixture_pre()\nr1 = mi_resize(640, 480)\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_resize(640, 480))\ncheck(e is None, 'with no document open: ' + show(e))\nfixture_pre()\ne2 = fails(mi_resize, 640)\ncheck(e2 is None, 'mi_resize(640): ' + show(e2))\nreturn 'returned None, drawing unchanged, no error with no document open, nor with one argument'")
run('mi_restore', "fixture_pre()\nr1 = mi_restore()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_restore())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_savebitmap', "fixture_pre()\nmi_zoomnatural()\nnames = ['/tmp/v5_sb.png', '/tmp/v5_sb.bmp', '/tmp/v5_sb.gif', '/tmp/v5_sb.jpg', '/tmp/v5_sb.jpeg']\nfor p in names:\n    if os.path.exists(p):\n        os.remove(p)\nr = [mi_savebitmap(p) for p in names]\nheadless = [os.path.exists(p) for p in names]\ncheck(r == [None] * 5 and headless == [False] * 5, show(r, headless))\ne = fails(mi_savebitmap)\ncheck(e is not None and 'positional argument' in e, show(e))\nsaved = femm._default\ngui = femm.FemmSession(port=48021, headless=False)\nfemm._default = gui\ntry:\n    fixture_pre()\n    mi_savebitmap('/tmp/v5_sb_first.png')\n    mi_zoomnatural()\n    heads = {}\n    for p in names:\n        mi_savebitmap(p)\n        with open(p, 'rb') as fh:\n            heads[p.rsplit('.', 1)[1]] = fh.read(24)\n    with open('/tmp/v5_sb_first.png', 'rb') as fh:\n        first = fh.read()\n    with open('/tmp/v5_sb.png', 'rb') as fh:\n        natural = fh.read()\n    mi_savebitmap('/tmp/v5_nodir/sb.png')\n    nodir = os.path.exists('/tmp/v5_nodir/sb.png')\nfinally:\n    femm._default = saved\n    gui.close()\nkinds = {k: ('PNG' if h[1:4] == b'PNG' else 'JPEG' if h[:2] == bytes([255, 216]) else str(h[:4])) for k, h in heads.items()}\npng = heads['png']\nw, h = int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big')\ncheck(kinds == {'png': 'PNG', 'bmp': 'PNG', 'gif': 'PNG', 'jpg': 'JPEG', 'jpeg': 'JPEG'}, show(kinds))\ncheck(first != natural, 'a new view already showed the whole model')\ncheck(not nodir, 'wrote into a missing folder')\ncheck(alive(), 'the headless FEMM did not answer after the GUI check')\nreturn 'headless: no file for .png .bmp .gif .jpg .jpeg, no error; no argument: ' + e + '; GUI build: first picture differs from the one after mi_zoomnatural(); ' + show(kinds) + ', %d x %d px; missing folder: no file, no error' % (w, h)")
run('mi_savemetafile', "fixture_pre()\np = '/tmp/v5_view.emf'\nif os.path.exists(p):\n    os.remove(p)\nr = mi_savemetafile(p)\ne = fails(mi_savemetafile)\ncheck(r is None and e is None and not os.path.exists(p), show(r, e, os.path.exists(p)))\nreturn 'returned None and wrote no ' + p + '; mi_savemetafile() with no argument raised nothing'")
run('mi_setgrid', 'blank()\nmi_setgrid(1, \'cart\')\nmi_saveas(\'/tmp/v5_grid_a.fem\')\nmi_setgrid(5, \'cart\')\nmi_saveas(\'/tmp/v5_grid_b.fem\')\ntexts = []\nfor p in (\'/tmp/v5_grid_a.fem\', \'/tmp/v5_grid_b.fem\'):\n    with open(p) as fh:\n        texts.append(fh.read())\ncheck(texts[0] == texts[1], \'the spacing changed the saved file\')\nmi_addboundprop(\'Uniform\', 0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0)\ncircle(40, 5)\narcprop(0, 40, 5, \'Uniform\')\narcprop(0, -40, 5, \'Uniform\')\nlabel(20, 20, \'Air\')\nmi_setgrid(1, \'polar\')\nmi_setgrid(2)\nsolve(\'v5_polar\')\nA, Bx, By = mo_getpointvalues(0, 0)[:3]\nmo_close()\nwith open(\'/tmp/v5_polar.fem\') as fh:\n    coords = re.search(r\'\\[Coordinates\\]\\s*=\\s*(\\w+)\', fh.read()).group(1)\nmi_setgrid(1, \'CARTESIAN\')\nmi_analyze(1)\nmi_loadsolution()\nBx2, By2 = mo_getb(0, 0)\nmo_close()\nbogus = fails(mi_setgrid, 1, \'bogus\')\nmi_close()\nclosed = fails(mi_setgrid, 1, \'cart\')\ncheck(coords == \'polar\', \'saved coordinates \' + show(coords))\ncheck(abs(Bx) < 1e-3 and abs(By) < 1e-3, \'polar: B = \' + show(Bx, By))\ncheck(abs(By2 - 1) < 1e-3, \'cartesian: B = \' + show(Bx2, By2))\ncheck(bogus is not None and \'Unknown option for set grid\' in bogus, show(bogus))\ncheck(closed is not None and \'No current magnetics input\' in closed, show(closed))\nreturn \'spacing 1 or 5: same .fem; [Coordinates] = %s (mi_setgrid(2) kept polar); boundary A1 = -0.001 gives B = (%.2g, %.2g) T at the centre in polar and (%.2g, %.5g) T after mi_setgrid(1, "CARTESIAN"); "bogus": %s; no document: %s\' % (coords, Bx, By, Bx2, By2, bogus, closed)')
run('mi_showgrid', "fixture_pre()\nr1 = mi_showgrid()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_showgrid())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_shownames', 'fixture_pre()\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = [mi_shownames(0), mi_shownames(1), mi_shownames()]\nfinally:\n    del femm._default.callfemm\ncheck(r == [None] * 3 and sent == [\'mi_shownames(0)\', \'mi_shownames(1)\', \'mi_shownames()\'], show(r, sent))\ne_str = fails(mi_shownames, \'off\')\ne_two = fails(mi_shownames, 2)\ncheck(e_str is not None and "can\'t convert str to float" in e_str, show(e_str))\ncheck(e_two is None, \'mi_shownames(2): \' + show(e_two))\nbox = mi_getboundingbox()\ncheck(near(box[0], -50) and near(box[1], 50), \'box \' + show(*box))\nmi_close()\ne = fails(mi_shownames, 1)\ncheck(e is None, \'with no document open: \' + show(e))\nreturn \'sent \' + \' \'.join(sent) + \', no error; mi_shownames(2) accepted; "off": \' + e_str + \'; drawing unchanged, no error with no document open\'')
run('mi_zoom', "fixture_pre()\nr1 = mi_zoom(-10, -10, 10, 10)\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_zoom(-10, -10, 10, 10))\ncheck(e is None, 'with no document open: ' + show(e))\nfixture_pre()\ne2 = fails(mi_zoom, -10, -10)\ncheck(e2 is None, 'mi_zoom(-10, -10): ' + show(e2))\nreturn 'returned None, drawing unchanged, no error with no document open, nor with two arguments'")
run('mi_zoomin', "fixture_pre()\nr1 = mi_zoomin()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_zoomin())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_zoomnatural', "fixture_pre()\nr1 = mi_zoomnatural()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_zoomnatural())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_zoomout', "fixture_pre()\nr1 = mi_zoomout()\nbox = mi_getboundingbox()\ncheck(r1 is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r1) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: mi_zoomout())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('mi_snapgridoff', 'blank()\nmi_setgrid(1, \'cart\')\nmi_gridsnap(\'on\')\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = mi_snapgridoff()\nfinally:\n    del femm._default.callfemm\ncheck(r is None and sent == [\'mi_gridsnap("off")\'], show(r, sent))\nraw = fails(callfemm, \'mi_snapgridoff()\')\ncheck(raw is not None and "isn\'t defined" in raw, show(raw))\nmi_addnode(0.3, 0.3)\nx, y = mi_selectnode(0.3, 0.3)\ncheck(near(x, 0.3) and near(y, 0.3), show(x, y))\nreturn \'sent \' + sent[0] + \', returned None; FEMM itself: \' + raw + \'; node at \' + show(x, y)')
run('mi_snapgridon', 'blank()\nmi_setgrid(1, \'cart\')\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = mi_snapgridon()\nfinally:\n    del femm._default.callfemm\ncheck(r is None and sent == [\'mi_gridsnap("on")\'], show(r, sent))\nraw = fails(callfemm, \'mi_snapgridon()\')\ncheck(raw is not None and "isn\'t defined" in raw, show(raw))\nmi_addnode(0.3, 0.3)\nx, y = mi_selectnode(0.3, 0.3)\ncheck(near(x, 0.3) and near(y, 0.3), \'node at \' + show(x, y))\nreturn \'sent \' + sent[0] + \', returned None; FEMM itself: \' + raw + \'; snapping on, 1 mm grid: mi_addnode(0.3, 0.3) stayed at \' + show(x, y)')
print("BATCH_DONE", flush=True)
closefemm()
