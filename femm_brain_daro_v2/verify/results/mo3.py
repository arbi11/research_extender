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
run('mo_hidegrid', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_hidegrid()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_hidemesh', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_hidemesh()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_hidenames', "fixture_post()\nB0 = mo_getb(5, 0)\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = mo_hidenames()\nfinally:\n    del femm._default.callfemm\ncheck(r is None and sent == ['mo_shownames(0)'] and mo_getb(5, 0) == B0, show(r, sent))\nreturn 'mo_hidenames() sent ' + sent[0] + ', returned None, B unchanged'")
run('mo_hidepoints', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_hidepoints()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_maximize', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_maximize()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_minimize', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_minimize()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_refreshview', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_refreshview()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_resize', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_resize(800, 600)]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\ne = [fails(mo_resize, 800), fails(mo_resize)]\ncheck(e == [None, None], 'wrong counts: ' + show(e))\nreturn 'returned None, B unchanged, no error with one or no argument either'")
run('mo_restore', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_restore()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_setgrid', 'fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_setgrid(1, \'cart\'), mo_setgrid(5, \'polar\'), mo_setgrid(1, \'CARTESIAN\')]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, \'returned \' + show(r) + \', B \' + show(B0, B1))\ne_bogus = fails(mo_setgrid, 1, \'bogus\')\ne_short = [fails(mo_setgrid, 1), fails(mo_setgrid)]\ncheck(e_bogus is not None and \'Unknown option for set grid\' in e_bogus, show(e_bogus))\ncheck(e_short == [None, None], \'short forms: \' + show(e_short))\nreturn \'returned None, B unchanged; "bogus": \' + e_bogus + \'; mo_setgrid(1) and mo_setgrid() raised nothing\'')
run('mo_showcontourplot', 'fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_showcontourplot(-1), mo_showcontourplot(19, 0, 3.5e-5, \'real\')]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, \'returned \' + show(r) + \', B \' + show(B0, B1))\ne = [fails(mo_showcontourplot, 19, 0, 3.5e-5, \'bogus\'), fails(mo_showcontourplot)]\ncheck(e == [None, None], show(e))\nreturn \'(-1) and (19, 0, 3.5e-5, "real") returned None, B unchanged; "bogus" and no argument raised nothing\'')
run('mo_showdensityplot', 'fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_showdensityplot(-1), mo_showdensityplot(1, 0, 0.01, 0, \'bmag\')]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, \'returned \' + show(r) + \', B \' + show(B0, B1))\ne = fails(mo_showdensityplot, 1, 0, 0.01, 0, \'bogus\')\ncheck(e is None, show(e))\nreturn \'(-1) and (1, 0, 0.01, 0, "bmag") returned None, B unchanged; "bogus" raised nothing\'')
run('mo_showgrid', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_showgrid()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_showmesh', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_showmesh()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_shownames', 'fixture_post()\nB0 = mo_getb(5, 0)\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = [mo_shownames(0), mo_shownames(1), mo_shownames()]\nfinally:\n    del femm._default.callfemm\ncheck(r == [None] * 3 and sent == [\'mo_shownames(0)\', \'mo_shownames(1)\', \'mo_shownames()\'], show(r, sent))\ne = fails(mo_shownames, \'on\')\ncheck(e is not None and "can\'t convert str to float" in e, show(e))\ncheck(mo_getb(5, 0) == B0, \'B changed\')\nreturn \'sent \' + \' \'.join(sent) + \', no error, B unchanged; "on": \' + e')
run('mo_showpoints', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_showpoints()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_showvectorplot', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_showvectorplot(1, 1), mo_showvectorplot(0), mo_showvectorplot(1)]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\ne = [fails(mo_showvectorplot, 9, 1), fails(mo_showvectorplot, -1, 1), fails(mo_showvectorplot)]\ncheck(e == [None] * 3, show(e))\nreturn '(1, 1), (0) and (1) returned None, B unchanged; 9, -1 and no argument raised nothing'")
run('mo_zoom', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_zoom(-20, -20, 20, 20)]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\ne = fails(mo_zoom, -20, -20)\ncheck(e is None, 'mo_zoom(-20, -20): ' + show(e))\nreturn 'returned None, B unchanged; mo_zoom(-20, -20) raised nothing'")
run('mo_zoomin', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_zoomin()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_zoomnatural', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_zoomnatural()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_zoomout', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_zoomout()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_snapgrid', 'fixture_post()\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = [mo_snapgrid(\'on\'), mo_snapgrid(\'Off\')]\nfinally:\n    del femm._default.callfemm\ncheck(r == [None, None] and sent == [\'mo_gridsnap("on")\', \'mo_gridsnap("Off")\'], show(r, sent))\ne_int = fails(mo_snapgrid, 1)\ne_none = fails(mo_snapgrid)\nraw = fails(callfemm, \'mo_snapgrid("on")\')\ncheck(e_int is not None and "can\'t convert \'int\'" in e_int, show(e_int))\ncheck(e_none is not None and \'mo_gridsnap()\' in e_none, show(e_none))\ncheck(raw is not None and "isn\'t defined" in raw, show(raw))\nreturn \'sent \' + \' \'.join(sent) + \', no error; 1: \' + e_int + \'; no argument: \' + e_none + \'; FEMM itself: \' + raw')
run('mo_snapgridoff', 'fixture_post()\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = mo_snapgridoff()\nfinally:\n    del femm._default.callfemm\ncheck(r is None and sent == [\'mo_gridsnap("off")\'], show(r, sent))\nraw = fails(callfemm, \'mo_snapgridoff()\')\ncheck(raw is not None and "isn\'t defined" in raw, show(raw))\nreturn \'sent \' + sent[0] + \', returned None; FEMM itself: \' + raw')
run('mo_snapgridon', 'fixture_post()\nmo_setgrid(1, \'cart\')\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = mo_snapgridon()\nfinally:\n    del femm._default.callfemm\ncheck(r is None and sent == [\'mo_gridsnap("on")\'], show(r, sent))\nraw = fails(callfemm, \'mo_snapgridon()\')\ncheck(raw is not None and "isn\'t defined" in raw, show(raw))\nmo_clearcontour()\nmo_addcontour(2.3, 0.3)\nmo_addcontour(7.6, 0.3)\nlength = mo_lineintegral(2)[0]\ncheck(near(length, 0.0053), \'contour length \' + show(length))\nreturn \'sent \' + sent[0] + \', returned None; FEMM itself: \' + raw + \'; snapping on, 1 mm grid: contour (2.3, 0.3)-(7.6, 0.3) measured \' + show(length) + \' m\'')
print("BATCH_DONE", flush=True)
closefemm()
