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
run('fact:python_route', 'import importlib.util\ncheck(sys.version_info[:2] == (3, 12), sys.version)\ncheck(femm.__file__ == \'/opt/femm/share/femm/pyfemm/femm.py\', femm.__file__)\ncheck(importlib.util.find_spec(\'numpy\') is None, \'numpy is installed\')\ncheck(isinstance(femm._default, femm.FemmSession) and femm._default._port == 48015, \'default session\')\ncheck(femm._find_exe(\'femm_headless\') == \'/opt/femm/bin/femm_headless\', str(femm._find_exe(\'femm_headless\')))\nconstants = [n for n in (\'mm\', \'inch\', \'muo\', \'uo\', \'Tesla\', \'Gauss\', \'Oersted\', \'PI\', \'I\') if hasattr(femm, n)]\ncheck(constants == [], \'pyFEMM defines \' + show(*constants))\nhere = os.getcwd()\nos.makedirs(\'/tmp/sub\', exist_ok=True)\nos.chdir(\'/tmp\')\ntry:\n    other = femm.FemmSession(port=48031, headless=True)\n    other.newdocument(0)\n    other.mi_saveas(\'relative.fem\')\n    os.chdir(\'/tmp/sub\')\n    other.mi_saveas(\'after_chdir.fem\')\n    other.callfemm(\'setcurrentdirectory("/tmp/sub")\')\n    other.mi_saveas(\'after_set.fem\')\n    other.close()\nfinally:\n    os.chdir(here)\ncheck(os.path.exists(\'/tmp/relative.fem\'), \'the relative save did not land in /tmp\')\ncheck(os.path.exists(\'/tmp/after_chdir.fem\') and not os.path.exists(\'/tmp/sub/after_chdir.fem\'), \'os.chdir moved FEMM\')\ncheck(os.path.exists(\'/tmp/sub/after_set.fem\'), \'setcurrentdirectory did not move FEMM\')\nsecond = fails(openfemm, 1)\ncheck(second and \'already connected to another client\' in second, str(second))\nreturn (\'Python \' + sys.version.split()[0] + \', pyFEMM at \' + femm.__file__ + \', no numpy, no Lua constants; \'\n        \'default session on port 48015; a second FemmSession on port 48031, started from /tmp, saved relative.fem into /tmp, \'\n        \'still into /tmp after os.chdir, and into /tmp/sub after setcurrentdirectory; a second openfemm(1): \' + second)')
run('closefemm', 'p = femm._default._proc\nr = closefemm()\ngone = femm._default is None\nerr = fails(mi_addnode, 0, 0)\nagain = fails(closefemm)\ncode = p.wait(10) if p else \'not launched by this session\'\ns = openfemm(1)\ncheck(r is None and gone and again is None, show(r, gone, again))\ncheck(err == "\'NoneType\' object has no attribute \'mi_addnode\'", show(err))\ncheck(code == 0, \'FEMM exit code \' + show(code))\ncheck(s is femm._default and callfemm(\'0\') == 0, \'openfemm(1) after closefemm gave no working session\')\nreturn \'closefemm() returned None; femm._default is None; mi_addnode then raised: \' + err + \'; a second closefemm() raised nothing; FEMM exited with code \' + show(code) + \'; openfemm(1) started a new FEMM\'')
run('mo_reload', 'fixture_geometry()\nsolve(\'b_reload1\')\nmo_close()\nmi_saveas(\'/tmp/b_reload2.fem\')\nmi_analyze(1)\nmi_loadsolution()\nbefore = mo_getb(5, 0)[1]\nmi_modifycircprop(\'Iw\', 1, 200)\nmi_analyze(1)\nstale = mo_getb(5, 0)[1]\nr = mo_reload()\nafter = mo_getb(5, 0)[1]\nok = callfemm(\'mo_setfocus("b_reload1")\')\nt = mo_gettitle()\nother = mo_getb(5, 0)[1]\ncheck(r is None and stale == before and near(after, 2 * before, 1e-3), show(before, stale, after, r))\ncheck(ok == [] and t == \'/tmp/b_reload1.ans\' and near(other, before, 1e-9), show(ok, t, other))\nmo_close()\nmi_loadsolution()\nt2, again = mo_gettitle(), mo_getb(5, 0)[1]\ncheck(t2 == \'/tmp/b_reload2.ans\' and near(again, after, 1e-9), show(t2, again))\nmo_reload()\nmi_loadsolution()\nnofocus = fails(mo_gettitle)\ncheck(nofocus is not None and \'No magnetics output in focus\' in nofocus, show(nofocus))\nmi_loadsolution()\nmo_reload()\ncrash = fails(callfemm, \'mo_setfocus("nosuch")\')\nlost = not alive()\nif lost:\n    openfemm(1)\ncheck(lost and crash is not None and \'Connection to FEMM lost\' in crash, show(crash, lost))\nreturn (f\'By at (5,0): {before:.5g} T at 100 A; re-solved at 200 A: {stale:.5g} T before mo_reload, {after:.5g} T after; mo_setfocus("b_reload1"), loaded before it, then worked (\' + t + f\'); mo_close and mi_loadsolution then gave {t2} with {again:.5g} T\'\n        + \'; mo_reload then mi_loadsolution: \' + nofocus + \'; then mo_setfocus("nosuch"): \' + crash + \', FEMM gone: \' + str(lost))')
run('openfemm', "import subprocess, time\nfirst = femm._default\nbusy = fails(openfemm, 1)\ncheck(busy is not None and 'already connected to another client' in busy, show(busy))\ncheck(femm._default is first and callfemm('0') == 0, 'the first session did not survive')\np = first._proc\nclosefemm()\nif p:\n    p.wait(10)\nt0 = time.time()\ns = openfemm(1)\ndt = time.time() - t0\nexe = os.path.join(os.environ['FEMM_DIR'], 'femm_headless')\ncheck(isinstance(s, femm.FemmSession) and s is femm._default, show(s))\nlaunched = list(s._proc.args) if s._proc else None\ncheck(launched == [exe, '--port', '48015'], show(launched))\ncwd = callfemm('getcwd()')\ncheck(cwd == os.getcwd(), show(cwd, os.getcwd()))\np = s._proc\nclosefemm()\np.wait(10)\next = subprocess.Popen([exe, '--port', '48015'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\nfor k in range(100):\n    with open('/proc/net/tcp') as f:\n        listening = re.search(r':BB8F [0-9A-F]+:0000 0A', f.read()) is not None\n    if listening:\n        break\n    time.sleep(0.05)\na = openfemm(1)\nattached = a._proc is None and callfemm('0') == 0\nclosefemm()\ncode = ext.wait(10)\nopenfemm(1)\ncheck(listening and attached and code == 0, show(listening, attached, code))\nf2 = femm.FemmSession(port=48016, headless=True)\ntry:\n    f2.newdocument(0)\n    f2.mi_addnode(1, 1)\n    other = f2.mi_getboundingbox()\n    newdocument(0)\n    mine = mi_getboundingbox()\nfinally:\n    p2 = f2._proc\n    f2.close()\ncheck(other == [1, 1, 1, 1] and mine == [], show(other, mine))\ncheck(p2.wait(10) == 0, 'the second FEMM did not exit')\nreturn 'while connected: ' + busy + '; the first session kept working; after closefemm, openfemm(1) ran ' + ' '.join(launched) + f' in {dt:.2f} s, FEMM directory ' + cwd + '; with femm_headless started by hand it attached (_proc None), and closefemm stopped that FEMM (exit 0); FemmSession(port=48016, headless=True) ran a second FEMM: its box ' + show(other) + ', the default one ' + show(mine)")
run('quit', 'p = femm._default._proc\nr = callfemm_noeval(\'quit()\')\ngone = fails(callfemm, \'0\')\ncode = p.wait(10) if p else \'not launched by this session\'\nopenfemm(1)\np = femm._default._proc\nraised = fails(callfemm, \'quit()\')\ncode2 = p.wait(10) if p else \'not launched by this session\'\nopenfemm(1)\ntry:\n    quit()\n    ended = None\nexcept SystemExit as error:\n    ended = \'SystemExit \' + repr(error.code)\nalive_now = callfemm(\'0\')\ncheck(r is None and gone is not None, show(r, gone))\ncheck(raised is not None and \'Connection to FEMM lost\' in raised, show(raised))\ncheck(code == 0 and code2 == 0, show(code, code2))\ncheck(ended is not None and alive_now == 0 and not hasattr(femm, \'quit\'), show(ended, alive_now))\nreturn "callfemm_noeval(\'quit()\') returned None and FEMM exited (code " + show(code) + \'; the next call: \' + gone + "); callfemm(\'quit()\') raised " + raised + \' (exit code \' + show(code2) + \'); quit() in the script raised \' + ended')
run('mo_gapintegral', "fixture_post()\nt0 = mo_gapintegral('AGap', 0)\nf1 = mo_gapintegral('AGap', 1)\nw = mo_gapintegral('AGap', 2)\nt3 = mo_gapintegral('AGap', 3)\nf4 = mo_gapintegral('AGap', 4)\nf6 = mo_gapintegral('AGap', 6)\nwrong = mo_gapintegral('Agap', 2)\nexpected = 0.010 * 1e-7 * 100 * 100 * math.log(11 / 10)\nobs = ('0 torque %r N m; 1 force %r N; 2 energy %r J (analytic %.6g J); 3 %r; 4 %r; 6 %r; misspelt name %r'\n       % (t0, f1, w, expected, t3, f4, f6, wrong))\ncheck(isinstance(t0, float) and isinstance(w, float) and len(f1) == 2 and f4 == [0, 0] and f6 == [0, 0], obs)\ncheck(near(w, expected, 0.01), obs)\ncheck(abs(t0) < 1e-8 and abs(f1[0]) < 1e-5 and abs(f1[1]) < 1e-5 and t3 == 0, obs)\ncheck(wrong == 0, obs)\n# Type 5 kills FEMM; openfemm(1) starts a new one.\nerror = fails(mo_gapintegral, 'AGap', 5)\nlost = not alive()\nif lost:\n    openfemm(1)\nfixture_post()\nw2 = mo_gapintegral('AGap', 2)\nmo_close()\nmi_close()\nobs += '; type 5: %s, FEMM gone: %s; after openfemm(1) energy again %r' % (error, lost, w2)\ncheck(lost and error is not None and 'Connection to FEMM lost' in error, obs)\ncheck(w2 == w, obs)\n# 50 Hz: the energy is a time average; types 3 and 4 complex; types 6 and 5 both kill FEMM.\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('gapac')\nwac = mo_gapintegral('AGap', 2)\nt3ac = mo_gapintegral('AGap', 3)\nf4ac = mo_gapintegral('AGap', 4)\nerror6 = fails(mo_gapintegral, 'AGap', 6)\nlost6 = not alive()\nif lost6:\n    openfemm(1)\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('gapac')\nerror5 = fails(mo_gapintegral, 'AGap', 5)\nlost5 = not alive()\nif lost5:\n    openfemm(1)\nobs += ('; at 50 Hz: energy %r J, type 3 %r, type 4 %r; type 6: %s, FEMM gone: %s; type 5: %s, FEMM gone: %s'\n        % (wac, t3ac, f4ac, error6, lost6, error5, lost5))\ncheck(near(wac, w / 2, 0.01), obs)\ncheck(isinstance(t3ac, complex) and isinstance(f4ac[0], complex), obs)\ncheck(lost6 and error6 is not None and 'Connection to FEMM lost' in error6, obs)\ncheck(lost5 and error5 is not None and 'Connection to FEMM lost' in error5, obs)\nreturn obs")
print("BATCH_DONE", flush=True)
closefemm()
