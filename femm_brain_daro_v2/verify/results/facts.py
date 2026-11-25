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
run('fact:builtin_python', 'import subprocess\nlines = [\'newdocument(0)\',\n         \'mi_probdef(0, "millimeters", "planar", 1e-8, 10, 30, 0)\',\n         \'mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)\',\n         \'mi_addboundprop("Uniform", 0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0)\',\n         \'mi_drawarc(20, 0, -20, 0, 180, 5)\',\n         \'mi_drawarc(-20, 0, 20, 0, 180, 5)\',\n         \'mi_selectarcsegment(0, 20)\',\n         \'mi_selectarcsegment(0, -20)\',\n         \'mi_setarcsegmentprop(5, "Uniform", 0, 0)\',\n         \'mi_clearselected()\',\n         \'mi_addblocklabel(0, 0)\',\n         \'mi_selectlabel(0, 0)\',\n         \'mi_setblockprop("Air", 1, 0, "<None>", 0, 0, 0)\',\n         \'mi_clearselected()\',\n         \'mi_saveas("/tmp/builtin.fem")\',\n         \'mi_analyze(1)\',\n         \'mi_loadsolution()\',\n         \'v = mo_getpointvalues(0, 0)\',\n         \'print("VALUES", type(v).__name__, len(v), v[2])\',\n         \'for name in ("mo_getb", "mo_geth", "mo_geta"):\',\n         \'    try:\',\n         \'        eval(name)\',\n         \'        print("HAS", name)\',\n         \'    except NameError:\',\n         \'        print("MISSING", name)\']\nwith open(\'/tmp/builtin.py\', \'w\') as f:\n    f.write(\'\\n\'.join(lines) + \'\\n\')\ndone = subprocess.run([\'/opt/femm/bin/femm_headless\', \'--port\', \'48032\', \'-run\', \'/tmp/builtin.py\'],\n                      capture_output=True, text=True, timeout=120)\nout = done.stdout + done.stderr\nmissing = re.findall(r\'MISSING\\s+(\\w+)\', out)\nvalues = re.search(r\'VALUES\\s+(\\w+)\\s+(\\d+)\\s+(\\S+)\', out)\ncheck(values and abs(float(values.group(3)) - 1) < 1e-3, out[-400:])\ncheck(missing == [\'mo_getb\', \'mo_geth\', \'mo_geta\'], out[-400:])\nreturn (\'femm_headless -run ran the script in the built-in Python: mo_getpointvalues gave a \' + values.group(1)\n        + \' of \' + values.group(2) + \' values, By = \' + values.group(3) + \' T; not defined there: \' + show(*missing))')
run('fact:errors_raise', 'blank()\nfirst = fails(mo_getpointvalues, 0, 0)\ncheck(first and first.startswith(\'FEMM returns:\') and "isn\'t defined" in first, str(first))\ncount = fails(mi_addnode, 1)\ncheck(count and \'positional argument\' in count, str(count))\ntry:\n    mi_addnode(1)\n    kind = \'nothing\'\nexcept Exception as error:\n    kind = type(error).__name__\ncheck(kind == \'Exception\', kind)\ntry:\n    mi_drawline(0, 0, 1)\n    helper = \'nothing\'\nexcept TypeError as error:\n    helper = \'TypeError: \' + str(error)\ncheck(helper.startswith(\'TypeError\'), helper)\nmi_close()\nfixture_post()\nmo_close()\nafter = fails(mo_getpointvalues, 0, 0)\ncheck(after and \'No magnetics output in focus\' in after, str(after))\nreturn (\'before any solution: \' + first + \' | wrong count: \' + count + \' | after mo_close: \' + after\n        + \' | raised as \' + kind + \' | mi_drawline with 3 arguments: \' + helper)')
run('fact:arguments', 'blank()\nmi_addnode(1 / 3, 2 / 3)\nx, y = mi_selectnode(1 / 3, 2 / 3)\ncheck(x == 1 / 3 and y == 2 / 3, show(repr(x), repr(y)))\nnone = fails(mi_addnode, None, 3)\ncheck(none and "can\'t convert NoneType to float" in none, str(none))\nmi_movetranslate(1, 0, None)\nmoved = mi_selectnode(4 / 3, 2 / 3)\ncheck(near(moved[0], 4 / 3), \'editaction None: the node is at \' + show(*moved))\nmi_addmaterial(\'Skip\', 500, None, 7)\nmi_probdef(0, \'millimeters\', \'planar\', None, 7)\nmi_saveas(\'/tmp/arguments.fem\')\ntext = open(\'/tmp/arguments.fem\').read()\nblock = text.split(\'"Skip"\')[1].split(\'<EndBlock>\')[0]\nvalues = dict(line.strip().split(\' = \') for line in block.strip().splitlines() if \' = \' in line)\ncheck(values.get(\'<Mu_y>\') == \'500\' and values.get(\'<H_c>\') == \'7\', \'stored \' + show(values.get(\'<Mu_y>\'), values.get(\'<H_c>\')))\ndepth = re.search(r\'\\[Depth\\]\\s*=\\s*(\\S+)\', text).group(1)\ncheck(depth == \'10\', \'mi_probdef with None before the depth 7 stored depth \' + depth)\nquote = fails(mi_addmaterial, \'a"b\', 1, 1)\ncheck(quote and \'invalid syntax\' in quote, str(quote))\nreturn (\'1/3 and 2/3 came back exactly (\' + repr(x) + \', \' + repr(y) + \'); None as a coordinate: \' + none\n        + \'; editaction None moved the selected node to \' + show(*moved) + "; mi_addmaterial(\'Skip\', 500, None, 7) stored mu_y "\n        + values.get(\'<Mu_y>\') + \' and H_c \' + values.get(\'<H_c>\') + "; mi_probdef(0, \'millimeters\', \'planar\', None, 7) kept depth "\n        + depth + \'; a double quote inside a string: \' + quote)')
run('fact:results_shapes', "blank()\nempty = mi_getboundingbox()\ncheck(empty == [], repr(empty))\nmi_close()\nfixture_pre()\nbox = mi_getboundingbox()\ncheck(isinstance(box, list) and len(box) == 4, repr(box))\nxmin, xmax, ymin, ymax = box\ncheck(near(xmin, -50) and near(xmax, 50) and near(ymin, -50) and near(ymax, 50), show(*box))\ngroup = mi_selectgroup(1)\nraw = callfemm('mi_selectgroup(1)')\ncheck(group is None and raw == [], show(repr(group), repr(raw)))\nmi_close()\nfixture_post()\nvalues = mo_getpointvalues(5, 0)\ncheck(isinstance(values, list) and len(values) == 14 and near(values[2], 0.0039744, 0.01), repr(values))\nmo_selectblock(0, 0)\narea = mo_blockintegral(5)\nmo_clearblock()\ncheck(isinstance(area, float), repr(area))\nreturn ('empty document: mi_getboundingbox() = [] ; fixture box ' + show(*box) + '; mo_getpointvalues: list of '\n        + str(len(values)) + '; mo_blockintegral(5): ' + type(area).__name__ + ' ' + repr(area)\n        + '; mi_selectgroup gives None, while FEMM replied ' + repr(raw))")
run('fact:text_results_garbled', 'blank()\nmi_saveas(\'/tmp/inside.fem\')\ntitle = mi_gettitle()\ncheck(title == \'/tmp/1jns1jde.fem\', repr(title))\nair = callfemm(\'"Air"\')\ncheck(air == \'A1jr\', repr(air))\ninfinity = callfemm(\'float("inf")\')\nnan = callfemm(\'float("nan")\')\ncheck(infinity == \'1jnf\' and nan == \'nan\', show(repr(infinity), repr(nan)))\nreturn (\'mi_gettitle() for /tmp/inside.fem: \' + repr(title) + \'; the text "Air": \' + repr(air)\n        + \'; infinity: \' + repr(infinity) + \'; NaN: \' + repr(nan))')
run('fact:angles', "blank()\nmi_drawarc(10, 0, -10, 0, 90, 1)\nxmin, xmax, ymin, ymax = mi_getboundingbox()\nexpected = 10 * math.sqrt(2) - 10\ncheck(near(ymax, expected, 1e-3), show(xmin, xmax, ymin, ymax))\ncheck(math.sin(math.pi / 2) == 1 and near(math.degrees(math.pi), 180), 'math')\nreturn ('a 90-degree arc between (10, 0) and (-10, 0) rises to y = ' + format(ymax, '.5g')\n        + ' (10 sqrt 2 - 10 = ' + format(expected, '.5g') + '): FEMM reads the angle in degrees')")
run('fact:complex_numbers', "fixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nmi_addcircprop('Ic', 3 + 4j, 1)\nsolve('ac')\ni, v, flux = mo_getcircuitproperties('Iw')\nic = mo_getcircuitproperties('Ic')[0]\nomega = 2 * math.pi * 50\ncheck(isinstance(v, complex) and near(v.real, 0.0054951, 0.01), repr(v))\ncheck(near(abs(v.imag), omega * abs(flux.real), 0.02), show(v, flux))\ncheck(ic == 3 + 4j, repr(ic))\nmo_close()\nmi_close()\nfixture_post()\nidc, vdc, fdc = mo_getcircuitproperties('Iw')\ncheck(not isinstance(vdc, complex), repr(vdc))\nreturn ('at 50 Hz the wire gives volts ' + format(v, '.5g') + ' V and flux linkage ' + format(flux, '.5g')\n        + ' Wb; a circuit set to 3+4j reads back ' + repr(ic) + '; at DC volts = ' + repr(vdc)\n        + ' (' + type(vdc).__name__ + '), current = ' + repr(idc) + ' (' + type(idc).__name__ + ')')")
run('fact:current_direction', "fixture_post()\nvalues = mo_getpointvalues(5, 0)\nb1, b2 = values[1], values[2]\ncheck(b2 > 0 and abs(b1) < 0.02 * b2, show(b1, b2))\nreturn 'B at (5,0) mm = (' + format(b1, '.3g') + ', ' + format(b2, '.5g') + ') T for +100 A'")
run('fact:results_in_si', "fixture_post()\nmo_selectblock(0, 0)\narea = mo_blockintegral(5)\nvolume = mo_blockintegral(10)\nmo_clearblock()\ncheck(3e-6 < area < 3.2e-6, 'area ' + repr(area) + ' m^2')\ncheck(near(volume, area * 0.010, 1e-6), 'volume ' + repr(volume) + ' vs area * depth')\nA, B1, B2, Sig, E, H1, H2, Je, Js = mo_getpointvalues(0.5, 0)[:9]\ncheck(abs(Sig - 58) < 0.01 and abs(Js * 1e6 * area / 100 - 1) < 1e-3, 'in the wire Sig = ' + repr(Sig) + ', Js = ' + repr(Js))\nreturn ('wire area ' + format(area, '.6g') + ' m^2, volume ' + format(volume, '.6g')\n        + ' m^3 = area x 10 mm depth; point values Sig = ' + format(Sig, '.4g') + ' (MS/m), Js = '\n        + format(Js, '.5g') + ' (MA/m^2)')")
run('fact:air_gap_band', "fixture_geometry()\nmi_selectlabel(0, 10.5)\nmi_deleteselectedlabels()\nmi_clearselected()\nnoband = fails(solve, 'noband')\ncheck(noband and '2560' in noband, 'without the no-mesh label: ' + str(noband))\nmi_close()\nfixture_post()\nbr, bt = mo_getgapb('AGap', 0)\ninband = mo_getpointvalues(0, 10.5)\ncheck(inband == [], 'mo_getpointvalues in the band gave ' + repr(inband))\nshortcut = fails(mo_getb, 0, 10.5)\ncheck(shortcut and 'index out of range' in shortcut, str(shortcut))\nmo_close()\nmi_close()\nfixture_geometry()\nmi_selectlabel(0, 10.5)\nmi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)\nmi_clearselected()\nsolve('airband')\nbrair, btair = mo_getgapb('AGap', 0)\nmo_close()\nmi_close()\ncheck(abs(btair / bt - 0.5) < 0.02, 'Air in the band: Bt ' + repr(btair) + ' against ' + repr(bt))\nfixture_geometry()\nmi_modifyboundprop('AGap', 9, 7)\nsolve('anti')\nbr7, bt7 = mo_getgapb('AGap', 0)\ncheck(bt > 1.8e-3 and abs(bt7) < 1e-9, 'periodic ' + repr(bt) + ', anti-periodic ' + repr(bt7))\nreturn ('no label -> ' + noband + '; <No Mesh> band, periodic: Bt = ' + format(bt, '.5g') + ' T; Air band: Bt = '\n        + format(btair, '.5g') + ' T, no error; anti-periodic Bt = ' + format(bt7, '.3g')\n        + ' T; in the band mo_getpointvalues = [] and mo_getb raises: ' + shortcut)")
run('fact:prescribed_a_units', "blank()\nmi_addboundprop('Uniform', 0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0)\ncircle(40, 5)\narcprop(0, 40, 5, 'Uniform')\narcprop(0, -40, 5, 'Uniform')\nlabel(20, 20, 'Air')\nsolve('uniform')\nbx, by = mo_getpointvalues(0, 0)[1:3]\ncheck(abs(by - 1) < 1e-3 and abs(bx) < 1e-3, show(bx, by))\nreturn 'A1 = -0.001 per mm -> B = (' + format(bx, '.3g') + ', ' + format(by, '.6g') + ') T'")
run('fact:torque_sign', "blank()\nmi_addboundprop('Uniform', 0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0)\nmi_addmaterial('PM', 1.05, 1.05, 900000, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)\ncircle(40, 5)\narcprop(0, 40, 5, 'Uniform')\narcprop(0, -40, 5, 'Uniform')\nmi_drawrectangle(-2, -2, 2, 2)\nmi_addblocklabel(0, 0)\nmi_selectlabel(0, 0)\nmi_setblockprop('PM', 0, 0.3, '<None>', 0, 0, 0)\nmi_clearselected()\nlabel(20, 20, 'Air', None, 0, 0, 1)\nsolve('torque')\nmo_selectblock(0, 0)\ntorque = mo_blockintegral(22)\nmo_clearblock()\ncheck(torque > 0, 'torque ' + repr(torque))\nreturn '+x magnet in a +1 T y field: torque ' + format(torque, '.4g') + ' N m (counter-clockwise)'")
run('fact:save_before_solve', "blank()\nmi_drawrectangle(-10, -10, 10, 10)\nlabel(0, 0, 'Air')\nunsaved = fails(mi_analyze, 1)\ncheck(unsaved is not None, 'analyze of an unsaved document did not fail')\nmi_saveas('/tmp/saved.fem')\nmi_selectsegment(0, -10)\nmi_setsegmentprop('A0', 0, 1, 0, 0)\nmi_clearselected()\nmi_analyze(1)\nmi_loadsolution()\nreturn 'unsaved: ' + unsaved + '; after mi_saveas it solved'")
run('fact:dxf_line_endings', "# Two LINE entities, (0,0)-(10,0) and (10,0)-(10,5): ymax 5 means both came in, 0 only the first.\ndxf = ['0', 'SECTION', '2', 'ENTITIES',\n       '0', 'LINE', '8', '0', '10', '0', '20', '0', '11', '10', '21', '0',\n       '0', 'LINE', '8', '0', '10', '10', '20', '0', '11', '10', '21', '5',\n       '0', 'ENDSEC', '0', 'EOF']\ndef ymax(pad, eol):\n    lines = ['%3s' % line if pad and i % 2 == 0 else line for i, line in enumerate(dxf)]\n    with open('/tmp/case.dxf', 'w', newline='') as f:\n        f.write(eol.join(lines) + eol)\n    blank()\n    mi_readdxf('/tmp/case.dxf')\n    top = mi_getboundingbox()[3]\n    mi_close()\n    return top\npadlf, padcrlf = ymax(True, '\\n'), ymax(True, '\\r\\n')\nlf, crlf = ymax(False, '\\n'), ymax(False, '\\r\\n')\n# FEMM's own output: padded codes and LF.\nblank()\nmi_drawline(0, 0, 10, 0)\nmi_drawline(10, 0, 10, 5)\nmi_savedxf('/tmp/saved.dxf')\nmi_close()\nblank()\nmi_readdxf('/tmp/saved.dxf')\nsaved = mi_getboundingbox()[3]\nmi_close()\ncheck(near(padcrlf, 5) and abs(padlf) < 1e-9, 'padded codes: CRLF ymax ' + repr(padcrlf) + ', LF ymax ' + repr(padlf))\ncheck(near(lf, 5) and abs(crlf) < 1e-9, 'unpadded codes: LF ymax ' + repr(lf) + ', CRLF ymax ' + repr(crlf))\ncheck(abs(saved) < 1e-9, 'mi_savedxf output read back: ymax ' + repr(saved))\nreturn ('ymax (5 = whole, 0 = first entity only): padded+LF %g, padded+CRLF %g, unpadded+LF %g, '\n        'unpadded+CRLF %g; mi_savedxf output read back %g' % (padlf, padcrlf, lf, crlf, saved))")
run('fact:selection', "blank()\nmi_addnode(0, 0)\nmi_addnode(10, 0)\nmi_selectnode(0, 0)\nmi_selectnode(0, 0)\nmi_deleteselectednodes()\nx1, y1 = mi_selectnode(0, 0)\ncheck(x1 == 0 and y1 == 0, 'the node selected twice was deleted; nearest is ' + show(x1, y1))\nmi_selectnode(10, 0)\nmi_deleteselectednodes()\ngone = mi_selectnode(0, 0)\ncheck(gone == [], 'a node survived: ' + repr(gone))\nleft = []\nfor way in ('rectangle', 'circle', 'group'):\n    mi_clearselected()\n    mi_drawline(0, 0, 10, 0)\n    if way == 'group':\n        mi_selectsegment(5, 0)\n        mi_setsegmentprop('<None>', 0, 1, 0, 4)\n        mi_clearselected()\n    for _ in range(2):\n        if way == 'rectangle':\n            mi_selectrectangle(-1, -1, 11, 1, 1)\n        elif way == 'circle':\n            mi_selectcircle(5, 0, 6, 1)\n        else:\n            mi_selectgroup(4)\n    mi_deleteselectedsegments()\n    left.append(mi_selectsegment(5, 0))\ncheck(left == [[], [], []], 'segments left after two rectangle, circle and group selects: ' + show(*left))\nmi_close()\nblank()\nmi_drawrectangle(0, 0, 10, 10)\nmi_selectrectangle(-1, -1, 11, 11, 4)\nmi_movetranslate(5, 0)\nmi_movetranslate(5, 0)\na, b = mi_getboundingbox()[:2]\ncheck(near(a, 5) and near(b, 15), 'after two moves of 5: x from ' + show(a, b))\nmi_close()\nfixture_post()\nmo_clearblock()\nmo_selectblock(0, 0)\nmo_selectblock(0, 0)\ntwice = fails(mo_blockintegral, 5)\ncheck(twice and 'No area selected' in twice, 'a block selected twice: ' + str(twice))\nmo_groupselectblock(1)\nmo_groupselectblock(1)\ngroup_twice = fails(mo_blockintegral, 5)\ncheck(group_twice and 'No area selected' in group_twice, 'group 1 selected twice: ' + str(group_twice))\nreturn ('a node selected twice was not deleted, and a third select returned []; a segment selected twice '\n        'by rectangle, circle or group was deleted; the second of two moves moved nothing (x ' + show(a, b)\n        + '); a block selected twice left ' + twice + '; group 1 selected twice left ' + group_twice)")
run('fact:edits_reset_node_properties', "def rows():\n    mi_saveas('/tmp/reset.fem')\n    text = open('/tmp/reset.fem').read()\n    def part(a, b=None):\n        body = text.split(a)[1]\n        body = body.split(b)[0] if b else body\n        return [row.split() for row in body.strip().splitlines()[1:]]\n    return (part('[NumPoints]', '[NumSegments]')[0], part('[NumSegments]', '[NumArcSegments]')[0],\n            part('[NumArcSegments]', '[NumHoles]')[0], part('[NumBlockLabels]')[0])\ndef setall():\n    mi_clearselected()\n    mi_selectnode(0, 0)\n    mi_setnodeprop('P', 3)\n    mi_clearselected()\n    mi_selectsegment(5, 5)\n    mi_setsegmentprop('A0', 0, 1, 0, 4)\n    mi_clearselected()\n    mi_selectarcsegment(5, 8)\n    mi_setarcsegmentprop(5, 'A0', 0, 6)\n    mi_clearselected()\n    mi_selectlabel(5, 20)\n    mi_setblockprop('Air', 1, 0, '<None>', 0, 7, 0)\n    mi_clearselected()\ndef others_kept(segment, arc, label):\n    return segment[3] == '1' and segment[5] == '4' and arc[4] == '1' and arc[6] == '6' and label[6] == '7'\nblank()\nmi_addpointprop('P', 0, 1)\nmi_addnode(0, 0)\nmi_addnode(10, 0)\nmi_drawline(0, 5, 10, 5)\nmi_drawarc(0, 10, 10, 10, 90, 5)\nmi_addblocklabel(5, 20)\nsetall()\npoint, segment, arc, label = rows()\ncheck(point[2:4] == ['1', '3'] and others_kept(segment, arc, label), 'set: ' + show(point, segment, arc, label))\nedits = [('mi_movetranslate', lambda: mi_movetranslate(1, 0)),\n         ('mi_moverotate', lambda: mi_moverotate(0, 0, 0)),\n         ('mi_copytranslate', lambda: mi_copytranslate(0, 20, 1)),\n         ('mi_copyrotate', lambda: mi_copyrotate(0, 0, 90, 1)),\n         ('mi_scale', lambda: mi_scale(0, 0, 1)),\n         ('mi_mirror', lambda: mi_mirror(0, -50, 1, -50))]\nreset = []\nfor name, edit in edits:\n    setall()\n    edit()\n    point, segment, arc, label = rows()\n    if point[2:4] == ['0', '0']:\n        reset.append(name)\n    check(others_kept(segment, arc, label), name + ' changed another object: ' + show(segment, arc, label))\ncheck(reset == [name for name, _ in edits], 'only these reset node (0, 0): ' + show(*reset))\nkept = []\nfor name, edit in [('mi_addnode', lambda: mi_addnode(50, 50)),\n                   ('mi_drawline', lambda: mi_drawline(60, 60, 70, 60)),\n                   ('mi_addblocklabel', lambda: mi_addblocklabel(80, 80))]:\n    setall()\n    edit()\n    if rows()[0][2:4] == ['1', '3']:\n        kept.append(name)\ncheck(len(kept) == 3, 'only these kept node (0, 0): ' + show(*kept))\nreturn ('with nothing selected, ' + ', '.join(reset) + ' each reset node (0, 0) from point property 1, group 3 to 0, 0, '\n        'while the segment kept boundary 1 and group 4, the arc boundary 1 and group 6, the label group 7; '\n        + ', '.join(kept) + ' reset nothing')")
run('fact:material_library', 'blank()\nloads = [\'N42\', \'N52\', \'SmCo24\', \'Ceramic 8A\', \'M-19 14 mil\', \'M-27\', \'Pure iron, annealed\',\n         \'1010 Steel (hot rolled)\', \'Copper\', \'Aluminum, 6061-T6\', \'18 AWG\']\nbad = [\'M-19 Steel\', \'NdFeB 40 MGOe\', \'Pure Iron\', \'1010 Steel\', \'copper\']\nfor name in loads:\n    error = fails(mi_getmaterial, name)\n    check(error is None, name + \': \' + str(error))\nfor name in bad:\n    error = fails(mi_getmaterial, name)\n    check(error and "Couldn\'t load" in error, name + \' loaded\')\nreturn (str(len(loads)) + \' femm-qt names loaded; \' + str(len(bad)) + \' FEMM 4.2 or miscased names failed, e.g. \' + error)')
print("BATCH_DONE", flush=True)
closefemm()
