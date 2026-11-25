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
run('mi_modifypointprop', "blank()\ncircle(40, 5)\narcprop(0, 40, 5, 'A0')\narcprop(0, -40, 5, 'A0')\nmi_addnode(0, 0)\nmi_addpointprop('P', 0, 100)\nmi_selectnode(0, 0)\nmi_setnodeprop('P', 0)\nmi_clearselected()\nlabel(20, 20, 'Air', None, 0, 0, 1)\nmi_modifypointprop('P', 2, 50)\nunknown = fails(mi_modifypointprop, 'NoSuch', 2, 5)\nsolve('modifypointprop')\nmo_clearcontour()\nmo_addcontour(5, 0)\nmo_addcontour(-5, 0)\nmo_bendcontour(180, 1)\nmo_addcontour(5, 0)\nmo_bendcontour(180, 1)\ntotal = mo_lineintegral(1)[0]\nmo_clearcontour()\nmo_close()\nmi_modifypointprop('P', 0, 'Q')\nsolve('renamepointprop')\nr = mo_getpointvalues(5, 0)\nmo_close()\ncheck(abs(total - 50) < 0.5, 'H.t ' + str(total))\ncheck(unknown is None, 'unknown name: ' + str(unknown))\ncheck(abs(r[1]) < 1e-12 and abs(r[2]) < 1e-12, 'after renaming: B ' + show(r[1], r[2]))\nreturn 'point current 100 A set to 50 A with propnum 2: H.t round r = 5 mm = %.5g A; renamed to Q with propnum 0, the node lost it (B at (5,0) = %s T); an unknown name raised nothing' % (total, show(r[1], r[2]))")
run('mi_probdef', 'newdocument(0)\nt0, f0, d0, u0 = mi_getprobleminfo()\ncheck(t0 == 0 and f0 == 0 and d0 == 1 and near(u0, 0.001), \'new document: \' + show(t0, f0, d0, u0))\nmi_probdef(-50, \'inches\', \'axi\', 1e-10, 2, 25, 1)\nt1, f1, d1, u1 = mi_getprobleminfo()\ncheck(t1 == 1 and f1 == 50 and d1 == 2 and near(u1, 0.0254), \'full call: \' + show(t1, f1, d1, u1))\nmi_probdef(0, \'millimeters\')\nt2, f2, d2, u2 = mi_getprobleminfo()\ncheck(t2 == 1 and f2 == 0 and d2 == 2 and near(u2, 0.001), \'two arguments: \' + show(t2, f2, d2, u2))\ndef saved(name):\n    mi_saveas(\'/tmp/\' + name + \'.fem\')\n    text = open(\'/tmp/\' + name + \'.fem\').read()\n    return [re.search(r\'\\[\' + k + r\'\\]\\s*=\\s*(\\S+)\', text).group(1) for k in (\'Precision\', \'MinAngle\', \'ACSolver\', \'LengthUnits\', \'ProblemType\')]\np1, m1, s1, l1, y1 = saved(\'b_probdef1\')\ncheck(near(float(p1), 1e-10) and float(m1) == 25 and float(s1) == 1, \'kept in the file: \' + show(p1, m1, s1))\nmi_probdef(0, \'millimeters\', \'planar\', 1e-5, 0, 40, 3)\nt3, f3, d3, u3 = mi_getprobleminfo()\np2, m2, s2, l2, y2 = saved(\'b_probdef2\')\ncheck(t3 == 0 and d3 == 1 and near(float(p2), 1e-8) and float(m2) == 25 and float(s2) == 1, \'out of range: \' + show(t3, d3, p2, m2, s2))\nmi_probdef(0, \'millimeters\', \'planar\', 1e-8, 5)\nnonedepth = fails(mi_probdef, 0, \'millimeters\', \'planar\', None, 7)\nt4, f4, d4, u4 = mi_getprobleminfo()\ncheck(nonedepth is None and d4 == 5, \'None before depth 7: \' + show(nonedepth, d4))\nmi_probdef(0, \'micrometers\', \'axi\')\np5, m5, s5, l5, y5 = saved(\'b_probdef3\')\ncheck(l5 == \'microns\' and y5 == \'axisymmetric\', \'file words \' + show(l5, y5))\nobs = []\nfor unit, metres in ((\'inches\', 0.0254), (\'millimeters\', 0.001), (\'centimeters\', 0.01), (\'meters\', 1), (\'mils\', 2.54e-5), (\'mills\', 2.54e-5), (\'micrometers\', 1e-6), (\'microns\', 0), (\'mm\', 0), (\'Millimeters\', 0)):\n    err = fails(mi_probdef, 0, unit)\n    got = mi_getprobleminfo()[3]\n    if metres == 0:\n        check(err is not None, unit + \' was accepted\')\n        obs.append(unit + \' rejected\')\n    else:\n        check(err is None and near(got, metres), unit + \': \' + str(err) + \' \' + str(got))\n        obs.append(unit + \' \' + str(got) + \' m\')\nfor ptype, code in ((\'planar\', 0), (\'axi\', 1), (\'axisymmetric\', -1), (\'Planar\', -1)):\n    err = fails(mi_probdef, 0, \'millimeters\', ptype)\n    got = mi_getprobleminfo()[0]\n    if code < 0:\n        check(err is not None, ptype + \' was accepted\')\n        obs.append(ptype + \' rejected\')\n    else:\n        check(err is None and got == code, ptype + \': \' + str(err) + \' \' + str(got))\n        obs.append(ptype + \' type \' + str(got))\nmm = fails(mi_probdef, 0, \'mm\')\nreturn \'new doc %s; left-off arguments kept; precision 1e-5 stored as %s, minangle 40 and acsolver 3 ignored; (..., None, 7) kept depth %g; micrometers/axi written as %s/%s; %s; "mm" error: %s\' % (show(t0, f0, d0, u0), p2, d4, l5, y5, \', \'.join(obs), mm)')
run('mi_setarcsegmentprop', "blank(1)\nmi_drawarc(10, 0, -10, 0, 180, 5)\nmi_drawarc(-10, 0, 10, 0, 180, 5)\nmi_selectarcsegment(0, 10)\nmi_selectarcsegment(0, -10)\nmi_setarcsegmentprop(90, 'A0', 0, 4)\nmi_clearselected()\nlabel(0, 0, 'Air')\nsolve('arcsegprop')\nmo_selectblock(0, 0)\nmm2 = mo_blockintegral(5) / 1e-6\nmo_clearblock()\nmo_close()\ns = open('/tmp/arcsegprop.fem').read()\narcs = ' '.join(re.search(r'\\[NumArcSegments\\]\\s*=\\s*\\d+\\s+([^\\[]*)', s).group(1).split())\nmi_selectarcsegment(0, 10)\nnonename = fails(mi_setarcsegmentprop, 5, None, 0, 4)\nmi_clearselected()\nmi_close()\nblank(1)\nmi_drawarc(10, 0, -10, 0, 180, 90)\nmi_drawarc(-10, 0, 10, 0, 180, 90)\nlabel(0, 0, 'Air')\nsolve('arcmaxseg90')\nmo_selectblock(0, 0)\nsquare = mo_blockintegral(5) / 1e-6\nmo_clearblock()\nmo_close()\npolygon = 18 * math.sin(math.pi / 18) * 100\ncheck(abs(mm2 / polygon - 1) < 1e-4, 'area %.4f mm^2, 36-gon %.4f' % (mm2, polygon))\ncheck('0 1 180 10 1 0 4' in arcs, 'arcs ' + arcs)\ncheck(abs(square / 200 - 1) < 1e-6, 'drawn with maxseg 90: area %.4f mm^2' % square)\ncheck(nonename is not None, 'None as the boundary name was accepted')\nreturn 'drawn with maxseg 5, set to 90: area %.4f mm^2 = 36-gon (%.4f); arc lines (n0 n1 angle maxseg boundary hide group ...): %s; drawn with maxseg 90 and left alone: area %.4f mm^2, a square; None as propname: %s' % (mm2, polygon, arcs, square, nonename)")
run('mi_setblockprop', "obs = []\nfor k, (direction, cx, cy, axis) in enumerate(((0, 0, 0, 'x'), (90, 0, 0, 'y'), ('theta', 0, 10, 'y'))):\n    blank()\n    mi_addmaterial('PM', 1.05, 1.05, 900000, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0)\n    mi_drawrectangle(cx - 2, cy - 2, cx + 2, cy + 2)\n    circle(40, 5)\n    arcprop(0, 40, 5, 'A0')\n    arcprop(0, -40, 5, 'A0')\n    mi_addblocklabel(cx, cy)\n    mi_selectlabel(cx, cy)\n    mi_setblockprop('PM', 1, 0, '<None>', direction, 0, 0)\n    mi_clearselected()\n    label(20, 20, 'Air')\n    solve('magnet' + str(k))\n    b1, b2 = mo_getpointvalues(cx, cy)[1:3]\n    mo_close()\n    mi_close()\n    if axis == 'x':\n        check(b1 > 0.1 and abs(b2) < 0.05 * b1, 'magdir ' + str(direction) + ': B = ' + show(b1, b2))\n    else:\n        check(b2 > 0.1 and abs(b1) < 0.05 * b2, 'magdir ' + str(direction) + ': B = ' + show(b1, b2))\n    obs.append('magdir %s at (%g,%g): B = (%.3f, %.3f) T' % (direction, cx, cy, b1, b2))\nfixture_geometry()\nmi_selectlabel(0, 0)\nmi_setblockprop('Copper', 0, 0.2, 'Iw', 0, 1, -1)\nmi_clearselected()\nsolve('turns')\nb2 = mo_getpointvalues(5, 0)[2]\nmo_close()\ncheck(b2 < -0.0035, 'turns -1: B at (5,0) = ' + str(b2))\nmi_selectlabel(0, 0)\nmi_setblockprop('Copper', 0, 0.2, None, 0, 7, 3)\nmi_clearselected()\nmi_saveas('/tmp/blockprop_none.fem')\ns = open('/tmp/blockprop_none.fem').read()\nrows = [line.split() for line in s[s.find('[NumBlockLabels]'):].splitlines()[1:] if line.startswith('0\\t0\\t')]\ncheck(len(rows) == 1 and rows[0][4] == '0' and rows[0][6] == '7' and rows[0][7] == '3', 'label line ' + str(rows))\nreturn '; '.join(obs) + '; turns -1 on the 100 A wire: By at (5,0) = %.4g T, reversed; incircuit None stored circuit 0 (none) with group %s and turns %s' % (b2, rows[0][6], rows[0][7])")
run('mi_setcomment', 'import socket\nimport time\nblank()\nmi_setcomment(\'rotor at 0 deg, 100 A\')\nmi_saveas(\'/tmp/b_comment.fem\')\nc = re.search(r\'\\[Comment\\]\\s*=\\s*"([^"]*)"\', open(\'/tmp/b_comment.fem\').read()).group(1)\nquote = fails(mi_setcomment, \'say "hi"\')\nsock = femm._default._sock\ndef pending(wait):\n    sock.settimeout(wait)\n    data = b\'\'\n    try:\n        while True:\n            chunk = sock.recv(4096)\n            if not chunk:\n                break\n            data += chunk\n    except socket.timeout:\n        pass\n    sock.settimeout(None)\n    return data\nsock.sendall(b\'mi_setcomment("line one\\nline two")\\n\')\ntime.sleep(0.5)\nraw = pending(0.5).decode().splitlines()\nfirst = fails(mi_setcomment, \'line one\\nline two\')\ntime.sleep(0.5)\nstale = fails(callfemm, \'7\')\nshifted = callfemm(\'8\') if stale else None\npending(0.5)\nin_step = callfemm(\'9\') == 9\nmi_setcomment(\'rotor at 30°\')\nmi_saveas(\'/tmp/b_comment2.fem\')\nc2 = re.search(r\'\\[Comment\\]\\s*=\\s*"([^"]*)"\', open(\'/tmp/b_comment2.fem\', encoding=\'utf-8\').read()).group(1)\ncheck(c == \'rotor at 0 deg, 100 A\', \'saved comment \' + str(c))\ncheck(quote is not None and \'invalid syntax\' in quote, \'double quote: \' + str(quote))\ncheck(len(raw) == 2 and all(\'invalid syntax\' in r for r in raw), \'replies to one line holding a newline: \' + repr(raw))\ncheck(first is not None and \'invalid syntax\' in first, \'newline through pyFEMM: \' + str(first))\ncheck(stale is None or shifted == 7, \'after a stale reply, callfemm("8") gave \' + repr(shifted))\ncheck(in_step, \'still out of step after draining the socket\')\ncheck(c2 == \'rotor at 30°\', \'degree sign: \' + ascii(c2))\nreturn \'[Comment] line of the saved .fem: "%s"; a double quote: %s; one line holding a newline got %d replies %s; through pyFEMM: %s, then the next call raised %s and callfemm("8") returned %r (the reply meant for "7"); %s saved as UTF-8\' % (c, quote, len(raw), raw, first, stale or \'nothing (both replies came in one read)\', shifted, ascii(c2))')
run('mi_setcurrent', 'fixture_geometry()\nmi_setcurrent(\'Iw\', 50)\nunknown = fails(mi_setcurrent, \'NoSuch\', 5)\nshort = fails(mi_setcurrent, \'Iw\')\nraw = fails(callfemm, \'mi_setcurrent("Iw", 50)\')\nsolve(\'setcurrent\')\ni = mo_getcircuitproperties(\'Iw\')[0]\nb2 = mo_getpointvalues(5, 0)[2]\nmo_close()\nmi_close()\nfixture_geometry()\nmi_modifycircprop(\'Iw\', 1, 50)\nmi_saveas(\'/tmp/modifycirc50.fem\')\na = open(\'/tmp/setcurrent.fem\').read()\nsame = a == open(\'/tmp/modifycirc50.fem\').read()\ncirc = \' \'.join(a[a.find(\'[CircuitProps]\'):a.find(\'[NumPoints]\')].split())\nexpected = 4e-7 * math.pi * 50 / (2 * math.pi * 0.005)\ncheck(near(i, 50, 1e-9) and \'<CircuitName> = "Iw" <TotalAmps_re> = 50 <TotalAmps_im> = 0 <CircuitType> = 1\' in circ, \'current \' + str(i) + \', \' + circ)\ncheck(same, \'the .fem after mi_setcurrent differs from the one after mi_modifycircprop(name, 1, 50)\')\ncheck(near(b2, expected, 0.02), \'By \' + str(b2) + \', expected \' + str(expected))\ncheck(unknown is None and short is not None and raw is not None and \'mi_setcurrent\' in raw, show(unknown, short, raw))\nreturn \'mi_setcurrent("Iw", 50): circuit current %g A, stored as %s; By at (5,0) = %.5g T, expected %.5g T; same .fem as mi_modifycircprop("Iw", 1, 50): %s; one argument: %s; sent to FEMM directly: %s; unknown name raised nothing\' % (i, circ, b2, expected, same, short, raw)')
run('mi_setgroup', "fixture_pre()\nmi_selectlabel(25, 0)\nmi_setgroup(5)\nmi_clearselected()\nmi_selectgroup(5)\nmi_deleteselectedlabels()\nmi_clearselected()\nx, y = mi_selectlabel(25, 0)\nmi_clearselected()\ncheck(x == 30 and y == 0, 'nearest label after deleting group 5: ' + show(x, y))\nreturn 'the iron label moved from group 2 to group 5: deleting group 5 removed it, nearest label now ' + show(x, y)")
run('mi_setnodeprop', 'blank()\nmi_addpointprop(\'P\', 0, 100)\nmi_addnode(0, 0)\nmi_addnode(10, 0)\nmi_addnode(20, 0)\nmi_selectnode(0, 0)\nmi_setnodeprop(\'P\', 3)\nmi_clearselected()\nmi_selectnode(20, 0)\nmi_setnodeprop(\'P\', 0)\nmi_clearselected()\nmi_selectnode(20, 0)\nmi_setnodeprop(\'<None>\', 0)\nnonename = fails(mi_setnodeprop, None, 0)\nmi_clearselected()\nmi_saveas(\'/tmp/nodeprop.fem\')\ns = open(\'/tmp/nodeprop.fem\').read()\npts = \' \'.join(re.search(r\'\\[NumPoints\\]\\s*=\\s*3\\s+([^\\[]*)\', s).group(1).split())\ncheck(pts.startswith(\'0 0 1 3 10 0 0 0 20 0 0 0\'), \'node lines \' + pts)\ncheck(nonename is not None, \'None as propname was accepted\')\nmi_selectgroup(3)\nmi_deleteselectednodes()\nmi_clearselected()\nx, y = mi_selectnode(0, 0)\nmi_clearselected()\ncheck(x == 10 and y == 0, \'nearest node after deleting group 3: \' + show(x, y))\nreturn \'node lines (x y property-index group): \' + pts + \'; "<None>" cleared the node at (20,0); deleting group 3 removed the node at (0,0); None as propname: \' + nonename')
run('mi_setprevious', 'def model(airmesh):\n    blank()\n    mi_addmaterial(\'Copper\', 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0, 1, 0)\n    mi_addcircprop(\'Iw\', 100, 1)\n    circle(1, 5)\n    circle(20, 5)\n    arcprop(0, 20, 5, \'A0\')\n    arcprop(0, -20, 5, \'A0\')\n    label(0, 0, \'Copper\', \'Iw\', 1, 0, 0.2)\n    label(10, 0, \'Air\', None, 0, 0, airmesh)\nmodel(1)\nsolve(\'b_prev0\')\nn0 = mo_numelements()\nby0 = mo_getpointvalues(5, 0)[2]\nmo_close()\nmi_close()\nmodel(1)\nmi_setprevious(\'/tmp/b_prev0.ans\')\nmi_saveas(\'/tmp/b_prev1.fem\')\ntext = open(\'/tmp/b_prev1.fem\').read()\nt1 = re.search(r\'\\[PrevType\\]\\s*=\\s*(\\d+)\', text).group(1)\ns1 = re.search(r\'\\[PrevSoln\\]\\s*=\\s*"([^"]*)"\', text).group(1)\ncheck(t1 == \'0\' and s1 == \'/tmp/b_prev0.ans\', \'without prevtype: \' + show(t1, s1))\nmi_setprevious(\'/tmp/b_prev0.ans\', 2)\nmi_analyze(1)\nmi_loadsolution()\nby1 = mo_getpointvalues(5, 0)[2]\nmo_close()\ncheck(near(by1, by0, 1e-6), \'frozen solve By \' + str(by1) + \' vs \' + str(by0))\nmi_setprevious(\'/tmp/b_nosuch.ans\', 2)\nerr = fails(mi_analyze, 1)\ncheck(err is not None, \'a missing previous solution did not stop the solve\')\nmi_close()\nmodel(0.3)\nsolve(\'b_fine\')\nnf = mo_numelements()\nmo_close()\nmi_setprevious(\'/tmp/b_prev0.ans\', 2)\nsolve(\'b_fineprev\')\nn2 = mo_numelements()\nmo_close()\ncheck(nf > 2 * n0 and n2 == n0, \'elements: coarse, fine, fine with the coarse .ans \' + show(n0, nf, n2))\nreturn \'no prevtype -> [PrevType] = %s; prevtype 2 on the solved .ans: By(5,0) = %.6g T, as before (%.6g); missing .ans: %s; a finer mesh (%d elements alone) solved with prevtype 2 on %d elements, those of the .ans\' % (t1, by1, by0, err, nf, n2)')
run('mi_setsegmentprop', "blank()\nmi_addboundprop('Uniform', 0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0)\nmi_drawrectangle(-10, -10, 10, 10)\nlabel(0, 0, 'Air')\ndef setall(size, auto):\n    for x, y in ((0, -10), (10, 0), (0, 10), (-10, 0)):\n        mi_selectsegment(x, y)\n    mi_setsegmentprop('Uniform', size, auto, 0, 0)\n    mi_clearselected()\nsetall(0, 1)\nsolve('segauto')\nbx, by = mo_getpointvalues(0, 0)[1:3]\nn1 = mo_numelements()\nmo_close()\nsetall(0.5, 0)\nsolve('seghalf')\nnh = mo_numelements()\nmo_close()\nsetall(0.1, 0)\nsolve('segfine')\nn2 = mo_numelements()\nmo_close()\nmi_selectsegment(0, -10)\nnonename = fails(mi_setsegmentprop, None, 0, 1, 0, 0)\nmi_clearselected()\ncheck(abs(by - 1) < 1e-3 and abs(bx) < 1e-3, 'B ' + show(bx, by))\ncheck(nh == n1 and n2 > n1, 'elements ' + show(n1, nh, n2))\ncheck(nonename is not None, 'None as propname was accepted')\nreturn 'Uniform on all four sides: B at the centre = (%.2g, %.6g) T; elements %d with automesh, %d with elementsize 0.5, %d with 0.1; None as propname: %s' % (bx, by, n1, nh, n2, nonename)")
run('mi_addbhpoints', 'fixture_pre()\nmu0 = 4e-7 * math.pi\nbs = [0, 0.5, 1.0, 1.5, 2.0]\ndef stored(path):\n    text = open(path).read()\n    part = text[text.find(\'<BlockName> = "Iron"\'):]\n    m = re.search(r\'<BHPoints> = (\\d+)([^<]*)\', part)\n    return int(m.group(1)), [float(x) for x in m.group(2).split()]\nmi_addbhpoints(\'Iron\', [[b, b / (500 * mu0)] for b in bs])\nsolve(\'bhpairs\')\nmu_pairs = mo_getpointvalues(25, 0)[9:11]\nmo_close()\nn1, p1 = stored(\'/tmp/bhpairs.fem\')\nmi_clearbhpoints(\'Iron\')\nmi_addbhpoints(\'Iron\', bs, [b / (250 * mu0) for b in bs])\nsolve(\'bhlists\')\nmu_lists = mo_getpointvalues(25, 0)[9:11]\nmo_close()\nn2, p2 = stored(\'/tmp/bhlists.fem\')\nmi_clearbhpoints(\'Iron\')\nmi_addbhpoints(\'Iron\', [0, 1, 2], [0, 1000])\nmi_addbhpoints(\'Iron\', ((1.5, 3000),))\nflat = fails(mi_addbhpoints, \'Iron\', [0, 1, 2])\nmi_saveas(\'/tmp/bhshort.fem\')\nn3, p3 = stored(\'/tmp/bhshort.fem\')\nf = open(\'/tmp/bh_kam.txt\', \'w\')\nf.write(\'\\n\'.join([\'0 0\', \'0.5 0.8\', \'1.0 1.6\', \'1.8 20\']) + \'\\n\')\nf.close()\npts = [[float(x) for x in line.split()] for line in open(\'/tmp/bh_kam.txt\') if line.strip()]\nmi_clearbhpoints(\'Iron\')\nmi_addbhpoints(\'Iron\', [[b, 1000 * hk] for b, hk in pts])\nmi_saveas(\'/tmp/bhfile.fem\')\nn4, p4 = stored(\'/tmp/bhfile.fem\')\nmissing = fails(open, \'/tmp/no_such_bh.txt\')\ncheck(n1 == 5 and n2 == 5 and p1[0::2] == p2[0::2] == bs, \'pairs \' + show(n1, p1) + \'; lists \' + show(n2, p2))\ncheck(all(near(hb, 2 * ha, 1e-12) for ha, hb in zip(p1[1::2], p2[1::2])), \'H values \' + show(p1, p2))\ncheck(near(mu_pairs[0], 500, 0.02) and near(mu_pairs[1], 500, 0.02), \'pairs: mu \' + show(*mu_pairs))\ncheck(near(mu_lists[0], 250, 0.02) and near(mu_lists[1], 250, 0.02), \'lists: mu \' + show(*mu_lists))\ncheck(n3 == 3 and p3 == [0, 0, 1, 1000, 1.5, 3000], \'zip and append: \' + show(n3, p3))\ncheck(flat is not None and \'subscriptable\' in flat, \'flat list: \' + str(flat))\ncheck(n4 == 4 and all(near(a, b, 1e-12) for a, b in zip(p4, [0, 0, 0.5, 800, 1, 1600, 1.8, 20000])), \'file: \' + show(n4, p4))\ncheck(missing is not None, \'a missing file opened\')\nreturn \'pairs B = 500 mu0 H: %d points, solved mu_r %.5g; lists B = 250 mu0 H: %d points, mu_r %.5g; lists of 3 B and 2 H plus one tuple pair stored %d points %s; flat list: %s; a B, H-in-kA/m file read in Python stored %s\' % (n1, mu_pairs[0], n2, mu_lists[0], n3, p3, flat, p4)')
run('mi_analyse', "fixture_geometry()\nmi_saveas('/tmp/b_analyse.fem')\ns = femm._default\nsent = []\nreal = s.callfemm\ns.callfemm = lambda c: (sent.append(c), real(c))[1]\ntry:\n    r = mi_analyse(1)\nfinally:\n    del s.callfemm\nmi_loadsolution()\ni = mo_getcircuitproperties('Iw')[0]\ncheck(sent == ['mi_analyze(1)'] and r is None, show(sent, r))\ncheck(near(i, 100), 'current ' + show(i))\nreturn 'mi_analyse(1) sent ' + show(*sent) + ' and solved: Iw = ' + show(i) + ' A'")
run('mi_analyze', "fixture_geometry()\nunsaved = fails(mi_analyze, 1)\nmi_saveas('/tmp/b_analyze.fem')\nmi_modifycircprop('Iw', 1, 200)\nr = mi_analyze()\nready = os.path.exists('/tmp/b_analyze.ans')\nmi_loadsolution()\ni = mo_getcircuitproperties('Iw')[0]\nmi_close()\ncheck(unsaved is not None and 'must be saved before analysis' in unsaved, show(unsaved))\ncheck(r is None and ready and near(i, 200), show(r, ready, i))\nblank()\nmi_drawrectangle(-10, -10, 10, 10)\nmi_addblocklabel(0, 0)\nmi_saveas('/tmp/b_nomat.fem')\nnomat = fails(mi_analyze, 1)\nmi_close()\nblank()\nmi_drawrectangle(-10, -10, 10, 10)\nmi_drawrectangle(-20, -20, 20, 20)\nlabel(0, 0, 'Air')\nmi_saveas('/tmp/b_nolabel.fem')\nnolabel = fails(mi_analyze, 1)\nmi_close()\nblank()\nmi_drawrectangle(-10, -10, 10, 10)\nlabel(0, 0, 'Air')\nmi_setprevious('/tmp/b_missing_previous.ans', 2)\nmi_saveas('/tmp/b_prev.fem')\nnoprev = fails(mi_analyze, 1)\ncheck(nomat is not None and 'no material' in nomat, show(nomat))\ncheck(nolabel is not None and 'error code 2560' in nolabel, show(nolabel))\ncheck(noprev is not None and 'error code 512' in noprev, show(noprev))\nreturn 'unsaved: ' + unsaved + '; saved: returned None with the .ans already written and solved the edit made after saving (Iw = ' + show(i) + ' A); label without material: ' + nomat + '; region without label: ' + nolabel + '; missing previous solution: ' + noprev")
run('mi_close', 'blank()\nmi_drawrectangle(0, 0, 1, 1)\nmi_saveas(\'/tmp/b_closea.fem\')\nblank()\nmi_drawrectangle(0, 0, 2, 2)\nmi_saveas(\'/tmp/b_closeb.fem\')\nmi_close()\nafter = fails(mi_getboundingbox)\nagain = fails(mi_close)\nmi_setfocus(\'b_closeb\')\ngone = fails(mi_getboundingbox)\nmi_setfocus(\'b_closea\')\nx1, x2, y1, y2 = mi_getboundingbox()\nmi_close()\ncheck(after is not None and \'No current magnetics input in focus\' in after, show(after))\ncheck(again is None and gone is not None, show(again, gone))\ncheck(near(x2 - x1, 1), \'width \' + show(x2 - x1))\nfixture_geometry()\nsolve(\'b_closeout\')\nmi_close()\ntitle = mo_gettitle()\ncheck(title == \'/tmp/b_closeout.ans\', show(title))\nreturn \'after mi_close: \' + after + \'; a second mi_close raised nothing; mi_setfocus("b_closeb") then found nothing; mi_setfocus("b_closea") made the other open document active; after closing a solved input, mo_gettitle() -> \' + title')
run('mi_createmesh', "fixture_pre()\nunsaved = fails(mi_createmesh)\nmi_saveas('/tmp/b_mesh.fem')\nmi_setcomment('edited after saving')\nn = mi_createmesh()\nwith open('/tmp/b_mesh.fem') as f:\n    saved = 'edited after saving' in f.read()\nraw = callfemm('mi_createmesh()')\nmi_analyze(1)\nmi_loadsolution()\nne = mo_numelements()\ncheck(unsaved is not None and 'must be saved before creating mesh' in unsaved, show(unsaved))\ncheck(n is None and raw == [] and saved, show(n, raw, saved))\ncheck(ne > 0, show(ne))\nreturn 'unsaved: ' + unsaved + '; saved: returned None (FEMM replied ' + show(raw) + ') and rewrote the .fem first; the solve then had ' + show(ne) + ' elements (mo_numelements)'")
run('mi_gettitle', 'newdocument(0)\nt0 = mi_gettitle()\nmi_saveas(\'/tmp/b_title.fem\')\nt1 = mi_gettitle()\nreal_text = \'\'.join(map(chr, callfemm(\'[ord(c) for c in mi_gettitle()]\')))\nnewdocument(0)\nmi_saveas(\'/tmp/b_other.fem\')\nmi_setfocus(\'b_title\')\nt2 = mi_gettitle()\nmi_close()\nmi_setfocus(\'b_other\')\ncallfemm(\'setcurrentdirectory("/tmp")\')\ntry:\n    mi_saveas(\'b_rel.fem\')\n    rel = mi_gettitle()\nfinally:\n    callfemm(\'setcurrentdirectory("\' + os.getcwd() + \'")\')\nmi_newdocument()\nt3 = mi_gettitle()\ncheck(t0 == [], \'unsaved title \' + show(t0))\ncheck(t1 == \'/tmp/b_t1jtle.fem\' and t1 == \'/tmp/b_title.fem\'.replace(\'I\', \'1j\').replace(\'i\', \'1j\'), show(t1))\ncheck(real_text == \'/tmp/b_title.fem\' and t2 == t1, show(real_text, t2))\ncheck(rel == \'b_rel.fem\' and os.path.exists(\'/tmp/b_rel.fem\'), show(rel))\ncheck(t3 == \'Unt1jtled.fem\', show(t3))\nreturn \'before saving: \' + show(t0) + \'; after mi_saveas("/tmp/b_title.fem"): \' + t1 + \' (real text \' + real_text + \'); after mi_setfocus("b_title") from another document: \' + t2 + \'; relative save: \' + rel + \'; after mi_newdocument: \' + t3')
run('mi_loadsolution', 'blank()\nmi_drawrectangle(-10, -10, 10, 10)\nlabel(0, 0, \'Air\')\nunsaved = fails(mi_loadsolution)\nmi_saveas(\'/tmp/b_unsolved.fem\')\nunsolved = fails(mi_loadsolution)\nmi_close()\nfixture_geometry()\nsolve(\'b_load1\')\nby1 = mo_getb(5, 0)[1]\nmi_saveas(\'/tmp/b_load2.fem\')\nmi_modifycircprop(\'Iw\', 1, 200)\nmi_analyze(1)\nmi_loadsolution()\nby2 = mo_getb(5, 0)[1]\ncallfemm(\'mo_setfocus("b_load1")\')\nt = mo_gettitle()\ncheck(unsaved is not None and \'No file path\' in unsaved, show(unsaved))\ncheck(unsolved is not None and \'No results file: /tmp/b_unsolved.ans\' in unsolved, show(unsolved))\ncheck(near(by2, 2 * by1, 1e-3) and t == \'/tmp/b_load2.ans\', show(by1, by2, t))\nreturn \'unsaved: \' + unsaved + \' | saved, never solved: \' + unsolved + \' | after mi_analyze it loads (By at (5,0) \' + f\'{by1:.5g}\' + \' T); loading b_load2 while b_load1 had the focus replaced it: mo_setfocus("b_load1") left the focus on \' + t')
run('mi_newdocument', "blank()\nmi_probdef(50, 'inches', 'axi', 1e-9, 3, 20, 1)\nmi_drawrectangle(0, 0, 5, 5)\nmi_saveas('/tmp/b_newdoc.fem')\nr = mi_newdocument()\ninfo = mi_getprobleminfo()\nbox = mi_getboundingbox()\ntitle = mi_gettitle()\nmi_saveas('/tmp/b_newdoc2.fem')\nwith open('/tmp/b_newdoc2.fem') as f:\n    text = f.read()\nnmat = re.search(r'\\[BlockProps\\]\\s*=\\s*(\\d+)', text).group(1)\nnbdry = re.search(r'\\[BdryProps\\]\\s*=\\s*(\\d+)', text).group(1)\ncheck(r is None and info[:3] == [0, 0, 1] and near(info[3], 0.001), 'problem after mi_newdocument: ' + show(*info))\ncheck(box == [], 'geometry left: ' + show(box))\ncheck(title == 'Unt1jtled.fem', show(title))\ncheck(nmat == '0' and nbdry == '0', show(nmat, nbdry))\nmi_close()\nnone = fails(mi_newdocument)\ncheck(none is not None and 'No current magnetics input in focus' in none, show(none))\nreturn 'returned None; reset in place: problem ' + show(*info) + ', no geometry, title ' + title + ', ' + nmat + ' materials, ' + nbdry + ' boundaries; with no document in focus: ' + none")
run('mi_purgemesh', "fixture_pre()\nmi_saveas('/tmp/b_purge.fem')\nmi_createmesh()\nr1 = mi_purgemesh()\nraw = callfemm('mi_purgemesh()')\nmi_analyze(1)\nmi_loadsolution()\nne = mo_numelements()\nmi_close()\nr2 = mi_purgemesh()\ncheck(r1 is None and raw == [] and r2 is None and ne > 0, show(r1, raw, r2, ne))\nreturn 'returned None (FEMM replied ' + show(raw) + '); a solve after it still had ' + show(ne) + ' elements; no error with no document in focus'")
run('mi_readdxf', "fixture_pre()\nmi_savedxf('/tmp/b_rd.dxf')\nmi_close()\nwith open('/tmp/b_rd.dxf', 'rb') as f:\n    text = f.read()\nwith open('/tmp/b_rd_crlf.dxf', 'wb') as f:\n    f.write(text.replace(b'\\n', b'\\r\\n'))\nblank()\nmi_readdxf('/tmp/b_rd.dxf')\nlf = mi_getboundingbox()\nmi_close()\ncheck(all(near(a, b) for a, b in zip(lf, [23, 27, -2, -2])), 'LF file: ' + show(*lf))\nblank()\nmi_addnode(80, 0)\nmi_readdxf('/tmp/b_rd_crlf.dxf')\nbox = mi_getboundingbox()\ncheck(near(box[0], -50) and near(box[1], 80) and near(box[3], 50), 'CRLF file added to a node at (80,0): ' + show(*box))\nmi_saveas('/tmp/b_rd.fem')\nwith open('/tmp/b_rd.fem') as f:\n    fem = f.read()\nm = re.search(r'\\[NumArcSegments\\]\\s*=\\s*(\\d+)\\n\\S+\\t\\S+\\t(\\S+)\\t(\\S+)', fem)\nnlab = re.search(r'\\[NumBlockLabels\\]\\s*=\\s*(\\d+)', fem).group(1)\nnarc, span, maxseg = m.group(1), float(m.group(2)), float(m.group(3))\ncheck(narc == '8' and abs(span - 180) < 1e-6 and maxseg == 1 and nlab == '0', show(narc, span, maxseg, nlab))\nmissing = fails(mi_readdxf, '/tmp/b_missing.dxf')\ncheck(missing is not None and 'Failed to read DXF file' in missing, show(missing))\nr = mi_readdxf('/tmp/b_rd_crlf.dxf', 0.01)\nagain = mi_getboundingbox()\ncheck(r is None and all(near(a, b) for a, b in zip(again, box)), 'box after a second read ' + show(*again))\nthree = fails(mi_readdxf, '/tmp/b_rd_crlf.dxf', 0.01, 5)\ncheck(three is not None and 'takes from 1 to 2 positional arguments' in three, show(three))\nreturn 'LF file from mi_savedxf: box ' + show(*lf) + ' (only its first LINE, the iron square\\'s bottom side); CRLF copy added to a node at (80,0): box ' + show(*box) + f'; {narc} arcs, the first spans {span:g} deg with maxseg {maxseg:g}; {nlab} labels; missing file: ' + missing + '; mi_readdxf(name, 0.01) returned None, box unchanged; three arguments: ' + three")
run('mi_saveas', 'blank()\nmi_drawrectangle(0, 0, 5, 5)\nlabel(2, 2, \'Air\')\nmi_saveas(\'/tmp/b_Save.FEM\')\ntitle = mi_gettitle()\nlower = os.path.exists(\'/tmp/b_Save.fem\') and not os.path.exists(\'/tmp/b_Save.FEM\')\nwith open(\'/tmp/b_Save.fem\') as f:\n    first = f.readline().strip()\nmi_drawline(1, 1, 3, 1)\nmi_createmesh()\nwith open(\'/tmp/b_Save.fem\') as f:\n    npoints = re.search(r\'\\[NumPoints\\]\\s*=\\s*(\\d+)\', f.read()).group(1)\nmi_saveas(\'/tmp/b_noext\')\nnoext = os.path.exists(\'/tmp/b_noext\') and not os.path.exists(\'/tmp/b_noext.fem\')\nbad = fails(mi_saveas, \'/nonexistent/dir/b.fem\')\ncheck(title == \'/tmp/b_Save.fem\' and lower, show(title, lower))\ncheck(npoints == \'6\', \'the file was not rewritten: \' + show(npoints))\ncheck(noext, \'no extension: a .fem was added\')\ncheck(bad is not None and \'Could not save to /nonexistent/dir/b.fem\' in bad, show(bad))\nreturn \'mi_saveas("/tmp/b_Save.FEM") wrote \' + title + \' (first line \' + first + \'); after a line was drawn, mi_createmesh rewrote it (\' + npoints + \' points); "/tmp/b_noext" got no extension; missing directory: \' + bad')
run('mi_savedxf', "fixture_pre()\nbox = mi_getboundingbox()\nmi_savedxf('/tmp/b_rt.dxf')\nmi_close()\nwith open('/tmp/b_rt.dxf', 'rb') as f:\n    text = f.read()\nnline, narc, ncr = text.count(b'\\nLINE\\n'), text.count(b'\\nARC\\n'), text.count(b'\\r')\nblank()\nmi_readdxf('/tmp/b_rt.dxf')\nlf = mi_getboundingbox()\nlarc = mi_selectarcsegment(0, 50)\nmi_saveas('/tmp/b_rt_lf.fem')\nwith open('/tmp/b_rt_lf.fem') as f:\n    fem = f.read()\nnseg = re.search(r'\\[NumSegments\\]\\s*=\\s*(\\d+)', fem).group(1)\nnarcs = re.search(r'\\[NumArcSegments\\]\\s*=\\s*(\\d+)', fem).group(1)\nmi_close()\nwith open('/tmp/b_rt_crlf.dxf', 'wb') as f:\n    f.write(text.replace(b'\\n', b'\\r\\n'))\nblank()\nmi_readdxf('/tmp/b_rt_crlf.dxf')\ncrlf = mi_getboundingbox()\nlx = mi_selectlabel(25, 0)\ncheck(nline == 4 and narc == 8 and ncr == 0, show(nline, narc, ncr))\ncheck(all(near(a, b) for a, b in zip(lf, [23, 27, -2, -2])) and larc == [] and nseg == '1' and narcs == '0', 'LF file read back as ' + show(lf, larc, nseg, narcs))\ncheck(all(near(a, b) for a, b in zip(crlf, box)), 'CRLF file read back as ' + show(*crlf))\ncheck(lx == [], 'a block label came back: ' + show(lx))\nreturn f'fixture (4 segments, 8 arcs) -> {nline} LINE + {narc} ARC entities with LF endings; read back: one segment (the first LINE, the iron square\\'s bottom side), box ' + show(*lf) + ', no arcs; the same file with CRLF endings: box ' + show(*crlf) + ', no block labels'")
run('mi_setfocus', "blank()\nmi_drawrectangle(0, 0, 1, 1)\nmi_saveas('/tmp/b_focusa.fem')\nblank()\nmi_drawrectangle(0, 0, 2, 2)\nmi_saveas('/tmp/b_focusb.fem')\nblank()\nmi_drawrectangle(0, 0, 3, 3)\nseen = []\nfor name, want in [('b_focusa', 1), ('b_focusb.fem', 2), ('/tmp/B_FOCUSA.FEM', 1), ('B_FocusB', 2), ('nosuch', 2), ('Untitled', 2)]:\n    r = mi_setfocus(name)\n    x1, x2, y1, y2 = mi_getboundingbox()\n    seen.append(name + ' -> width ' + show(x2 - x1))\n    check(r is None and near(x2 - x1, want), '; '.join(seen))\nmi_close()\nmi_setfocus('b_focusa')\nmi_close()\nreturn '; '.join(seen) + '; the unsaved third document (width 3) was never reached'")
run('mi_showmesh', "fixture_pre()\nmi_saveas('/tmp/b_showmesh.fem')\nmi_createmesh()\nr1 = mi_showmesh()\nraw = callfemm('mi_showmesh()')\nmi_close()\nr2 = mi_showmesh()\ncheck(r1 is None and raw == [] and r2 is None, show(r1, raw, r2))\nreturn 'headless: returned None (FEMM replied ' + show(raw) + '), also after mi_close with no document in focus'")
run('mi_smartmesh', "def flag():\n    with open('/tmp/b_sm.fem') as f:\n        return re.search(r'\\[DoSmartMesh\\]\\s*=\\s*(\\d+)', f.read()).group(1)\nfixture_pre()\nmi_saveas('/tmp/b_sm.fem')\nflag0 = flag()\nmi_smartmesh(0)\nmi_analyze(1)\nflagoff = flag()\nmi_loadsolution()\nnoff = mo_numelements()\nmi_smartmesh(1)\nmi_analyze(1)\nmi_loadsolution()\nnon = mo_numelements()\ncheck(flag0 == '1' and flagoff == '0', show(flag0, flagoff))\ncheck(noff == 8356 and non == 15538, show(noff, non))\nreturn 'new document [DoSmartMesh] = ' + flag0 + ', after mi_smartmesh(0) = ' + flagoff + '; fixture mesh: ' + show(noff) + ' elements off, ' + show(non) + ' on'")
print("BATCH_DONE", flush=True)
closefemm()
