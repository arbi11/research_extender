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
run('mo_getgapharmonics', "fixture_post()\nnmax = mo_getgapharmonics('AGap')\nh0 = mo_getgapharmonics('AGap', 0)\nh1 = mo_getgapharmonics('AGap', 1)\nabove = mo_getgapharmonics('AGap', nmax + 1)\nnegative = mo_getgapharmonics('AGap', -1)\nwrong = mo_getgapharmonics('Agap', 1)\na = mo_getgapa('AGap', 0)\nbt = mo_getgapb('AGap', 0)[1]\nobs = ('name only -> %r; n = 0 -> %r; n = 1 -> %r; n = %d -> %r; n = -1 -> %r; misspelt name -> %r; '\n       'mo_getgapa(0) %.6g; Bt(0) %.6g' % (nmax, h0, h1, nmax + 1, above, negative, wrong, a, bt))\ncheck(nmax == 180 and isinstance(nmax, int), obs)\ncheck(len(h0) == 6 and len(h1) == 6 and near(h0[0], a, 1e-3), obs)\ncheck(h0[4] == 0 and h0[2] == 0 and bt > 1.9e-3, obs)\ncheck(above == [0] * 6 and negative == [] and wrong == [], obs)\nreturn obs")
run('mo_getnode', "fixture_post()\nnn = mo_numnodes()\nfirst = mo_getnode(1)\nlast = mo_getnode(nn)\nzero, beyond = mo_getnode(0), mo_getnode(nn + 1)\na, b, c, xc, yc = mo_getelement(1)[:5]\nax, ay = mo_getnode(a)\nbx, by = mo_getnode(b)\ncx, cy = mo_getnode(c)\ndx, dy = (ax + bx + cx) / 3 - xc, (ay + by + cy) / 3 - yc\nobs = ('%d nodes; node 1 at %r; node %d at %r; mo_getnode(0) -> %r; mo_getnode(%d) -> %r; the nodes of element 1 '\n       'average to its centroid within (%.2g, %.2g) mm' % (nn, first, nn, last, zero, nn + 1, beyond, dx, dy))\ncheck(isinstance(first, list) and len(first) == 2 and len(last) == 2, obs)\ncheck(zero == [] and beyond == [], obs)\ncheck(abs(dx) < 1e-6 and abs(dy) < 1e-6, obs)\ncheck(max(abs(v) for v in first + last) > 1, obs)\nreturn obs")
run('mo_getpointvalues', "# 5 mm from a +100 A wire, |B| = mu0 I / (2 pi r) = 4 mT, along +y at (5,0): positive\n# current flows out of the page (+z). A = 0 at r = 50 mm, so A(5 mm) = mu0 I/(2 pi) ln(10).\nmu0 = 4e-7 * math.pi\nfixture_post()\nvalues = mo_getpointvalues(5, 0)\ncheck(isinstance(values, list) and len(values) == 14, repr(values))\nA, B1, B2, Sig, E, H1, H2, Je, Js, Mu1, Mu2, Pe, Ph, ff = values\nexpected = mu0 * 100 / (2 * math.pi * 0.005)\ncheck(abs(B1) < 0.02 * expected, 'B1 = ' + repr(B1))\ncheck(abs(B2 / expected - 1) < 0.02, 'B2 = ' + repr(B2) + ', expected ' + repr(expected))\ncheck(abs(H2 * mu0 / B2 - 1) < 0.01, 'H2 is not B2 / mu0 in air')\ncheck(Mu1 == 1 and Mu2 == 1 and type(Mu1) is int and type(Sig) is int and ff == -1, 'ints in air: ' + repr(values))\naexp = mu0 * 100 / (2 * math.pi) * math.log(10)\ncheck(abs(A / aexp - 1) < 0.02, 'A = ' + repr(A) + ', expected ' + repr(aexp))\neexp = (B1 * B1 + B2 * B2) / (2 * mu0)\ncheck(abs(E / eexp - 1) < 0.02, 'E = ' + repr(E) + ', expected ' + repr(eexp))\n# Inside the copper wire (58 MS/m, 100 A over 3.1376e-6 m^2): Sig and Js are scaled.\nwire = mo_getpointvalues(0.5, 0)\nwSig, wJe, wJs, wPe = wire[3], wire[7], wire[8], wire[11]\njs = 100 / (36 * math.sin(math.radians(5)) * 1e-6)\ncheck(abs(wSig / 58 - 1) < 0.01, 'Sig = ' + repr(wSig) + ', expected 58 (MS/m)')\ncheck(abs(wJs / (js / 1e6) - 1) < 0.01, 'Js = ' + repr(wJs) + ', expected ' + repr(js / 1e6) + ' (MA/m^2)')\ncheck(wPe == 0 and wJe == 0, 'Pe = ' + repr(wPe) + ', Je = ' + repr(wJe) + ' at 0 Hz')\n# No mesh: outside the model and inside the <No Mesh> air-gap band.\noutside = mo_getpointvalues(100, 100)\nband = mo_getpointvalues(10.5, 0)\ncheck(outside == [] and band == [], 'outside ' + repr(outside) + ', band ' + repr(band))\nmo_close()\nmi_close()\n# 50 Hz: phasors are complex; E and Pe are real time averages for peak amplitudes.\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('pointac')\nac = mo_getpointvalues(0.5, 0)\nb1, b2, sig, e, h1, h2, je, jsac, pe = ac[1], ac[2], ac[3], ac[4], ac[5], ac[6], ac[7], ac[8], ac[11]\neavg = (b1 * h1.conjugate() + b2 * h2.conjugate()).real / 4\npeavg = abs((je + jsac) * 1e6) ** 2 / (2 * sig * 1e6)\ne5 = mo_getpointvalues(5, 0)[4]\nobs = ('at (5,0) mm: B = (%.4g, %.5g) T (expected By %.5g), A = %.5g Wb/m, E = %.4g J/m^3; in the wire: Sig = %r (MS/m), '\n       'Js = %.5g (MA/m^2), Pe = %r at 0 Hz; outside the model and inside the gap band: []; at 50 Hz in the wire: '\n       'B2 = %r, E = %.5g (Re(B.H*)/4 = %.5g), Pe = %.6g W/m^3 (|J|^2/(2 sigma) = %.6g); E at (5,0) %.5g, half of DC %.5g'\n       % (B1, B2, expected, A, E, wSig, wJs, wPe, b2, e, eavg, pe, peavg, e5, E))\ncheck(all(isinstance(v, complex) for v in (ac[0], b1, b2, h1, h2, je, jsac)), obs)\ncheck(isinstance(e, float) and isinstance(pe, float) and not isinstance(ac[12], complex), obs)\ncheck(near(e, eavg, 1e-6) and near(pe, peavg, 1e-6) and near(e5, E / 2, 1e-3), obs)\nreturn obs")
run('mo_lineintegral', "# Type 1 (H.t) round a closed contour gives the enclosed current (Ampere's law).\nfixture_post()\nmo_clearcontour()\nmo_addcontour(5, 0)\nmo_addcontour(-5, 0)\nmo_bendcontour(180, 1)\nmo_addcontour(5, 0)\nmo_bendcontour(180, 1)\ntotal, average = mo_lineintegral(1)\nlength, surface = mo_lineintegral(2)\nforce = mo_lineintegral(3)\ntorque = mo_lineintegral(4)\nunknown = mo_lineintegral(6)\nmo_clearcontour()\nmo_close()\nmi_close()\nobs = ('contour (5,0) -> (-5,0) -> (5,0), each edge bent +180: H.t = %.5g A; length %.5g m, surface %.5g m^2; '\n       'static type 3 %r, type 4 %r, type 6 %r' % (total, length, surface, force, torque, unknown))\ncheck(abs(total - 100) < 1, obs)\ncheck(abs(length / (2 * math.pi * 0.005) - 1) < 0.01, obs)\ncheck(len(force) == 2 and len(torque) == 2 and torque[1] == 0 and unknown == [], obs)\n# 50 Hz: type 3 gives four values, the 2x pair complex; H.t is complex.\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('lineac')\nmo_addcontour(5, 0)\nmo_addcontour(-5, 0)\nmo_bendcontour(180, 1)\nmo_addcontour(5, 0)\nmo_bendcontour(180, 1)\nhtac = mo_lineintegral(1)[0]\nforceac = mo_lineintegral(3)\nmo_clearcontour()\nobs += '; at 50 Hz H.t = %r, type 3 gives %d values %r' % (htac, len(forceac), forceac)\ncheck(isinstance(htac, complex) and abs(htac - 100) < 1, obs)\ncheck(len(forceac) == 4 and isinstance(forceac[2], complex), obs)\nreturn obs")
run('mo_makeplot', "fixture_post()\nmo_clearcontour()\nnone = fails(mo_makeplot, 1, 8, '/tmp/mo_noplot.csv')\nmo_addcontour(2, 0)\nmo_addcontour(9, 0)\nr = mo_makeplot(1, 8, '/tmp/mo_plot.csv')\nlines = open('/tmp/mo_plot.csv').read().splitlines()\nr4 = mo_makeplot(1, 8, '/tmp/mo_plot2.csv', 0)\nsame = open('/tmp/mo_plot2.csv').read().splitlines() == lines\nscreen = mo_makeplot(1, 8)\ndefault = mo_makeplot(1)\ntwo = mo_makeplot(1, 1, '/tmp/mo_plot1.csv')\ntwo_lines = open('/tmp/mo_plot1.csv').read().splitlines()\nbogus = mo_makeplot(9, 3, '/tmp/mo_plot9.csv')\nbogus_lines = open('/tmp/mo_plot9.csv').read().splitlines()\nunwritable = fails(mo_makeplot, 1, 3, '/nonexistent/plot.csv')\nmo_clearcontour()\nmo_close()\nmi_close()\nrows = [[float(v) for v in line.split(',')] for line in lines[1:]]\nobs = ('no contour: %s; returned %r; header %r; %d rows; first %r; last %r; with a 4th argument returned %r, same file: %s; '\n       'to the screen %r; PlotType only %r; NumPoints 1 -> %r, rows %r; PlotType 9 -> %r, %r; unwritable path: %s'\n       % (none, r, lines[0], len(rows), lines[1], lines[-1], r4, same, screen, default, two, two_lines[1:], bogus, bogus_lines, unwritable))\ncheck(none is not None and 'No contour defined' in none, obs)\ncheck(r == 8 and r4 == 8 and screen == 8 and default == 150 and lines[0] == 'Position, |B|' and len(rows) == 8 and same, obs)\ncheck(abs(rows[0][0]) < 1e-9 and near(rows[-1][0], 7), obs)\ncheck(near(rows[0][1], 2e-7 * 100 / 0.002, 0.02) and near(rows[-1][1], 2e-7 * 100 / 0.009, 0.02), obs)\ncheck(two == 2 and len(two_lines) == 3, obs)\ncheck(bogus_lines[0] == 'Position, Value' and all(float(l.split(',')[1]) == 0 for l in bogus_lines[1:]), obs)\ncheck(unwritable is not None and 'Could not open output file' in unwritable, obs)\n# 50 Hz: the current-density plots add real and imaginary columns (MA/m^2).\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('plotac')\nmo_addcontour(-0.9, 0)\nmo_addcontour(0.9, 0)\nmo_makeplot(8, 3, '/tmp/mo_plotac.csv')\nmo_clearcontour()\nac = open('/tmp/mo_plotac.csv').read().splitlines()\nobs += '; at 50 Hz PlotType 8 header %r, middle row %r' % (ac[0], ac[2])\ncheck(ac[0] == 'Position, |Js+J_eddy|, Re(Js+J_eddy), Im(Js+J_eddy)' and len(ac) == 4, obs)\ncheck(near(float(ac[2].split(',')[1]), 100 / (36 * math.sin(math.radians(5))), 0.01), obs)\nreturn obs")
run('mo_numelements', "fixture_post()\nne = mo_numelements()\nnn = mo_numnodes()\nlast = mo_getelement(ne)\nbeyond = mo_getelement(ne + 1)\nobs = ('mo_numelements() -> %r (%d nodes); mo_getelement(%d) -> %r; mo_getelement(%d) -> %r'\n       % (ne, nn, ne, last, ne + 1, beyond))\ncheck(isinstance(ne, int) and ne > 0 and len(last) == 7 and beyond == [], obs)\nreturn obs")
run('mo_numnodes', "fixture_post()\nnn = mo_numnodes()\nlast = mo_getnode(nn)\nbeyond = mo_getnode(nn + 1)\nobs = 'mo_numnodes() -> %r; mo_getnode(%d) -> %r; mo_getnode(%d) -> %r' % (nn, nn, last, nn + 1, beyond)\ncheck(isinstance(nn, int) and nn > 0 and len(last) == 2 and beyond == [], obs)\nreturn obs")
run('mo_selectpoint', "fixture_post()\nmo_clearcontour()\nr = mo_selectpoint(1.2, 0.1)\nmo_selectpoint(9.7, 0.2)\nradial = mo_lineintegral(2)[0]\nmo_clearcontour()\nmo_selectpoint(10, 0)\nmo_selectpoint(-10, 0)\narc = mo_lineintegral(2)[0]\nht = mo_lineintegral(1)[0]\nmo_clearcontour()\nobs = ('(1.2,0.1) then (9.7,0.2): length %.6g m (nodes (1,0) to (10,0) = 0.009); (10,0) then (-10,0): length %.6g m '\n       '(half circle %.6g), H.t %.5g A; mo_selectpoint returned %r' % (radial, arc, math.pi * 0.010, ht, r))\ncheck(near(radial, 0.009), obs)\ncheck(near(arc, math.pi * 0.010, 1e-3) and ht > 45, obs)\ncheck(r == [], obs)\nreturn obs")
run('mo_setweightingscheme', "fixture_post()\nmo_groupselectblock(1)\nparts = []\nworst = 0\nfor k in range(5):\n    r = mo_setweightingscheme(k)\n    fx = mo_blockintegral(18)\n    t = mo_blockintegral(22)\n    worst = max(worst, abs(fx), abs(t))\n    parts.append('scheme %d: Fx %.3g N, T %.3g N m' % (k, fx, t))\nr5 = mo_setweightingscheme(5)\nrm = mo_setweightingscheme(-1)\nmo_setweightingscheme(0)\nmo_clearblock()\nobs = '; '.join(parts) + '; n = 5 and n = -1 returned ' + show(r5, rm) + ' without an error'\ncheck(worst < 1e-5, obs)\ncheck(r is None and r5 is None and rm is None, obs)\nreturn obs")
run('mo_smooth', 'fixture_post()\nx, y = 5.03, 0.07\nA0, bx0, by0 = mo_getpointvalues(x, y)[:3]\nr = mo_smooth(\'off\')\nA1, bx1, by1 = mo_getpointvalues(x, y)[:3]\nmo_smooth(\'on\')\nA2, bx2, by2 = mo_getpointvalues(x, y)[:3]\nbogus = fails(mo_smooth, \'maybe\')\nempty = fails(mo_smooth)\nobs = (\'By at (5.03, 0.07): before any call %.7g, off %.7g, on %.7g T; A %.8g, %.8g, %.8g; mo_smooth returned %r; \'\n       \'flag "maybe": %s; no flag: %s\' % (by0, by1, by2, A0, A1, A2, r, bogus, empty))\ncheck(by0 == by2, obs)\ncheck(abs(by1 - by2) > 1e-3 * abs(by2), obs)\ncheck(near(A1, A2, 1e-9), obs)\ncheck(r is None and bogus is not None and \'Unknown option for smoothing\' in bogus, obs)\ncheck(empty is not None and \'Unknown option for smoothing\' in empty, obs)\nreturn obs')
run('mo_smoothoff', 'fixture_post()\nmo_smooth(\'off\')\na = mo_getpointvalues(5.03, 0.07)[2]\nmo_smooth(\'on\')\nr = mo_smoothoff()\nb = mo_getpointvalues(5.03, 0.07)[2]\nmo_smooth(\'on\')\nc = mo_getpointvalues(5.03, 0.07)[2]\nobs = \'By: mo_smooth("off") %.7g, mo_smoothoff() %.7g, smoothing on %.7g; mo_smoothoff returned %r\' % (a, b, c, r)\ncheck(a == b and b != c and r is None, obs)\nreturn obs')
run('mo_smoothon', 'fixture_post()\nmo_smooth(\'on\')\na = mo_getpointvalues(5.03, 0.07)[2]\nmo_smooth(\'off\')\noff = mo_getpointvalues(5.03, 0.07)[2]\nr = mo_smoothon()\nb = mo_getpointvalues(5.03, 0.07)[2]\nobs = \'By: mo_smooth("on") %.7g, off %.7g, mo_smoothon() after "off" %.7g; mo_smoothon returned %r\' % (a, off, b, r)\ncheck(a == b and off != b and r is None, obs)\nreturn obs')
run('mo_geta', "fixture_post()\na = mo_geta(5, 0)\nfull = mo_getpointvalues(5, 0)[0]\ntry:\n    mo_geta(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nexpected = 2e-7 * 100 * math.log(50 / 5)\nobs = ('A at (5,0) mm %r Wb/m (mo_getpointvalues item 0 %r; mu0 I ln(10) / (2 pi) = %.5g); in the gap band: %s'\n       % (a, full, expected, band))\ncheck(a == full and near(a, expected, 0.02), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getb', "fixture_post()\nb = mo_getb(5, 0)\nfull = mo_getpointvalues(5, 0)[1:3]\ntry:\n    mo_getb(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nexpected = 2e-7 * 100 / 0.005\nobs = ('B at (5,0) mm %r T (mo_getpointvalues items 1, 2: %r; mu0 I / (2 pi r) = %.4g T along +y); in the gap band: %s'\n       % (b, full, expected, band))\ncheck(b == full and abs(b[0]) < 0.02 * expected and near(b[1], expected, 0.02), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getconductivity', "fixture_post()\ncopper, air = mo_getconductivity(0.5, 0), mo_getconductivity(5, 0)\ntry:\n    mo_getconductivity(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nmo_close()\nmi_close()\n# A magnet-wire winding (wire type 3): 20 turns of 0.5 mm copper in a 4 x 4 mm block,\n# and a laminated block: 0.5 mm laminations, fill 0.9, 2 MS/m.\nfixture_geometry()\nmi_addmaterial('Wire', 1, 1, 0, 0, 58, 0, 0, 1, 3, 0, 0, 1, 0.5)\nmi_addcircprop('Ic', 1, 1)\nmi_drawrectangle(-27, -2, -23, 2)\nlabel(-25, 0, 'Wire', 'Ic', 20, 3)\nmi_addmaterial('Lam', 1000, 1000, 0, 0, 2, 0.5, 0, 0.9, 0, 0, 0, 1, 0)\nmi_drawrectangle(-2, 23, 2, 27)\nlabel(0, 25, 'Lam', None, 0, 4)\nsolve('sigdc')\ncoil = mo_getconductivity(-25, 0)\nfill = mo_getfill(-25, 0)\nlamdc = mo_getconductivity(0, 25)\nmo_close()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('sigac')\nlamac = mo_getconductivity(0, 25)\nobs = ('copper %r, air %r (MS/m); magnet-wire winding %.6g = 58 x fill factor %.6g; laminated block %r at DC, %r at 50 Hz; '\n       'in the gap band: %s' % (copper, air, coil, fill, lamdc, lamac, band))\ncheck(copper == 58 and air == 0 and type(air) is int, obs)\ncheck(near(coil, 58 * fill, 1e-6) and near(fill, 20 * math.pi * 0.25 ** 2 / 16, 1e-6), obs)\ncheck(lamdc == 2 and lamac == 0, obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getenergydensity', "mu0 = 4e-7 * math.pi\nfixture_post()\ne = mo_getenergydensity(5, 0)\nbx, by = mo_getb(5, 0)\ntry:\n    mo_getenergydensity(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nmo_close()\nmi_close()\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('energyac')\neac = mo_getenergydensity(5, 0)\nb1, b2 = mo_getb(5, 0)\nh1, h2 = mo_geth(5, 0)\naverage = (b1 * h1.conjugate() + b2 * h2.conjugate()).real / 4\ndc = (bx ** 2 + by ** 2) / (2 * mu0)\nobs = ('DC at (5,0) mm: %.5g J/m^3 (|B|^2 / (2 mu0) = %.5g); at 50 Hz: %.5g J/m^3 (Re(B.H*)/4 = %.5g), a %s; '\n       'in the gap band: %s' % (e, dc, eac, average, type(eac).__name__, band))\ncheck(near(e, dc, 1e-3), obs)\ncheck(isinstance(eac, float) and near(eac, average, 1e-6) and near(eac, e / 2, 1e-3), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getfill', "fixture_post()\ncopper, air = mo_getfill(0.5, 0), mo_getfill(5, 0)\ntry:\n    mo_getfill(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nmo_close()\nmi_close()\nfixture_geometry()\nmi_addmaterial('Wire', 1, 1, 0, 0, 58, 0, 0, 1, 3, 0, 0, 1, 0.5)\nmi_addcircprop('Ic', 1, 1)\nmi_drawrectangle(-27, -2, -23, 2)\nlabel(-25, 0, 'Wire', 'Ic', 20, 3)\nmi_addmaterial('Lam', 1000, 1000, 0, 0, 2, 0.5, 0, 0.9, 0, 0, 0, 1, 0)\nmi_drawrectangle(-2, 23, 2, 27)\nlabel(0, 25, 'Lam', None, 0, 4)\nsolve('fill')\ncoil = mo_getfill(-25, 0)\nlam = mo_getfill(0, 25)\nexpected = 20 * math.pi * 0.25 ** 2 / 16\nobs = ('solid copper %r, air %r; 20 turns of 0.5 mm magnet wire in 16 mm^2: %.5g (20 pi 0.25^2 / 16 = %.5g); '\n       'laminated block (fill 0.9): %r; in the gap band: %s' % (copper, air, coil, expected, lam, band))\ncheck(copper == -1 and air == -1 and lam == -1, obs)\ncheck(near(coil, expected, 1e-6), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_geth', "mu0 = 4e-7 * math.pi\nfixture_pre()\nsolve('hfield')\nhair = mo_geth(5, 0)\nbair = mo_getb(5, 0)\nhiron = mo_geth(25, 0)\nbiron = mo_getb(25, 0)\nfull = mo_getpointvalues(25, 0)[5:7]\ntry:\n    mo_geth(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nrair = bair[1] / (mu0 * hair[1])\nriron = biron[1] / (mu0 * hiron[1])\nobs = ('air (5,0): H %r A/m, B / (mu0 H) = %.6g; iron (25,0), mu_r 1000: H %r A/m, B / (mu0 H) = %.6g; '\n       'mo_getpointvalues items 5, 6: %r; in the gap band: %s' % (hair, rair, hiron, riron, full, band))\ncheck(hiron == full, obs)\ncheck(near(rair, 1, 1e-6) and near(riron, 1000, 1e-6), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getj', "fixture_post()\nj = mo_getj(0.5, 0)\nje, js = mo_getpointvalues(0.5, 0)[7:9]\njair = mo_getj(5, 0)\ntry:\n    mo_getj(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nmo_close()\nmi_close()\nuniform = 100 / (36 * math.sin(math.radians(5)))\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('jac')\njac = mo_getj(0.5, 0)\njeac, jsac = mo_getpointvalues(0.5, 0)[7:9]\nobs = ('DC in the wire: mo_getj %r MA/m^2 = Je %r + Js %r (100 A over 3.1376 mm^2 = %.5g); in air %r; '\n       'at 50 Hz: mo_getj %r = Je %r + Js %r; in the gap band: %s' % (j, je, js, uniform, jair, jac, jeac, jsac, band))\ncheck(j == je + js and je == 0 and near(j, uniform, 1e-3) and jair == 0, obs)\ncheck(near(jac, jeac + jsac, 1e-12) and abs(jeac) > 1 and abs(jac - uniform) < 0.01 * uniform, obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getmu', "fixture_pre()\nmi_addmaterial('Lam', 1000, 1000, 0, 0, 0, 0.5, 0, 0.9, 0, 0, 0, 1, 0)\nmi_drawrectangle(-2, 23, 2, 27)\nlabel(0, 25, 'Lam', None, 0, 4)\nsolve('mu')\nair = mo_getmu(5, 0)\niron = mo_getmu(25, 0)\nlam = mo_getmu(0, 25)\ntry:\n    mo_getmu(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nmo_close()\nmi_close()\n# 50 Hz, a linear material with a 20-degree hysteresis lag.\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nmi_addmaterial('Lag', 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 20, 20, 1, 0)\nmi_drawrectangle(23, -2, 27, 2)\nlabel(25, 0, 'Lag', None, 0, 2)\nsolve('muac')\nlag = mo_getmu(25, 0)\nexpected = 1000 * complex(math.cos(math.radians(20)), -math.sin(math.radians(20)))\nobs = ('air %r, iron (mu_r 1000) %r, laminated block (fill 0.9) %r; at 50 Hz with a 20-degree lag %r '\n       '(1000 e^-j20deg = %r); in the gap band: %s' % (air, iron, lam, lag, expected, band))\ncheck(air == [1, 1] and iron == [1000, 1000], obs)\ncheck(near(lam[0], 900.1, 1e-6) and near(lam[1], 900.1, 1e-6), obs)\ncheck(near(lag[0], expected, 1e-6) and near(lag[1], expected, 1e-6), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getpe', "fixture_post()\npe = mo_getpe(0.5, 0)\nmo_selectblock(0, 0)\nloss = mo_blockintegral(4)\nmo_clearblock()\ntry:\n    mo_getpe(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nmo_close()\nmi_close()\nfixture_geometry()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('peac')\npeac = mo_getpe(0.5, 0)\nj = mo_getj(0.5, 0) * 1e6\nsigma = mo_getconductivity(0.5, 0) * 1e6\nexpected = abs(j) ** 2 / (2 * sigma)\nobs = ('DC in the wire: Pe %r although mo_blockintegral(4) gives %.5g W; at 50 Hz: Pe %.6g W/m^3, '\n       '|J|^2 / (2 sigma) = %.6g; in the gap band: %s' % (pe, loss, peac, expected, band))\ncheck(pe == 0 and loss > 0.5, obs)\ncheck(isinstance(peac, float) and near(peac, expected, 1e-6), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_getph', "w = 2 * math.pi * 50\nfixture_geometry()\nmi_addmaterial('Lag', 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 20, 20, 1, 0)\nmi_drawrectangle(23, -2, 27, 2)\nlabel(25, 0, 'Lag', None, 0, 2)\nsolve('phdc')\ndc = mo_getph(25, 0)\ntry:\n    mo_getph(10.5, 0)\n    band = 'no error'\nexcept IndexError as error:\n    band = 'IndexError: ' + str(error)\nmo_close()\nmi_probdef(50, 'millimeters', 'planar', 1e-8, 10, 30, 0)\nsolve('phac')\nmo_smooth('off')\nph = mo_getph(25.3, 0.4)\nb1, b2 = mo_getb(25.3, 0.4)\nh1, h2 = mo_geth(25.3, 0.4)\nexpected = w / 2 * (h1 * b1.conjugate() + h2 * b2.conjugate()).imag\nrows = [r for r in mo_getallelements() if r[6] == 2]\ntotal = sum(mo_getph(r[3], r[4]) * r[5] * 1e-6 * 0.010 for r in rows)\nmo_smooth('on')\nmo_groupselectblock(2)\nblock = mo_blockintegral(3)\nmo_clearblock()\nobs = ('DC: Ph %r; 50 Hz, 20-degree lag, smoothing off: Ph %.6g W/m^3 at (25.3, 0.4), omega/2 Im(H.B*) = %.6g; '\n       'Ph x element volume over the %d elements of the block %.6g W, mo_blockintegral(3) %.6g W; in the gap band: %s'\n       % (dc, ph, expected, len(rows), total, block, band))\ncheck(dc == 0, obs)\ncheck(ph > 0 and near(ph, expected, 1e-6), obs)\ncheck(near(total, block, 1e-6), obs)\ncheck('index out of range' in band, obs)\nreturn obs")
run('mo_gridsnap', 'fixture_post()\nB0 = mo_getb(5, 0)\nsent = []\nreal = femm._default.callfemm\nfemm._default.callfemm = lambda s: (sent.append(s), real(s))[1]\ntry:\n    r = [mo_gridsnap(\'on\'), mo_gridsnap(\'OFF\')]\nfinally:\n    del femm._default.callfemm\ncheck(r == [None, None] and sent == [\'mo_gridsnap("on")\', \'mo_gridsnap("OFF")\'], show(r, sent))\ne_int = fails(mo_gridsnap, 1)\ne_word = fails(mo_gridsnap, \'yes\')\ncheck(e_int is not None and "can\'t convert \'int\'" in e_int, show(e_int))\ncheck(e_word is not None and \'Unknown option for grid snap\' in e_word, show(e_word))\nmo_setgrid(1, \'cart\')\nmo_gridsnap(\'on\')\nmo_clearcontour()\nmo_addcontour(2.3, 0.3)\nmo_addcontour(7.6, 0.3)\nlength = mo_lineintegral(2)[0]\ncheck(near(length, 0.0053), \'contour length \' + show(length))\ncheck(mo_getb(5, 0) == B0, \'B changed\')\nreturn \'sent \' + \' \'.join(sent) + \', no error; 1: \' + e_int + \'; "yes": \' + e_word + \'; snapping on, 1 mm grid: contour (2.3, 0.3)-(7.6, 0.3) measured \' + show(length) + \' m\'')
run('mo_hidecontourplot', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_hidecontourplot()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
run('mo_hidedensityplot', "fixture_post()\nB0 = mo_getb(5, 0)\nr = [mo_hidedensityplot()]\nB1 = mo_getb(5, 0)\ncheck(r == [None] * len(r) and B1 == B0, 'returned ' + show(r) + ', B ' + show(B0, B1))\nreturn 'returned None and B is unchanged'")
print("BATCH_DONE", flush=True)
closefemm()
