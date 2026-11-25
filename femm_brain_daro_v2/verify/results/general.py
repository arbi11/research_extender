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
run('AWG', 'std = 0.127 * math.exp((36 - 18) / 39 * math.log(92))\ns = femm._default\nsent = []\nreal = s.callfemm\ns.callfemm = lambda c: (sent.append(c), real(c))[1]\ntry:\n    d18, d10 = AWG(18), AWG(10)\nfinally:\n    del s.callfemm\nserver = callfemm(\'AWG(18)\')\ncheck(near(d18, std, 1e-3) and near(d10, 2.5882, 1e-4), show(d18, std, d10))\ncheck(sent == [] and near(server, d18, 1e-12), show(sent, server))\nreturn f\'AWG(18) = {d18:.5g} mm (standard {std:.5g} mm), AWG(10) = {d10:.5g} mm; nothing sent to FEMM; callfemm("AWG(18)") = {server:.8g}\'')
run('callfemm', 'newdocument(0)\nr = callfemm(\'mi_addnode(3, 4)\')\ncallfemm(\'mi_addnode(7, 8)\')\nxy = mi_selectnode(3, 4)\nbox = callfemm(\'mi_getboundingbox()\')\nn = callfemm(\'len([1, 2, 3])\')\nt = callfemm(\'type(1.5)\')\ncallfemm(\'g3x = 5\')\nkept = callfemm(\'g3x * 2\')\nvalues = [callfemm(\'(1, 2.5, "ab")\'), callfemm(\'3+4j\'), callfemm(\'"Air"\'), callfemm(\'"1e3"\'), callfemm(\'1 == 1\'), callfemm(\'None\')]\nlua_form = fails(callfemm, \'mi_addnode\', 9, 9)\nunknown = fails(callfemm, \'mi_nosuchcommand(1)\')\ncheck(r == [] and xy == [3, 4] and box == [3, 7, 4, 8], show(r, xy, box))\ncheck(n == 3 and t == "<class \'float\'>" and kept == 10, show(n, t, kept))\ncheck(values == [[1, 2.5, \'ab\'], 3 + 4j, \'A1jr\', 1000.0, \'true\', []], show(*values))\ncheck(lua_form is not None and \'positional argument\' in lua_form, show(lua_form))\ncheck(unknown == "FEMM returns: name \'mi_nosuchcommand\' isn\'t defined", show(unknown))\nreturn \'statement -> \' + show(r) + \'; mi_getboundingbox() -> \' + show(box) + \'; len -> 3, type(1.5) -> \' + t + \', names persist (g3x * 2 -> 10); (1, 2.5, "ab"), 3+4j, "Air", "1e3", 1 == 1, None -> \' + show(*values) + \'; Lua form: \' + lua_form + \'; unknown: \' + unknown')
run('clearconsole', 'missing = fails(lambda: clearconsole())\nr = callfemm(\'clearconsole()\')\ncheck(missing == "name \'clearconsole\' is not defined" and not hasattr(femm, \'clearconsole\'), show(missing))\ncheck(r == [], show(r))\nreturn "pyFEMM: " + missing + "; callfemm(\'clearconsole()\') -> " + show(r)')
run('hideconsole', "s = femm._default\nsent = []\nreal = s.callfemm\ns.callfemm = lambda c: (sent.append(c), real(c))[1]\ntry:\n    r = hideconsole()\nfinally:\n    del s.callfemm\nraw = callfemm('hideconsole()')\ncheck(r is None and sent == ['hideconsole()'] and raw == [], show(r, sent, raw))\nreturn 'hideconsole() sent ' + show(*sent) + ', returned None; FEMM replied with nothing (' + show(raw) + ') and raised no error headless'")
run('messagebox', 'import time\ns = femm._default\nsent = []\nreal = s.callfemm\ns.callfemm = lambda c: (sent.append(c), real(c))[1]\ntry:\n    t0 = time.time()\n    r = messagebox(\'catalogue test message\')\n    dt = time.time() - t0\nfinally:\n    del s.callfemm\nnumber = fails(messagebox, 5)\nnone = fails(messagebox)\nquote = fails(messagebox, \'say "hi"\')\ncheck(r is None and sent == [\'messagebox("catalogue test message")\'] and dt < 1, show(r, sent, dt))\ncheck(number is not None and "can\'t convert \'int\' object to str" in number, show(number))\ncheck(none is not None and \'takes 1 positional arguments but 0 were given\' in none, show(none))\ncheck(quote is not None and \'invalid syntax\' in quote, show(quote))\nreturn f\'headless: returned None after {dt:.3f} s, no dialog; messagebox(5): {number}; messagebox(): {none}; a double quote: {quote}\'')
run('newdocument', 'r = newdocument(0)\nmi_addnode(1, 2)\nxy = mi_selectnode(1, 2)\nmi_close()\nnewdocument(1)\nei_addnode(1, 2)\nexy = ei_selectnode(1, 2)\nei_close()\nnewdocument(2)\nhi_addnode(1, 2)\nhxy = hi_selectnode(1, 2)\nhi_close()\nnewdocument(3)\nci_addnode(1, 2)\ncxy = ci_selectnode(1, 2)\nci_close()\nfour = fails(newdocument, 4)\nnoarg = fails(newdocument)\nc = callfemm(\'create(0)\')\nmi_addnode(3, 4)\ncr = mi_selectnode(3, 4)\ncheck(r is None and xy == [1, 2] and exy == [1, 2] and hxy == [1, 2] and cxy == [1, 2], show(xy, exy, hxy, cxy))\ncheck(four is None, \'newdocument(4): \' + show(four))\ncheck(noarg is not None and \'missing 1 required positional argument\' in noarg, show(noarg))\ncheck(c == [] and cr == [3, 4], show(c, cr))\nreturn \'returned None; nodes added and selected back in types 0 (mi_), 1 (ei_), 2 (hi_), 3 (ci_); newdocument(4) raised no error; newdocument(): \' + noarg + "; callfemm(\'create(0)\') opened a magnetics document"')
run('open', 'import builtins\nfixture_pre()\nmi_saveas(\'/tmp/open.fem\')\nmi_close()\nr = callfemm(\'open("/tmp/open.fem")\')\nbox = mi_getboundingbox()\nxy = mi_selectlabel(25, 0)\nmi_close()\ns = femm._default\nsent = []\nreal = s.callfemm\ns.callfemm = lambda c: (sent.append(c), real(c))[1]\ntry:\n    opendocument(\'/tmp/open.fem\')\nfinally:\n    del s.callfemm\ncheck(open is builtins.open and not hasattr(femm, \'open\'), \'open in the script is not Python file open\')\ncheck(r == [] and xy == [25, 0] and near(box[0], -50) and near(box[3], 50), show(r, xy, box))\ncheck(sent == [\'open("/tmp/open.fem")\'], show(*sent))\nreturn "callfemm(\'open(\\"/tmp/open.fem\\")\') returned " + show(r) + \'; bounding box \' + show(*box) + \'; iron label found at (25,0); in the script open is Python file open; opendocument sent \' + show(*sent)')
run('opendocument', 'fixture_pre()\nmi_saveas(\'/tmp/opendoc.fem\')\nmi_close()\nr = opendocument(\'/tmp/opendoc.fem\')\nxy = mi_selectlabel(25, 0)\nt = mi_gettitle()\ncheck(r is None and xy == [25, 0] and t == \'/tmp/opendoc.fem\', show(r, xy, t))\nmi_addnode(70, 0)\nopendocument(\'/tmp/opendoc.fem\')\ncopy = mi_selectnode(70, 0)\nmi_close()\nmi_setfocus(\'opendoc\')\nmi_close()\ncheck(near(copy[0], 50) and abs(copy[1]) < 1e-9, \'the second copy has the unsaved node: \' + show(copy))\nfixture_geometry()\nmi_setcurrent(\'Iw\', 50)\nmi_saveas(\'/tmp/opendoc2.fem\')\nmi_analyze(1)\nmi_close()\nans = fails(opendocument, \'/tmp/opendoc2.ans\')\nnoin = fails(mi_gettitle)\nnoout = fails(mo_gettitle)\ncheck(ans is None and noin is not None and noout is not None, show(ans, noin, noout))\nopendocument(\'/tmp/opendoc2.fem\')\nmi_loadsolution()\nby = mo_getb(5, 0)[1]\ncheck(near(by, 0.0019872, 1e-3), show(by))\nmi_close()\nhere = os.getcwd()\ncwd = callfemm(\'getcwd()\')\nos.chdir(\'/tmp\')\nmoved = callfemm(\'getcwd()\')\nos.chdir(here)\ncallfemm(\'setcurrentdirectory("/tmp")\')\ntry:\n    opendocument(\'opendoc.fem\')\n    rel = mi_gettitle()\n    lxy = mi_selectlabel(25, 0)\nfinally:\n    callfemm(\'setcurrentdirectory("\' + here + \'")\')\nmi_close()\ncheck(cwd == here and moved == here, \'FEMM directory \' + show(cwd, moved, here))\ncheck(rel == \'opendoc.fem\' and lxy == [25, 0], show(rel, lxy))\nmissing = fails(opendocument, \'/tmp/missing.fem\')\ncheck(missing is not None and \'Failed to open /tmp/missing.fem\' in missing, show(missing))\nreturn \'reopened /tmp/opendoc.fem with its iron label, title \' + t + \'; opening it again gave a copy without the unsaved node; opening the .ans: no error, then \' + noin + \' / \' + noout + f\'; its .fem with mi_loadsolution(): By {by:.5g} T; FEMM directory \' + cwd + \' (unchanged by os.chdir); after setcurrentdirectory("/tmp"), opendocument("opendoc.fem") worked, title \' + rel + \'; missing file: \' + missing')
run('print', 'import io, contextlib, builtins\nbuf = io.StringIO()\nwith contextlib.redirect_stdout(buf):\n    r = print(\'catalogue test: print() output line\')\nraw = callfemm(\'print("printed inside FEMM")\')\ncheck(print is builtins.print and r is None and buf.getvalue() == \'catalogue test: print() output line\\n\', show(r, buf.getvalue()))\ncheck(raw == [], show(raw))\nreturn "print is Python\'s own and wrote " + repr(buf.getvalue()) + " to the script\'s standard output; callfemm(\'print(...)\') returned " + show(raw) + \' and the text did not come back\'')
run('showconsole', "s = femm._default\nsent = []\nreal = s.callfemm\ns.callfemm = lambda c: (sent.append(c), real(c))[1]\ntry:\n    r = showconsole()\nfinally:\n    del s.callfemm\nraw = callfemm('showconsole()')\ncheck(r is None and sent == ['showconsole()'] and raw == [], show(r, sent, raw))\nreturn 'showconsole() sent ' + show(*sent) + ', returned None; FEMM replied with nothing (' + show(raw) + ') and raised no error headless'")
run('smartmesh', "fixture_pre()\nmi_saveas('/tmp/smart.fem')\ncounts = []\ntry:\n    for state in (0, 1, -1):\n        smartmesh(state)\n        mi_analyze(1)\n        mi_loadsolution()\n        counts.append(mo_numelements())\nfinally:\n    smartmesh(-1)\ncheck(counts == [8356, 15538, 15538], show(*counts))\nreturn 'elements with smartmesh 0, 1, -1: ' + show(*counts)")
run('IEC', 'r20 = [8.0, 7.1, 6.3, 5.6, 5.0, 4.5, 4.0, 3.55, 3.15, 2.8, 2.5, 2.24, 2.0, 1.8, 1.6, 1.4, 1.25, 1.12, 1.0, 0.9]\nworst = max(abs(IEC(n) / (r20[n % 20] / 10 ** (n // 20)) - 1) for n in range(40))\nserver = fails(callfemm, \'IEC(10)\')\nratios = [AWG(n) / IEC(n) for n in (0, 40)]\ncheck(near(IEC(10), 7.959159641581004 * math.exp(-1.1519673572274754)) and near(IEC(10), 2.5152, 1e-4), show(IEC(10)))\ncheck(worst < 0.013, f\'worst deviation from R20 {worst:.4f}\')\ncheck(server is not None and "name \'IEC\' isn\'t defined" in server, show(server))\ncheck(near(ratios[0], 1.0367, 1e-3) and near(ratios[1], 1.0062, 1e-3), show(*ratios))\nreturn f\'IEC(10) = {IEC(10):.5g} mm, IEC(18) = {IEC(18):.5g} mm, IEC(20) = {IEC(20):.4g} mm; within {100 * worst:.2f}% of the R20 sizes for n = 0..39; AWG/IEC = {ratios[0]:.4f} at n = 0, {ratios[1]:.4f} at n = 40; FEMM itself: {server}\'')
run('callfemm_noeval', 'newdocument(0)\nr1 = callfemm_noeval(\'mi_addnode(1, 2)\')\nr2 = callfemm_noeval(\'1 + 1\')\nr3 = callfemm_noeval(\'mi_nosuchcommand()\')\nafter = callfemm(\'2 + 2\')\nxy = mi_selectnode(1, 2)\ncheck(r1 is None and r2 is None and r3 is None, show(r1, r2, r3))\ncheck(after == 4 and xy == [1, 2], show(after, xy))\nreturn "returned None for a command, an expression and an unknown command (no error raised); callfemm(\'2 + 2\') then gave " + show(after) + \'; the node was added at \' + show(*xy)')
run('hidepointprops', 'blank()\nsaved = femm._default\nfemm._default = None\ntry:\n    r = hidepointprops()\nfinally:\n    femm._default = saved\nraw = fails(callfemm, \'hidepointprops()\')\ncheck(r is None, \'returned \' + show(r))\ncheck(raw is not None and "name \'hidepointprops\' isn\'t defined" in raw, show(raw))\nreturn \'returned None with no FEMM connected (it sends nothing); FEMM itself: \' + raw')
run('main_maximize', "fixture_pre()\nr = main_maximize()\nbox = mi_getboundingbox()\ncheck(r is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: main_maximize())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('main_minimize', "fixture_pre()\nr = main_minimize()\nbox = mi_getboundingbox()\ncheck(r is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: main_minimize())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('main_resize', "fixture_pre()\nr = main_resize(800, 600)\nbox = mi_getboundingbox()\ncheck(r is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: main_resize(800, 600))\ncheck(e is None, 'with no document open: ' + show(e))\ne2 = [fails(main_resize, 800), fails(main_resize)]\ncheck(e2 == [None, None], 'wrong counts: ' + show(e2))\nreturn 'returned None, drawing unchanged, no error with no document open, nor with one or no argument'")
run('main_restore', "fixture_pre()\nr = main_restore()\nbox = mi_getboundingbox()\ncheck(r is None and near(box[0], -50) and near(box[1], 50), 'returned ' + show(r) + ', box ' + show(*box))\nmi_close()\ne = fails(lambda: main_restore())\ncheck(e is None, 'with no document open: ' + show(e))\nreturn 'returned None, drawing unchanged, and no error with no document open'")
run('showpointprops', 'blank()\nsaved = femm._default\nfemm._default = None\ntry:\n    r = showpointprops()\nfinally:\n    femm._default = saved\nraw = fails(callfemm, \'showpointprops()\')\ncheck(r is None, \'returned \' + show(r))\ncheck(raw is not None and "name \'showpointprops\' isn\'t defined" in raw, show(raw))\nreturn \'returned None with no FEMM connected (it sends nothing); FEMM itself: \' + raw')
print("BATCH_DONE", flush=True)
closefemm()
