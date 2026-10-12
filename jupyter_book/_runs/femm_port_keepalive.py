"""Like femm_port.py, plus closefemm() keeps the FEMM session open.
Usage: python _runs/femm_port_keepalive.py <port> <module> [args...]
The C4/C5 environments open and close FEMM around every solve (fine for Windows pyFEMM).
femm-qt relaunches a server on the same port each time, which races and resets the
connection, so here the first session stays up and is reused. No project code is modified."""
import os, runpy, sys, femm
sys.path.insert(0, os.getcwd())
port = int(sys.argv[1])
femm.FemmSession.__init__.__defaults__ = (port, '127.0.0.1', False, True)

_callfemm, _noeval = femm.FemmSession.callfemm, femm.FemmSession.callfemm_noeval
_fix = lambda s: s.replace('"M-19 Steel"', '"M-19 14 mil"')
femm.FemmSession.callfemm = lambda self, s: _callfemm(self, _fix(s))
femm.FemmSession.callfemm_noeval = lambda self, s: _noeval(self, _fix(s))

_openfemm = femm.openfemm
femm.openfemm = lambda *arg: femm._default if femm._default is not None else _openfemm(*arg)
femm.closefemm = lambda *arg: None

sys.argv = [sys.argv[2]] + sys.argv[3:]
runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)
