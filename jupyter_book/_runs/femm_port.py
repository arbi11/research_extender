"""Run a module against femm-qt with runtime compat shims (no project code is modified).
Usage: python _runs/femm_port.py <port> <module> [args...]
  - FEMM on <port>, so it doesn't share another job's instance (C2 uses 48015).
  - "M-19 Steel" (stock FEMM 4.2 name) -> "M-19 14 mil" (femm-qt library), per user choice 2026-10-09.
  - openfemm() reuses the open session: femm-qt allows one client per instance, while
    Windows pyFEMM let scripts call openfemm() once per design."""
import os, runpy, sys, femm
sys.path.insert(0, os.getcwd())
port = int(sys.argv[1])
femm.FemmSession.__init__.__defaults__ = (port, '127.0.0.1', False, True)

_callfemm, _noeval = femm.FemmSession.callfemm, femm.FemmSession.callfemm_noeval
_fix = lambda s: s.replace('"M-19 Steel"', '"M-19 14 mil"')
femm.FemmSession.callfemm = lambda self, s: _callfemm(self, _fix(s))
femm.FemmSession.callfemm_noeval = lambda self, s: _noeval(self, _fix(s))

_openfemm = femm.openfemm
def openfemm(*arg):
    return femm._default if femm._default is not None else _openfemm(*arg)
femm.openfemm = openfemm

sys.argv = [sys.argv[2]] + sys.argv[3:]
runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)
