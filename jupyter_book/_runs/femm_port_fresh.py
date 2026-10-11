"""Run a module against femm-qt on its own port, one FEMM process per open/close.
Usage: python _runs/femm_port_fresh.py <port> <module> [args...]
Replaces femm_port_keepalive.py, whose shared session kept every model and solution in one
femm_headless process and grew it to 11+ GB. Here closefemm() really quits FEMM and waits
for the process to exit, so the next openfemm() can't race a server that is still shutting
down. No project code is modified."""
import os, runpy, subprocess, sys, time, femm
sys.path.insert(0, os.getcwd())
port = int(sys.argv[1])
femm.FemmSession.__init__.__defaults__ = (port, '127.0.0.1', False, True)

_callfemm, _noeval = femm.FemmSession.callfemm, femm.FemmSession.callfemm_noeval
_fix = lambda s: s.replace('"M-19 Steel"', '"M-19 14 mil"')
femm.FemmSession.callfemm = lambda self, s: _callfemm(self, _fix(s))
femm.FemmSession.callfemm_noeval = lambda self, s: _noeval(self, _fix(s))

_openfemm, _closefemm = femm.openfemm, femm.closefemm

def closefemm():
    proc = getattr(femm._default, '_proc', None) if femm._default else None
    _closefemm()
    if proc is not None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()

def openfemm(*arg):
    if femm._default is not None:      # scripts that open twice without closing
        return femm._default
    for attempt in range(40):
        try:
            return _openfemm(*arg)
        except Exception:
            femm._default = None
            if attempt == 39:
                raise
            time.sleep(0.5)

femm.openfemm, femm.closefemm = openfemm, closefemm

sys.argv = [sys.argv[2]] + sys.argv[3:]
runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)
