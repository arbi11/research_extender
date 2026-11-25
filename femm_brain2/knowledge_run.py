"""Run graph-generated Lua in isolated real FEMM; preserve and re-audit evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from .catalogue import load_batch
from .knowledge import digest, validate_knowledge
from .knowledge_check import evaluate_control, evaluate_summary
from .knowledge_compile import generate
from .source import DEFAULT_CACHE, fingerprint, load_sources, source_file, verified_text

ROOT=Path(__file__).resolve().parents[1]
CODE=('knowledge.py','recipes.py','knowledge_compile.py','knowledge_check.py','knowledge_run.py','paper_source.py')


def code_fingerprints():
    return {name:fingerprint((Path(__file__).parent/name).read_bytes()) for name in CODE}


def _runtime(image):
    result=subprocess.run(['docker','run','--rm','--network','none','--read-only','--entrypoint','cat',image,
                           '/opt/femm/mesh-count-manifest.json'],capture_output=True,check=True,timeout=30)
    return result.stdout


def _artifacts(folder):
    return {p.relative_to(folder).as_posix():fingerprint(p.read_bytes()) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name not in ('binding.lua','report.json')}


def run(output,knowledge,paper_id,cache=DEFAULT_CACHE):
    validate_knowledge(knowledge,source_check=True)
    paper=next((p for p in knowledge['papers'] if p['id']==paper_id),None)
    if paper is None:raise ValueError('Unknown paper: '+paper_id)
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise ValueError('Use an empty run directory; earlier evidence is preserved')
    profile=paper['engine'];image=profile['image_id']
    actual=subprocess.run(['docker','image','inspect','--format','{{.Id}}',image],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    if actual!=image:raise ValueError('Unexpected FEMM image')
    source=next(s for s in load_sources() if s['id']==profile['binding_source_id'])
    verified_text(source,cache)
    runtime=_runtime(image)
    if fingerprint(runtime)!=profile['runtime_manifest_sha256']:raise ValueError('Runtime manifest differs from frozen image profile')
    (folder/'runtime-build.json').write_bytes(runtime)
    (folder/'knowledge.json').write_text(json.dumps(knowledge,indent=2)+'\n')
    started=datetime.now(timezone.utc).isoformat();engines=[];checks=[]
    for control in paper['controls']:
        directory=folder/control['id'];directory.mkdir()
        script,trace,_=generate(knowledge,paper_id,control)
        (directory/'native.lua').write_text(script)
        (directory/'api-trace.json').write_text(json.dumps(trace,indent=2)+'\n')
        shutil.copyfile(source_file(source,cache),directory/'binding.lua')
        args=['docker','run','--rm','--init','--name','femm-brain2-paper-'+uuid.uuid4().hex[:12],
              '--network','none','--read-only','--memory',profile['memory'],'--cpus','2','--pids-limit','256',
              '--tmpfs','/tmp:rw,size=512m','--user',f'{os.getuid()}:{os.getgid()}',
              '-e','HOME=/tmp','-e','QT_QPA_PLATFORM=offscreen','-e','FEMM_DIR=/opt/femm/bin',
              '-e','FEMM_MESH_EXPORT=1','-v',str(directory)+':/work','-w','/work',image,
              '/opt/femm/bin/lua4','-e',"dofile('/work/binding.lua'); _femm_port=48015; openfemm(1)",'-f','/work/native.lua']
        t=time.monotonic()
        try:
            result=subprocess.run(args,capture_output=True,text=True,timeout=profile['timeout_seconds'])
            returncode=result.returncode;log=result.stdout+result.stderr
        except subprocess.TimeoutExpired as exc:
            returncode=-1
            log='\n'.join(v.decode(errors='replace') if isinstance(v,bytes) else (v or '') for v in [exc.stdout,exc.stderr])
            log+='\nFrozen native time limit exceeded.\n'
            subprocess.run(['docker','rm','-f',args[6]],capture_output=True,timeout=15)
        (directory/'native.log').write_text(log)
        engine={'id':control['id'],'kind':'native_femm_docker','image_id':image,'network':'none','read_only':True,
                'returncode':returncode,'elapsed_seconds':time.monotonic()-t,'argv':args}
        engines.append(engine)
        check=evaluate_control(directory,knowledge,paper,control,engine);checks.append(check)
        print(json.dumps({'control':control['id'],'status':check['status'],'elapsed_seconds':round(engine['elapsed_seconds'],2)}),flush=True)
        if control['id']=='pilot' and check['status']!='passed':break
    assessment=evaluate_summary(paper,checks)
    report={'schema_version':1,'paper_id':paper_id,'scope':'One learning paper, Example 1 geometry and no-load EMF only; not a held-out evaluation or complete paper replication.',
            'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),'status':assessment['status'],
            'knowledge_sha256':digest(knowledge),'api_batch_sha256':digest(load_batch()),'code_sha256':code_fingerprints(),
            'binding_sha256':source['sha256'],'engines':engines,'checks':checks,'assessment':assessment,'artifacts':_artifacts(folder)}
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    audit(folder/'report.json',knowledge)
    return {'status':report['status'],'report':str(folder/'report.json'),'native_solves':sum(paper['controls'][i]['samples'] for i in range(len(checks))),
            'assessment':assessment}


def audit(path,knowledge):
    path=Path(path).resolve();folder=path.parent;report=json.loads(path.read_text())
    if report['knowledge_sha256']!=digest(knowledge) or report['api_batch_sha256']!=digest(load_batch()):
        raise ValueError('Paper evidence belongs to another knowledge/API snapshot')
    if report['code_sha256']!=code_fingerprints():raise ValueError('Construction/checker code differs from the recorded run')
    if json.loads((folder/'knowledge.json').read_text())!=knowledge:raise ValueError('Saved knowledge snapshot differs')
    paper=next(p for p in knowledge['papers'] if p['id']==report['paper_id']); profile=paper['engine']
    if fingerprint((folder/'runtime-build.json').read_bytes())!=profile['runtime_manifest_sha256']:
        raise ValueError('Runtime manifest differs from frozen profile')
    binding=next(s for s in load_sources() if s['id']==profile['binding_source_id'])
    if report['binding_sha256']!=binding['sha256']:raise ValueError('Unexpected Lua binding')
    for name,sha in report['artifacts'].items():
        file=folder/name
        if file.is_symlink() or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe evidence artifact path')
        if fingerprint(file.read_bytes())!=sha:raise ValueError('Evidence artifact fingerprint mismatch: '+name)
    if _artifacts(folder)!=report['artifacts']:raise ValueError('Missing or unhashed native evidence artifacts')
    if len(report['engines'])>len(paper['controls']):raise ValueError('Unexpected native control')
    checks=[]
    for control,engine in zip(paper['controls'],report['engines']):
        if engine['id']!=control['id'] or engine['kind']!='native_femm_docker' or engine['image_id']!=profile['image_id'] or engine['network']!='none' or engine['read_only'] is not True:
            raise ValueError('Native isolation profile differs')
        argv=engine['argv']
        if '--read-only' not in argv or argv[argv.index('--network')+1]!='none' or profile['image_id'] not in argv:
            raise ValueError('Native invocation lacks the frozen isolation settings')
        directory=folder/control['id'];script,trace,_=generate(knowledge,paper['id'],control)
        if (directory/'native.lua').read_text()!=script or json.loads((directory/'api-trace.json').read_text())!=trace:
            raise ValueError('Saved Lua/API trace differs from graph-generated plan')
        checks.append(evaluate_control(directory,knowledge,paper,control,engine))
    assessment=evaluate_summary(paper,checks)
    if checks!=report['checks'] or assessment!=report['assessment'] or assessment['status']!=report['status']:
        raise ValueError('Recorded checks do not agree with the native artifacts')
    return {'status':report['status'],'paper_id':paper['id'],'scope':report['scope'],'assessment':assessment,
            'procedures':[n['id'] for n in knowledge['nodes'] if n['level']==5],
            'report':str(path),'report_sha256':fingerprint(path.read_bytes())}
