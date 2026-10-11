#!/usr/bin/env python
"""Stage the book sources for the PDF build (run from _book/, then ./build_pdf.sh).

Copies every chapter notebook into _book/<chapter>/ without touching the originals:
  * a book notebook with executed outputs is used as is;
  * one without outputs is replaced by its executed copy from ../_runs/executed/ when that copy has no error outputs;
  * build-only rewrites on markdown cells, because the MyST typst exporter drops them otherwise:
      {admonition} Title  ->  {note} Title            (7 blocks in C3)
      ::::{grid} ... :::{grid-item} ...  -> the inner figures, unwrapped  (C4/02)
  * grey placeholder images are written for figures the Notion export does not contain (_static/figures/...),
  * data_gen/ and designs/ are symlinked for the rendered PNGs the notebooks reference, references.bib is copied.
"""
import json, os, re, glob, shutil, subprocess, sys
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); B=os.path.dirname(os.path.abspath(__file__))
CH=['C1','C2','C3','C4','C5','AppA']
def n_err(nb): return sum(1 for c in nb['cells'] if c['cell_type']=='code' for o in c.get('outputs',[]) if o.get('output_type')=='error')
def n_out(nb): return sum(1 for c in nb['cells'] if c['cell_type']=='code' if c.get('outputs'))
def rewrite(src):
    out=[]; in_grid=False
    for line in src.split('\n'):
        if re.match(r'^:{4,}\{grid\}',line): in_grid=True; continue
        if in_grid:
            if re.match(r'^:{4,}\s*$',line): in_grid=False; continue
            if re.match(r'^:{3,}(\{grid-item\}.*)?\s*$',line) or re.match(r'^:gutter:',line): continue
        line=re.sub(r'^([:`]{3,})\{admonition\}',r'\1{note}',line)
        out.append(line)
    return '\n'.join(out)
report=[]
for ch in CH:
    os.makedirs(f'{B}/{ch}',exist_ok=True)
    for f in sorted(glob.glob(f'{ROOT}/{ch}/*.ipynb')):
        name=os.path.basename(f); nb=json.load(open(f)); src='book copy'
        if n_out(nb)==0:
            ex=f'{ROOT}/_runs/executed/{ch}__{name}'
            if os.path.exists(ex):
                nbx=json.load(open(ex))
                if n_err(nbx)==0 and n_out(nbx)>0: nb=nbx; src='executed copy'
        for c in nb['cells']:
            if c['cell_type']=='markdown':
                s=''.join(c['source']); s2=rewrite(s)
                if s2!=s: c['source']=s2
        json.dump(nb,open(f'{B}/{ch}/{name}','w'),indent=1)
        report.append(f'{ch}/{name:45s} {src:14s} outputs={n_out(nb):3d} errors={n_err(nb)}')
for ch,d in [('C3','data_gen'),('C4','designs'),('C5','designs')]:
    link=f'{B}/{ch}/{d}'
    if not os.path.islink(link): os.symlink(f'{ROOT}/{ch}/{d}',link)
shutil.copy(f'{ROOT}/C2/references.bib',f'{B}/references.bib')
# real thesis figures from <project>/_static (copied from research_extender/thesis_latex/Figures on 2026-10-11)
if os.path.isdir(f'{ROOT}/_static'): shutil.copytree(f'{ROOT}/_static', f'{B}/_static', dirs_exist_ok=True)
# placeholders for figures that are still missing
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
made=0
for nb in sorted(glob.glob(f'{B}/C[1-5]/*.ipynb')+glob.glob(f'{B}/AppA/*.ipynb')):
    d=json.load(open(nb))
    for cell in d['cells']:
        if cell['cell_type']!='markdown': continue
        for m in re.finditer(r'^[:`]{3,}\{figure\}[ \t]*(.+?)[ \t]*$',''.join(cell['source']),flags=re.M):
            p=os.path.normpath(os.path.join(os.path.dirname(nb),m.group(1)))
            if os.path.exists(p): continue
            os.makedirs(os.path.dirname(p),exist_ok=True)
            label=f"FIGURE MISSING FROM NOTION EXPORT\n{os.path.relpath(p,B)}\n(referenced by {os.path.relpath(nb,B)})"
            if p.lower().endswith('.pdf'):
                typ=p+'.typ'; open(typ,'w').write('#set page(width: 16cm, height: 6cm, fill: rgb("#eeeeee"))\n#set align(center+horizon)\n#text(size: 14pt, fill: rgb("#555555"))['+label.replace('\n','\\\n').replace('_','\\_')+']\n')
                subprocess.run([f'{B}/tools/bin/typst','compile',typ,p],check=True); os.remove(typ)
            else:
                fig,ax=plt.subplots(figsize=(8,3)); ax.set_facecolor('#eeeeee'); ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5,0.5,label,ha='center',va='center',fontsize=11,color='#555555',transform=ax.transAxes)
                fig.savefig(p,dpi=100,bbox_inches='tight'); plt.close(fig)
            made+=1
print('\n'.join(report)); print(f'{made} placeholder figures written')
