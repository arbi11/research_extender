#!/bin/bash
# Build _build/exports/compem_book.pdf with Jupyter Book 2.1.7 (MyST) + Typst 0.14.1 (tools/bin), no LaTeX.
#   ./build_pdf.sh                # full: MyST typst export, fix-ups, compile
#   ./build_pdf.sh --compile-only # skip the MyST export, re-run fix-ups + compile on _build/typst
# Fix-ups needed for the bundled MyST v1.11 with typst 0.14:
#   - LaTeX macros MyST passes through verbatim get a typst definition (table below; e.g. \xrightarrow, \infty)
#   - markdown links to missing anchors become #link("") which typst rejects -> #link("#")
#   - `diff` (\partial) is only deprecated in typst 0.14 (removed in 0.15), warnings are filtered out
set -uo pipefail
cd "$(dirname "$0")"
T=_build/typst; IMP=$T/myst-imports.typ; OUT=_build/exports/compem_book.pdf
if [ "${1:-}" != "--compile-only" ]; then
  ./build.sh --typst || { echo "MyST typst export failed"; exit 1; }
fi
[ -f $T/compem_book.typ ] || { echo "no $T/compem_book.typ"; exit 1; }
mkdir -p _build/exports
grep -q 'compem shims' $IMP || printf '\n// --- compem shims (added by build_pdf.sh) ---\n#let xrightarrow(top) = math.attach(math.limits(math.arrow.r.long), t: top)\n' >> $IMP
sed -i 's/#link("")/#link("#")/g' $T/*.typ
# citations whose key is not in the bib: typst errors on them, so render the key in red instead
../.venv/bin/python - "$T" <<'PY'
import re, glob, sys
T=sys.argv[1]
bib=set(re.findall(r'^@\w+\{([^,\s]+),', open(f'{T}/main.bib',encoding='utf-8',errors='replace').read(), flags=re.M))
missing=set(); n=0
for f in glob.glob(f'{T}/compem_book-*.typ'):
    s=open(f).read()
    def sub(m):
        global n
        k=m.group(1)
        if k in bib: return m.group(0)
        missing.add(k); n+=1
        return f'#text(fill: red)[\\[{k}\\]]'
    s2=re.sub(r'#cite\(<([^>]+)>[^)]*\)', sub, s)
    if s2!=s: open(f,'w').write(s2)
print(f"missing-bib citations replaced: {n} occurrences, {len(missing)} keys: {sorted(missing)}")
# cross-references to labels that no longer exist (e.g. figures inside an unsupported {grid}): keep the text only
files=glob.glob(f'{T}/*.typ'); text={f:open(f).read() for f in files}
from collections import Counter
defined=Counter()
for s in text.values():
    defined.update(re.findall(r'(?<![(,\s])\s*<([\w:.-]+)>', s))
    defined.update(re.findall(r'^<([\w:.-]+)>$', s, flags=re.M))
dangling=set(); m=0
for f,s in text.items():
    def sub2(mm):
        global m
        if defined[mm.group(1)]==1: return mm.group(0)   # unique label: keep the link; missing or duplicated: text only
        dangling.add(mm.group(1)); m+=1; return mm.group(2)
    s2=re.sub(r'#link\(<([\w:.-]+)>\)\[([^\]]*)\]', sub2, s)
    if s2!=s: open(f,'w').write(s2)
print(f"dangling cross-references flattened: {m} occurrences, labels: {sorted(dangling)}")
PY
declare -A MAP=( [infty]='sym.infinity' [ldots]='math.dots' [cdots]='math.dots.h.c' [vdots]='math.dots.v' [ddots]='math.dots.down'
  [mathbb]='math.bb' [mathcal]='math.cal' [mathrm]='math.upright' [mathbf]='math.bold' [boldsymbol]='math.bold' [operatorname]='math.op'
  [textbf]='math.bold' [varepsilon]='sym.epsilon.alt' [varphi]='sym.phi.alt' [implies]='sym.arrow.r.double.long' [iff]='sym.arrow.l.r.double.long'
  [xleftarrow]='(top) => math.attach(math.limits(math.arrow.l.long), t: top)' [leq]='sym.lt.eq' [geq]='sym.gt.eq' [neq]='sym.eq.not'
  [langle]='sym.angle.l' [rangle]='sym.angle.r' [cdot]='sym.dot.op' [partial]='sym.diff' [rightarrow]='sym.arrow.r' [leftarrow]='sym.arrow.l'
  [Rightarrow]='sym.arrow.r.double' [Leftrightarrow]='sym.arrow.l.r.double' [mapsto]='sym.arrow.r.bar' [approx]='sym.approx' [sim]='sym.tilde.op'
  [propto]='sym.prop' [forall]='sym.forall' [exists]='sym.exists' [nabla]='sym.nabla' [hbar]='sym.planck.reduce' [ell]='sym.ell' [emptyset]='sym.nothing' [boxed]='""' [argmax]='math.op("arg max", limits: true)' [argmin]='math.op("arg min", limits: true)' [diag]='math.op("diag")' [sign]='math.op("sign")' [sgn]='math.op("sgn")' [tr]='math.op("tr")' [softmax]='math.op("softmax")' [Var]='math.op("Var")' [Cov]='math.op("Cov")' [KL]='math.op("KL")' [mathit]='math.italic' [textit]='math.italic' [textrm]='math.upright' [mathsf]='math.sans' [mathtt]='math.mono' [dagger]='sym.dagger' [star]='sym.star' [circ]='sym.circle.small' [bullet]='sym.bullet' [prime]='sym.prime' [top]='sym.top' [bot]='sym.bot' [parallel]='sym.parallel' [perp]='sym.perp' [subseteq]='sym.subset.eq' [supseteq]='sym.supset.eq' [setminus]='sym.without' [mid]='sym.divides' [vert]='sym.bar.v' [lVert]='sym.bar.v.double' [rVert]='sym.bar.v.double' [lvert]='sym.bar.v' [rvert]='sym.bar.v' [ll]='sym.lt.double' [gg]='sym.gt.double' [equiv]='sym.ident' [simeq]='sym.tilde.eq' [cong]='sym.tilde.equiv' [textbackslash]='sym.backslash' )
for i in $(seq 1 25); do
  ERR=$(tools/bin/typst compile --root $T $T/compem_book.typ $OUT 2>&1 | grep -v 'deprecated' | grep -E -A4 '^error' | head -12)
  [ -z "$ERR" ] && { echo "compiled OK (attempt $i)"; break; }
  V=$(echo "$ERR" | grep -oE 'unknown variable: [A-Za-z]+' | head -1 | awk '{print $3}')
  if [ -n "$V" ] && [ -n "${MAP[$V]:-}" ]; then echo "shim: $V -> ${MAP[$V]}"; printf '#let %s = %s\n' "$V" "${MAP[$V]}" >> $IMP
  else echo "UNHANDLED typst error:"; echo "$ERR"; exit 1; fi
done
ls -la $OUT && pdfinfo $OUT 2>/dev/null | grep -E 'Pages|File size'
