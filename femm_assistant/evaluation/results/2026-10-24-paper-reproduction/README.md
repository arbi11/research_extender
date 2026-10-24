# Paper reproduction evidence

Source: Lubin, Berger and Rezzoug, PIER B 41 (2012), 377–396,
[doi:10.2528/PIERB12051105](https://www.jpier.org/PIERB/pier.php?paper=12051105).
The publisher PDF and original native artifacts are retained locally; their
digests are recorded here. Copyrighted PDFs and full page text are not included.

`native-reproduction.json` records three independently transcribed gold cases,
each with base, refined and larger-domain native executions. All published
targets pass 2%, and mesh/domain changes pass 0.5%. Final far-domain quantities
are 87.0082 mH (87 mH reference), 76.9670 N (77.9 N reference) and 48.9767 mH
(49 mH reference). These gold inputs do not establish AI extraction accuracy.

The original coarse-profile failures and strict auxiliary energy diagnostic
failures are preserved. The documented v3 re-audit verifies native artifact
hashes and rereads CSVs, using an explicit 1e-4 auxiliary energy bound justified
by the pinned FEMM axisymmetric quadratures. It makes no new native calls and
does not change the published values, 2% target tolerances or 0.5% convergence
criterion.

`paper-extraction-attempts.json` retains the actual live progression:

1. Extraction stopped at source/schema verification; original diagnostic detail
   was not retained and usage remains unknown.
2. Unit spellings, verbatim quotation and reference classification required fixes.
3. A numeric boundary claim invented a unit; it was rejected.
4. The revised importer extracted the real source correctly. Planning emitted
   10 A while its explanation derived 20 A and asked for clarification. No native
   execution was allowed. Linear inductance alone would not detect that error.
5. A separately recorded continuation reused the verified extraction. The normal
   bounded planner, with the new source arithmetic contract, produced 20 A and
   passed all 19 independent physical checks. Actual native mesh and solve gave
   86.830578 mH against the original frozen 87 mH target: 0.194738% error.

The continuation made no new extraction call. It is not a wholly fresh
extraction-to-native acceptance trial. This is a public development case used to
improve the implementation, not an independent holdout or evidence that arbitrary
papers can be reproduced. Geometry, physical inputs and original operating
conditions still need independent review; mesh, solver and target agreement are
necessary checks, not sufficient proof of reproduction.

Known reported extraction costs total $0.016790425. Initial extraction and
planning usage remain unknown, not zero. `manifest.json` records report hashes.
