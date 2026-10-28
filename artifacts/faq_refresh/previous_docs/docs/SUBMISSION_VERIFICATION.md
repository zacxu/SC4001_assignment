# Step 15: verification and submission packaging

Completed the local verification and packaging work on October 3, 2026. Student-name replacement and NTULearn upload/receipt remain pending; the full step's portal completion evidence is therefore not yet satisfied.

On **October 4**, [the full requirements recheck](FINAL_REQUIREMENTS_RECHECK.md) reread all 13 brief/slide pages and all 10 report pages. Corrected the housing standardization sentence and explicitly flagged the final-FC activation interpretation, then rebuilt/re-audited the four files. All 96 root tests passed again; the additional saved-evidence audit checked62 histories and all questions, preserving938 experiment files.

## Deliverables

The [submission folder](../submission/README.md) contains the two five-page PDFs and separate Part A/Part B code ZIPs. [Its manifest](../submission/manifest.json) identifies the four upload files, hashes, sizes and submission status; [checksums](../submission/SHA256SUMS.txt) support a final upload comparison. ZIPs contain approximately **19 MiB / 75 MiB**, including each required raw dataset and prepared data. The final sizes/hashes in the manifest take precedence over these rounded values.

Each ZIP has its own root README covering hardware/Python/library versions, venv setup, seeds, data locations/source hashes, exact preparation/training/export commands, output directories, controls and limitations. Layout preserves `assignment_common/` plus the relevant `part_a/` or `part_b/` namespace. Each includes that part's tests, checkpoints, numeric evidence, figures and an embedded SHA-256 file manifest. Part A retains supplied baseline source for exact parity tests. Part B retains the small initial B4 consistency-failure record. No venv, caches, setup probes, console logs, opposite-part code or opaque Ray tuner/event state are bundled.

## Extracted-folder checks

Both packages passed ZIP CRC/integrity, safe-member path, file-manifest, Python syntax, dependency pin and pip consistency checks. Each was extracted to a new directory and launched with isolated Python search paths using the existing project venv; dependencies were **not freshly installed**. Package code/data/imports reside in those extractions and do not require the other part or personal workspace paths.

| Evidence | Part A | Part B |
| --- | --- | --- |
| Behavioral tests | 57 passed | 62 passed |
| Preprocessing reproduced from bundled raw source | Both protocols; all four tensor/statistic artifacts exactly equal | Both tensor/statistic artifacts exactly equal |
| Checkpoint loading/synthetic CUDA forward | A1 and A3 seed-42 final models | Three selected CNNs |
| All saved asset exporters | A1–A4 passed; tables retained their values | B1–B4 passed; tables retained their values |
| Ray worker | Local Part A import, CUDA and synthetic model forward passed | Not required |
| Other part needed | No | No |

The common 23 tests are present in both packages; total distinct coverage is **96 behavioral tests**. The verification reuses prior experimental results, runs synthetic/subset behavior checks, and performs **zero new full experiment training runs or test-set model inference**. It does not claim that the full 62-run workload was repeated from the extracted folders. Production entry points and a sequential `run_all.py` are provided for full reproduction into a new destination.

Found and corrected two actual portability issues: A1's exporter read a historical absolute prepared-data path, and B3's exporter read original absolute curve directories. A1 now accepts the current prepared directory (CLI and training entry points pass it); B3 receives local source directories and checks their selected-checkpoint hashes. Historic JSON/PT lineage records remain unchanged to preserve original hashes and are used as provenance, not dependencies.

## Report and artifact checks

Opened both PDFs: five pages each, declaration first, nonempty/searchable content, author metadata consistent with the report manifest. The report audit checks all requested answers, clarified plotting rules, saved full-precision metrics/tables, references, text bounds/glyphs/fonts and page count. All **938 earlier experiment output hashes**, **12 supplied input hashes**, grid/lambda rows, seed statistics, feature rankings and Grad-CAM titles/IDs remain intact. No public test-metric epoch plots were introduced; the A2 ablations remain separate, and final RMSE/R² remain in tables/barplots.

The two reports now also declare submission-packaging assistance. Their regenerated first pages were rendered and visually reviewed after that small declaration update; the other pages retain the prior full visual audit and all pages pass the repeated geometry/content checks.

## Rebuild and remaining actions

```powershell
# Rebuild using the current report manifest's author, then perform fresh checks.
.\.venv\Scripts\python.exe -B scripts/build_submission.py
.\.venv\Scripts\python.exe -B scripts/verify_submission.py

# Replace the placeholder everywhere that determines submitted identity/files.
.\.venv\Scripts\python.exe -B scripts/finalize_submission.py --name "Your Full Name"
```

Final author replacement requires the student's actual name, which has not been supplied. There is no authenticated NTULearn capability available here, so upload/final submit/receipt remain outstanding. The written brief states **October 12, 2026** without a cutoff time; confirm the portal time. Upload exactly the manifest's four files, verify the portal lists the correct names/sizes, complete its final submission action, then retain the receipt. The manifest explicitly records `ntulearn_submitted: false` and `submission_receipt: null` until actual evidence exists.

Evidence: [latest submission audit](../artifacts/submission_checks/latest.json), [report audit](../reports/verification.json), and [package builder](../scripts/build_submission.py). The latest audit points to exact extracted-folder logs and verification JSON. Packaging is reproducible from explicit allowlists; generated ZIP entries have fixed timestamps.
