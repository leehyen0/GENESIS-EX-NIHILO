# Evaluation Round 001 — Anonymized Candidates

Open this file **only after** committing the SHA-256 hash of your independent baseline answer as described in `TASKS.md`.

Candidates are randomized independently for each case. Letters do not correspond to the same underlying policy across cases. Internal policy names are intentionally withheld.

## python/cpython#154874

### Candidate A

- **Diagnostic locus:** type/model semantics boundary
- **Question / probe:** Is compiler/type lowering changing the numeric result?
- **Verification:** Try type-level/compiler variations.
- **Recommended action:** Change type/model implementation.

### Candidate B

- **Diagnostic locus:** multiple competing loci
- **Question / probe:** Check representation, lifecycle, observation, external environment and type-model causes in parallel.
- **Verification:** Run all available representation, timing, environment and type checks.
- **Recommended action:** Consider signedness conversion, API split, environment guard, extra observations, and broader refactor.

### Candidate C

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Which surface representation changed the meaning of the return value?
- **Verification:** Compare old/new outputs and wrong signedness.
- **Recommended action:** Repair the semantic conversion at the return boundary.

### Candidate D

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Is termattrs semantically a status-returning int like baudrate, or an unsigned capability bitmask whose high bit is valid data?
- **Verification:** Cross-check termattrs against term_attrs/slk_attr/window.getattrs, include a bit-31 terminal, and compare pre-regression behavior.
- **Recommended action:** Separate the semantic paths: preserve ERR/status handling where it is contractual, but convert termattrs as an unsigned attribute mask.

## pandas-dev/pandas#66639

### Candidate A

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Which method/engine is semantically applicable to sep=None?
- **Verification:** Compare engine outcomes under sep=None.
- **Recommended action:** Route sep=None to the semantically applicable parser.

### Candidate B

- **Diagnostic locus:** multiple competing loci
- **Question / probe:** Check engine semantics, parallelism, lifecycle, external parser dependency and measurement issues.
- **Verification:** Run all parser/engine/lifecycle/environment checks.
- **Recommended action:** Consider fallback, parser changes, docs changes, engine guards, and broader parser refactor.

### Candidate C

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Is sep=None a literal delimiter value, or a semantic request for delimiter sniffing that must route to an engine supporting that contract?
- **Verification:** Matched matrix: default engine, explicit c/python/pyarrow, old release vs current main, plus documented expected behavior.
- **Recommended action:** Restore semantic routing for sep=None to the supported engine; fail closed with a clear unsupported-option error for explicit incompatible engines.

### Candidate D

- **Diagnostic locus:** observation/test-oracle boundary
- **Question / probe:** Is the failing TypeError merely an inadequate error oracle?
- **Verification:** Change error assertions.
- **Recommended action:** Improve the error message/test without restoring routing.

## curl/curl#22272

### Candidate A

- **Diagnostic locus:** temporal/lifecycle ownership boundary
- **Question / probe:** At which API transition is wakeup ownership consumed, and does the documented multi API require the wakeup state to survive perform until the subsequent wait/poll boundary?
- **Verification:** Ordered eventfd/socket trace across wakeup -> perform -> poll, with before/after version control and a finite-time assertion that poll observes the pending wakeup.
- **Recommended action:** Repair the smallest wakeup-state ownership/consumption boundary so perform does not erase a wakeup that the following wait is contractually expected to observe.

### Candidate B

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Did the semantic applicability of wakeup vs poll change?
- **Verification:** Compare old/new call results.
- **Recommended action:** Adjust wakeup handling to restore compatible behavior.

### Candidate C

- **Diagnostic locus:** multiple competing loci
- **Question / probe:** Check representation, event lifecycle, socket backend, resolver environment and polling observations simultaneously.
- **Verification:** Run every timing/backend/semantic/environment check.
- **Recommended action:** Consider wakeup buffering, poll changes, resolver changes, timeout workarounds, and API changes.

### Candidate D

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Is this only an API semantic mismatch independent of event ordering?
- **Verification:** Compare return values without ordered event-state trace.
- **Recommended action:** Change API behavior generically.

## scipy/scipy#23177

### Candidate A

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Which backend semantics changed for this input?
- **Verification:** Compare JAX outputs.
- **Recommended action:** Adapt the local computation to the changed semantic behavior.

### Candidate B

- **Diagnostic locus:** multiple competing loci
- **Question / probe:** Check SciPy logic, JAX backend semantics, CI environment, dtype representation, test oracle and timing.
- **Verification:** Run all backend/environment/representation tests.
- **Recommended action:** Consider SciPy patch, JAX pin, backend special case, oracle change, and CI change.

### Candidate C

- **Diagnostic locus:** API/semantic contract boundary
- **Question / probe:** Treat it as a local array-semantic mismatch.
- **Verification:** Test only current JAX output.
- **Recommended action:** Patch SciPy around the observed value.

### Candidate D

- **Diagnostic locus:** dependency/environment ownership boundary
- **Question / probe:** Is the changed result caused by SciPy product logic or by a version-specific JAX backend semantic change exposed only on the dependency/environment path?
- **Verification:** Reproduce the exact SciPy test across JAX 0.6.0, 0.6.1 and 0.6.2 under matched code, then identify whether NumPy/other array backends retain the expected nan contract.
- **Recommended action:** Route the fix to the dependency/backend compatibility locus first; only change SciPy if the cross-backend contract shows SciPy is relying on unsupported behavior.
