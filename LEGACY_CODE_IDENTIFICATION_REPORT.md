# Legacy Code Identification Report

**Date:** 2025-12-19
**Context:** Post-SOLID Refactoring Cleanup
**Goal:** Identify redundant files that can be safely removed without impacting benchmarking or critical functionality

---

## Executive Summary

After completing the SOLID refactoring of DPR-RC's three monolithic files into Clean Architecture, a comprehensive scan identified legacy code that can be safely removed or requires updating.

**Key Findings:**
- ✅ **3 backup files** can be safely deleted (after cloud benchmark validation)
- ⚠️ **4 files** require updating to work with refactored code (including 1 newly discovered)
- ✅ **200+ critical files** must be preserved (all functional code, benchmarks, deployment)

**Total Removable:** 3 files (~3,136 lines, ~50 KB)
**Total Requiring Updates:** 4 files (legacy tests + baseline executor)

---

## Category 1: SAFE TO REMOVE - Backup Files

These files are complete duplicates of pre-refactoring code and can be deleted immediately after cloud benchmark validation.

### Files to Remove (3 files)

| File | Lines | Type | Replaced By |
|------|-------|------|-------------|
| `dpr_rc/passive_agent.py.backup` | 1,232 | Backup | `passive_agent.py` (315 lines) + domain/application/infrastructure |
| `dpr_rc/active_agent.py.backup` | 957 | Backup | `active_agent.py` (162 lines) + domain/application/infrastructure |
| `dpr_rc/slm_service.py.backup` | 947 | Backup | `slm_service.py` (285 lines) + domain/infrastructure |

**Details:**

1. **`dpr_rc/passive_agent.py.backup`**
   - Original monolithic passive worker implementation
   - Contains god class `PassiveWorker` with 42+ methods
   - Mixed FastAPI, business logic, GCS, ChromaDB, Redis
   - Completely replaced by refactored architecture

2. **`dpr_rc/active_agent.py.backup`**
   - Original monolithic active controller
   - Contains 320-line god function `handle_query()`
   - Mixes routing, consensus, worker communication, synthesis
   - Completely replaced by refactored architecture

3. **`dpr_rc/slm_service.py.backup`**
   - Original SLM service with mixed concerns
   - Inline prompt engineering, model loading, parsing, HTTP
   - Completely replaced by refactored architecture

**Recommendation:** Delete after cloud benchmark passes
**Risk Level:** NONE - Git history preserves files if rollback needed
**Disk Space Saved:** ~50 KB

---

## Category 2: NEEDS UPDATING - Legacy Code Dependencies

These files depend on removed classes and will BREAK without updates. They require refactoring to use the new Clean Architecture.

### Files Requiring Updates (4 files)

| File | Issue | Current Error | Fix Required |
|------|-------|---------------|--------------|
| `tests/test_passive_unit.py` | Imports `PassiveWorker` | `ImportError` | Rewrite to test `ProcessRFIUseCase` |
| `tests/test_active_unit.py` | Imports `RouteLogic` | `ImportError` | Rewrite to test `HandleQueryUseCase` |
| `tests/test_integration_local.py` | Uses `PassiveWorker` class | `AttributeError` | Rewrite to test HTTP endpoints |
| `benchmark/infrastructure/executors/baseline_executor.py` | Imports and instantiates `PassiveWorker` | Will fail at runtime | Update to use `PassiveAgentFactory` |

**Critical Discovery:**

#### `baseline_executor.py` - BREAKING CHANGE

**Location:** `/Users/alangarcia/Documents/dpr-rc/benchmark/infrastructure/executors/baseline_executor.py`

**Problem:** Lines 52-53 actually import and instantiate the removed `PassiveWorker` class:
```python
from dpr_rc.passive_agent import PassiveWorker
self._worker = PassiveWorker()
```

**Impact:** This will cause runtime failures when baseline benchmarks are executed.

**Fix Required:**
```python
# OLD (broken):
from dpr_rc.passive_agent import PassiveWorker
self._worker = PassiveWorker()

# NEW (fixed):
from dpr_rc.infrastructure.passive_agent.factory import PassiveAgentFactory
self._use_case = PassiveAgentFactory.create_from_env()

# Then update method calls:
# OLD: self._worker.retrieve(query, shard_id, timestamp_context)
# NEW: Use ProcessRFIUseCase or create a retrieval-only use case
```

**Priority:** HIGH - This affects baseline benchmark execution

**Recommendation:** Update this file before running baseline vs DPR-RC comparisons

---

## Category 3: CRITICAL - MUST KEEP

All other files are essential and must NOT be flagged for removal.

### Protected File Categories

#### 1. Core Refactored Code (KEEP ALL)
- **Domain Layer:** `dpr_rc/domain/` (38+ files)
  - Entities, services, repository interfaces
  - Pure business logic with no infrastructure dependencies

- **Application Layer:** `dpr_rc/application/` (20+ files)
  - Use cases, DTOs, interfaces
  - Orchestration logic

- **Infrastructure Layer:** `dpr_rc/infrastructure/` (25+ files)
  - Concrete implementations, factories
  - GCS, ChromaDB, HTTP clients

- **Facades:** Current thin facades (3 files)
  - `passive_agent.py` (315 lines)
  - `active_agent.py` (162 lines)
  - `slm_service.py` (285 lines)

- **Utilities:** Core utilities (4 files)
  - `logging_utils.py`, `debug_utils.py`, `embedding_utils.py`, `models.py`

#### 2. Benchmark System (CRITICAL - KEEP ALL)
- **Benchmark Core:** `benchmark/` directory (100+ files)
  - Domain, application, infrastructure, presentation layers
  - Synthetic data generation, evaluation services
  - Query executors (HTTP, DPR-RC)

- **Benchmark Results:**
  - `benchmark_results_research/` - Research results and analysis
  - `benchmark_results_cloud/` - Cloud deployment results
  - `benchmark_data/` - Generated datasets

- **Benchmark Scripts:**
  - `generate_and_upload.py` - Data generation
  - `research_benchmark.py` - Benchmark execution
  - `synthetic_history.py` - Synthetic history generation

#### 3. Deployment & Infrastructure (KEEP ALL)
- **Shell Scripts:** Automation and deployment (10+ files)
  - `run_cloud_benchmark.sh` - Cloud benchmark execution
  - `run_tests.sh` - Test automation
  - `deploy_commands.sh` - Deployment commands
  - `entrypoint.sh` - Container entrypoint
  - `infrastructure.sh` - Infrastructure setup
  - `download_results.sh` - Results retrieval
  - `view_cloud_logs.sh`, `view_debug_logs.sh` - Log viewing

- **Container Configuration:**
  - `Dockerfile` - Container image definition
  - `requirements.txt` - Python dependencies
  - `pytest.ini` - Test configuration

#### 4. API Specifications (KEEP ALL)
- `openapi/openapi_active_controller.yaml` - Active Agent API spec
- `openapi/openapi_passive_worker.yaml` - Passive Worker API spec
- `openapi/openapi_slm_service.yaml` - SLM Service API spec

#### 5. Working Tests (KEEP ALL)
- **Passing Unit Tests:** `tests/unit/benchmark/` (86 tests)
  - Domain interfaces, services, value objects
  - Infrastructure executors
  - All passing, no dependencies on removed classes

- **Passing Integration Tests:**
  - `tests/integration/benchmark/test_run_benchmark_use_case.py`

- **Passing Regression Tests:**
  - `tests/test_benchmark_scale.py` - Scale configuration tests
  - `tests/test_hallucination_detection.py` - Hallucination detection
  - `tests/test_batch_regression.py` - Batch processing regression
  - `tests/test_executor_equivalence.py` - Executor equivalence

- **Test Configuration:**
  - `tests/conftest.py` - Pytest fixtures
  - `tests/unit/benchmark/conftest.py` - Benchmark fixtures

#### 6. Documentation (KEEP ALL)
- **Primary Documentation:**
  - `CLAUDE.md` - Comprehensive codebase guide (critical for LLM context)

- **Refactoring Documentation:**
  - `SOLID_REFACTORING_COMPLETE.md` - Overall refactoring summary
  - `PHASE1_COMPLETE_SUMMARY.md` - Passive Agent refactoring
  - `PHASE2_COMPLETE_SUMMARY.md` - Active Agent refactoring
  - `PHASE3_COMPLETE_SUMMARY.md` - SLM Service refactoring
  - `TEST_RESULTS_POST_REFACTORING.md` - Test results and update plan

- **Phase Documentation:**
  - `PHASE2_ARCHITECTURE.md`, `PHASE2_IMPLEMENTATION.md`, `PHASE2_SUMMARY.md`
  - `BENCHMARK_DECOUPLING_PLAN.md`
  - `REGRESSION_ANALYSIS.md`

#### 7. Utility Scripts (KEEP ALL)
- `scripts/download_query_history.py` - Exchange history downloader

---

## Summary Matrix

| Category | Files | Lines | Action | Priority | Risk |
|----------|-------|-------|--------|----------|------|
| **Safe to Remove** | 3 backup files | 3,136 | DELETE after cloud benchmark | LOW | None |
| **Needs Updating** | 4 files | ~500 | REFACTOR to use new architecture | HIGH | Medium if not updated |
| **Critical - Keep** | 200+ files | ~15,000 | KEEP ALL | N/A | High if removed |

---

## Recommended Removal Plan

### Phase 1: Validation (REQUIRED FIRST)

Before removing any files, validate the refactored system:

```bash
# 1. Run cloud benchmark
BENCHMARK_SCALE=small ./run_cloud_benchmark.sh

# 2. Verify accuracy matches baseline
# Check: benchmark_results_cloud/small/comparison.json

# 3. If passing → Proceed to Phase 2
```

### Phase 2: Update Baseline Executor (REQUIRED)

**Before running baseline comparisons, fix `baseline_executor.py`:**

```python
# File: benchmark/infrastructure/executors/baseline_executor.py

# Update _ensure_worker method:
def _ensure_worker(self):
    """Lazy-load the use case"""
    if self._worker is None:
        try:
            from dpr_rc.infrastructure.passive_agent.factory import PassiveAgentFactory
            # Create use case instead of PassiveWorker
            self._use_case = PassiveAgentFactory.create_from_env()
        except ImportError as e:
            raise RuntimeError(
                "PassiveAgent not available. Install dpr-rc dependencies."
            ) from e

# Update execute method to use use case instead of direct worker calls
```

**Test after update:**
```bash
pytest tests/unit/benchmark/infrastructure/executors/test_executors_simple.py -v
```

### Phase 3: Remove Backup Files (SAFE)

**Only after cloud benchmark passes:**

```bash
# Remove backup files
rm dpr_rc/passive_agent.py.backup
rm dpr_rc/active_agent.py.backup
rm dpr_rc/slm_service.py.backup

# Commit removal
git add -A
git commit -m "chore: Remove pre-refactoring backup files

Removed backup files after successful cloud benchmark validation:
- passive_agent.py.backup (1,232 lines)
- active_agent.py.backup (957 lines)
- slm_service.py.backup (947 lines)

Files preserved in git history (commit: $(git rev-parse HEAD~1))
Refactored code validated via cloud benchmark."
```

### Phase 4: Archive Legacy Tests (OPTIONAL)

```bash
# Option 1: Archive for reference
mkdir -p tests/legacy_archived
mv tests/test_passive_unit.py tests/legacy_archived/
mv tests/test_active_unit.py tests/legacy_archived/
mv tests/test_integration_local.py tests/legacy_archived/

# Option 2: Delete (preserved in git history)
rm tests/test_passive_unit.py
rm tests/test_active_unit.py
rm tests/test_integration_local.py
```

---

## What NOT to Remove - Explicit Exclusions

**NEVER remove files in these categories:**

1. ❌ Any file under `benchmark/` - Critical for validation
2. ❌ Any file under `dpr_rc/domain/`, `dpr_rc/application/`, `dpr_rc/infrastructure/` - Core refactored code
3. ❌ Any `.sh` script - Deployment automation
4. ❌ `Dockerfile`, `requirements.txt`, `pytest.ini` - Infrastructure
5. ❌ Any OpenAPI spec - API documentation
6. ❌ `CLAUDE.md` - Primary documentation
7. ❌ Any passing test file - Test coverage
8. ❌ Any `conftest.py` - Test fixtures
9. ❌ `scripts/download_query_history.py` - Utility script

---

## Risk Assessment

### Low Risk (Safe to Proceed)
- ✅ Removing backup files after cloud benchmark validation
- ✅ Archiving legacy tests (preserved in git history)
- ✅ Updating baseline_executor.py with proper testing

### Medium Risk (Needs Attention)
- ⚠️ Running baseline benchmarks before updating baseline_executor.py
- ⚠️ Removing backup files without cloud benchmark validation

### High Risk (DO NOT DO)
- ❌ Removing any file under `benchmark/` directory
- ❌ Removing any refactored code in `dpr_rc/domain|application|infrastructure/`
- ❌ Removing deployment scripts or Docker configuration
- ❌ Removing `CLAUDE.md` or passing tests

---

## Validation Checklist

### Before ANY Removal

- [ ] Cloud benchmark executed successfully
- [ ] Benchmark accuracy matches baseline (±2%)
- [ ] All HTTP endpoints tested and working
- [ ] All changes committed to git
- [ ] `baseline_executor.py` updated and tested
- [ ] Backup files verified as exact duplicates

### After Removal

- [ ] System builds: `docker build -t dpr-rc .`
- [ ] Tests pass: `pytest tests/unit/benchmark/ -v`
- [ ] Cloud deployment works: `./run_cloud_benchmark.sh`
- [ ] Baseline benchmarks work (after baseline_executor.py update)
- [ ] Git history preserves deleted files

### Emergency Rollback

```bash
# Restore specific files
git checkout HEAD~1 -- dpr_rc/passive_agent.py.backup
git checkout HEAD~1 -- dpr_rc/active_agent.py.backup
git checkout HEAD~1 -- dpr_rc/slm_service.py.backup

# Or full rollback
git revert HEAD
```

---

## Conclusion

**Summary of Findings:**
- **3 backup files** are safe to remove after validation
- **4 files** need updating to work with refactored architecture
- **1 critical file** (`baseline_executor.py`) will break baseline benchmarks if not updated
- **200+ files** are essential and must be preserved

**Recommended Priority:**
1. **HIGH:** Update `baseline_executor.py` before running baseline comparisons
2. **MEDIUM:** Run cloud benchmark to validate refactored system
3. **LOW:** Remove backup files after successful validation
4. **LOW:** Archive legacy tests (optional)

**Next Steps:**
1. Review and approve this report
2. Update `baseline_executor.py` to use `PassiveAgentFactory`
3. Run cloud benchmark for validation
4. Remove backup files if benchmark passes
5. Document cleanup in git commit

**Risk Level:** LOW - All changes are reversible via git history, and critical files are protected.
