#!/bin/bash
# Sequential mutation testing runner for tip-of-the-top projects.
# Runs one project at a time, each with --max-workers=20.
# On baseline test failure (rc=2), calls claude to diagnose and fix, then retries once.
# Appends to summary.log and regenerates summary.md after every project.
#
# Usage: bash "$HOME/src/python/mutation/run-sequential.sh"

set -uo pipefail

# Re-exec under a systemd memory-capped scope if not already inside one
if [ -z "${_UNDER_SYSTEMD_SCOPE:-}" ] && command -v systemd-run &>/dev/null; then
    exec systemd-run --user --scope \
        -p MemoryMax=50% \
        -p MemorySwapMax=0 \
        env _UNDER_SYSTEMD_SCOPE=1 bash "$0" "$@"
fi

MUTATION_PY="$HOME/src/python/mutation/mutation.py"
TMP_DIR="$HOME/tmp/mutation/tip-of-the-top"
LOG_DIR="$TMP_DIR/logs"
SUMMARY_LOG="$TMP_DIR/summary.log"
SUMMARY_MD="$TMP_DIR/summary.md"
WORKERS=$(nproc --ignore=3)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="$SCRIPT_DIR/tip-of-the-top.txt"

mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M'; }
ts_long() { date '+%Y-%m-%d %H:%M:%S'; }

# ─── Record one line to summary.log ───────────────────────────────────────────
_record() {
    local repo_name="$1"
    local status="$2"
    local project_dir="$TMP_DIR/$repo_name"
    local db="$project_dir/.mutation.db"

    if [ "$status" = "SUCCESS" ] && [ -f "$db" ]; then
        read -r total survived killed rate < <(python3 -c "
import sqlite3
db = '$db'
con = sqlite3.connect(db)
cur = con.cursor()
try:
    cur.execute('SELECT COUNT(*) FROM mutations')
    total = cur.fetchone()[0]
    cur.execute(\"SELECT COUNT(*) FROM results WHERE status=1\")
    survived = cur.fetchone()[0]
    killed = total - survived
    rate = (survived / total * 100) if total > 0 else 0.0
    print(total, survived, killed, f'{rate:.1f}%')
except Exception as e:
    print('0 0 0 0.0%')
con.close()
" 2>/dev/null || echo "0 0 0 0.0%")
        printf '%s  %-8s  %-20s  total=%-6s  survived=%-6s  killed=%-6s  rate=%s\n' \
            "$(ts)" "$status" "$repo_name" "$total" "$survived" "$killed" "$rate" \
            >> "$SUMMARY_LOG"
    else
        printf '%s  %-8s  %-20s  total=%-6s  survived=%-6s  killed=%-6s  rate=%s\n' \
            "$(ts)" "$status" "$repo_name" "-" "-" "-" "-" \
            >> "$SUMMARY_LOG"
    fi
}

# ─── Regenerate summary.md from all .mutation.db files ────────────────────────
_update_summaries() {
    local progress_done total_urls
    progress_done=$(grep -c '' "$SUMMARY_LOG" 2>/dev/null || echo 0)
    total_urls=$(grep -cP 'https://\S+' "$URLS_FILE" 2>/dev/null || echo 67)

    local _py_script
    _py_script=$(cat <<'PYEOF'
import sys, sqlite3, os, glob, subprocess
from datetime import datetime

tmp_dir = sys.argv[1]
progress = int(sys.argv[2])
total = int(sys.argv[3])

def get_coverage(project_dir):
    venv_python = os.path.join(project_dir, "venv", "bin", "python3")
    cov_file = os.path.join(project_dir, ".coverage")
    if not os.path.exists(venv_python) or not os.path.exists(cov_file):
        return None
    try:
        r = subprocess.run(
            [venv_python, "-m", "coverage", "report",
             "--format=total", "--ignore-errors", "--fail-under=0",
             "--include=*.py", "--omit=venv/*,mutation.py,conftest.py,setup.py"],
            cwd=project_dir, capture_output=True, text=True, timeout=15
        )
        val = r.stdout.strip()
        return int(val) if val.isdigit() else None
    except Exception:
        return None

def get_loc(project_dir):
    total = 0
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in ("venv", ".git", "__pycache__", ".tox", "build", "dist")]
        for f in files:
            if not f.endswith(".py"):
                continue
            if f in ("mutation.py", "conftest.py", "setup.py"):
                continue
            path = os.path.join(root, f)
            # skip test files
            rel = os.path.relpath(path, project_dir)
            if any(p in rel for p in ("test", "tests", "testing")):
                continue
            try:
                with open(path, "rb") as fh:
                    total += sum(1 for _ in fh)
            except Exception:
                pass
    return total

rows = []
for db_path in sorted(glob.glob(f"{tmp_dir}/*/.mutation.db")):
    project_dir = os.path.dirname(db_path)
    repo_name = os.path.basename(project_dir)
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM mutations")
        total_mut = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM results WHERE status=1")
        survived = cur.fetchone()[0]
        killed = total_mut - survived
        rate = (survived / total_mut * 100) if total_mut > 0 else 0.0
        con.close()
        cov = get_coverage(project_dir)
        loc = get_loc(project_dir)
        rows.append((repo_name, total_mut, killed, survived, rate, cov, loc, "done"))
    except Exception:
        rows.append((repo_name, 0, 0, 0, 0.0, None, 0, "error"))

rows.sort(key=lambda r: r[4], reverse=True)

now = datetime.now().strftime("%Y-%m-%d %H:%M")
lines = []
lines.append("# Mutation Testing — tip-of-the-top")
lines.append(f"Updated: {now} | Progress: {progress}/{total}")
lines.append("")
lines.append("| # | Project | LOC | Coverage | Mutations | Killed | Survived | Survive% |")
lines.append("|---|---------|----:|--------:|--------:|-------:|---------:|---------:|")
for i, (name, tot, killed, survived, rate, cov, loc, status) in enumerate(rows, 1):
    cov_str = f"{cov}%" if cov is not None else "—"
    loc_str = f"{loc:,}" if loc else "—"
    lines.append(
        f"| {i} | {name} | {loc_str} | {cov_str} | {tot:,} | {killed:,} | {survived:,} | {rate:.1f}% |"
    )

print("\n".join(lines))
PYEOF
)
    python3 -c "$_py_script" "$TMP_DIR" "$progress_done" "$total_urls" \
        > "$SUMMARY_MD.tmp" 2>/dev/null \
        && mv "$SUMMARY_MD.tmp" "$SUMMARY_MD"
}

# ─── Call claude to debug baseline test failures ───────────────────────────────
_claude_debug() {
    local project_dir="$1"
    local repo_name="$2"
    echo "--- Calling claude to debug $repo_name ---"
    # Unset CLAUDECODE so nested claude invocation is allowed; cap at 30 min
    CLAUDECODE="" timeout 1800 claude --dangerously-skip-permissions -p \
        "You are fixing a Python project so its test suite passes baseline.
Project: $repo_name
Directory: $project_dir
Venv: $project_dir/venv

Steps:
1. Run: cd '$project_dir' && venv/bin/python3 -m pytest --no-header --tb=short -q -x 2>&1 | head -40
2. Diagnose what's failing (missing deps, import errors, config issues, submodules, etc.)
3. Fix it: install missing packages with venv/bin/pip, add collect_ignore to conftest.py,
   run git submodule update --init, patch pyproject.toml/setup.cfg as needed.
4. Do NOT modify test logic or source code — only install deps or add pytest ignores.
5. Verify: venv/bin/python3 -m pytest --no-header --tb=no -q 2>&1 | tail -5
6. Stop once tests pass (exit 0) or conclude it is unfixable." 2>&1
}

# Projects to skip (unfixable in this environment):
#   anyio — OpenSSL 3.x TLS shutdown regression; test_send_eof_not_implemented fails
SKIP_REPOS=( anyio )

# ─── Core per-project logic ────────────────────────────────────────────────────
_run_project() {
    local url="$1"
    local repo_name log_file project_dir

    repo_name="$(basename "$url" .git)"
    project_dir="$TMP_DIR/$repo_name"
    log_file="$LOG_DIR/${repo_name}.log"

    for _skip in "${SKIP_REPOS[@]}"; do
        if [ "$repo_name" = "$_skip" ]; then
            echo "[$(ts_long)]  SKIPPED (skip list): $repo_name"
            _record "$repo_name" "SKIPPED"
            _update_summaries
            return
        fi
    done

    {
        echo "=== START: $repo_name  [$(ts_long)] ==="

        # 1. Clone --depth=1 (skip if dir already exists)
        if [ ! -d "$project_dir" ]; then
            echo "--- Cloning $url ---"
            git clone --depth=1 "$url" "$project_dir" \
                || { echo "FAILED: git clone"; _record "$repo_name" "FAILED"; _update_summaries; return; }
        else
            echo "--- Already cloned ---"
        fi

        cd "$project_dir" || { echo "FAILED: cd"; _record "$repo_name" "FAILED"; _update_summaries; return; }

        # 2. Skip if .mutation.db already exists
        if [ -f ".mutation.db" ]; then
            echo "SKIPPED: .mutation.db exists"
            _record "$repo_name" "SKIPPED"
            _update_summaries
            return
        fi

        # 3. Create venv
        if [ ! -d "venv" ]; then
            echo "--- Creating venv ---"
            python3 -m venv venv \
                || { echo "FAILED: venv creation"; _record "$repo_name" "FAILED"; _update_summaries; return; }
        fi

        local PIP="$project_dir/venv/bin/pip"
        local PYTHON="$project_dir/venv/bin/python3"

        # 4. Install mutation.py deps
        echo "--- Installing mutation.py deps ---"
        "$PIP" install -q --upgrade pip 2>&1
        "$PIP" install -q \
            aiostream docopt humanize loguru pygments \
            pytest-cov pytest-randomly pytest-xdist python-ulid \
            termcolor tqdm zstandard coverage 2>&1

        # 5. Install project with extras fallback chain
        echo "--- Installing project ---"
        "$PIP" install -q -e ".[dev,test,tests,testing]" 2>/dev/null \
            || "$PIP" install -q -e ".[dev,test,tests]" 2>/dev/null \
            || "$PIP" install -q -e ".[dev,test]" 2>/dev/null \
            || "$PIP" install -q -e ".[dev]" 2>/dev/null \
            || "$PIP" install -q -e ".[test]" 2>/dev/null \
            || "$PIP" install -q -e ".[testing]" 2>/dev/null \
            || "$PIP" install -q -e "." 2>&1 \
            || echo "WARNING: project install partial"

        # 6a. Install any extra requirement files
        for req in \
            requirements-dev.txt requirements-test.txt requirements_test.txt \
            test-requirements.txt requirements/test.txt requirements/dev.txt \
            requirements/testing.txt dev-requirements.txt; do
            [ -f "$req" ] && "$PIP" install -q -r "$req" 2>/dev/null || true
        done

        # 6b. Install PEP 735 dependency-groups (pip 24.3+ supports --group)
        for grp in tests test testing dev; do
            "$PIP" install -q --group "$grp" 2>/dev/null || true
        done

        # 6c. Project-specific fixups
        case "$repo_name" in
            black)
                # needs aiohttp (the [d] extra) so test_blackd.py can be collected
                "$PIP" install -q "black[d]" aiohttp 2>&1 || true
                ;;
            cattrs)
                # bench/ requires msgspec + PyYAML; ignore bench/ so baseline passes
                "$PIP" install -q msgspec PyYAML 2>&1 || true
                if ! grep -q "collect_ignore" conftest.py 2>/dev/null; then
                    printf '\ncollect_ignore_glob = ["bench/*"]\n' >> conftest.py
                fi
                ;;
            dpath-python)
                "$PIP" install -q nose2 hypothesis 2>&1 || true
                ;;
            gitlint)
                "$PIP" install -q arrow click 2>&1 || true
                ;;
            "jaraco.text")
                "$PIP" install -q inflect 2>&1 || true
                ;;
            jedi)
                # parso + missing git submodule (typeshed)
                "$PIP" install -q parso 2>&1 || true
                git submodule update --init --recursive 2>&1 || true
                ;;
            "jmespath.py")
                # extra/ requires hypothesis; ignore it so baseline collects cleanly
                "$PIP" install -q hypothesis 2>&1 || true
                if ! grep -q "collect_ignore" conftest.py 2>/dev/null; then
                    printf '\ncollect_ignore_glob = ["extra/*"]\n' >> conftest.py
                fi
                ;;
            klein)
                "$PIP" install -q treq 2>&1 || true
                ;;
            blist)
                # ez_setup.py tries to download setuptools; pre-install it to avoid network fetch
                "$PIP" install -q setuptools 2>&1 || true
                "$PIP" install -q -e . 2>&1 || true
                ;;
            mpmath)
                # editable install produces broken version metadata; reinstall as non-editable
                "$PIP" install -q --force-reinstall . 2>&1 || true
                ;;
            pyparsing)
                # railroad-diagrams + jinja2 needed for diagram tests
                "$PIP" install -q railroad-diagrams jinja2 2>&1 || true
                ;;
            pyrsistent)
                "$PIP" install -q hypothesis 2>&1 || true
                ;;
            pytz)
                # zdump.out is a generated file that can't be reproduced; ignore the test
                "$PIP" install -q pytz 2>&1 || true
                if ! grep -q "collect_ignore" conftest.py 2>/dev/null; then
                    printf '\ncollect_ignore_glob = ["*/zdump*"]\n' >> conftest.py
                fi
                ;;
            result)
                "$PIP" install -q mypy pytest-mypy-plugins 2>&1 || true
                ;;
            returns)
                "$PIP" install -q covdefaults pytest-mypy-plugins 2>&1 || true
                # typesafety/ tests check exact mypy output format and fail on mypy version mismatch
                echo 'collect_ignore_glob = ["typesafety/*"]' >> "$project_dir/conftest.py"
                ;;
            thefuzz)
                "$PIP" install -q pycodestyle hypothesis 2>&1 || true
                ;;
            glom)
                "$PIP" install -q PyYAML 2>&1 || true
                ;;
            whenever)
                "$PIP" install -q tzdata 2>&1 || true
                # pytest.ini has filterwarnings=error which turns pytest-benchmark's
                # xdist-incompatibility warning into a hard crash; suppress it
                if ! grep -q "PytestBenchmarkWarning" pytest.ini 2>/dev/null; then
                    printf '    ignore::pytest_benchmark.logger.PytestBenchmarkWarning\n' >> pytest.ini
                fi
                ;;
            tomlkit)
                # tests/toml-test is a git submodule; init it now
                git submodule update --init --recursive 2>&1 || true
                ;;
            "python-ftfy")
                # doctest in source files has stale unicode-width expectation; skip doctests
                "$PIP" install -q ftfy 2>&1 || true
                if ! grep -qE "addopts.*doctest|doctest_optionflags" setup.cfg pytest.ini pyproject.toml 2>/dev/null; then
                    printf '\n[tool.pytest.ini_options]\naddopts = "--ignore-glob=**/*.py"\n' >> pyproject.toml 2>/dev/null || true
                fi
                ;;
        esac

        # 7. Re-pin mutation.py deps (project may have downgraded something)
        "$PIP" install -q \
            aiostream docopt humanize loguru pygments \
            pytest-cov pytest-randomly pytest-xdist python-ulid \
            termcolor tqdm zstandard coverage 2>&1

        # 8. Copy mutation.py to project root
        local _copied=0
        if [ ! -f "mutation.py" ]; then
            cp "$MUTATION_PY" "mutation.py"
            _copied=1
        fi

        # 9. Run mutation testing
        echo "--- Running mutation.py play (workers=$WORKERS) ---"
        PATH="$project_dir/venv/bin:$PATH" \
        "$PYTHON" "$MUTATION_PY" play --max-workers="$WORKERS" 2>&1
        local _rc=$?

        # 11. If rc==2 (baseline tests red) → call claude, then retry once
        if [ "$_rc" = "2" ]; then
            echo "--- Baseline tests failed (rc=2). Invoking claude debugger... ---"
            _claude_debug "$project_dir" "$repo_name"
            echo "--- Retrying mutation.py play after claude fix... ---"
            PATH="$project_dir/venv/bin:$PATH" \
            "$PYTHON" "$MUTATION_PY" play --max-workers="$WORKERS" 2>&1
            _rc=$?
        fi

        [ "$_copied" = "1" ] && rm -f "mutation.py"

        # 12. Record result and update summaries
        if [ "$_rc" = "0" ]; then
            echo "=== DONE: $repo_name  [$(ts_long)] ==="
            _record "$repo_name" "SUCCESS"
        else
            echo "FAILED: mutation.py play (rc=$_rc)"
            _record "$repo_name" "FAILED"
        fi
        _update_summaries

    } > "$log_file" 2>&1

    # Echo progress to stdout (outside the log redirect)
    local result
    result=$(tail -1 "$SUMMARY_LOG" 2>/dev/null || echo "?")
    echo "[$(ts_long)]  $result"
}

# ─── Main: sequential execution ───────────────────────────────────────────────
mapfile -t ALL_URLS < <(grep -oP 'https://\S+' "$URLS_FILE")
TOTAL="${#ALL_URLS[@]}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Sequential Mutation Testing — tip-of-the-top        ║"
printf "║  Projects: %-3d | Workers/project: %-3d | %s  ║\n" "$TOTAL" "$WORKERS" "$(date '+%Y-%m-%d')"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "TMP dir:  $TMP_DIR"
echo "Log dir:  $LOG_DIR"
echo "Summary:  $SUMMARY_LOG"
echo ""

# Initialize summary log header if new
if [ ! -f "$SUMMARY_LOG" ]; then
    printf '%-16s  %-8s  %-20s  %-12s  %-12s  %-12s  %s\n' \
        "timestamp" "status" "project" "total" "survived" "killed" "rate" \
        >> "$SUMMARY_LOG"
    printf '%-16s  %-8s  %-20s  %-12s  %-12s  %-12s  %s\n' \
        "----------------" "--------" "--------------------" "------------" "------------" "------------" "----" \
        >> "$SUMMARY_LOG"
fi

LAUNCHED=0
for url in "${ALL_URLS[@]}"; do
    repo_name="$(basename "$url" .git)"
    LAUNCHED=$((LAUNCHED + 1))
    echo "[$(ts_long)]  [$LAUNCHED/$TOTAL]  Starting: $repo_name"
    _run_project "$url"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ALL DONE — $(ts_long)"
echo "  Results: $SUMMARY_LOG"
echo "  Report:  $SUMMARY_MD"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
