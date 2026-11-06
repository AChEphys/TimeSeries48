assert COMPARE, "No baseline set: skip this section."
import difflib  # noqa: E402
import io  # noqa: E402
import tokenize  # noqa: E402


def code_without_comments(path: Path) -> list[str]:
    """Source lines with comments and blank lines removed (NOTE comments aren't setup changes)."""
    out, last_line = [], -1
    tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
    kept = [t for t in tokens if t.type != tokenize.COMMENT]
    src = tokenize.untokenize(kept)
    for line in src.splitlines():
        if line.strip():
            out.append(line.rstrip())
    return out


def archived_script(run_dir: Path) -> Path | None:
    """The training script copy archived in the run folder (largest .py, if several)."""
    scripts = [p for p in run_dir.glob("*.py")]
    return max(scripts, key=lambda p: p.stat().st_size) if scripts else None


run_script, base_script = archived_script(RUN_DIR), archived_script(BASELINE_DIR)
print(f"run script:      {run_script.name if run_script else 'NOT ARCHIVED'}")
print(f"baseline script: {base_script.name if base_script else 'NOT ARCHIVED'}")
if run_script and base_script:
    diff = list(difflib.unified_diff(code_without_comments(base_script), code_without_comments(run_script),
                                     fromfile=f"baseline/{base_script.name}", tofile=f"run/{run_script.name}",
                                     n=1, lineterm=""))
    print(f"\n--- code diff (comments and blank lines ignored): {sum(l.startswith(('+', '-')) and not l.startswith(('+++', '---')) for l in diff)} changed lines ---")
    print("\n".join(diff) if diff else "(no code differences)")

# Per-window data and selection facts, matched by test start date.
b_by_start = {r["test_start"]: r for r in b_results}
rows = []
for k, r in enumerate(results):
    b = b_by_start.get(r["test_start"], {})
    rows.append({"window": k + 1, "test_start": r["test_start"]}
                | {f"{key} base": b.get(key) for key in ("train_rows", "validation_rows", "test_rows", "features")}
                | {f"{key} run": r.get(key) for key in ("train_rows", "validation_rows", "test_rows", "features")})
setup_facts = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    display(setup_facts)  # noqa: F821


def selected_features(run_dir: Path) -> dict[str, set[str]]:
    """Selected features per window (keyed by test start), from models/*.features.json."""
    out = {}
    for f in (run_dir / "models").glob("*.features.json"):
        m = re.search(r"_test_(\d{4}-\d{2}-\d{2})_", f.name)
        if m:
            out[m[1]] = set(json.loads(f.read_text())["features"])
    return out


run_feats, base_feats = selected_features(RUN_DIR), selected_features(BASELINE_DIR)
for k, r in enumerate(results):
    a, b = run_feats.get(r["test_start"]), base_feats.get(r["test_start"])
    if a is None or b is None:
        print(f"W{k + 1}: feature lists not available")
        continue
    print(f"W{k + 1}: {len(b)} -> {len(a)} features; added {sorted(a - b) or '-'}; removed {sorted(b - a) or '-'}")
