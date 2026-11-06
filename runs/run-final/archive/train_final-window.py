#!/usr/bin/env python3
"""
Final evaluation (Stage 6, issue I-6): train every setup on the reserved final window.

All output goes to runs/run-final/, one subfolder per setup. The setups are the two
baselines (run 2: the author's original script; run 5: the purged k-fold baseline)
and every model the final-evaluation candidates use (claude/project-notes.md,
"Candidates for the final evaluation"). Each setup is trained exactly like its
archived run, on window 7 (the final window, 84 test dates) only:

- Every head setting the training depends on is compared with the archived run's
  script before training; any difference stops the setup.
- With VERIFY, window 6 is trained too and its predictions are compared with the
  archived run's (reproduction check, verify.json). Run 2's original script cannot
  pick windows: it trains all seven (about 3 min) and checks windows 1-6.

Usage (from the repo root):
    python train_final-window.py              # every unfinished setup, one process each
    python train_final-window.py --setup run-22
    python train_final-window.py --dry-run    # check the settings only, train nothing

Next steps: runs/run-final/post-processing.ipynb (candidates' final predictions), then
final-evaluation_performances.ipynb (repo root).
"""  # BY: Claude

from __future__ import annotations  # BY: Claude

import argparse  # BY: Claude
import ast  # BY: Claude
import importlib  # BY: Claude
import importlib.util  # BY: Claude
import json  # BY: Claude
import shutil  # BY: Claude
import subprocess  # BY: Claude
import sys  # BY: Claude
import time  # BY: Claude
from pathlib import Path  # BY: Claude
from typing import TYPE_CHECKING, NamedTuple  # BY: Claude

import numpy as np  # BY: Claude
import polars as pl  # BY: Claude

if TYPE_CHECKING:  # BY: Claude
    from types import ModuleType  # BY: Claude

ROOT = Path(__file__).resolve().parent  # BY: Claude
DATA_FILE = ROOT / "data.parquet"  # BY: Claude
FINAL_DIR = ROOT / "runs/run-final"  # BY: Claude
FINAL_WINDOW = 7  # BY: Claude  # NOTE: Test dates after 2025-12-18 (I-6).
VERIFY = True  # BY: Claude  # NOTE: Also train window 6 and compare with the archived run (about 2x the cost).
VERIFY_WINDOW = 6  # BY: Claude
TRAINER_SCRIPTS = {  # BY: Claude  # NOTE: The current scripts the "baseline" and "multi-day" trainers import.
    "baseline": ROOT / "train_baseline.py",  # BY: Claude
    "multi-day": ROOT / "train_multi-day-feature-sets.py",  # BY: Claude
}  # BY: Claude


class Setup(NamedTuple):  # BY: Claude
    """How to train one setup: which trainer, and the head settings it overrides."""  # BY: Claude

    trainer: str  # BY: Claude  # NOTE: "original" (run 2's archived script), "baseline", "multi-day".
    head: dict  # BY: Claude


MULTI_DAY_OFF = {  # BY: Claude  # NOTE: Settings added after runs 11-23; these values reproduce them.
    "LAGS": (1, 5), "FEATURE_FRACTION": 1.0, "RECENCY_HALF_LIFE": None,  # BY: Claude
    "CORRECTION_OF": None,  # BY: Claude
}  # BY: Claude
SETUPS = {  # BY: Claude  # NOTE: Setup name = its archived run (runs/<name>_*).
    "run-2": Setup("original", {}),  # BY: Claude  # NOTE: Baseline: the author's script.
    "run-5": Setup("baseline", {}),  # BY: Claude  # NOTE: Baseline: purged 5-fold, all features.
    "run-11": Setup("multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+lags"}),  # BY: Claude  # NOTE: Lag mix; mix 11 + 17.
    "run-13": Setup("multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+lags-1-2"}),  # BY: Claude  # NOTE: Lag mix.
    "run-14": Setup("multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+lags-1-3"}),  # BY: Claude  # NOTE: Lag mix.
    "run-15": Setup("multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+lags-1-4"}),  # BY: Claude  # NOTE: Lag mix.
    "run-16": Setup("multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+lags-1-6"}),  # BY: Claude  # NOTE: Lag mix.
    "run-17": Setup("multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+labelraw-6-10"}),  # BY: Claude  # NOTE: Mix 11 + 17.
    "run-22": Setup(  # BY: Claude  # NOTE: Primary candidate.
        "multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+lags+labelraw-6-10-scaled"}  # BY: Claude
    ),  # BY: Claude
    "run-23": Setup(  # BY: Claude
        "multi-day", {**MULTI_DAY_OFF, "FEATURE_SET": "raw+lags+labelraw-6-10-rank"}  # BY: Claude
    ),  # BY: Claude
}  # BY: Claude
CHECKED = (  # BY: Claude  # NOTE: Head settings compared with the archived run's script.
    "RANDOM_SEED", "CORR_BAR", "LOOKBACK", "GAP", "TEST_DAYS", "VALIDATION_FOLD",
    "PURGE", "SELECT_FEATURES", "MAX_ROUNDS", "EARLY_STOPPING_ROUNDS", "FEATURE_SET",
    "LABEL_LAG", "LABEL_MEANS", "LABEL_SPREAD", "DEV_WINDOWS", "LAGS", "STALE_LIMIT",
    "FEATURE_FRACTION", "RECENCY_HALF_LIFE", "CORRECTION_OF",
)  # BY: Claude


def archived_run(name: str) -> Path:  # BY: Claude
    """Return the setup's archived run folder (runs/run-22_*, not its seed variants)."""  # BY: Claude
    found = sorted((ROOT / "runs").glob(f"{name}_*"))  # BY: Claude
    if len(found) != 1:  # BY: Claude
        msg = f"Expected one folder runs/{name}_*, found {found}"  # BY: Claude
        raise FileNotFoundError(msg)  # BY: Claude
    return found[0]  # BY: Claude


def archived_script(name: str) -> Path:  # BY: Claude
    """Return the training script archived in the setup's run folder."""  # BY: Claude
    return next(archived_run(name).glob("train*.py"))  # BY: Claude


def load_module(path: Path, module_name: str) -> ModuleType:  # BY: Claude
    """Import a script by path (file names with hyphens can't be imported by name)."""  # BY: Claude
    spec = importlib.util.spec_from_file_location(module_name, path)  # BY: Claude
    if spec is None or spec.loader is None:  # BY: Claude
        msg = f"Can't import {path}"  # BY: Claude
        raise ImportError(msg)  # BY: Claude
    module = importlib.util.module_from_spec(spec)  # BY: Claude
    spec.loader.exec_module(module)  # BY: Claude
    return module  # BY: Claude


def constants(path: Path) -> dict[str, str]:  # BY: Claude
    """Module-level `NAME = value` assignments of a script, values as source text."""  # BY: Claude
    tree = ast.parse(path.read_text(encoding="utf-8"))  # BY: Claude
    return {  # BY: Claude
        node.targets[0].id: ast.unparse(node.value)  # BY: Claude
        for node in tree.body  # BY: Claude
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)  # BY: Claude
    }  # BY: Claude


def lgbm_params_source(path: Path) -> dict[str, str]:  # BY: Claude
    """Return an old script's `params = {...}` literal, values as source text."""  # BY: Claude
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):  # BY: Claude
        if (  # BY: Claude
            isinstance(node, ast.Assign)  # BY: Claude
            and getattr(node.targets[0], "id", None) == "params"  # BY: Claude
            and isinstance(node.value, ast.Dict)  # BY: Claude
        ):  # BY: Claude
            return {  # BY: Claude
                ast.literal_eval(k): ast.unparse(v)  # BY: Claude
                for k, v in zip(node.value.keys, node.value.values, strict=True)  # BY: Claude
                if k is not None  # BY: Claude  # NOTE: None = a `**other` entry; params literals have none.
            }  # BY: Claude
    msg = f"No params literal in {path}"  # BY: Claude
    raise ValueError(msg)  # BY: Claude


def check_archive(name: str, module: ModuleType, source: Path) -> list[str]:  # BY: Claude
    """
    Differences between the configured module and the setup's archived script.

    Compares the CHECKED head settings (archived literal vs the module's value, or the
    source text when the archive uses an expression such as PURGE = GAP), and the
    LightGBM parameters: run 5's literal vs train_baseline.lgbm_params(); family runs
    must call tb.lgbm_params(RANDOM_SEED + i).
    """  # BY: Claude
    archive = archived_script(name)  # BY: Claude
    old, new_source = constants(archive), constants(source)  # BY: Claude
    problems = []  # BY: Claude
    for key in CHECKED:  # BY: Claude
        if key not in old:  # BY: Claude
            continue  # BY: Claude  # NOTE: Setting added later; SETUPS pins its "off" value.
        try:  # BY: Claude
            same = ast.literal_eval(old[key]) == getattr(module, key)  # BY: Claude
        except ValueError:  # BY: Claude
            same = old[key] == new_source.get(key)  # BY: Claude
        if not same:  # BY: Claude
            now = getattr(module, key)  # BY: Claude
            problems.append(f"{key}: archive {old[key]} vs now {now!r}")  # BY: Claude
    text = archive.read_text(encoding="utf-8")  # BY: Claude
    if SETUPS[name].trainer == "original" or "tb.lgbm_params(RANDOM_SEED + i)" in text:  # BY: Claude
        return problems  # BY: Claude  # NOTE: Run 2 trains with its own literal; family runs call lgbm_params().
    import train_baseline as tb  # BY: Claude  # noqa: PLC0415
    params = lgbm_params_source(archive)  # BY: Claude
    now = tb.lgbm_params(0)  # BY: Claude
    for key, value in params.items():  # BY: Claude
        if key in {"seed", "num_threads"}:  # BY: Claude
            continue  # BY: Claude  # NOTE: seed = RANDOM_SEED + i in both; threads = this machine's.
        if ast.literal_eval(value) != now.get(key):  # BY: Claude
            problems.append(f"param {key}: archive {value} vs now {now.get(key)!r}")  # BY: Claude
    if set(params) != set(now):  # BY: Claude
        problems.append(f"param keys differ: {sorted(set(params) ^ set(now))}")  # BY: Claude
    return problems  # BY: Claude


def configure(name: str, out: Path, data_file: Path) -> tuple[ModuleType, list[int]]:  # BY: Claude
    """Import the setup's trainer, point it at `out`, apply its head; windows to run."""  # BY: Claude
    setup = SETUPS[name]  # BY: Claude
    windows = [VERIFY_WINDOW, FINAL_WINDOW] if VERIFY else [FINAL_WINDOW]  # BY: Claude
    if setup.trainer == "original":  # BY: Claude  # NOTE: The archived script itself, untouched.
        module = load_module(archived_script(name), "original_baseline")  # BY: Claude
        settings = {"MODEL_DIR": out / "models"}  # BY: Claude  # NOTE: Predictions and curves go flat into out/.
        windows = list(range(1, FINAL_WINDOW + 1))  # BY: Claude  # NOTE: No window selection in this script.
    else:  # BY: Claude
        if setup.trainer == "baseline":  # BY: Claude  # NOTE: By name: the same module the family script imports.
            module = importlib.import_module("train_baseline")  # BY: Claude
        else:  # BY: Claude
            module = load_module(TRAINER_SCRIPTS["multi-day"], "multi_day")  # BY: Claude
        settings = {  # BY: Claude
            "MODEL_DIR": out / "models",  # BY: Claude
            "PREDICTIONS_DIR": out / "predictions",  # BY: Claude
            "CURVES_DIR": out / "curves",  # BY: Claude
            "WINDOWS": tuple(windows),  # BY: Claude
            **setup.head,  # BY: Claude
        }  # BY: Claude
    settings |= {  # BY: Claude
        "DATA_FILE": data_file,  # BY: Claude
        "RUN_DIR": out,  # BY: Claude
        "RESULT_FILE": out / "results_lgbm.json",  # BY: Claude
        "SKIP_LAST_WINDOW": False,  # BY: Claude
        "SMOKE_TEST": False,  # BY: Claude
    }  # BY: Claude
    for key, value in settings.items():  # BY: Claude
        if not hasattr(module, key):  # BY: Claude
            msg = f"{name}: trainer has no setting {key}"  # BY: Claude
            raise AttributeError(msg)  # BY: Claude
        setattr(module, key, value)  # BY: Claude
    if setup.trainer == "multi-day":  # BY: Claude
        module.share_head()  # BY: Claude  # NOTE: Pushes RUN_DIR, CURVES_DIR, ... into train_baseline.
    return module, windows  # BY: Claude


def window_of(path: Path, starts: dict[str, int]) -> int | None:  # BY: Claude
    """Window number of a predictions file, from the test start date in its name."""  # BY: Claude
    start = path.stem.split("_test_")[1].split("_")[0]  # BY: Claude
    return starts.get(start)  # BY: Claude


def verify(name: str, out: Path, windows: list[int]) -> list[dict]:  # BY: Claude
    """Compare the re-trained windows before the final one with the archived run's."""  # BY: Claude
    archive = archived_run(name)  # BY: Claude
    starts = {  # BY: Claude
        r["test_start"]: k  # BY: Claude
        for k, r in enumerate(  # BY: Claude
            json.loads((archive / "results_lgbm.json").read_text()), start=1  # BY: Claude
        )  # BY: Claude
    }  # BY: Claude
    old = {window_of(p, starts): p for p in archive.rglob("predictions_*.parquet")}  # BY: Claude
    new = {window_of(p, starts): p for p in out.rglob("predictions_*.parquet")}  # BY: Claude
    report = []  # BY: Claude
    for w in windows:  # BY: Claude
        if w == FINAL_WINDOW or w not in old or w not in new:  # BY: Claude
            continue  # BY: Claude
        pair = pl.read_parquet(old[w]).join(  # BY: Claude
            pl.read_parquet(new[w]), on=["Date", "Code"], suffix="_new"  # BY: Claude
        )  # BY: Claude
        if pair.is_empty():  # BY: Claude  # NOTE: Nothing to compare (e.g. a test on synthetic data).
            report.append({"window": w, "rows_matched": pair.height})  # BY: Claude
            continue  # BY: Claude
        a, b = pair["Prediction"].to_numpy(), pair["Prediction_new"].to_numpy()  # BY: Claude
        daily = pair.group_by("Date").agg(  # BY: Claude
            pl.corr(pl.col(c).rank(), pl.col("label").rank()).alias(k)  # BY: Claude
            for k, c in (("old", "Prediction"), ("new", "Prediction_new"))  # BY: Claude
        )  # BY: Claude
        report.append({  # BY: Claude
            "window": w,  # BY: Claude
            "rows_archived": pl.read_parquet(old[w]).height,  # BY: Claude
            "rows_matched": pair.height,  # BY: Claude
            "max_abs_diff": float(np.abs(a - b).max()),  # BY: Claude
            "corr": float(np.corrcoef(a, b)[0, 1]),  # BY: Claude
            "rank_ic_archived": float(daily["old"].to_numpy().mean()),  # BY: Claude
            "rank_ic_new": float(daily["new"].to_numpy().mean()),  # BY: Claude
        })  # BY: Claude
    return report  # BY: Claude


def run_setup(name: str, data_file: Path, *, dry_run: bool) -> None:  # BY: Claude
    """Train one setup into FINAL_DIR/<name>/ (in this process), then verify it."""  # BY: Claude
    out = FINAL_DIR / name  # BY: Claude
    module, windows = configure(name, out, data_file)  # BY: Claude
    trainer = SETUPS[name].trainer  # BY: Claude
    original = trainer == "original"  # BY: Claude
    source = archived_script(name) if original else TRAINER_SCRIPTS[trainer]  # BY: Claude
    problems = check_archive(name, module, source)  # BY: Claude
    head = {k: getattr(module, k) for k in CHECKED if hasattr(module, k)}  # BY: Claude
    print(f"{name}: {SETUPS[name].trainer} trainer, windows {windows}, head {head}")  # BY: Claude  # noqa: T201
    if problems:  # BY: Claude
        msg = f"{name} differs from its archived run: " + "; ".join(problems)  # BY: Claude
        raise RuntimeError(msg)  # BY: Claude
    print(f"{name}: settings match {archived_script(name).relative_to(ROOT)}")  # BY: Claude  # noqa: T201
    if dry_run:  # BY: Claude
        return  # BY: Claude
    if out.exists():  # BY: Claude
        msg = f"{out} exists but isn't finished; inspect it, delete it, then rerun"  # BY: Claude
        raise FileExistsError(msg)  # BY: Claude
    out.mkdir(parents=True)  # BY: Claude
    started = time.time()  # BY: Claude
    module.train()  # BY: Claude
    report = verify(name, out, windows) if VERIFY or original else []  # BY: Claude
    (out / "verify.json").write_text(json.dumps(report, indent=2), encoding="utf-8")  # BY: Claude
    for r in (r for r in report if "corr" in r):  # BY: Claude
        print(  # BY: Claude  # noqa: T201
            f"{name} W{r['window']} vs archive: corr {r['corr']:.6f}, max |diff| "  # BY: Claude
            f"{r['max_abs_diff']:.2e}, Rank IC {r['rank_ic_archived']:.4f} -> "  # BY: Claude
            f"{r['rank_ic_new']:.4f}, rows {r['rows_matched']}/{r['rows_archived']}"  # BY: Claude
        )  # BY: Claude
    done = {  # BY: Claude
        "setup": name,  # BY: Claude
        "trainer": SETUPS[name].trainer,  # BY: Claude
        "archived_run": str(archived_run(name).relative_to(ROOT)),  # BY: Claude
        "head": {k: list(v) if isinstance(v, tuple) else v for k, v in head.items()},  # BY: Claude
        "windows": windows,  # BY: Claude
        "data_file": str(data_file),  # BY: Claude
        "minutes": round((time.time() - started) / 60, 2),  # BY: Claude
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),  # BY: Claude
    }  # BY: Claude
    (out / "done.json").write_text(json.dumps(done, indent=2), encoding="utf-8")  # BY: Claude  # NOTE: Marks the setup as finished.


def run_all(args: argparse.Namespace) -> int:  # BY: Claude
    """Archive the scripts, then train every unfinished setup in its own subprocess."""  # BY: Claude
    if not args.dry_run:  # BY: Claude
        archive = FINAL_DIR / "archive"  # BY: Claude
        archive.mkdir(parents=True, exist_ok=True)  # BY: Claude
        scripts = ("train_final-window.py", "train_baseline.py",  # BY: Claude
                   "train_multi-day-feature-sets.py")  # BY: Claude
        for script in scripts:  # BY: Claude
            shutil.copy2(ROOT / script, archive / script)  # BY: Claude
    status = {}  # BY: Claude
    for name in SETUPS:  # BY: Claude
        if (FINAL_DIR / name / "done.json").exists():  # BY: Claude
            status[name] = "already done"  # BY: Claude
            continue  # BY: Claude
        cmd = [sys.executable, str(Path(__file__).resolve()), "--setup", name,  # BY: Claude
               "--data-file", str(args.data_file), "--final-dir", str(FINAL_DIR)]  # BY: Claude
        if args.dry_run:  # BY: Claude
            cmd.append("--dry-run")  # BY: Claude
        code = subprocess.run(cmd, check=False, cwd=ROOT).returncode  # BY: Claude  # noqa: S603
        status[name] = "ok" if code == 0 else f"FAILED (exit {code})"  # BY: Claude
    print("\n".join(f"{k:7} {v}" for k, v in status.items()))  # BY: Claude  # noqa: T201
    return int(any(v.startswith("FAILED") for v in status.values()))  # BY: Claude


def main() -> int:  # BY: Claude
    """Entry point: one setup (--setup) or all of them."""  # BY: Claude
    global FINAL_DIR  # BY: Claude  # noqa: PLW0603  # NOTE: Overridable for tests (--final-dir).
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # BY: Claude
    parser.add_argument("--setup", choices=list(SETUPS), help="train only this setup")  # BY: Claude
    parser.add_argument("--dry-run", action="store_true", help="check settings only")  # BY: Claude
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)  # BY: Claude  # NOTE: For tests on synthetic data.
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)  # BY: Claude  # NOTE: For tests.
    args = parser.parse_args()  # BY: Claude
    FINAL_DIR = args.final_dir.resolve()  # BY: Claude
    if args.setup is None:  # BY: Claude
        return run_all(args)  # BY: Claude
    run_setup(args.setup, args.data_file.resolve(), dry_run=args.dry_run)  # BY: Claude
    return 0  # BY: Claude


if __name__ == "__main__":  # BY: Claude
    sys.exit(main())  # BY: Claude
