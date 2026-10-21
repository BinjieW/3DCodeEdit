"""Step 1: extract each edited program and render it next to its source program.

Input: a JSONL file with one editor reply per task,

    {"case_id": "...", "response": "<the editor's full reply>"}     # the program is extracted (evaluation/extract.py)
    {"case_id": "...", "code": "<the edited program>"}              # used as is

and an optional "finish_reason". A reply that was cut off or refused ("length", "refused", ...; anything
other than "stop") counts as a failed edit, as does a task with no line. Each task is rendered from six
matched views (Cycles, CPU, 64 samples, 512x512) into <run>/renders/<case_id>/. A program that raises,
times out (480 s), or crashes Blender fails the task (error.log). Finished cases are skipped.

    python -m evaluation.render --run runs/my-model --predictions my-model.jsonl --blender /path/to/blender --workers 16
"""
import argparse
import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .data import add_task_filters, load_sources, load_tasks, read_json, tasks_from_args, write_json
from .extract import extract_program
from .prompts import VIEWS

BLENDER_SCRIPT = Path(__file__).resolve().parent / "blender_render.py"
RENDER_SETTINGS = {"samples": 64, "resolution": 512}
TIMEOUT_SEC = 480
ATTEMPTS = 3
# Crashes of Blender itself (not Python errors in the program) are retried.
CRASH_MARKERS = ("blender.crash.txt", "segmentation fault", "fatal python error", "aborted (core dumped)")


def ingest(predictions, tasks, run):
    """Write <run>/programs/<case_id>.json for every task: status "ok" with the program, or the failure."""
    rows = {}
    for line in open(predictions):
        if line.strip():
            row = json.loads(line)
            rows[row["case_id"]] = row
    unknown = set(rows) - {t["case_id"] for t in load_tasks()}
    if unknown:
        raise SystemExit(f"{len(unknown)} unknown case ids in {predictions}, e.g. {sorted(unknown)[0]}")
    status = Counter()
    for task in tasks:
        row = rows.get(task["case_id"])
        if row is None:
            rec = dict(status="missing")
        elif row.get("finish_reason") not in (None, "stop"):
            rec = dict(status=f"edit_failed:{row['finish_reason']}")
        elif "code" in row:
            rec = dict(status="ok", code=row["code"])
        else:
            code = extract_program(row.get("response"))
            rec = dict(status="ok", code=code) if code.strip() else dict(status="no_program")
        status[rec["status"]] += 1
        write_json(run / "programs" / f"{task['case_id']}.json", dict(case_id=task["case_id"], **rec))
    print("programs:", dict(status))


def check_blender(blender, python_paths):
    """Fail early unless Blender runs and its Python can import scipy (five source programs need it)."""
    code = f"import sys; sys.path[:0] = {python_paths!r}; import scipy"
    try:
        r = subprocess.run([blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python-expr", code],
                           capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise SystemExit(f"Blender not found at {blender!r}; pass --blender /path/to/blender (Blender 5.1.0).")
    if r.returncode != 0:
        raise SystemExit("Blender's Python cannot import scipy, which five source programs need. Install it with\n"
                         "  <blender dir>/5.1/python/bin/python3.13 -m pip install --target blender_deps scipy\n"
                         "in the repository folder (or pass its location with --blender-python-path).\n\n" + (r.stdout + r.stderr)[-2000:])


def render_done(case_dir):
    return (case_dir / "frame.json").exists() and all(
        (case_dir / f"{side}_{view}.png").exists() for side in ("original", "modified") for view in VIEWS)


def render_case(original, modified, case_dir, blender, python_paths):
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "original_script.py").write_text(original)
    (case_dir / "modified_script.py").write_text(modified)
    task = dict(original_script=str(case_dir / "original_script.py"), modified_script=str(case_dir / "modified_script.py"),
                output_dir=str(case_dir), views=VIEWS, render=RENDER_SETTINGS, target_coverage=0.95,
                python_paths=python_paths)
    write_json(case_dir / "task.json", task)
    try:
        compile(modified, str(case_dir / "modified_script.py"), "exec")
    except Exception as error:  # a program that does not compile is an execution failure without running Blender
        (case_dir / "error.log").write_text(f"{type(error).__name__}: {error}\n")
        return "error"
    cmd = [blender, "--background", "--threads", "1", "--python", str(BLENDER_SCRIPT), "--", "--task", str(case_dir / "task.json")]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    logs = []
    for attempt in range(1, ATTEMPTS + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC, env=env)
        except subprocess.TimeoutExpired:
            logs.append(f"=== attempt {attempt}: timed out after {TIMEOUT_SEC}s ===")
            break
        logs.append(f"=== attempt {attempt}: returncode {r.returncode} ===\n{r.stdout}\n{r.stderr}")
        if r.returncode == 0 and render_done(case_dir):
            (case_dir / "error.log").unlink(missing_ok=True)
            return "ok"
        crashed = r.returncode < 0 or any(m in (r.stdout + r.stderr).lower() for m in CRASH_MARKERS)
        if not crashed:
            break
    (case_dir / "error.log").write_text("\n\n".join(logs))
    return "error"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, help="editor replies (JSONL); omit to reuse <run>/programs")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    ap.add_argument("--blender-python-path", action="append", default=[],
                    help="extra site-packages for Blender's Python (default: ./blender_deps if it exists, see requirements.txt)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--retry-errors", action="store_true", help="re-render cases that previously failed")
    add_task_filters(ap)
    args = ap.parse_args()

    tasks = tasks_from_args(args)
    if args.predictions:
        ingest(args.predictions, tasks, args.run)
    sources = load_sources()
    todo = []
    for task in tasks:
        rec = read_json(args.run / "programs" / f"{task['case_id']}.json")
        case_dir = args.run / "renders" / task["case_id"]
        if rec is None or rec["status"] != "ok" or render_done(case_dir):
            continue
        if (case_dir / "error.log").exists() and not args.retry_errors:
            continue
        todo.append((sources[task["source_id"]], rec["code"], case_dir))
    print(f"{len(tasks)} tasks, {len(todo)} to render")
    if not args.blender_python_path and Path("blender_deps").is_dir():
        args.blender_python_path = ["blender_deps"]  # the folder suggested in requirements.txt
    python_paths = [str(Path(p).resolve()) for p in args.blender_python_path]
    if todo:
        check_blender(args.blender, python_paths)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(render_case, *job, args.blender, python_paths) for job in todo]
        print(dict(Counter(j.result() for j in tqdm(as_completed(jobs), total=len(jobs)))))


if __name__ == "__main__":
    main()
