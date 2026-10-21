"""Step 3: aggregate verdicts into all-pass rates.

A task passes only if the edited program was extracted, it executed and rendered a non-empty scene, and every
checklist item was judged "yes". Tasks without a verdict count as failures, so partial runs are
reported as such.

    python -m evaluation.score runs/my-model [runs/other-model ...]
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from .data import EDIT_TYPES, add_task_filters, read_json, tasks_from_args, write_json


def score_run(run, tasks):
    by_type, by_category = defaultdict(list), defaultdict(list)
    stages = Counter()
    for task in tasks:
        verdict = read_json(run / "verdicts" / f"{task['case_id']}.json")
        program = read_json(run / "programs" / f"{task['case_id']}.json")
        if verdict:
            stage = verdict["stage"]
        elif program and program["status"] != "ok":
            stage = "edit_failed"  # fails without verification
        else:
            stage = "not_verified"
        passed = bool(verdict and verdict["passed"])
        stages[stage] += 1
        by_type[task["edit_type"]].append(passed)
        by_category[task["category"]].append(passed)
    rate = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    executed = stages["judged"] + stages["verifier_error"]
    return {
        "run": run.name,
        "tasks": len(tasks),
        "all_pass": rate([p for v in by_type.values() for p in v]),
        "by_edit_type": {k: rate(by_type[k]) for k in EDIT_TYPES if by_type[k]},
        "by_category": {k: rate(v) for k, v in sorted(by_category.items())},
        "exec_rate": round(100 * executed / len(tasks), 1),
        "stages": dict(stages),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    add_task_filters(ap)
    args = ap.parse_args()
    tasks = tasks_from_args(args)
    rows = [score_run(run, tasks) for run in args.runs]
    header = ["run", "all-pass", *rows[0]["by_edit_type"], "exec"]
    print(" | ".join(header))
    for run, r in zip(args.runs, rows):
        print(" | ".join(str(x) for x in [r["run"], r["all_pass"], *r["by_edit_type"].values(), r["exec_rate"]]))
        write_json(run / "score.json", r)
        missing = r["stages"].get("not_verified", 0)
        if missing:
            print(f"  warning: {missing} of {r['tasks']} tasks have no verdict and count as failures")


if __name__ == "__main__":
    main()
