"""Step 2: judge each checklist item from the six ORIGINAL/MODIFIED render pairs.

One verifier call per checklist item, each with the task's full `eval_query` as context. The verifier sees
the renders only, never the programs. A reply that cannot be parsed, or that hits the output limit, is
retried up to 5 more times.
Results: <run>/verdicts/<case_id>.json. The defaults are the paper's verifier (Gemini 3.8 Flash, thinking level high):

    GEMINI_API_KEY=... python -m evaluation.verify --run runs/my-model --workers 32
"""
import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .data import add_task_filters, read_json, tasks_from_args, write_json
from .llm import add_verifier_args, verifier_from_args
from .prompts import PRESERVATION_GUIDANCE, VERIFIER_SYSTEM, VERIFIER_USER, VIEWS

ATTEMPTS = 6


def parse_verdict(text):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty verifier response")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        for block in re.findall(r"```(?:json)?\s*\n?(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
            try:
                value = json.loads(block.strip())
                break
            except json.JSONDecodeError:
                continue
        else:
            raise error
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict) or set(value) != {"answer", "view", "evidence"}:
        raise ValueError("response must be one JSON object with exactly answer, view, evidence")
    answer, view, evidence = (value[k] for k in ("answer", "view", "evidence"))
    if not isinstance(answer, str) or answer.strip().lower() not in {"yes", "no"}:
        raise ValueError("answer must be yes or no")
    if not isinstance(view, str) or view.strip().lower() not in VIEWS:
        raise ValueError(f"view must be one of {list(VIEWS)}")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("evidence must be a non-empty string")
    return {"answer": answer.strip().lower(), "view": view.strip().lower(), "evidence": evidence.strip()}


def item_prompt(task, item):
    geometric = task["edit_type"] in ("G1", "G2", "G3", "G4")
    return VERIFIER_USER.format(
        full_query=task["eval_query"], query_clause=item["clause"], checklist_item=item["question"],
        item_guidance=PRESERVATION_GUIDANCE if geometric and item["preservation"] else "")


def case_images(case_dir):
    return [(f"[{side} — {view} view]", (case_dir / f"{side.lower()}_{view}.png").read_bytes())
            for view in VIEWS for side in ("ORIGINAL", "MODIFIED")]


def judge_item(llm, task, index, images):
    user = item_prompt(task, task["checklist"][index])
    attempts = []
    for _ in range(ATTEMPTS):
        reply = llm.complete(VERIFIER_SYSTEM, user, images)
        try:
            if reply.finish != "stop":
                raise ValueError(f"finish reason {reply.finish}")
            verdict = parse_verdict(reply.text)
            attempts.append(dict(text=reply.text, error=None))
            return dict(index=index, **verdict, attempts=attempts)
        except (json.JSONDecodeError, ValueError) as error:
            attempts.append(dict(text=reply.text, finish=reply.finish, error=f"{type(error).__name__}: {error}"))
    return dict(index=index, answer=None, view=None, evidence="", error="verifier_error", attempts=attempts)


def verify_case(llm, task, program, case_dir, path, pool):
    """Returns the case stage; failed edits and failed renders score 0 without verifier calls."""
    record = dict(case_id=task["case_id"], edit_type=task["edit_type"], category=task["category"], verifier=llm.model,
                  program_status=program["status"])
    frame = read_json(case_dir / "frame.json") if program["status"] == "ok" else None
    if program["status"] != "ok":
        record.update(stage="edit_failed", passed=False)
    elif frame is None:
        record.update(stage="render_failed", passed=False)
    elif not frame["modified_has_geometry"]:
        record.update(stage="empty_scene", passed=False)
    else:
        images = case_images(case_dir)
        items = [f.result() for f in [pool.submit(judge_item, llm, task, i, images) for i in range(len(task["checklist"]))]]
        errors = sum(item["answer"] is None for item in items)
        record.update(stage="verifier_error" if errors else "judged", items=items,
                      passed=not errors and all(item["answer"] == "yes" for item in items))
    write_json(path, record)
    return record["stage"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16, help="concurrent verifier calls")
    ap.add_argument("--retry-errors", action="store_true", help="re-judge cases that ended in verifier_error")
    add_task_filters(ap)
    add_verifier_args(ap)
    args = ap.parse_args()

    llm = verifier_from_args(args)
    todo = []
    for task in tasks_from_args(args):
        program = read_json(args.run / "programs" / f"{task['case_id']}.json")
        if program is None:
            continue
        out = args.run / "verdicts" / f"{task['case_id']}.json"
        done = read_json(out)
        if done and not (args.retry_errors and done["stage"] == "verifier_error"):
            continue
        case_dir = args.run / "renders" / task["case_id"]
        if program["status"] == "ok" and not (case_dir / "frame.json").exists() and not (case_dir / "error.log").exists():
            continue  # not rendered yet
        todo.append((task, program, case_dir, out))
    print(f"{len(todo)} cases to verify")
    # Items go to an inner pool; cases are fed a few at a time so items from one case run together.
    with ThreadPoolExecutor(args.workers) as item_pool, ThreadPoolExecutor(max(1, args.workers // 4)) as case_pool:
        jobs = [case_pool.submit(verify_case, llm, *job, item_pool) for job in todo]
        print(dict(Counter(j.result() for j in tqdm(as_completed(jobs), total=len(jobs)))))


if __name__ == "__main__":
    main()
