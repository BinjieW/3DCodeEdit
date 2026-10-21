"""Example inference: send each task's prompt to a model once and save the replies for evaluation.render.

Queries an OpenAI-compatible chat endpoint with the paper's settings (one call per task, at most 64,000
output tokens, reasoning effort medium). Finished tasks are skipped, so an interrupted run can be resumed.

    OPENAI_API_KEY=... python -m evaluation.inference --model my-model --out my-model.jsonl
"""
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .data import add_task_filters, load_sources, tasks_from_args
from .llm import OpenAILLM
from .prompts import editor_messages

EMPTY_RETRIES = 10  # only empty replies are retried


def run_task(llm, task, source):
    system, user = editor_messages(task, source)
    for _ in range(1 + EMPTY_RETRIES):
        reply = llm.complete(system, user)
        if reply.text.strip() or reply.finish != "stop":
            break
    return dict(case_id=task["case_id"], response=reply.text, finish_reason=reply.finish)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True, help="predictions file (JSONL), appended to")
    ap.add_argument("--base-url", help="defaults to $OPENAI_BASE_URL or api.openai.com")
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="only the first N tasks")
    add_task_filters(ap)
    args = ap.parse_args()

    llm = OpenAILLM(args.model, base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY"),
                    max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
    tasks = tasks_from_args(args)[:args.limit]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["case_id"] for line in open(args.out) if line.strip()}
    todo = [t for t in tasks if t["case_id"] not in done]
    print(f"{len(tasks)} tasks, {len(todo)} to run")
    sources = load_sources()
    finish = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool, open(args.out, "a") as out:
        jobs = [pool.submit(run_task, llm, t, sources[t["source_id"]]) for t in todo]
        for job in tqdm(as_completed(jobs), total=len(jobs)):
            row = job.result()
            finish[row["finish_reason"]] += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
    print("finish reasons:", dict(finish))


if __name__ == "__main__":
    main()
