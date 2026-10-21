"""Benchmark data loading and small file helpers."""
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDIT_TYPES = ("G1", "G2", "G3", "G4", "S", "C")


def load_tasks(data_dir=DATA_DIR, edit_types=None, categories=None, subset=None):
    """subset: None for all 2,592 tasks, or "analysis" for the 312-task analysis subset."""
    tasks = [json.loads(line) for line in open(data_dir / "tasks.jsonl")]
    if edit_types:
        tasks = [t for t in tasks if t["edit_type"] in edit_types]
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]
    if subset == "analysis":
        keep = set(json.load(open(data_dir / "analysis_subset.json"))["case_ids"])
        tasks = [t for t in tasks if t["case_id"] in keep]
    elif subset:
        raise ValueError(f"unknown subset {subset!r}")
    return tasks


def load_sources(data_dir=DATA_DIR):
    """source_id -> original Blender program."""
    rows = [json.loads(line) for line in open(data_dir / "sources.jsonl")]
    return {r["source_id"]: (data_dir / r["script"]).read_text() for r in rows}


def add_task_filters(parser):
    parser.add_argument("--edit-types", nargs="+", choices=EDIT_TYPES)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--subset", choices=("analysis",), help="only the 312-task analysis subset")


def tasks_from_args(args):
    return load_tasks(edit_types=args.edit_types, categories=args.categories, subset=args.subset)


def write_json(path, value):
    """Atomic write, so an interrupted run never leaves a half-written record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None
