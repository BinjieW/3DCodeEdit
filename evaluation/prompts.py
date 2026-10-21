"""Frozen editor and verifier prompts.

Write the editor prompt of every task (the paper's single-shot prompt) as JSONL, one
{"case_id", "system", "user"} object per line, for running your own editor:

    python -m evaluation.prompts > prompts.jsonl
"""

EDITOR_SYSTEM = """\
You are an expert Blender Python programmer. Given a base script and a
modification instruction, return the complete modified script implementing every
requested clause. Preserve unrelated behavior, definitions, imports, helpers,
and geometry. Make no additional visible edits beyond those required by the
instruction.

Return only Python code.
"""

EDITOR_USER = """\
## Base Script
```python
{script}
```

## Modification Instruction
{query}

Return the complete modified Blender Python script.
"""

VIEWS = {  # name: (camera azimuth, elevation) in degrees; azimuth 0 puts the camera on -Y, 90 on +X
    "front": (0, 20),
    "back": (180, 20),
    "left": (270, 20),
    "right": (90, 20),
    "top": (0, 85),
    "bottom": (0, -85),
}

VERIFIER_SYSTEM = """\
You are a visual evaluation judge for 3D rendered objects. You receive six
matched ORIGINAL/MODIFIED view pairs, the full modification query, one aligned
query clause, and one checklist item. Inspect all six pairs and judge only the
current checklist item from directly visible evidence.

Use the full query only to understand intended edits, target roles, and which
parts are targeted versus non-targeted. The query is not evidence that an edit
succeeded. Do not score any other clause, and do not let success or failure of
another requested edit change the current answer.

The checklist item is the complete success criterion. Answer "yes" only when
every explicit qualifier in it is visibly satisfied. Ground the target in the
ORIGINAL renders using its shape and stable neighboring geometry, then compare
the same target role in MODIFIED. Hidden geometry, implementation state, names,
metadata, and inferred connectivity are not visual evidence. For relative
changes, compare matching camera directions. For counts and relationships,
require visibly distinguishable members and direct spatial evidence.
"""

# Added to checklist item 0 of G1-G4 tasks (the framework-owned preservation item).
PRESERVATION_GUIDANCE = """\
This is the framework-owned preservation item. Treat every change explicitly
requested anywhere in the full query as intended and acceptable, even when the
change is substantial. Do not require targeted parts to retain their original
geometry. Answer "no" only when major non-targeted parts are severely corrupted
or lost, or when the source object becomes unidentifiable or visually
incoherent.
"""

VERIFIER_USER = """\
## Full modification query (context only; not visual evidence)
{full_query}

## Current independently scored query clause
{query_clause}

## Current checklist item
{checklist_item}

{item_guidance}
Six labeled matched ORIGINAL/MODIFIED pairs are attached: front, back, left,
right, top, and bottom. Every pair uses the same camera direction and shared
union framing. Inspect all pairs and cite the one with the clearest direct
evidence.

Return one JSON object with exactly these fields:
```json
{{
  "answer": "yes",
  "view": "right",
  "evidence": "A concise matched-view comparison supporting the answer."
}}
```

`answer` is exactly `yes` or `no`. `view` is exactly one of `front`, `back`,
`left`, `right`, `top`, or `bottom`. For a negative answer, cite the view that
most clearly demonstrates the missing or contradictory result. Return only the
JSON object.
"""


def editor_messages(task, source):
    return EDITOR_SYSTEM, EDITOR_USER.format(script=source, query=task["instruction"])


if __name__ == "__main__":
    import json
    import sys

    from .data import load_sources, load_tasks

    sources = load_sources()
    for task in load_tasks():
        system, user = editor_messages(task, sources[task["source_id"]])
        sys.stdout.write(json.dumps(dict(case_id=task["case_id"], system=system, user=user), ensure_ascii=False) + "\n")
