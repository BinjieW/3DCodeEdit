"""Extract the edited program from an editor reply (the rule used for all results in the paper).

Applied to the reply after dropping everything up to the first </think>:
 1. Complete fenced blocks tagged python/py/python3/bpy or untagged: the longest one that compiles
    (if none compiles, the longest one).
 2. An opening fence that is never closed: everything after it, if that compiles.
 3. Otherwise: drop leading lines until the first line that looks like Python, and drop trailing fence
    markers, blank lines and prose sentences, never real code lines.
"""
import re

FENCE = re.compile(r'^[ \t]*```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)^[ \t]*```[ \t]*$', re.S | re.M)
OPEN = re.compile(r'^[ \t]*```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n', re.M)
PROSE = re.compile(r'^(?!\s)(?!(import|from|def|class|return|if|for|while|with|try|except|else|elif|print|raise|assert|del|pass)\b)'
                   r'[A-Z*>-][^=()\[\]{}]*\s\S+\s\S+[^=()\[\]{}:,]*$')
CODE_START = re.compile(r'^(import |from \S+ import |def |class |#|@|bpy\.|if __name__|"""|\'\'\'|'
                        r'[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\s*(=|\+=|\()|\s*$)')
PY_TAGS = {"", "python", "py", "python3", "bpy"}


def compiles(code):
    try:
        compile(code, "<edited>", "exec")
        return True
    except Exception:  # SyntaxError, ValueError, or RecursionError on pathological input
        return False


def extract_program(reply):
    """Returns the extracted program ("" if nothing usable)."""
    text = (reply or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    blocks = sorted((b for tag, b in FENCE.findall(text) if tag.lower() in PY_TAGS), key=len, reverse=True)
    if blocks:
        return next((b for b in blocks if compiles(b)), blocks[0]).strip("\n")
    m = OPEN.search(text)
    if m and m.group(1).lower() in PY_TAGS and compiles(text[m.end():]):
        return text[m.end():].strip("\n")
    lines = text.split("\n")
    start = 0
    while start < len(lines) and not (lines[start].strip() and CODE_START.match(lines[start])):
        start += 1
    body = lines[start:]
    while body and (not body[-1].strip() or body[-1].strip().startswith("```") or PROSE.match(body[-1])):
        body.pop()
    return "\n".join(body).strip("\n")
