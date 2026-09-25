"""Extract submittable agent sources from pulled Kaggle notebooks/scripts and fingerprint their lineage.

python -m tools.harvest_agents <pulled_dir> <out_dir>
For every pulled kernel (one sub-directory each) it looks for, in order:
  1. `%%writefile <x>.py` cells (the largest one that defines an agent callable),
  2. base64-embedded tar(.gz) archives containing main.py,
  3. `SOURCE_BYTES = b''.join((...))` byte literals,
  4. script kernels / single code cells that define `def agent(`.
Each extracted source is compile-checked, deduplicated by sha256 and written to <out_dir>/<kernel>.py together
with index.json (kernel, sha, size, lineage markers, entry callable name).
"""
import ast
import base64
import hashlib
import io
import json
import os
import re
import sys
import tarfile

MARKERS = ["V39", "shop-router", "Shop Router", "yhay81", "thomastschinkel", "tetsutani", "Ahmed Berat Ozer",
           "prvsiyan", "aurax7", "destbreso", "Chassis", "make_agent", "_R108_DATA", "native schedules",
           "kaitofukami", "boatlee", "raykkretzschmar", "pilkwang", "romantamrazov", "flexonafft"]


def cells(path):
    if path.endswith(".ipynb"):
        nb = json.load(open(path, encoding="utf-8"))
        return ["".join(c.get("source", "")) for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    return [open(path, encoding="utf-8", errors="ignore").read()]


def defines_agent(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and len(n.args.args) >= 1 and ("agent" in n.name or n.args.args[0].arg in ("obs", "observation")):
            return True
    return False


def from_writefile(cs):
    out = []
    for c in cs:
        m = re.match(r"\s*%%writefile\s+(-a\s+)?(\S+\.py)\s*\n", c)
        if m:
            body = c[m.end():]
            if defines_agent(body):
                out.append(body)
    return max(out, key=len) if out else None


def from_b64(cs):
    for c in cs:
        for blob in re.findall(r'"""([A-Za-z0-9+/=\s]{2000,})"""', c) + re.findall(r"'''([A-Za-z0-9+/=\s]{2000,})'''", c):
            try:
                data = base64.b64decode("".join(blob.split()))
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
                    for mname in t.getnames():
                        if mname.endswith("main.py"):
                            return t.extractfile(mname).read().decode("utf-8", "ignore")
            except Exception:
                continue
    return None


def from_source_bytes(cs):
    for c in cs:
        m = re.search(r"SOURCE_BYTES\s*=\s*b''\.join\(\((.*?)\)\)", c, re.S)
        if m:
            try:
                parts = ast.literal_eval("(" + m.group(1) + ")")
                return b"".join(parts).decode("utf-8", "ignore")
            except Exception:
                continue
    return None


def from_plain(cs):
    cands = [c for c in cs if not c.lstrip().startswith("%") and defines_agent(c)]
    return max(cands, key=len) if cands else None


def entry_of(src):
    env_order = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            if n.name not in env_order:
                env_order.append(n.name)
    return env_order[-1] if env_order else None


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    index, seen = [], {}
    for k in sorted(os.listdir(root)):
        d = os.path.join(root, k)
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith((".ipynb", ".py"))]
        if not files:
            continue
        cs = sum((cells(f) for f in files), [])
        src, how = None, None
        for fn in (from_writefile, from_b64, from_source_bytes, from_plain):
            src = fn(cs)
            if src:
                how = fn.__name__
                break
        if not src:
            index.append({"kernel": k, "status": "no_agent"})
            continue
        try:
            compile(src, k, "exec")
        except SyntaxError as e:
            index.append({"kernel": k, "status": f"syntax:{e.msg}"})
            continue
        sha = hashlib.sha256(src.encode()).hexdigest()[:16]
        rec = {"kernel": k, "status": "ok", "how": how, "sha": sha, "bytes": len(src), "lines": src.count("\n"),
               "markers": [mk for mk in MARKERS if mk in src], "dup_of": seen.get(sha)}
        if sha not in seen:
            seen[sha] = k
            open(os.path.join(out, k + ".py"), "w").write(src)
        index.append(rec)
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
    ok = [r for r in index if r["status"] == "ok"]
    print(f"{len(index)} kernels, {len(ok)} with agent code, {len(seen)} unique sources")


if __name__ == "__main__":
    main()
