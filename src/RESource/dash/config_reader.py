"""Read RESource YAML configs supplied next to the stores.

Scenario files carry ``extends: ../base.yaml``. The chain is resolved by file name
among the files the user supplied; a parent that was not supplied is reported, not
guessed.
"""

from __future__ import annotations

import json

import yaml


def _merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = _merge(base.get(k), v) if k in base else v
        return out
    return over


def _basename(p):
    return str(p).replace("\\", "/").rsplit("/", 1)[-1]


def read_configs(texts):
    """``texts`` maps a relative path to YAML text. Returns a JSON list of resolved configs."""
    parsed, errors = {}, {}
    for path, text in texts.items():
        try:
            doc = yaml.safe_load(text)
            if isinstance(doc, dict):
                parsed[path] = doc
        except yaml.YAMLError as exc:
            errors[path] = str(exc)[:300]
    by_name = {}
    for p in parsed:
        by_name.setdefault(_basename(p), []).append(p)

    def _norm(p):
        out = []
        for part in str(p).replace("\\", "/").split("/"):
            if part == "..":
                if out:
                    out.pop()
            elif part not in ("", "."):
                out.append(part)
        return "/".join(out)

    norm = {_norm(p): p for p in parsed}

    def parent_of(path, ext):
        here = _norm(path)
        if "/" in here:  # files came with folders: resolve the relative path exactly
            return norm.get(_norm(here.rsplit("/", 1)[0] + "/" + str(ext)))
        cands = by_name.get(_basename(ext), [])  # loose files: accept a unique name match
        return cands[0] if len(cands) == 1 else None

    def resolve(path, seen=()):
        doc = parsed[path]
        ext = doc.get("extends")
        chain, missing = [path], []
        if ext and path not in seen:
            par = parent_of(path, ext)
            if par and par != path:
                base, pchain, pmissing = resolve(par, seen + (path,))
                merged = _merge(base, {k: v for k, v in doc.items() if k != "extends"})
                return merged, pchain + chain, pmissing
            missing.append(str(ext))
        return {k: v for k, v in doc.items() if k != "extends"}, chain, missing

    out = []
    for path in parsed:
        merged, chain, missing = resolve(path)
        sc = merged.get("Scenario") if isinstance(merged.get("Scenario"), dict) else {}
        out.append({"path": path, "chain": chain, "missing_parents": missing,
                    "run_id": sc.get("run_id"), "description": sc.get("Description"),
                    "country": merged.get("country"), "is_scenario": "extends" in parsed[path],
                    "config": merged})
    return json.dumps({"configs": out, "errors": errors}, default=str)
