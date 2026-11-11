"""Reproducible no-human workflow experiments. No credentials embedded or printed.

audit: recompute legacy image metrics, not a new-model effectiveness experiment.
preflight: inventory and blockers; never calls a network endpoint.
live: explicit bounded API run, requires real input manifest and configured endpoints.
"""
import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_local_credentials():
    """Load only the local OpenRouter key; never execute dotenv contents."""
    path = ROOT / "runtime" / "openrouter.env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "OPENROUTER_API_KEY":
                value = value.strip().strip('\"').strip("'")
                if value and not os.environ.get("OPENROUTER_API_KEY"):
                    os.environ["OPENROUTER_API_KEY"] = value


def endpoint(cfg, kind):
    url = os.environ.get(cfg[kind + "_endpoint_env"], "").strip() or cfg.get(kind + "_endpoint", "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query:
        raise ValueError("Endpoint must be HTTPS without credentials or query parameters")
    if cfg.get("backend") == "openrouter" and parsed.hostname != "openrouter.ai":
        raise ValueError("OpenRouter credentials may only be sent to openrouter.ai")
    return url


def output_bytes(data, cfg):
    if cfg.get("backend") == "openrouter":
        items = data.get("data", [])
        if len(items) != 1 or not items[0].get("b64_json"):
            raise ValueError("Expected exactly one base64 OpenRouter image")
        return base64.b64decode(items[0]["b64_json"], validate=True)
    import requests
    items = data["output"]["choices"][0]["message"]["content"]
    urls = [item["image"] for item in items if "image" in item]
    if len(urls) != 1 or urlparse(urls[0]).scheme != "https":
        raise ValueError("Expected exactly one HTTPS output image")
    downloaded = requests.get(urls[0], timeout=cfg["timeout_seconds"])
    downloaded.raise_for_status()
    return downloaded.content


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def fields_check(obj, fields):
    if not isinstance(obj, dict) or set(obj) != set(fields):
        raise ValueError("Expected exactly the configured clothing fields")
    if any(v is not None and not isinstance(v, str) for v in obj.values()):
        raise ValueError("Field values must be strings or null")
    return {k: obj[k] for k in fields}


def resolve(profile, explicit, fields):
    fields_check(profile, fields)
    if not isinstance(explicit, dict) or set(explicit) - set(fields):
        raise ValueError("Invalid explicit field keys")
    if any(not isinstance(v, str) or not v.strip() for v in explicit.values()):
        raise ValueError("Explicit values must be nonempty strings")
    return {k: explicit[k] if k in explicit else profile[k] for k in fields}


def flatten(profile, fields):
    return "；".join(profile[k] for k in fields if profile[k] is not None)


def file_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Manifest references a missing local image")
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def image_uri(path):
    from PIL import Image
    with Image.open(path) as img:
        mime = Image.MIME.get(img.format)
        img.verify()
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise ValueError("Only verified PNG/JPEG/WebP supported")
    return "data:" + mime + ";base64," + base64.b64encode(Path(path).read_bytes()).decode()


def validate_tasks(tasks, cfg):
    if not tasks:
        raise ValueError("No real reference/task manifest: live experiment blocked")
    ids = set()
    ref_hashes, eval_hashes, ref_groups, eval_groups = set(), set(), set(), set()
    for t in tasks:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", t.get("task_id", "")) or t["task_id"] in ids:
            raise ValueError("Task IDs must be unique safe identifiers")
        ids.add(t["task_id"])
        if t.get("split") not in ("development", "test"):
            raise ValueError("Task split must be development or test")
        if not isinstance(t.get("brief"), str) or not t["brief"].strip():
            raise ValueError("Task brief is required")
        for key in ("hard_constraints", "preserve"):
            if not isinstance(t.get(key), list) or any(not isinstance(s, str) for s in t[key]):
                raise ValueError("Constraint lists must contain strings")
        if not t["hard_constraints"]:
            raise ValueError("Task requires observable constraints")
        if not isinstance(t.get("explicit_fields"), dict):
            raise ValueError("Predefined explicit_fields are required")
        refs = t.get("reference_images", [])
        evs = t.get("evaluation_images", [])
        if not refs or not evs:
            raise ValueError("Both reference and held-out evaluation images required")
        count = len(refs) + bool(t.get("input_image"))
        if count > cfg["max_input_images"]:
            raise ValueError("Common image-input limit exceeded; never silently drop references")
        for records, hashes, groups in ((refs, ref_hashes, ref_groups), (evs, eval_hashes, eval_groups)):
            for entry in records:
                if not all(entry.get(k) for k in ("image_id", "path", "source", "rights_status", "garment_group_id")):
                    raise ValueError("Image provenance and garment_group_id required")
                if entry["rights_status"] not in ("owned", "licensed_for_research_api", "public_domain"):
                    raise ValueError("Image usage permission is unresolved")
                hashes.add(file_record(entry["path"])["sha256"])
                groups.add(entry["garment_group_id"])
                image_uri(entry["path"])
        if t.get("input_image"):
            image_uri(t["input_image"])
    if ref_hashes & eval_hashes or ref_groups & eval_groups:
        raise ValueError("Reference/evaluation image or garment-group leakage")


def preflight(cfg, tasks):
    blockers = []
    names = (cfg["api_key_env"],)
    presence = {name: bool(os.environ.get(name, "").strip()) for name in names}
    blockers.extend("Missing environment variable: " + n for n, present in presence.items() if not present)
    for kind in ("vlm", "image"):
        try:
            endpoint(cfg, kind)
        except ValueError as exc:
            blockers.append(kind + ": " + str(exc))
    try:
        validate_tasks(tasks, cfg)
    except (ValueError, OSError) as exc:
        blockers.append(str(exc))
    return {"timestamp_utc": stamp(), "status": "blocked" if blockers else "ready_for_development_probe",
            "network_calls": 0, "credential_values_logged": False, "configuration_presence": presence,
            "models": {k: cfg[k] for k in ("vlm_model", "image_model")},
            "task_count": len(tasks), "local_image_count": len(list((HERE / "input").glob("*.*"))),
            "modules": {m: importlib.util.find_spec(m) is not None for m in ("numpy", "cv2", "requests", "torch", "transformers")},
            "blockers": blockers,
            "note": "Legacy generated images are not a reference style dataset."}


def metric_values(rgb):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    prob = hist / hist.sum()
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = gray.shape
    mask = np.ones((h, w))
    mask[h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8] = 0
    return {"sobel_variance": float(np.var(np.sqrt(sx*sx + sy*sy))),
            "contrast": float(np.sqrt(np.mean((gray - np.mean(gray))**2))),
            "entropy": float(-np.sum(prob * np.log2(prob + 1e-7))),
            "blur_score": float((mag * mask).sum() / mag.sum()) if mag.sum() else None}


def audit(out):
    import cv2
    import numpy as np
    from PIL import Image
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    legacy = {}
    with (HERE / "image_quality_results.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            legacy[row["filename"]] = {k: float(row[k]) for k in ("sobel_variance", "contrast", "entropy", "blur_score")}
    records = []
    for path in sorted((HERE / "input").iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        with Image.open(path) as im:
            fmt, size = im.format, im.size
            im.verify()
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError("Cannot decode image: " + path.name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        values = metric_values(rgb)
        matched = {}
        for key, value in values.items():
            old = legacy.get(path.name, {}).get(key)
            # CSV is rounded; permit 0.1% numerical difference, not substantive drift.
            matched[key] = {"csv": old, "recomputed": value,
                            "difference": None if old is None or value is None else value-old,
                            "matches": old is not None and value is not None and math.isclose(old, value, rel_tol=1e-3, abs_tol=1e-5)}
        # Sensitivity only: decode/re-encode in memory. Original files remain unchanged.
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError("JPEG sensitivity encode failed")
        jpg_rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        scale = min(1.0, 512 / max(size))
        resized = cv2.resize(rgb, (max(1, round(size[0]*scale)), max(1, round(size[1]*scale))), interpolation=cv2.INTER_AREA)
        records.append({"filename": path.name, "sha256": digest(path), "format": fmt,
                        "width": size[0], "height": size[1], "method_label": "unverified",
                        "original": values, "comparison": matched,
                        "jpeg95_in_memory": metric_values(jpg_rgb),
                        "long_edge_max512": metric_values(resized)})
    report = {"timestamp_utc": stamp(), "kind": "legacy_reproducibility_and_sensitivity_audit",
              "status": "completed", "new_generation_count": 0, "human_evaluation_count": 0,
              "opencv": cv2.__version__, "numpy": np.__version__,
              "matched_counts": {k: sum(r["comparison"][k]["matches"] for r in records) for k in legacy[next(iter(legacy))]},
              "image_count": len(records), "records": records,
              "limitations": ["No verified original/reference/generated mapping", "No designer style or preference conclusion",
                              "Metrics depend on encoding and resolution", "No new-model baseline has been executed"]}
    save(out / "legacy_quality_audit.json", report)
    with (out / "legacy_quality_audit.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        cols = ["filename", "format", "width", "height", "metric", "csv", "recomputed", "difference", "matches", "jpeg95", "max512"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in records:
            for key, c in r["comparison"].items():
                writer.writerow({**{k: r[k] for k in ("filename", "format", "width", "height")},
                                 "metric": key, **c, "jpeg95": r["jpeg95_in_memory"][key], "max512": r["long_edge_max512"][key]})
    print(json.dumps({"images": len(records), "matched_counts": report["matched_counts"], "output": str(out)}, ensure_ascii=False))
    return report


def render(task, condition, profile, summary, prompts):
    fields = prompts["fields"]
    resolved = resolve(profile, task["explicit_fields"], fields)
    content = ""
    if condition == "B":
        content = summary
    elif condition == "C":
        content = json.dumps(resolved, ensure_ascii=False)
    elif condition == "D":
        content = flatten(resolved, fields)
    elif condition == "C_no_priority":
        content = json.dumps(profile, ensure_ascii=False)
    elif condition != "A":
        raise ValueError("Unknown condition")
    roles = []
    if task.get("input_image"):
        roles.append("图1为待编辑对象")
    for i, entry in enumerate(task["reference_images"], start=1 + bool(task.get("input_image"))):
        roles.append(f"图{i}为服装参考{entry['image_id']}")
    return prompts["common"].format(brief=task["brief"], hard_constraints=json.dumps(task["hard_constraints"], ensure_ascii=False),
             preserve=json.dumps(task["preserve"], ensure_ascii=False), roles="；".join(roles),
             condition_block=prompts[condition].format(content=content))


class Client:
    def __init__(self, cfg, out):
        self.cfg, self.out, self.last, self.calls = cfg, Path(out), 0.0, 0
        self.key = os.environ[cfg["api_key_env"]]

    def request(self, kind, payload):
        import requests
        env = self.cfg["vlm_endpoint_env"] if kind == "vlm" else self.cfg["image_endpoint_env"]
        url = endpoint(self.cfg, "vlm" if kind == "vlm" else "image")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query:
            raise ValueError("Endpoint must be HTTPS without credentials or query parameters")
        delay = self.cfg["min_interval_seconds"] - (time.monotonic() - self.last)
        if delay > 0:
            time.sleep(delay)
        self.calls += 1
        if self.cfg.get("backend") == "openrouter":
            routing = self.cfg.get("image_provider", self.cfg.get("provider", {"allow_fallbacks": False})) if kind == "image" else self.cfg.get("provider", {"allow_fallbacks": False})
            payload = dict(payload, provider=routing)
        call_id = f"call_{self.calls:05d}"
        start = time.monotonic()
        try:
            response = requests.post(url, headers={"Authorization": "Bearer " + self.key}, json=payload,
                                     timeout=(self.cfg.get("connect_timeout_seconds", 15), self.cfg["timeout_seconds"]), allow_redirects=False)
            meta = {"call_id": call_id, "kind": kind, "requested_model": payload["model"],
                    "endpoint_host": parsed.hostname, "timestamp": stamp(), "http_status": response.status_code,
                    "elapsed_seconds": time.monotonic()-start}
            save(self.out / (call_id + ".metadata.json"), meta)
            if response.status_code != 200:
                try:
                    error = response.json().get("error", {})
                except ValueError:
                    error = {}
                save(self.out / (call_id + ".error.json"), {"kind": kind, "http_status": response.status_code,
                     "error_code": error.get("code"), "error_message": str(error.get("message", "Non-JSON HTTP error")).replace(self.key, "[REDACTED]")[:1200],
                     "timestamp": stamp(), "elapsed_seconds": time.monotonic()-start})
                raise RuntimeError(f"HTTP {response.status_code}; stopped without automatic retry")
            result = response.json()
            # Store usage and response identifiers, not authorization headers or payload base64.
            meta.update({"response_id": result.get("request_id", result.get("id")), "usage": result.get("usage"),
                         "returned_model": result.get("model"), "provider": result.get("provider")})
            save(self.out / (call_id + ".metadata.json"), meta)
            return result
        except requests.RequestException as exc:
            chain, current = [], exc
            for _ in range(8):
                if current is None:
                    break
                chain.append(type(current).__name__)
                current = current.__cause__ or current.__context__
            save(self.out / (call_id + ".error.json"), {"kind": kind, "error": "network_error", "exception_type": type(exc).__name__,
                 "exception_chain_types": chain,
                 "timestamp": stamp(), "elapsed_seconds": time.monotonic()-start,
                 "connect_timeout_seconds": self.cfg.get("connect_timeout_seconds", 15), "read_timeout_seconds": self.cfg["timeout_seconds"]})
            raise RuntimeError("Network error; stopped without automatic retry") from None
        finally:
            self.last = time.monotonic()

    def text(self, prompt, paths):
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": image_uri(p)}} for p in paths)
        data = self.request("vlm", {"model": self.cfg["vlm_model"], "messages": [{"role": "user", "content": content}],
                                   "max_tokens": self.cfg.get("text_max_tokens", 2048), **self.cfg.get("vlm_extra", {})})
        if data["choices"][0].get("finish_reason") == "length":
            raise ValueError("Text truncated by token limit; no silent acceptance")
        answer = data["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model returned no text")
        save(self.out / f"call_{self.calls:05d}.text.json", {"prompt": prompt, "response": answer,
                                                          "images": [file_record(p) for p in paths]})
        return answer

    def image(self, prompt, paths, seed=None):
        if self.cfg.get("backend") == "openrouter":
            payload = {"model": self.cfg["image_model"], "prompt": prompt,
                "input_references": [{"type": "image_url", "image_url": {"url": image_uri(p)}} for p in paths],
                **self.cfg["image_parameters"]}
            if seed is not None:
                payload["seed"] = seed
            return self.request("image", payload)
        content = [{"image": image_uri(p)} for p in paths] + [{"text": prompt}]
        return self.request("image", {"model": self.cfg["image_model"],
                      "input": {"messages": [{"role": "user", "content": content}]},
                      "parameters": self.cfg["image_parameters"]})


def strict_json(text):
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    return json.loads(stripped)


def judge_checks(answer, expected_ids):
    if not isinstance(answer, dict) or not isinstance(answer.get("checks"), list):
        raise ValueError("Judge must return checks array")
    rows = answer["checks"]
    if len(rows) != len(expected_ids) or {r.get("constraint_id") for r in rows} != set(expected_ids):
        raise ValueError("Judge omitted or duplicated constraint IDs")
    if any(r.get("status") not in ("met", "violated", "unknown") or not isinstance(r.get("evidence"), str) for r in rows):
        raise ValueError("Judge status/evidence invalid")
    total = len(rows)
    met = sum(r["status"] == "met" for r in rows)
    unknown = sum(r["status"] == "unknown" for r in rows)
    return {"constraint_count": total, "met_count": met, "unknown_count": unknown,
            "conservative_adherence": met/total, "unknown_rate": unknown/total,
            "known_only_adherence": met/(total-unknown) if total > unknown else None}


def judge(cfg, prompts, runs, out, confirmed):
    if not confirmed:
        raise ValueError("Explicit --execute-api required")
    if cfg.get("backend") == "openrouter":
        cfg = dict(cfg, judge_model=os.environ.get("FASHION_JUDGE_MODEL") or cfg["judge_model"])
        required = (cfg["api_key_env"],)
    else:
        required = ("FASHION_JUDGE_API_KEY", "FASHION_JUDGE_ENDPOINT", "FASHION_JUDGE_MODEL")
    if any(not os.environ.get(k) for k in required):
        raise ValueError("Missing independent judge configuration: " + ", ".join(required))
    runs, out = Path(runs), Path(out)
    source_cfg = load(runs / "frozen_config.json")
    tasks = {t["task_id"]: t for t in load(runs / "frozen_tasks.json")}
    judge_model = cfg["judge_model"] if cfg.get("backend") == "openrouter" else os.environ["FASHION_JUDGE_MODEL"]
    if judge_model in (source_cfg["vlm_model"], source_cfg["image_model"]):
        raise ValueError("Judge model must differ from generator and profiler")
    if cfg.get("backend") != "openrouter":
        cfg = dict(cfg, api_key_env="FASHION_JUDGE_API_KEY", vlm_endpoint_env="FASHION_JUDGE_ENDPOINT")
    cfg = dict(cfg, vlm_model=judge_model, vlm_extra={})
    out.mkdir(parents=True, exist_ok=False)
    client = Client(cfg, out)
    rows = []
    save(out / "status.json", {"status": "running", "kind": "machine_constraint_evaluation"})
    try:
        for path in sorted(runs.glob("*.json")):
            record = load(path)
            if not isinstance(record, dict) or record.get("status") != "completed" or "run_id" not in record:
                continue
            task = tasks[record["task_id"]]
            constraints = [{"constraint_id": f"c{i+1}", "text": s} for i, s in enumerate(task["hard_constraints"])]
            prompt = prompts["judge"].format(task=json.dumps({"brief": task["brief"], "constraints": constraints}, ensure_ascii=False))
            # Judge gets no method label, extraction text, historical score, or expected outcome.
            answer = strict_json(client.text(prompt, [record["output"]["path"]]))
            score = judge_checks(answer, [c["constraint_id"] for c in constraints])
            result = {"run_id": record["run_id"], "task_id": record["task_id"], "condition": record["condition"],
                      "judge_model": judge_model, "kind": "machine_proxy_only", "raw": answer, **score}
            save(out / (record["run_id"] + ".judge.json"), result)
            rows.append(result)
        if not rows:
            raise ValueError("No completed runs available for evaluation")
        save(out / "machine_scores.json", rows)
        save(out / "status.json", {"status": "completed", "scored_images": len(rows), "human_evaluations": 0})
    except Exception as exc:
        save(out / "status.json", {"status": "stopped", "scored_images": len(rows), "error_type": type(exc).__name__})
        raise RuntimeError("Judge stopped; inspect records, no automatic retries") from None


def style_checks(answer, expected_dimensions):
    if not isinstance(answer, dict) or set(answer) != {"style_checks"} or not isinstance(answer["style_checks"], list):
        raise ValueError("Expected style_checks object")
    rows = answer["style_checks"]
    if len(rows) != len(expected_dimensions) or {r.get("dimension") for r in rows} != set(expected_dimensions):
        raise ValueError("Missing or duplicate style dimensions")
    weights = {"retained": 1.0, "partial": 0.5, "not_retained": 0.0, "unknown": 0.0}
    for row in rows:
        if row.get("status") not in weights or any(not isinstance(row.get(k), str) for k in ("reference_evidence", "generated_evidence")):
            raise ValueError("Invalid style status or evidence")
    count = len(rows)
    unknown = sum(r["status"] == "unknown" for r in rows)
    total = sum(weights[r["status"]] for r in rows)
    return {"eligible_dimensions": count, "style_retention_proxy": total/count if count else None,
            "unknown_rate": unknown/count if count else None,
            "known_only_style_proxy": total/(count-unknown) if count > unknown else None}


def style_judge(cfg, prompts, runs, out, confirmed):
    if not confirmed or cfg.get("backend") != "openrouter":
        raise ValueError("Explicit OpenRouter style evaluation required")
    runs, out = Path(runs), Path(out)
    source_cfg = load(runs / "frozen_config.json")
    tasks = {t["task_id"]: t for t in load(runs / "frozen_tasks.json")}
    model = os.environ.get("FASHION_JUDGE_MODEL") or cfg["judge_model"]
    if model in (source_cfg["vlm_model"], source_cfg["image_model"]):
        raise ValueError("Style judge must differ from profiler and generator")
    out.mkdir(parents=True, exist_ok=False)
    client = Client(dict(cfg, vlm_model=model, vlm_extra={}), out / "calls")
    rows = []
    save(out / "frozen_config.json", dict(cfg, actual_judge_model=model))
    save(out / "frozen_prompts.json", prompts)
    try:
        for path in sorted(runs.glob("*.json")):
            record = load(path)
            if not isinstance(record, dict) or record.get("status") != "completed" or "run_id" not in record:
                continue
            task = tasks[record["task_id"]]
            if len(task["reference_images"]) != 2:
                raise ValueError("Style template requires exactly two references")
            excluded = sorted(set(task["explicit_fields"]) | set(task.get("style_excluded_dimensions", [])))
            dimensions = [k for k in prompts["fields"] if k not in excluded]
            if not dimensions:
                rows.append({"run_id": record["run_id"], "status": "not_computable_no_eligible_dimensions"})
                continue
            prompt = prompts["style_judge"].format(excluded_dimensions=json.dumps(excluded, ensure_ascii=False),
                                                    eligible_dimensions=json.dumps(dimensions, ensure_ascii=False))
            paths = [r["path"] for r in task["reference_images"]] + [record["output"]["path"]]
            raw = strict_json(client.text(prompt, paths))
            result = {"run_id": record["run_id"], "task_id": task["task_id"], "condition": record["condition"],
                      "kind": "machine_style_proxy_not_designer_assessment", "model": model,
                      "raw": raw, **style_checks(raw, dimensions)}
            save(out / (record["run_id"] + ".style.json"), result)
            rows.append(result)
        if not rows:
            raise ValueError("No completed outputs to evaluate")
        save(out / "style_scores.json", rows)
        save(out / "status.json", {"status": "completed", "scored_outputs": len(rows)})
    except Exception as exc:
        save(out / "status.json", {"status": "stopped", "scored_outputs": len(rows), "error_type": type(exc).__name__})
        raise RuntimeError("Style evaluation stopped; inspect logs") from None


def live(cfg, tasks, prompts, out, confirmed):
    if not confirmed:
        raise ValueError("Explicit --execute-api required for image uploads and billable calls")
    check = preflight(cfg, tasks)
    if check["blockers"]:
        raise ValueError("; ".join(check["blockers"]))
    for task in tasks:
        resolve({k: None for k in prompts["fields"]}, task["explicit_fields"], prompts["fields"])
    if cfg["repeats"] < 1 or not set(cfg["conditions"]) <= {"A", "B", "C", "D", "C_no_priority"}:
        raise ValueError("Invalid repeats or experiment conditions")
    if "C_no_priority" in cfg["conditions"] and not all(t.get("conflict_fields") for t in tasks):
        raise ValueError("Conflict ablation requires a predeclared conflict-only task manifest")
    if len(tasks) > cfg["max_live_tasks"]:
        raise ValueError("Task budget exceeded")
    planned = len(tasks)*len(cfg["conditions"])*cfg["repeats"]
    if planned > cfg["max_live_outputs"]:
        raise ValueError("Output budget exceeded")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / "frozen_config.json", cfg)
    save(out / "frozen_prompts.json", prompts)
    save(out / "frozen_tasks.json", tasks)
    save(out / "status.json", {"status": "running", "planned_outputs": planned, "started": stamp()})
    client = Client(cfg, out)
    fields = prompts["fields"]
    completed = 0
    try:
        profiles, prepared = {}, {}
        if cfg.get("prepared_profiles_dir"):
            cache_dir = Path(cfg["prepared_profiles_dir"])
            cache_prompts = load(cache_dir / "frozen_prompts.json")
            if any(cache_prompts[k] != prompts[k] for k in ("fields", "observe", "aggregate", "free_summary")):
                raise ValueError("Prepared profile prompt version does not match")
            cache_cfg = load(cache_dir / "frozen_config.json")
            if cache_cfg["vlm_model"] != cfg["vlm_model"] or cache_cfg.get("vlm_extra") != cfg.get("vlm_extra"):
                raise ValueError("Prepared profile extraction configuration does not match")
            for path in cache_dir.glob("*.profile.json"):
                record = load(path)
                actual_key = text_hash(json.dumps([file_record(r["path"])["sha256"] for r in record["reference_records"]]))
                if actual_key != record["cache_key"]:
                    raise ValueError("Prepared reference bytes have changed")
                profile = fields_check(record["profile"], fields)
                if any(v is not None and len(v) > 60 for v in profile.values()) or len(record["summary"]) > 480:
                    raise ValueError("Prepared profile exceeds frozen length limits")
                profiles[actual_key] = (profile, record["summary"])
                save(out / path.name, record)
            save(out / "profile_cache_provenance.json", {"directory": str(cache_dir.resolve()),
                "extraction_prompt_version": cache_prompts["version"], "generation_prompt_version": prompts["version"],
                "extraction_templates_matched": True})
        for task in tasks:
            refs = task["reference_images"]
            paths = [r["path"] for r in refs]
            cache_key = text_hash(json.dumps([file_record(p)["sha256"] for p in paths]))
            if cache_key not in profiles:
                observations = []
                for r in refs:
                    prompt = prompts["observe"].format(fields=",".join(fields), image_id=r["image_id"])
                    obj = fields_check(strict_json(client.text(prompt, [r["path"]])), fields)
                    observations.append({"image_id": r["image_id"], "fields": obj})
                profile = fields_check(strict_json(client.text(prompts["aggregate"].format(fields=",".join(fields),
                          observations=json.dumps(observations, ensure_ascii=False)), [])), fields)
                summary = client.text(prompts["free_summary"], paths)
                if any(v is not None and len(v) > 60 for v in profile.values()) or len(summary) > 480:
                    raise ValueError("Frozen profile/summary length limit exceeded; no silent truncation")
                profiles[cache_key] = (profile, summary)
                save(out / (cache_key[:12] + ".profile.json"), {"observations": observations, "profile": profile, "summary": summary})
            profile, summary = profiles[cache_key]
            prepared[task["task_id"]] = (task, profile, summary)
        schedule = [(t["task_id"], c, repeat) for t in tasks for c in cfg["conditions"] for repeat in range(cfg["repeats"])]
        random.Random(cfg["schedule_seed"]).shuffle(schedule)
        save(out / "schedule.json", schedule)
        import requests
        from PIL import Image
        for task_id, condition, repeat in schedule:
            task, profile, summary = prepared[task_id]
            prompt = render(task, condition, profile, summary, prompts)
            paths = ([task["input_image"]] if task.get("input_image") else []) + [r["path"] for r in task["reference_images"]]
            run_id = f"{task_id}_{condition}_{repeat}"
            seed = None
            if cfg.get("image_seed_base") is not None:
                seed = cfg["image_seed_base"] + next(i for i, t in enumerate(tasks) if t["task_id"] == task_id)*cfg["repeats"] + repeat
            metadata = {"run_id": run_id, "task_id": task_id, "condition": condition, "repeat": repeat,
                        "prompt": prompt, "prompt_sha256": text_hash(prompt), "images": [file_record(p) for p in paths],
                        "seed": seed, "seed_supported": "advertised_by_endpoint_not_determinism_verified" if seed is not None else "not_tested", "status": "started", "timestamp": stamp()}
            save(out / (run_id + ".json"), metadata)
            data = client.image(prompt, paths, seed=seed)
            blob = output_bytes(data, cfg)
            with Image.open(io.BytesIO(blob)) as im:
                fmt = im.format
                im.verify()
            extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(fmt)
            if not extension:
                raise ValueError("Unsupported response image format")
            dest = out / (run_id + extension)
            dest.write_bytes(blob)
            metadata.update({"status": "completed", "output": file_record(dest), "usage": data.get("usage"),
                             "image_request_id": data.get("request_id", data.get("id")),
                             "returned_model": data.get("model"), "provider": data.get("provider")})
            save(out / (run_id + ".json"), metadata)
            completed += 1
            print(json.dumps({"completed": completed, "planned": planned, "run_id": run_id}), flush=True)
        save(out / "status.json", {"status": "completed", "output_count": completed, "timestamp": stamp(), "calls": client.calls})
    except Exception as exc:
        save(out / "status.json", {"status": "stopped", "output_count": completed, "error_type": type(exc).__name__, "timestamp": stamp()})
        # Do not serialize exception strings from HTTP/download clients containing signed URLs.
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__, "completed": completed}), flush=True)
        raise RuntimeError("Live run stopped; inspect metadata, no automatic retry") from None


def selfcheck(prompts):
    """Small deterministic fixture checks, never presented as model experiments."""
    fields = prompts["fields"]
    profile = {k: None for k in fields}
    color = "color_palette" if "color_palette" in fields else "color_tone_usage_characteristics"
    cut = "structure_cut" if "structure_cut" in fields else "cut_style_characteristics"
    profile.update({color: "black", cut: "asymmetric"})
    resolved = resolve(profile, {color: "blue", cut: "symmetric"}, fields)
    assert resolved[color] == "blue" and resolved[cut] == "symmetric"
    assert profile[color] == "black"
    assert set(flatten(resolved, fields).split("；")) == {"symmetric", "blue"}
    caught = 0
    for bad in ({"extra": "x"}, {k: 1 for k in fields}):
        try:
            fields_check(bad, fields)
        except ValueError:
            caught += 1
    assert caught == 2
    task = {"brief": "sample", "hard_constraints": ["blue"], "preserve": [],
            "explicit_fields": {color: "blue"}, "reference_images": [{"image_id": "REF1"}]}
    outputs = {c: render(task, c, profile, "summary", prompts) for c in ("A", "B", "C", "D")}
    assert len(set(outputs.values())) == 4
    assert all("REF1" in p for p in outputs.values())
    assert all("{{" not in p for p in outputs.values())
    score = judge_checks({"checks": [{"constraint_id": "c1", "status": "met", "evidence": "fixture"},
                                     {"constraint_id": "c2", "status": "unknown", "evidence": "fixture"}]}, ["c1", "c2"])
    assert score["conservative_adherence"] == 0.5 and score["unknown_rate"] == 0.5
    try:
        judge_checks({"checks": [{"constraint_id": "c1", "status": "met", "evidence": "fixture"}]}, ["c1", "c2"])
    except ValueError:
        pass
    else:
        raise AssertionError("Missing judge constraints must fail")
    try:
        validate_tasks([], {"max_input_images": 3})
    except ValueError:
        pass
    else:
        raise AssertionError("Empty manifests must block execution")
    # Use existing files solely to test validation, not as new reference data.
    fixture = dict(task, task_id="VALIDATION_ONLY", split="development")
    entry = {"image_id": "REF1", "path": str(HERE / "input" / "11.png"),
             "source": "software-validation-fixture", "rights_status": "owned", "garment_group_id": "G1"}
    fixture.update(reference_images=[entry], evaluation_images=[dict(entry, image_id="EVAL1")])
    try:
        validate_tasks([fixture], {"max_input_images": 3})
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:
        raise AssertionError("Identical reference/evaluation input must fail")
    return {"kind": "deterministic_software_checks", "status": "passed", "new_generation_count": 0,
            "checks": ["explicit_override", "source_immutable", "lossless_flatten_order", "schema_rejects_invalid",
                       "four_distinct_conditions", "shared_reference_roles", "no_unresolved_template",
                       "empty_manifest_blocks", "reference_evaluation_leakage_blocks",
                       "unknown_not_discarded", "missing_judge_constraint_blocks"], "timestamp": stamp()}


def main():
    load_local_credentials()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "audit", "selfcheck", "live", "judge", "stylejudge"))
    parser.add_argument("--config", default=str(HERE / "workflow_config.json"))
    parser.add_argument("--tasks", default=str(HERE / "workflow_tasks.json"))
    parser.add_argument("--prompts", default=str(HERE / "workflow_prompts.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--runs", help="Directory containing completed live run records")
    args = parser.parse_args()
    cfg, task_doc, prompts = load(args.config), load(args.tasks), load(args.prompts)
    tasks = task_doc["tasks"]
    if args.mode == "audit":
        audit(args.out)
    elif args.mode == "preflight":
        result = preflight(cfg, tasks)
        save(Path(args.out) / "preflight.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return 2 if result["blockers"] else 0
    elif args.mode == "selfcheck":
        result = selfcheck(prompts)
        save(Path(args.out) / "software_checks.json", result)
        print(json.dumps(result, ensure_ascii=False))
    elif args.mode in ("judge", "stylejudge"):
        if not args.runs:
            raise ValueError("--runs is required for machine judging")
        (style_judge if args.mode == "stylejudge" else judge)(cfg, prompts, args.runs, args.out, args.execute_api)
    else:
        live(cfg, tasks, prompts, args.out, args.execute_api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
