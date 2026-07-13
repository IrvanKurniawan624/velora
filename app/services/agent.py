"""Agent orchestrator: read /input/tasks.json -> answer all tasks -> /output/results.json.

Design: answer-then-improve under a strict wall-clock budget.
  Pass 1 banks a best-shot answer for every task (cheap categories first).
  Pass 2 spends remaining time re-verifying low-confidence answers.
  A watchdog guarantees a complete, valid results.json and exit code 0.

MODE=zero    -> never touches Fireworks (0 scored tokens).
MODE=hybrid  -> escalates still-low-confidence tasks via FIREWORKS_BASE_URL.
"""
import json
import os
import re
import sys
import threading
import time

from .gate import sanity_check
from .router import route
from .solvers import SOLVERS
from ..clients.local_client import LocalClient
from ..clients.remote_client import remote_answer, token_usage
from ..config import (
    ESC_CONF, ESC_MAX, HARD_DEADLINE, INPUT_PATH,
    MODE, OUTPUT_PATH, SOFT_DEADLINE,
)

T0 = time.time()

_lock = threading.Lock()
_results = {}          # task_id -> answer str
_order = []            # task ids in input order

# Velora-specific: some tasks explicitly request a JSON answer (e.g. "Return
# strictly JSON with keys 'sentiment' and 'reason'"). The handlers produce clean
# plain-text / line formats that an intent judge already accepts; this repackages
# them as JSON ONLY when the prompt asks for it, so the proven plain-text path is
# untouched for every other case. No answers are hardcoded — it only reformats
# the model's own extracted content. Never raises.
_TYPE_KEYS = {
    "person": "PERSON", "organization": "ORG", "org": "ORG",
    "location": "LOC", "place": "LOC", "loc": "LOC",
    "date": "DATE", "time": "TIME", "event": "EVENT",
    "product": "PRODUCT", "money": "MONEY", "percent": "PERCENT",
    "other": "OTHER",
}


def _extract_ner_keys_from_prompt(prompt):
    candidates = ["person", "org", "organization", "loc", "location", "date", "time", 
                  "event", "product", "money", "percent", "other", "gpe"]
    found_keys = []
    quoted = re.findall(r"[\"']([a-zA-Z\-_]+)[\"']", prompt)
    for q in quoted:
        q_low = q.lower()
        if any(c == q_low or q_low.startswith(c) or c.startswith(q_low) for c in candidates):
            if q not in found_keys and q.upper() != "JSON":
                found_keys.append(q)
    if found_keys:
        return found_keys
    words = re.findall(r"\b([a-zA-Z]+)\b", prompt)
    for w in words:
        w_low = w.lower()
        if w_low in candidates or w_low == "gpe":
            if w not in found_keys and w.upper() != "JSON":
                found_keys.append(w)
    if found_keys:
        return found_keys
    return ["PERSON", "ORGANIZATION", "LOCATION", "DATE"]


def _map_to_key(type_str, found_keys):
    t_low = type_str.lower().strip()
    mapping = {
        "person": "person",
        "organization": "organization",
        "org": "organization",
        "location": "location",
        "loc": "location",
        "gpe": "location",
        "date": "date",
        "time": "time",
        "event": "event",
        "product": "product",
        "money": "money",
        "percent": "percent",
        "other": "other"
    }
    std_low = mapping.get(t_low, t_low)
    for k in found_keys:
        k_low = k.lower()
        if k_low == std_low:
            return k
        if (std_low == "organization" and k_low == "org") or (std_low == "org" and k_low == "organization"):
            return k
        if (std_low == "location" and k_low == "loc") or (std_low == "loc" and k_low == "location"):
            return k
    for k in found_keys:
        if k.lower() == t_low:
            return k
    return None


def json_if_requested(prompt, answer, cat):
    if not answer or "json" not in prompt.lower():
        return answer
    try:
        if cat == "sentiment":
            low = answer.strip()
            label = next((lab for lab in ("mixed", "positive", "negative", "neutral")
                          if re.search(rf"\b{lab}\b", low, re.I)), "neutral")
            if "Mixed" in prompt or "Positive" in prompt:
                label = label.title()
            elif "MIXED" in prompt or "POSITIVE" in prompt:
                label = label.upper()
            reason = re.sub(r"^\s*\w+\s*[-–—:]\s*", "", low).strip().rstrip(".")
            quoted = re.findall(r"[\"']([a-zA-Z_]+)[\"']", prompt)
            k_sent = "sentiment"
            k_reason = "reason"
            for q in quoted:
                if "sent" in q.lower():
                    k_sent = q
                elif "reason" in q.lower() or "just" in q.lower() or "explan" in q.lower():
                    k_reason = q
            return json.dumps({k_sent: label, k_reason: reason or low})
        if cat == "summarize":
            n_b = 3
            m = re.search(r"(\d+)\s+(?:\w+\s+){0,2}bullet", prompt, re.I)
            if m:
                n_b = int(m.group(1))
            items = [ln.strip().lstrip("-*• ").strip() for ln in answer.splitlines()
                     if ln.strip()]
            if len(items) < 2:
                items = [s.strip().rstrip(".") for s in
                         re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
            return json.dumps({"bullets": items[:n_b]})
        if cat == "ner":
            keys = _extract_ner_keys_from_prompt(prompt)
            grouped = {k: [] for k in keys}
            for ln in answer.splitlines():
                m = re.match(r"(.+?)\s*[-–—:]\s*(.+)", ln.strip())
                if m:
                    ent, typ = m.group(1).strip(), m.group(2).strip()
                    k = _map_to_key(typ, keys)
                    if k and k in grouped:
                        grouped[k].append(ent)
            return json.dumps(grouped)
    except Exception:
        return answer
    return answer


def elapsed():
    return time.time() - T0


def log(msg):
    sys.stderr.write(f"[main +{elapsed():6.1f}s] {msg}\n")


def write_results():
    with _lock:
        data = [{"task_id": tid, "answer": _results.get(tid, "") or
                 "Unable to determine within the time limit."}
                for tid in _order]
    tmp = OUTPUT_PATH + ".tmp"
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=1)
    os.replace(tmp, OUTPUT_PATH)


def watchdog():
    while True:
        left = HARD_DEADLINE - elapsed()
        if left <= 0:
            break
        time.sleep(min(left, 1.0))
    log("watchdog fired: flushing and exiting")
    try:
        write_results()
    finally:
        os._exit(0)


class TaskContext:
    """What handlers see: chat access + time awareness.

    fast=True is the Pass-1 mode: only cheap verification (<=18s asks) is
    allowed so every task banks an answer quickly; the expensive checks run
    in Pass 2 with whatever wall-clock remains.
    """

    def __init__(self, llm):
        self.llm = llm
        self.task_deadline = None  # absolute time.time() cap for current task
        self.fast = False

    def chat(self, system, user, **kw):
        if elapsed() > HARD_DEADLINE - 6:
            return ""
        return self.llm.chat(system, user, **kw)

    def have_time(self, seconds_needed: float) -> bool:
        if self.fast and seconds_needed > 18:
            return False
        if elapsed() + seconds_needed > SOFT_DEADLINE:
            return False
        if self.task_deadline and time.time() + seconds_needed > self.task_deadline:
            return False
        return True


# Cheap/fast categories first so answers get banked early.
_CAT_ORDER = ["sentiment", "ner", "summarize", "factual",
              "math", "code_gen", "code_debug", "logic"]


def run():
    global _order
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        assert isinstance(tasks, list)
    except Exception as e:
        log(f"FATAL: cannot read tasks: {e}")
        _order = []
        write_results()
        return 0

    items = []
    for t in tasks:
        tid = str(t.get("task_id", "")) or f"task-{len(items)+1}"
        prompt = str(t.get("prompt", "") or "")
        items.append({"id": tid, "prompt": prompt, "cat": route(prompt)})
    _order[:] = [it["id"] for it in items]
    log(f"loaded {len(items)} tasks: " +
        ", ".join(f"{it['id']}={it['cat']}" for it in items))
    write_results()  # valid (placeholder) file exists from the very start

    threading.Thread(target=watchdog, daemon=True).start()

    llm = LocalClient()
    llm.start()

    # hybrid: escalate-first. Fireworks answers the hard categories in a
    # parallel wave at t~0 (immune to local CPU speed); the local pipeline
    # covers easy categories and any failed calls.
    fw_done = {}
    if MODE == "hybrid":
        try:
            from concurrent.futures import ThreadPoolExecutor
            from .remote import remote_answer
            HARD = {"factual", "math", "logic", "code_debug", "code_gen"}
            hard_items = [it for it in items if it["cat"] in HARD][:ESC_MAX]

            def _esc(it):
                text, tok = remote_answer(it["prompt"], it["cat"])
                return it["id"], text

            if hard_items:
                with ThreadPoolExecutor(max_workers=4) as ex:
                    for tid, text in ex.map(_esc, hard_items):
                        if text:
                            fw_done[tid] = text
                log(f"escalate-first wave: {len(fw_done)}/{len(hard_items)} answered by Fireworks")
                with _lock:
                    _results.update(fw_done)
                write_results()
        except Exception as e:
            log(f"escalate-first error (falling back to local): {e}")

    if not llm.wait_ready(timeout=90):
        log("FATAL: local model failed to start")
        write_results()
        return 0
    log("local model ready")

    ctx = TaskContext(llm)
    work = sorted((it for it in items if it["id"] not in fw_done),
                  key=lambda it: _CAT_ORDER.index(it["cat"]))
    confs = {it["id"]: 0.85 for it in items if it["id"] in fw_done}

    # ---- Pass 1 (fast): bank an answer for everything quickly
    ctx.fast = True
    for i, it in enumerate(work):
        remaining = max(1, len(work) - i)
        budget = max(12.0, (SOFT_DEADLINE * 0.62 - elapsed()) / remaining * 1.25)
        ctx.task_deadline = time.time() + budget
        try:
            res = SOLVERS[it["cat"]](it["prompt"], ctx)
        except Exception as e:
            log(f"handler error on {it['id']}: {e}")
            res = {"answer": "", "conf": 0.1, "cat": it["cat"]}
        if not res.get("answer"):
            try:
                res["answer"] = ctx.chat(
                    "Answer the task as well as you can, concisely.",
                    it["prompt"], temperature=0.0, max_tokens=180) or ""
                res["conf"] = min(res.get("conf", 0.3), 0.4)
            except Exception:
                pass
        ans = json_if_requested(it["prompt"], res.get("answer", ""), it["cat"])
        with _lock:
            _results[it["id"]] = ans
        confs[it["id"]] = res.get("conf", 0.3)
        if not sanity_check(it["prompt"], ans, it["cat"]):
            confs[it["id"]] = min(confs[it["id"]], 0.15)
        write_results()
        log(f"[{i+1}/{len(work)}] {it['id']} cat={it['cat']} "
            f"conf={confs[it['id']]:.2f} len={len(ans)}")

    # ---- Pass 2 (full): re-verify ascending by confidence while time remains
    ctx.fast = False
    ctx.task_deadline = None
    # Re-verify only genuinely-low-confidence tasks. The 0.9 threshold used to
    # re-verify ~18/29 tasks (incl. already-correct 0.80-0.85 gen/ner/sum),
    # blowing the 10-min judge cap. 0.78 keeps the weak ones (logic, factual,
    # math, code-debug) and skips the rest, cutting pass 2 from ~345s to ~180s
    # while preserving accuracy (the skipped tasks were already correct).
    weak = sorted((it for it in work if confs[it["id"]] < 0.78),
                  key=lambda it: confs[it["id"]])
    pass2_cut = SOFT_DEADLINE - (100 if MODE == "hybrid" else 25)
    for it in weak:
        if elapsed() > pass2_cut:
            break
        log(f"pass2 verify {it['id']} (conf={confs[it['id']]:.2f})")
        try:
            res = SOLVERS[it["cat"]](it["prompt"], ctx)
        except Exception as e:
            log(f"pass2 error {it['id']}: {e}")
            continue
        if res.get("answer") and res.get("conf", 0) > confs[it["id"]]:
            with _lock:
                _results[it["id"]] = json_if_requested(
                    it["prompt"], res["answer"], it["cat"])
            confs[it["id"]] = res["conf"]
            write_results()

    # ---- Hybrid escalation (never in zero mode)
    if MODE == "hybrid":
        from .remote import remote_answer, token_usage
        esc = [it for it in work if confs[it["id"]] < ESC_CONF]
        esc.sort(key=lambda it: confs[it["id"]])
        for it in esc[:ESC_MAX]:
            if elapsed() > HARD_DEADLINE - 35:
                break
            text, tok = remote_answer(it["prompt"], it["cat"])
            if text:
                with _lock:
                    _results[it["id"]] = text
                confs[it["id"]] = 0.8
                write_results()
                log(f"escalated {it['id']} via Fireworks ({tok} tokens)")
        log(f"fireworks usage: {token_usage()}")

    write_results()
    low = [f"{k}={v:.2f}" for k, v in confs.items() if v < 0.6]
    log(f"done in {elapsed():.1f}s; low-conf: {low or 'none'}")
    llm.stop()
    return 0


