"""
Collect baseline + OLIVE results into one JSON (+ video thumbnails) for the report pages.

    python report/collect_results.py --out report/build

Reads
  docs/baseline_libero_eval.md                        frozen OpenVLA SR (2026-09-01, 20 trials/task = 200 ep/suite)
  result/baseline/logs/<alias>-<stamp>/EVAL-*.txt     this run's baseline (1 trial/task: videos + sanity check)
  result/baseline/rollouts/*/*.mp4                    baseline rollout videos
  result/olive-<suite>-main/metrics.jsonl             PPO per-iteration metrics + in-train evals
  result/olive-<suite>-main/final_eval.json           final deterministic eval (03_eval.sh)
  result/olive-<suite>-main/logs/01_bc_*.log          BC warm-up summary line
  result/olive-<suite>-main/videos/*.mp4              OLIVE rollout videos
Writes
  <out>/results.json, <out>/media/*.mp4 (copied, one per task per method), <out>/media/*.jpg (thumbnails)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

ROOT = os.path.expanduser("~/olive/result")
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
ALIAS = {"libero_spatial": "spatial", "libero_object": "object", "libero_goal": "goal", "libero_10": "long"}
# OLIVE 논문 Table 2 (frozen OpenVLA vanilla, OLIVE) — docs/paper/
PAPER = {"libero_spatial": (60.0, 91.9), "libero_object": (88.0, 38.5),
         "libero_goal": (70.0, 71.4), "libero_10": (40.0, 50.0)}


def _slug(desc: str) -> str:
    return desc.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]


DOC_HEAD = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
            "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-Long"}


def parse_baseline_doc(suite: str) -> dict | None:
    """Per-task SR tables in docs/baseline_libero_eval.md (### Per-task success rate)."""
    path = os.path.expanduser("~/olive/docs/baseline_libero_eval.md")
    if not os.path.exists(path):
        return None
    lines = open(path).read().split("\n")
    start = next((i for i, l in enumerate(lines)
                  if l.startswith(f"**{DOC_HEAD[suite]}") and "%" in l), None)
    if start is None:
        return None
    tasks = []
    for l in lines[start + 1:]:
        if l.startswith("**") or l.startswith("#"):
            break
        m = re.match(r"\|\s*(.+?)\s*\|\s*([0-9.]+)\s*\|$", l)
        if m:
            tasks.append({"task_id": len(tasks), "task": m.group(1), "sr": float(m.group(2)), "n": 20})
    total = sum(t["sr"] for t in tasks) / len(tasks) if tasks else 0.0
    return {"log": path, "total_sr": total, "n_episodes": 20 * len(tasks), "per_task": tasks,
            "source": "2026-09-01 baseline run (docs/baseline_libero_eval.md)"}


def parse_baseline(suite: str) -> dict | None:
    logs = sorted(glob.glob(f"{ROOT}/baseline/logs/{ALIAS[suite]}-*/EVAL-{suite}-*.txt"))
    if not logs:
        return None
    path = logs[-1]                                  # latest run of this suite
    per_task, task, succ, videos = {}, None, [], {}
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("Task: "):
            task = line[len("Task: "):].strip()
            per_task.setdefault(task, [])
        elif line.startswith("Success: ") and task is not None:
            ok = line.endswith("True")
            per_task[task].append(ok)
            succ.append(ok)
        elif line.startswith("Saved rollout MP4 at path ") and task is not None:
            videos.setdefault(task, []).append(line.split("path ", 1)[1].strip())
    tasks = [{"task_id": i, "task": t, "sr": (sum(v) / len(v)) if v else 0.0, "n": len(v),
              "videos": videos.get(t, [])} for i, (t, v) in enumerate(per_task.items())]
    return {"log": path, "total_sr": (sum(succ) / len(succ)) if succ else 0.0,
            "n_episodes": len(succ), "per_task": tasks}


def parse_olive(suite: str) -> dict | None:
    rd = f"{ROOT}/olive-{suite}-main"
    if not os.path.isdir(rd):
        return None
    out = {"run_dir": rd, "iters": [], "evals": [], "final": None, "bc": None, "videos": {}}
    mpath = os.path.join(rd, "metrics.jsonl")
    if os.path.exists(mpath):
        for line in open(mpath):
            rec = json.loads(line)
            (out["iters"] if rec.get("kind") == "iter" else out["evals"]).append(rec)
    fpath = os.path.join(rd, "final_eval.json")
    if os.path.exists(fpath):
        out["final"] = json.load(open(fpath))
    for lp in sorted(glob.glob(os.path.join(rd, "logs", "01_bc_*.log")))[-1:]:
        m = [l for l in open(lp, errors="replace") if "[OLIVE][bc] done" in l]
        if m:
            nums = dict(re.findall(r"'(\w+_loss)': ([0-9.eE+-]+)", m[-1]))
            out["bc"] = {k: float(v) for k, v in nums.items()}
    for v in sorted(glob.glob(os.path.join(rd, "videos", "*.mp4"))):
        m = re.search(r"--task(\d+)--", os.path.basename(v))
        if m:
            out["videos"].setdefault(int(m.group(1)), []).append(v)
    return out


def _thumb(src_mp4: str, dst_jpg: str, frac: float = 0.6) -> bool:
    """Grab one frame (at frac of the clip) as a JPEG thumbnail."""
    try:
        import imageio.v2 as iio
        r = iio.get_reader(src_mp4)
        n = r.count_frames()
        frame = r.get_data(max(0, int(n * frac) - 1))
        r.close()
        iio.imwrite(dst_jpg, frame, quality=85)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[collect] thumbnail failed for {src_mp4}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/olive/report/build"))
    a = ap.parse_args()
    media = os.path.join(a.out, "media")
    os.makedirs(media, exist_ok=True)

    res = {"suites": []}
    for s in SUITES:
        b_doc, b_run, o = parse_baseline_doc(s), parse_baseline(s), parse_olive(s)
        b = b_doc or b_run                              # SR 은 200-ep 문서 수치 우선
        if b_doc and b_run:                             # 영상은 이번 run 것
            vids = {t["task"]: t["videos"] for t in b_run["per_task"]}
            for t in b_doc["per_task"]:
                t["videos"] = vids.get(t["task"], [])
        entry = {"suite": s, "alias": ALIAS[s], "paper_baseline": PAPER[s][0], "paper_olive": PAPER[s][1],
                 "baseline": None, "olive": None, "tasks": []}
        if b:
            entry["baseline"] = {k: b[k] for k in ("total_sr", "n_episodes", "log")}
            entry["baseline"]["source"] = b.get("source", "this run")
        if b_run:                                       # 이번 run 1-trial 결과 (sanity check)
            entry["baseline_rerun"] = {"total_sr": b_run["total_sr"], "n_episodes": b_run["n_episodes"],
                                       "per_task_success": [t["sr"] for t in b_run["per_task"]]}
        if o:
            fin = o["final"] or {}
            entry["olive"] = {"total_sr": fin.get("total_sr"), "n_episodes": fin.get("n_episodes"),
                              "bc": o["bc"], "iters": o["iters"],
                              "evals": [{k: e[k] for k in ("iter", "global_step", "total_sr")} for e in o["evals"]]}
        # per-task join: task order is the LIBERO suite order in both harnesses
        b_tasks = b["per_task"] if b else []
        o_tasks = (o["final"] or {}).get("per_task", []) if o else []
        n = max(len(b_tasks), len(o_tasks))
        for i in range(n):
            bt = b_tasks[i] if i < len(b_tasks) else None
            ot = o_tasks[i] if i < len(o_tasks) else None
            desc = (ot or bt)["task"]
            t = {"task_id": i, "task": desc,
                 "baseline_sr": bt["sr"] if bt else None, "baseline_n": bt["n"] if bt else None,
                 "olive_sr": ot["sr"] if ot else None, "olive_n": ot["n"] if ot else None}
            for method, vids in (("baseline", (bt or {}).get("videos", [])),
                                 ("olive", (o or {}).get("videos", {}).get(i, []))):
                vids = [v for v in vids if os.path.exists(v)]
                if not vids:
                    continue
                # prefer a successful episode for the showcase clip
                pick = next((v for v in vids if "success=True" in v), vids[0])
                name = f"{s}_t{i:02d}_{method}"
                shutil.copy2(pick, os.path.join(media, name + ".mp4"))
                t[f"{method}_video"] = f"media/{name}.mp4"
                t[f"{method}_video_success"] = "success=True" in pick
                if _thumb(pick, os.path.join(media, name + ".jpg")):
                    t[f"{method}_thumb"] = f"media/{name}.jpg"
            entry["tasks"].append(t)
        res["suites"].append(entry)

    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for e in res["suites"]:
        b = e["baseline"]["total_sr"] if e["baseline"] else None
        o = e["olive"]["total_sr"] if e["olive"] else None
        rr = e.get("baseline_rerun", {}).get("total_sr")
        print(f"{e['suite']:15s} baseline={b} (rerun {rr})  olive={o}  tasks={len(e['tasks'])}")
    print("wrote", os.path.join(a.out, "results.json"))


if __name__ == "__main__":
    main()
