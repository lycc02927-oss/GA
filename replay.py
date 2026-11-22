"""从一次 run 的 CSV 取出某个模型的入选参数，在新种子上逐次复验。"""
import argparse
import csv
import json
import sys
from pathlib import Path

from tune_ed import initialize, trial


def model_row(path, name, model):
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if int(row[name]) == model:
                return row
    raise ValueError(f"{path} 没有模型 {model}")


def replay(run, model, mode, seeds):
    cfg = json.loads((run/"config.json").read_text(encoding="utf-8"))
    row = model_row(run/"selected.csv", "model", model)
    ref = model_row(run/"references.csv", "data_model", model)
    if mode == "plateau":
        p = {"platform_error": float(row["platform_error"])}
    else:
        p = {k: float(v) for k, v in row.items() if k not in ("model", "platform_error")}
        for k in ("gp_restarts", "window"):
            p[k] = int(p[k])
    for seed in seeds:
        result = trial((model, mode, seed, p, cfg["initial"], cfg["rounds"], float(ref["reference_value"])))
        yield dict(model=model, mode=mode, **result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("../test"))
    p.add_argument("--run", type=Path, default=Path("results"))
    p.add_argument("--model", type=int, choices=range(1,7), required=True)
    p.add_argument("--mode", choices=("maximum", "plateau"), default="maximum")
    p.add_argument("--seeds", type=int, nargs="+", default=[242])
    a = p.parse_args()
    initialize(a.source.resolve())
    rows = list(replay(a.run, a.model, a.mode, a.seeds))
    out = csv.DictWriter(sys.stdout, list(rows[0]))
    out.writeheader()
    out.writerows(rows)
