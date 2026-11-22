"""按模型做轮盘赌 GA。用 --source 指向单独检出的 test 仓库。"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

PIN = "32f0d18437fbc73a0c1458bb121286c59f4ed2b6"
WINDOW_BASELINE = dict(gp_restarts=10, xi_init=.01, xi_final=0., sigma3_rel=0.,
                       global_prob=.4, ls_lo=40., ls_hi=2000., window=5, tol=.10, min_dis=.5)
PLATEAU = (.005, .01, .02, .03, .05, .08)


def space(ed_range, profile="standard"):
    # 用无抖动原始数据的极差把 EI 候选换算为本模型的 ED 单位。
    grid = dict(gp_restarts=(2, 5, 10), xi_init=(.005*ed_range, .02*ed_range, .08*ed_range, .01),
                xi_final=(0., .0005*ed_range, .002*ed_range), sigma3_rel=(0., .03, .10),
                global_prob=(.15, .40, .70), ls_lo=(20., 40., 80.), ls_hi=(200., 600., 2000.),
                window=(3, 5, 8), tol=(.01, .03, .06, .10), min_dis=(.5, 2., 6., 12.))
    if profile == "multi_peak":
        grid.update(global_prob=(.4, .7, .95), window=(8, 12, 20),
                    tol=(.002, .005, .01), min_dis=(.25, .5, 2.))
    return grid


def draw(grid, rng):
    return tuple(rng.choice(values) for values in grid.values())


def breed(ranked, grid, rng, mutation):
    """冠军保留；父母由排名轮盘赌选出；逐基因随机交叉和变异。"""
    size = len(ranked)
    children = [ranked[0]]
    while len(children) < size:
        a, b = rng.choices(ranked, weights=range(size, 0, -1), k=2)
        child = tuple(rng.choice(values) if rng.random() < mutation else rng.choice((x, y))
                      for x, y, values in zip(a, b, grid.values()))
        if child not in children:
            children.append(child)
    return children


def initialize(source):
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    sys.path.insert(0, str(source))


def trial(job):
    from ed_trials import maximum, plateau
    model, mode, seed, param, initial, rounds, reference = job
    if mode == "maximum":
        x, observed, clean, count, stop = maximum(model, seed, param, initial, rounds)
    else:
        x, observed, clean, count, stop = plateau(model, seed, param["platform_error"])
    error = 100 * abs(clean - reference) / abs(reference)
    return dict(seed=seed, evaluations=count, bo_rounds=count-initial if mode == "maximum" else "",
                stop=stop, x=x[0], y=x[1], z=x[2], observed_ed=observed,
                scoring_ed=clean, reference_ed=reference, error_pct=error, hit=int(error <= 10))


def metrics(rows):
    hit = sum(r["hit"] for r in rows)
    error = mean(r["error_pct"] for r in rows)
    count = mean(r["evaluations"] for r in rows)
    return dict(hits=hit, runs=len(rows), mean_error_pct=error, mean_evaluations=count,
                loss=error + .12*count)


def write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.DictWriter(f, list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def run_config(args, source, numpy_version, scipy_version):
    code = sorted(source.glob("*.py")) + sorted((source/"fit_models").glob("*.py"))
    core = {"search.py", "batch_test.py", "LHS.py", "kernel.py", "GPR.py", "EI.py",
            "fitter.py", "formula_models.py", "measurement_data.py"}
    hashes = {str(p.relative_to(source).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in code if p.name in core or p.parent.name == "fit_models"}
    return dict(vars(args), source=source.name, output=args.output.name, upstream_expected=PIN,
                source_sha256=hashes, python=sys.version.split()[0], numpy=numpy_version,
                scipy=scipy_version)


def main(args):
    source = args.source.resolve()
    if not (source/"search.py").is_file():
        raise SystemExit("--source 应指向包含 search.py 的 test 仓库")
    if args.population < 3 or args.population > 100 or min(args.generations, args.workers, args.rounds) < 1 or args.initial < 2:
        raise SystemExit("种群限 3..100，代数、进程数、轮数须为正，初始点至少 2")
    train, valid = args.train_seeds, args.valid_seeds
    if len(set(train+valid)) != len(train+valid) or any(s < 0 or s >= 2**32 for s in train+valid):
        raise SystemExit("随机种子须在 uint32 范围内；训练与验证使用独立且各不重复的种子")
    if len(set(args.models)) != len(args.models):
        raise SystemExit("模型编号重复")
    initialize(source)
    import numpy as np
    import scipy
    import batch_test
    from measurement_data import MODEL_DATA
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = run_config(args, source, np.__version__, scipy.__version__)
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    refs = [r for r in batch_test.make_references(args.models) if r["target"] == "ed"]
    write(out/"references.csv", refs)
    ref = {r["data_model"]: r["reference_value"] for r in refs}
    results, rankings, summaries, choices = [], [], [], []
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(source,)) as pool:
        for model in args.models:
            grid = space(float(np.ptp(MODEL_DATA[model][:, 3])), args.profile)
            rng = random.Random(args.ga_seed + model)
            cache = {}
            def evaluate(mode, params, seeds, stage):
                jobs = [(model, mode, s, p, args.initial, args.rounds, ref[model])
                        for p in params for s in seeds]
                rows = list(pool.map(trial, jobs))
                groups = []
                for j, p in enumerate(params):
                    group = rows[j*len(seeds):(j+1)*len(seeds)]
                    label = json.dumps(p, sort_keys=True)
                    for r in group:
                        results.append(dict(model=model, mode=mode, stage=stage, parameters=label, **r))
                    groups.append(metrics(group))
                return groups

            def order(m):
                return -m["hits"], m["loss"]
            start = dict(WINDOW_BASELINE)
            if args.profile == "multi_peak":
                start.update(global_prob=.7, window=12, tol=.005)
            population = [tuple(start[k] for k in grid)]
            while len(population) < args.population:
                g = draw(grid, rng)
                if g not in population:
                    population.append(g)
            for gen in range(args.generations):
                fresh = [g for g in population if g not in cache]
                params = [dict(zip(grid, g)) for g in fresh]
                cache.update(zip(fresh, evaluate("maximum", params, train, "train")))
                ranked = sorted(population, key=lambda g: order(cache[g]))
                for rank, g in enumerate(ranked, 1):
                    rankings.append(dict(model=model, generation=gen+1, rank=rank, **dict(zip(grid,g)), **cache[g]))
                if gen+1 < args.generations:
                    population = breed(ranked, grid, rng, .18)
            best = dict(zip(grid, min(cache, key=lambda g: order(cache[g]))))
            thresholds = [dict(platform_error=v) for v in PLATEAU]
            scored = evaluate("plateau", thresholds, train, "train")
            pick = min(range(len(scored)), key=lambda i: order(scored[i]))
            flat = thresholds[pick]
            choices.append(dict(model=model, **best, **flat))
            for mode, champion, control in [("maximum", best, None), ("plateau", flat, {"platform_error":.02})]:
                cases = [("GA" if mode=="maximum" else "sweep", champion), ("baseline", control)]
                if mode == "maximum":
                    cases.append(("window_baseline", WINDOW_BASELINE))
                for role, p in cases:
                    score, = evaluate(mode, [p], valid, "validation_"+role)
                    summaries.append(dict(model=model, mode=mode, role=role, **score))
            # 每个完整工况落盘，便于查看进度。
            for name, rows in [("selected", choices), ("trials", results), ("generations", rankings),
                               ("validation", summaries)]:
                write(out/(name+".csv"), rows)
            print(f"模型 {model} 完成，已评估 {len(cache)} 套 Maximum 参数", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("../test"))
    p.add_argument("--output", type=Path, default=Path("run_ed"))
    p.add_argument("--models", type=int, nargs="+", choices=range(1,7), default=list(range(1,7)))
    p.add_argument("--profile", choices=("standard", "multi_peak"), default="standard")
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--initial", type=int, default=20)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--ga-seed", type=int, default=2026)
    p.add_argument("--train-seeds", type=int, nargs="+", default=[42,43,44,45])
    p.add_argument("--valid-seeds", type=int, nargs="+", default=list(range(242,252)))
    main(p.parse_args())
