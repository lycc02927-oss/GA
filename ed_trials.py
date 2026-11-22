"""离线试验适配层。所有模型、GP 拟合、EI 选点从指定的上游仓库导入。"""
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import search
from fitter import EDFitter, EDPFModel


@lru_cache(maxsize=6)
def surface(model):
    return EDFitter(model, "rbf")


class StopTrial(Exception):
    """一次量测达到外层停止条件。"""


def maximum(model, seed, p, initial, rounds):
    """保留上游迭代和单发抖动，用设备端的窗口、近点规则包装每次量测。"""
    spatial = surface(model)
    sensor = EDPFModel(spatial, seed=seed)
    xs, ys = [], []
    near = 0
    reason = "budget"

    def observe(x):
        nonlocal near
        i = len(xs) - initial                 # 第一个 BO 候选为 i=0
        if p and i >= 0:
            closest = np.linalg.norm(np.asarray(xs) - x, axis=1).min()
            near = near + 1 if closest < p["min_dis"] else 0
            if near >= 3:
                raise StopTrial("short_distance")
        y = sensor.measure(x)[0]
        xs.append(x.copy())
        ys.append(y)
        if p and i >= p["window"]:
            recent = ys[-p["window"]:]
            peak = max(recent)
            if peak > 0 and (peak - min(recent)) / peak < p["tol"]:
                raise StopTrial("window")
        return y

    kwargs = dict(search.BO_PARAMETERS, init_points=initial, max_iter=rounds)
    gp_restarts = 10
    if p:
        gp_restarts = p["gp_restarts"]
        kwargs.update(xi_init=p["xi_init"], xi_final=p["xi_final"],
                      sigma3_rel=p["sigma3_rel"], global_prob=p["global_prob"],
                      ls_bounds=(p["ls_lo"], p["ls_hi"]))
    # 上游每轮显式传入 n_restarts=10，只在本次调用内更换参数，退出即恢复。
    upstream_fit = search.fit_gp

    def fit_with_restarts(*args, **kw):
        kw["n_restarts"] = gp_restarts
        return upstream_fit(*args, **kw)

    try:
        with patch.object(search, "fit_gp", fit_with_restarts):
            search.bayesian_optimization(observe, search.BOUNDS, verbose=False,
                                         random_seed=seed, **kwargs)
        if len(ys) < initial + rounds:
            reason = "no_improve"
    except StopTrial as stop:
        reason = str(stop)
    best = int(np.argmax(ys))
    return xs[best], ys[best], spatial.predict(xs[best]), len(ys), reason


def plateau(model, seed, tolerance):
    """设备端三轴二分的离线对照。固定原点、步长、扫描顺序和三轴对称权重。"""
    spatial = surface(model)
    sensor = EDPFModel(spatial, seed=seed)
    seen = 0

    def read(x):
        nonlocal seen
        seen += 1
        return sensor.measure(x)[0]

    centers = np.zeros(3)
    # 原点共享，三轴先完成初扫，再按 X/Y/Z 顺序二分。
    origin = read(np.zeros(3))
    scans = []
    for axis in range(3):
        coords = np.arange(-300., 301., 50.)
        values = [origin if c == 0 else read(np.eye(3)[axis] * c) for c in coords]
        scans.append((coords, np.asarray(values)))
    detected = 0
    for axis, (coords, values) in enumerate(scans):
        peak_idx = int(np.argmax(values))
        peak = values[peak_idx]
        if peak <= 0:
            continue
        gap = np.abs(values - peak) / peak
        if min(gap[0], gap[-1]) < tolerance:
            continue
        detected += 1
        # 与设备端相同，正、负两个区间均按其 left/right 更新规则处理。
        def edge(reverse):
            subset = np.arange(peak_idx + 1) if reverse else np.arange(peak_idx, len(coords))
            left_matches = subset[gap[subset] >= tolerance if reverse else gap[subset] <= tolerance]
            right_matches = subset[gap[subset] <= tolerance if reverse else gap[subset] >= tolerance]
            if not len(left_matches) or not len(right_matches):
                return None
            left, right = coords[left_matches[-1]], coords[right_matches[0]]
            mid = None
            while right - left > (4., 4., 1.)[axis]:
                mid = (left + right) / 2
                if abs(read(np.eye(3)[axis] * mid) - peak) / peak <= tolerance:
                    left = mid
                else:
                    right = mid
            return mid

        plus, minus = edge(False), edge(True)
        if plus is not None or minus is not None:
            centers[axis] = ((coords[-1] if plus is None else plus)
                             + (coords[0] if minus is None else minus)) / 2
    observed = read(centers)
    # 独立于候选 tolerance 的质量标准，考察中心及轴向 ±10 的无抖动最差 ED。
    probes = np.vstack([centers, centers + 10*np.eye(3), centers - 10*np.eye(3)])
    clean = min(spatial.predict(x) for x in np.clip(probes, -300, 300))
    return centers, observed, clean, seen, f"axes_{detected}"
