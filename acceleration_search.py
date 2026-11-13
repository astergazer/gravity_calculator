#!/usr/bin/env python3
"""Search constant speeds of two nested frames using a finite-time vector mean.

Python 3.9+, standard library only. See README.md for the coordinate convention.
R = Rz(theta) Rx(phi), gravity = (-g0, 0, 0), theta(0) = phi(0) = 0.
The outer z axis is horizontal; the inner x axis is initially vertical upward.
"""

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

Vector = Tuple[float, float, float]
Bounds = Tuple[float, float]
G0 = 9.80665
DURATION_S = 72.0 * 3600.0
RPM_TO_RAD_S = math.tau / 60.0
MODEL = "Rz_outer_Rx_inner; gravity=-x; zero_initial_angles; intersecting_axes"


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in values)


@dataclass(frozen=True)
class Problem:
    point_mm: Vector
    target_g: float
    outer_rpm: Bounds
    inner_rpm: Bounds
    tolerance_g: float = 1e-8
    duration_s: float = DURATION_S

    def __post_init__(self) -> None:
        if len(self.point_mm) != 3 or not _finite(self.point_mm):
            raise ValueError("評価点は有限の数値3個で指定してください。")
        if not math.isfinite(self.target_g) or self.target_g < 0:
            raise ValueError("目標値は0以上の有限の数値にしてください。")
        for name, bounds in (("外枠", self.outer_rpm), ("内枠", self.inner_rpm)):
            if len(bounds) != 2 or not _finite(bounds) or bounds[0] > bounds[1]:
                raise ValueError(name + "の範囲は有限の数値で、下限 <= 上限としてください。")
            if not math.isfinite(bounds[1] - bounds[0]):
                raise ValueError(name + "の探索範囲が計算可能な数値範囲を超えています。")
        if not math.isfinite(self.tolerance_g) or self.tolerance_g <= 0:
            raise ValueError("許容誤差は0より大きい有限の数値にしてください。")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("評価時間は0より大きい有限の数値にしてください。")

    @property
    def point_m(self) -> Vector:
        return tuple(v / 1000.0 for v in self.point_mm)


def trig_means(rpm: float, duration_s: float) -> Tuple[float, float]:
    """Return <cos(w*t)>, <sin(w*t)> on [0,T], including w=0.

    The half-angle form avoids cancellation in 1-cos(x). Taylor expansions
    remove the removable singularity at x=0. No infinite-time approximation.
    """
    phase = rpm * RPM_TO_RAD_S * duration_s
    if abs(phase) < 1e-4:
        squared = phase * phase
        return (1.0 - squared / 6.0 + squared * squared / 120.0,
                phase * (0.5 - squared / 24.0 + squared * squared / 720.0))
    return math.sin(phase) / phase, 2.0 * math.sin(phase / 2.0) ** 2 / phase


def mean_acceleration_g(point_m: Vector, outer_rpm: float, inner_rpm: float,
                        duration_s: float = DURATION_S) -> Vector:
    """Exact finite-time mean of effective acceleration in INNER coordinates, g.

    a_eff = R.T*g - omega x (omega x r) - alpha x r.
    omega = (v, u*sin(phi), u*cos(phi)); alpha = d(omega)/dt in body components.
    u=outer angular speed, v=inner angular speed, both constant relative speeds.
    """
    x, y, z = point_m
    u = outer_rpm * RPM_TO_RAD_S
    v = inner_rpm * RPM_TO_RAD_S
    co, _ = trig_means(outer_rpm, duration_s)
    ci, si = trig_means(inner_rpm, duration_s)
    c2i, s2i = trig_means(2.0 * inner_rpm, duration_s)
    cp, sp = trig_means(outer_rpm + inner_rpm, duration_s)
    cm, sm = trig_means(outer_rpm - inner_rpm, duration_s)

    # These combined inertial terms include BOTH centrifugal and Euler terms.
    inertial_x = u * u * x - 2.0 * u * v * (si * y + ci * z)
    inertial_y = (v * v + 0.5 * u * u * (1.0 + c2i)) * y - 0.5 * u * u * s2i * z
    inertial_z = (v * v + 0.5 * u * u * (1.0 - c2i)) * z - 0.5 * u * u * s2i * y
    answer = (-co + inertial_x / G0,
              0.5 * (sp + sm) + inertial_y / G0,
              -0.5 * (cm - cp) + inertial_z / G0)
    if not _finite(answer):
        raise ValueError("座標または速度が計算可能な数値範囲を超えています。")
    return answer


def instantaneous_acceleration_g(point_m: Vector, outer_rpm: float,
                                 inner_rpm: float, time_s: float) -> Vector:
    """Instantaneous body-frame effective acceleration; useful for validation."""
    u, v = outer_rpm * RPM_TO_RAD_S, inner_rpm * RPM_TO_RAD_S
    theta, phi = u * time_s, v * time_s
    s, c = math.sin(phi), math.cos(phi)
    omega = (v, u * s, u * c)
    alpha = (0.0, u * v * c, -u * v * s)

    def cross(a: Vector, b: Vector) -> Vector:
        return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0])

    centrifugal = cross(omega, cross(omega, point_m))
    euler = cross(alpha, point_m)
    gravity_g = (-math.cos(theta), math.sin(theta) * c, -math.sin(theta) * s)
    return tuple(gravity_g[j] - (centrifugal[j] + euler[j]) / G0 for j in range(3))


@dataclass(frozen=True)
class Candidate:
    outer_rpm: float
    inner_rpm: float
    mean_g: Vector

    @property
    def magnitude_g(self) -> float:
        return math.hypot(*self.mean_g)


@dataclass(frozen=True)
class SearchResult:
    candidate: Candidate
    error_g: float
    target_met: bool
    evaluations: int
    seed: int
    random_samples: int


class _Search:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.cache: Dict[Tuple[float, float], Candidate] = {}
        self.best: Optional[Candidate] = None
        self.low: Optional[Candidate] = None
        self.high: Optional[Candidate] = None

    def error(self, c: Candidate) -> float:
        return abs(c.magnitude_g - self.problem.target_g)

    def evaluate(self, outer: float, inner: float) -> Candidate:
        # Clipping also removes roundoff just outside an input bound.
        outer = max(self.problem.outer_rpm[0], min(self.problem.outer_rpm[1], outer))
        inner = max(self.problem.inner_rpm[0], min(self.problem.inner_rpm[1], inner))
        key = (outer, inner)
        if key not in self.cache:
            c = Candidate(outer, inner, mean_acceleration_g(
                self.problem.point_m, outer, inner, self.problem.duration_s))
            self.cache[key] = c
            if self.best is None or self.error(c) < self.error(self.best):
                self.best = c
            if self.low is None or c.magnitude_g < self.low.magnitude_g:
                self.low = c
            if self.high is None or c.magnitude_g > self.high.magnitude_g:
                self.high = c
        return self.cache[key]

    def can_finish(self) -> bool:
        if self.best is None:
            return False
        return (self.error(self.best) <= self.problem.tolerance_g or
                self.low.magnitude_g <= self.problem.target_g <= self.high.magnitude_g)

    def bisect_target(self) -> None:
        """A line segment stays inside the rectangular speed bounds.

        Once its endpoints bracket the target, continuity suffices: monotonicity
        is unnecessary, even for the narrow finite-time resonance features.
        """
        low, high = self.low, self.high
        if not (low.magnitude_g <= self.problem.target_g <= high.magnitude_g):
            return
        for _ in range(120):
            if self.error(self.best) <= self.problem.tolerance_g:
                return
            outer = low.outer_rpm + (high.outer_rpm - low.outer_rpm) * 0.5
            inner = low.inner_rpm + (high.inner_rpm - low.inner_rpm) * 0.5
            if (outer, inner) in ((low.outer_rpm, low.inner_rpm),
                                 (high.outer_rpm, high.inner_rpm)):
                return
            middle = self.evaluate(outer, inner)
            if middle.magnitude_g < self.problem.target_g:
                low = middle
            else:
                high = middle

    def refine_extreme(self, initial: Candidate, maximize: bool) -> None:
        """Bounded, derivative-free local search from sampled extrema."""
        spans = (self.problem.outer_rpm[1] - self.problem.outer_rpm[0],
                 self.problem.inner_rpm[1] - self.problem.inner_rpm[0])
        current = initial
        step = 0.08
        sign = -1.0 if maximize else 1.0
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (1, -1), (-1, 1), (-1, -1))
        for _ in range(180):
            winner = current
            for dx, dy in directions:
                c = self.evaluate(current.outer_rpm + dx * step * spans[0],
                                  current.inner_rpm + dy * step * spans[1])
                if self.can_finish():
                    return
                if sign * c.magnitude_g < sign * winner.magnitude_g:
                    winner = c
            if winner == current:
                step *= 0.5
                if step < 1e-13:
                    return
            else:
                current = winner


def _lerp(bounds: Bounds, fraction: float) -> float:
    return bounds[0] + (bounds[1] - bounds[0]) * fraction


def _seed_points(problem: Problem, samples: int, seed: int) -> Iterator[Bounds]:
    outer, inner = problem.outer_rpm, problem.inner_rpm
    yield _lerp(outer, 0.5), _lerp(inner, 0.5)
    for u in outer:
        for v in inner:
            yield u, v

    # A beat makes one revolution in T when its speed difference is 60/T rpm.
    # Sample strips around u=0, v=0 and u=+/-v, rather than hoping a grid hits them.
    offsets = [0.0]
    for cycles in (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0):
        delta = cycles * 60.0 / problem.duration_s
        offsets.extend((delta, -delta))
    for delta in offsets:
        for j in range(65):
            f = j / 64.0
            if outer[0] <= delta <= outer[1]:
                yield delta, _lerp(inner, f)
            if inner[0] <= delta <= inner[1]:
                yield _lerp(outer, f), delta
            same = (max(outer[0], inner[0] - delta),
                    min(outer[1], inner[1] - delta))
            if same[0] <= same[1]:
                u = _lerp(same, f)
                yield u, u + delta
            opposite = (max(outer[0], delta - inner[1]),
                        min(outer[1], delta - inner[0]))
            if opposite[0] <= opposite[1]:
                u = _lerp(opposite, f)
                yield u, -u + delta
    for i in range(33):
        for j in range(33):
            yield _lerp(outer, i / 32.0), _lerp(inner, j / 32.0)
    rng = random.Random(seed)
    for _ in range(samples):
        yield _lerp(outer, rng.random()), _lerp(inner, rng.random())


def search(problem: Problem, random_samples: int = 4096, seed: int = 20261113) -> SearchResult:
    """Return a tolerance-satisfying pair, or the best pair found (not a proof).

    Reproducible sampling, explicit resonance strips, local extrema refinement,
    and bracketed bisection. A global optimality/infeasibility certificate is
    intentionally not claimed by this finite numerical search.
    """
    if not isinstance(random_samples, int) or random_samples < 0:
        raise ValueError("追加探索点数は0以上の整数にしてください。")
    engine = _Search(problem)
    for u, v in _seed_points(problem, random_samples, seed):
        engine.evaluate(u, v)
        if engine.can_finish():
            break
    if not engine.can_finish():
        ordered = sorted(engine.cache.values(), key=lambda c: c.magnitude_g)
        starts = [(c, False) for c in ordered[:8]]
        starts += [(c, True) for c in ordered[-8:]]
        for c, maximize in starts:
            engine.refine_extreme(c, maximize)
            if engine.can_finish():
                break
    if engine.can_finish():
        engine.bisect_target()
    best = engine.best
    error = engine.error(best)
    return SearchResult(best, error, error <= problem.tolerance_g,
                        len(engine.cache), seed, random_samples)


def save_csv(problem: Problem, result: SearchResult, requested: Path) -> Path:
    """Save full precision; use a numbered filename instead of overwriting."""
    c = result.candidate
    row = {
        "status": "target_met" if result.target_met else "best_found",
        "outer_rpm": c.outer_rpm, "inner_rpm": c.inner_rpm,
        "outer_rad_s": c.outer_rpm * RPM_TO_RAD_S,
        "inner_rad_s": c.inner_rpm * RPM_TO_RAD_S,
        "target_g": problem.target_g, "mean_magnitude_g": c.magnitude_g,
        "absolute_error_g": result.error_g, "tolerance_g": problem.tolerance_g,
        "mean_x_g": c.mean_g[0], "mean_y_g": c.mean_g[1], "mean_z_g": c.mean_g[2],
        "point_x_mm": problem.point_mm[0], "point_y_mm": problem.point_mm[1],
        "point_z_mm": problem.point_mm[2], "duration_s": problem.duration_s,
        "outer_min_rpm": problem.outer_rpm[0], "outer_max_rpm": problem.outer_rpm[1],
        "inner_min_rpm": problem.inner_rpm[0], "inner_max_rpm": problem.inner_rpm[1],
        "g0_m_s2": G0, "model": MODEL, "averaging": "analytic_finite_time_vector_mean",
        "evaluations": result.evaluations, "seed": result.seed,
        "random_samples_requested": result.random_samples,
    }
    requested = Path(requested).expanduser()
    requested.parent.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        path = requested if index == 0 else requested.with_name(
            requested.stem + "_" + str(index) + requested.suffix)
        try:
            stream = path.open("x", encoding="utf-8-sig", newline="")
        except FileExistsError:
            continue
        with stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path.resolve()
    raise OSError("同名ファイルが多すぎるため、出力先を変更してください。")


def _prompt_numbers(prompt: str, count: int, validator) -> Tuple[float, ...]:
    while True:
        try:
            text = input(prompt).replace(",", " ").replace("，", " ")
            values = tuple(float(v) for v in text.split())
            if len(values) != count or not _finite(values):
                raise ValueError("有限の数値を" + str(count) + "個入力してください。")
            validator(values)
            return values
        except ValueError as error:
            print("入力エラー:", error, file=sys.stderr)


def _check_bounds(values: Sequence[float]) -> None:
    if values[0] > values[1]:
        raise ValueError("下限 <= 上限としてください。")


def _check_target(values: Sequence[float]) -> None:
    if values[0] < 0:
        raise ValueError("目標値は0以上にしてください。")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="二軸回転装置の72時間平均有効加速度を探索します。")
    parser.add_argument("--point-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--target-g", type=float)
    parser.add_argument("--outer-rpm", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--inner-rpm", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--tolerance-g", type=float, default=1e-8, help="絶対許容誤差（既定: 1e-8 g）")
    parser.add_argument("--samples", type=int, default=4096, help="追加の乱数探索点数（既定: 4096）")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--output", type=Path, default=Path("acceleration_result.csv"))
    args = parser.parse_args(argv)
    try:
        print("二軸回転装置: 72時間のベクトル平均有効加速度を探索")
        print("座標: x=初期の内軸（上向き）、z=水平な外軸、y=右手系。原点=軸の交点。")
        print("両角度は初期値0。反時計回りを正（各軸の正側から原点を見た場合）。")
        point = args.point_mm
        if point is None:
            point = _prompt_numbers("評価点 x y z [mm]: ", 3, lambda v: None)
        target = args.target_g
        if target is None:
            target = _prompt_numbers("目標値 [g]（平均ベクトルの大きさ）: ", 1, _check_target)[0]
        outer = args.outer_rpm
        if outer is None:
            outer = _prompt_numbers("外枠の下限 上限 [rpm]: ", 2, _check_bounds)
        inner = args.inner_rpm
        if inner is None:
            inner = _prompt_numbers("内枠の下限 上限 [rpm]: ", 2, _check_bounds)
        problem = Problem(tuple(point), target, tuple(outer), tuple(inner), args.tolerance_g)
        if args.samples < 0:
            raise ValueError("追加探索点数は0以上にしてください。")
        print("探索中...", flush=True)
        result = search(problem, args.samples, args.seed)
        path = save_csv(problem, result, args.output)
        c = result.candidate
        print("\n" + ("目標値に許容誤差内で到達しました。" if result.target_met else
                       "許容誤差内の解は見つかりませんでした。探索で得た最良候補を保存します。"))
        print("外枠: {:.17g} rpm / {:.17g} rad/s".format(c.outer_rpm, c.outer_rpm * RPM_TO_RAD_S))
        print("内枠: {:.17g} rpm / {:.17g} rad/s".format(c.inner_rpm, c.inner_rpm * RPM_TO_RAD_S))
        print("平均ベクトル [g]: ({:.12g}, {:.12g}, {:.12g})".format(*c.mean_g))
        print("平均ベクトルの大きさ: {:.12g} g".format(c.magnitude_g))
        print("目標との差: {:.12g} g（許容誤差 {:.12g} g）".format(result.error_g, problem.tolerance_g))
        print("評価した速度組数:", result.evaluations)
        print("CSV:", path)
        if not result.target_met:
            print("この結果は、範囲内での解の不存在や大域最適性を証明するものではありません。")
        return 0 if result.target_met else 2
    except (ValueError, OverflowError, OSError) as error:
        print("エラー:", error, file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\n入力または探索を中断しました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
