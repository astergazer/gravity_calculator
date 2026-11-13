"""Physics checks use R'' directly, independently of the implemented cross terms."""

import csv
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acceleration_search import (
    DURATION_S, G0, RPM_TO_RAD_S, Problem, instantaneous_acceleration_g,
    mean_acceleration_g, save_csv, search, trig_means,
)


def mmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def transpose(a):
    return tuple(zip(*a))


def mvec(a, x):
    return tuple(sum(row[k] * x[k] for k in range(3)) for row in a)


def inertial_reference(point_m, outer_rpm, inner_rpm, time_s):
    """Compute R.T*(gravity - d2(R*r)/dt2) using product derivatives."""
    u, v = outer_rpm * RPM_TO_RAD_S, inner_rpm * RPM_TO_RAD_S
    co, so = math.cos(u * time_s), math.sin(u * time_s)
    ci, si = math.cos(v * time_s), math.sin(v * time_s)
    a = ((co, -so, 0), (so, co, 0), (0, 0, 1))
    b = ((1, 0, 0), (0, ci, -si), (0, si, ci))
    da = ((-u*so, -u*co, 0), (u*co, -u*so, 0), (0, 0, 0))
    db = ((0, 0, 0), (0, -v*si, -v*ci), (0, v*ci, -v*si))
    dda = ((-u*u*co, u*u*so, 0), (-u*u*so, -u*u*co, 0), (0, 0, 0))
    ddb = ((0, 0, 0), (0, -v*v*ci, v*v*si), (0, -v*v*si, -v*v*ci))
    terms = (mmul(dda, b), mmul(da, db), mmul(a, ddb))
    ddr = tuple(tuple(terms[0][i][j] + 2*terms[1][i][j] + terms[2][i][j]
                     for j in range(3)) for i in range(3))
    position_acceleration = mvec(ddr, point_m)
    effective_world = tuple((-G0 if j == 0 else 0) - position_acceleration[j]
                            for j in range(3))
    return tuple(x / G0 for x in mvec(transpose(mmul(a, b)), effective_world))


def simpson_reference(point_m, outer_rpm, inner_rpm, duration_s, steps=2400):
    totals = [0.0, 0.0, 0.0]
    for i in range(steps + 1):
        weight = 1 if i in (0, steps) else (4 if i % 2 else 2)
        value = inertial_reference(point_m, outer_rpm, inner_rpm, duration_s*i/steps)
        for j in range(3):
            totals[j] += weight * value[j]
    return tuple(x / (3 * steps) for x in totals)


class PhysicsTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, tolerance=1e-11):
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, delta=tolerance)

    def test_stationary_and_vertical_axis_rotation(self):
        self.assertVectorClose(mean_acceleration_g((0.01, 0.02, -0.03), 0, 0), (-1, 0, 0))
        v = 6.25 * RPM_TO_RAD_S
        self.assertVectorClose(mean_acceleration_g((0.01, 0.02, -0.03), 0, 6.25),
                               (-1, v*v*0.02/G0, -v*v*0.03/G0))

    def test_instantaneous_matches_independent_inertial_derivatives(self):
        for outer, inner in ((7.31, 2.17), (-6.2, 5.5), (4, -4), (0, 3), (2, 0)):
            for t in (0, 0.123, 3.6, 45.7):
                with self.subTest(outer=outer, inner=inner, t=t):
                    point = (0.013, -0.024, 0.031)
                    self.assertVectorClose(
                        instantaneous_acceleration_g(point, outer, inner, t),
                        inertial_reference(point, outer, inner, t))

    def test_analytic_mean_matches_numeric_integration(self):
        point = (0.013, -0.024, 0.031)
        cases = ((2.31, -3.17, 43.25), (-1.6, 1.6, 67.3),
                 (0, -2.7, 53.7), (1.83, 0, 73.2),
                 (0.00317, 0.003170001, DURATION_S),
                 (0.000000000001, -0.000000000002, DURATION_S))
        for outer, inner, duration in cases:
            with self.subTest(outer=outer, inner=inner, duration=duration):
                self.assertVectorClose(
                    mean_acceleration_g(point, outer, inner, duration),
                    simpson_reference(point, outer, inner, duration), 3e-8)

    def test_72h_known_periodic_and_resonant_results(self):
        # 72 h contains an integer number of periods for these actual rpm values.
        point = (0.013, -0.024, 0.031)
        u, v = 7.0 * RPM_TO_RAD_S, 3.5 * RPM_TO_RAD_S
        expected = (u*u*point[0]/G0,
                    (v*v + u*u/2)*point[1]/G0,
                    (v*v + u*u/2)*point[2]/G0)
        self.assertVectorClose(mean_acceleration_g(point, 7, 3.5), expected)
        self.assertVectorClose(mean_acceleration_g((0, 0, 0), 7, 7), (0, 0, -0.5))
        self.assertVectorClose(mean_acceleration_g((0, 0, 0), 7, -7), (0, 0, 0.5))

    def test_vector_mean_precedes_magnitude(self):
        point = (0, 0, 0)
        self.assertLess(math.hypot(*mean_acceleration_g(point, 1, 2)), 1e-12)
        for t in (0, 0.1, 5, 29.3):
            self.assertAlmostEqual(math.hypot(*instantaneous_acceleration_g(point, 1, 2, t)), 1)

    def test_zero_and_near_zero_trig_means(self):
        self.assertEqual(trig_means(0, DURATION_S), (1, 0))
        c, s = trig_means(1e-17, DURATION_S)
        self.assertAlmostEqual(c, 1)
        self.assertAlmostEqual(s, 1e-17*RPM_TO_RAD_S*DURATION_S/2, delta=1e-25)


class SearchTests(unittest.TestCase):
    def assertSolution(self, problem, result):
        self.assertTrue(result.target_met)
        self.assertLessEqual(result.error_g, problem.tolerance_g)
        c = result.candidate
        self.assertTrue(problem.outer_rpm[0] <= c.outer_rpm <= problem.outer_rpm[1])
        self.assertTrue(problem.inner_rpm[0] <= c.inner_rpm <= problem.inner_rpm[1])
        self.assertAlmostEqual(math.hypot(*mean_acceleration_g(
            problem.point_m, c.outer_rpm, c.inner_rpm)), problem.target_g,
            delta=problem.tolerance_g)

    def test_target_reached_near_narrow_resonance(self):
        p = Problem((0, 0, 0), 0.123456789, (1, 10), (1, 10))
        self.assertSolution(p, search(p, random_samples=100))

    def test_rotational_acceleration_target_with_negative_speeds(self):
        p = Problem((13, -24, 31), 0.005, (-10, -1), (1, 10))
        self.assertSolution(p, search(p, random_samples=100))

    def test_inner_speed_fixed_and_exact_zero_target(self):
        p = Problem((0, 0, 0), 0, (1, 10), (2, 2))
        self.assertSolution(p, search(p, random_samples=0))

    def test_tiny_bounds_around_known_reachable_target(self):
        outer, inner = 3.271828, -4.123456
        target = math.hypot(*mean_acceleration_g((0.01, 0.02, 0.03), outer, inner))
        p = Problem((10, 20, 30), target, (outer-1e-9, outer+2e-9),
                    (inner-2e-9, inner+1e-9), tolerance_g=1e-12)
        self.assertSolution(p, search(p, random_samples=10))

    def test_unmet_target_and_fixed_bounds(self):
        p = Problem((0, 0, 0), 2, (7, 7), (7, 7))
        r = search(p, random_samples=0)
        self.assertFalse(r.target_met)
        self.assertAlmostEqual(r.candidate.magnitude_g, 0.5)
        self.assertAlmostEqual(r.error_g, 1.5)

    def test_unmet_target_with_variable_bounds(self):
        p = Problem((0, 0, 0), 2, (1, 2), (-2, -1))
        r = search(p, random_samples=32)
        self.assertFalse(r.target_met)
        self.assertLessEqual(r.candidate.magnitude_g, 1.0)
        self.assertGreaterEqual(r.error_g, 1.0)

    def test_reproducible(self):
        p = Problem((0, 0, 0), 0.23456, (1, 10), (1, 10))
        self.assertEqual(search(p, 10, 42), search(p, 10, 42))

    def test_invalid_inputs(self):
        cases = (dict(target_g=-1), dict(target_g=float("nan")),
                 dict(point_mm=(0, float("inf"), 0)), dict(outer_rpm=(3, 2)),
                 dict(inner_rpm=(0, float("inf"))), dict(tolerance_g=0))
        for change in cases:
            values = dict(point_mm=(0, 0, 0), target_g=0.1, outer_rpm=(1, 10), inner_rpm=(1, 10))
            values.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                Problem(**values)


class OutputAndCliTests(unittest.TestCase):
    def test_csv_roundtrip_and_no_overwrite(self):
        p = Problem((0, 0, 0), 0.2, (1, 10), (1, 10))
        r = search(p, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            first = save_csv(p, r, path)
            original = first.read_bytes()
            second = save_csv(p, r, path)
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), original)
            with first.open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "target_met")
            self.assertEqual(float(row["outer_rpm"]), r.candidate.outer_rpm)
            self.assertEqual(float(row["duration_s"]), DURATION_S)
            mean = mean_acceleration_g(p.point_m, float(row["outer_rpm"]), float(row["inner_rpm"]))
            self.assertLessEqual(abs(math.hypot(*mean) - p.target_g), p.tolerance_g)

    def test_interactive_input_retry_and_csv(self):
        script = Path(__file__).with_name("acceleration_search.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interactive.csv"
            process = subprocess.run(
                [sys.executable, str(script), "--output", str(output)],
                input="nan 0 0\n0, 0, 0\n-1\n0.2\n10 1\n1 10\n1 10\n",
                text=True, capture_output=True, timeout=20)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(output.exists())
            self.assertIn("入力エラー", process.stderr)
            self.assertIn("目標値に許容誤差内で到達", process.stdout)

    def test_cli_failure_and_unmet_exit_codes(self):
        script = Path(__file__).with_name("acceleration_search.py")
        base = [sys.executable, str(script), "--point-mm", "0", "0", "0",
                "--outer-rpm", "7", "7", "--inner-rpm", "7", "7"]
        invalid = subprocess.run(base + ["--target-g", "-1"], text=True, capture_output=True)
        self.assertEqual(invalid.returncode, 1)
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run(base + ["--target-g", "2", "--samples", "0", "--output",
                                            str(Path(directory) / "unmet.csv")],
                                     text=True, capture_output=True, timeout=20)
            self.assertEqual(process.returncode, 2)
            self.assertIn("最良候補", process.stdout)


if __name__ == "__main__":
    unittest.main()
