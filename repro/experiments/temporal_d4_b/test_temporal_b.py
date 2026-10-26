#!/usr/bin/env python3
"""CPU tests for the task-B intervention, selection and response metric.

Run: python -m unittest discover -s repro/experiments/temporal_d4_b -p "test_*.py"
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/mcy/Projects/GLANCE/repro")

import b_common as B  # noqa: E402
import run_forward as RF  # noqa: E402

RUN = os.environ.get("B_RUN", "")


class TestOrderMarker(unittest.TestCase):
    def test_neutral(self):
        for t in ["The video", "In a cozy, warmly lit indoor setting",
                  "A man in a yellow shirt", "The focus is on a young girl"]:
            flag, hits = B.has_order_marker(t)
            self.assertFalse(flag, (t, hits))

    def test_committal(self):
        for t in ["The video begins with a man",
                  "The video opens in a cozy kitchen",
                  "The scene then transitions to a fire pit",
                  "As the video progresses, the fire ignites"]:
            flag, _ = B.has_order_marker(t)
            self.assertTrue(flag, t)

    def test_returns_hits(self):
        flag, hits = B.has_order_marker("The video begins with a man")
        self.assertTrue(flag)
        self.assertTrue(any("begins" in h for h in hits))


class TestResponseMetric(unittest.TestCase):
    def test_ratio_and_na(self):
        toks = [1, 2, 3]
        lt_o = {1: -1.0, 2: -2.0, 3: -3.0}
        ld_o = {1: -1.0, 2: -2.0, 3: -3.0}
        lt_x = {1: -1.1, 2: -2.1, 3: -3.1}
        ld_x = {1: -1.05, 2: -2.05, 3: -3.05}
        r = RF.response(toks, lt_o, ld_o, lt_x, ld_x)
        self.assertAlmostEqual(r["L1_target"], 0.3, places=6)
        self.assertAlmostEqual(r["L1_draft"], 0.15, places=6)
        self.assertAlmostEqual(r["R"], 0.5, places=6)
        self.assertAlmostEqual(r["cos"], 1.0, places=6)

    def test_na_below_threshold(self):
        toks = [1, 2]
        lt_o = {1: -1.0, 2: -2.0}
        lt_x = {1: -1.001, 2: -2.001}
        r = RF.response(toks, lt_o, lt_o, lt_x, lt_x)
        self.assertLess(r["L1_target"], B.TARGET_L1_NA)
        self.assertIsNone(r["R"])
        self.assertIn("target L1", r["R_na_reason"])

    def test_cos_na_on_zero_vector(self):
        toks = [1, 2]
        lt = {1: -1.0, 2: -2.0}
        r = RF.response(toks, lt, lt, lt, lt)
        self.assertIsNone(r["cos"])

    def test_opposite_direction(self):
        toks = [1, 2]
        lt_o = {1: -1.0, 2: -2.0}
        ld_o = {1: -1.0, 2: -2.0}
        lt_x = {1: -2.0, 2: -2.0}   # target delta (-1, 0)
        ld_x = {1: -0.5, 2: -2.0}   # draft delta (+0.5, 0) -> cosine -1
        r = RF.response(toks, lt_o, ld_o, lt_x, ld_x)
        self.assertAlmostEqual(r["cos"], -1.0, places=6)
        self.assertAlmostEqual(r["R"], 0.5, places=6)


    def test_prob_space_secondary(self):
        toks = [1, 2]
        lt_o = {1: -1.0, 2: -2.0}
        lt_x = {1: -2.0, 2: -2.0}
        pt_o = {1: 0.5, 2: 0.25}
        pt_x = {1: 0.25, 2: 0.25}
        r = RF.response(toks, lt_o, lt_o, lt_x, lt_x, pt_o, pt_o, pt_x, pt_x)
        self.assertAlmostEqual(r["prob_tv_target"], 0.125, places=6)
        self.assertAlmostEqual(r["prob_R"], 1.0, places=6)

    def test_prob_na(self):
        toks = [1]
        lt = {1: -1.0}
        p = {1: 0.5}
        r = RF.response(toks, lt, lt, lt, lt, p, p, p, p)
        self.assertEqual(r["prob_tv_target"], 0.0)
        self.assertIsNone(r["prob_R"])


class TestConditionsOnDisk(unittest.TestCase):
    """Only runs when B_RUN points at a built run directory."""

    def setUp(self):
        if not RUN or not os.path.exists(os.path.join(RUN, "conditions",
                                                      "conditions.json")):
            self.skipTest("no built conditions (set B_RUN)")

    def test_T_is_the_reverse_and_K_is_the_mirror(self):
        import cv2
        import numpy as np
        rep = B.read_json(os.path.join(RUN, "conditions", "conditions.json"))
        for vid, e in rep["videos"].items():
            def rd(name):
                cap = cv2.VideoCapture(e["conditions"][name]["path"])
                fr = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    fr.append(f)
                cap.release()
                return fr
            o, t, k = rd("O_rec"), rd("T"), rd("K")
            self.assertEqual(len(o), 16)
            for i in range(16):
                self.assertLess(float(np.abs(
                    t[i].astype(np.int16) - o[15 - i].astype(np.int16)).mean()),
                    8.0, (vid, i))
                self.assertLess(float(np.abs(
                    k[i].astype(np.int16) - o[i][:, ::-1].astype(np.int16)).mean()),
                    8.0, (vid, i))

    def test_hashes_recorded(self):
        rep = B.read_json(os.path.join(RUN, "conditions", "conditions.json"))
        for vid, e in rep["videos"].items():
            for name in ("O_rec", "T", "K"):
                c = e["conditions"][name]
                self.assertEqual(c["sha256"], B.sha256_file(c["path"]))


class TestSelectionOnDisk(unittest.TestCase):
    def setUp(self):
        if not RUN or not os.path.exists(os.path.join(RUN, "stageA",
                                                      "selection.json")):
            self.skipTest("no selection (set B_RUN)")

    def test_caps_and_families(self):
        sel = B.read_json(os.path.join(RUN, "stageA", "selection.json"))
        for fam in ("N", "C"):
            per = {}
            for u in sel[f"family_{fam}_units"]:
                per[u["video"]] = per.get(u["video"], 0) + 1
                self.assertEqual(u["family"], fam)
                self.assertGreaterEqual(u["target_entropy_offset1_C6"], 0.30)
                if fam == "N":
                    self.assertFalse(u["order_marker"])
                else:
                    self.assertTrue(u["order_marker"])
            self.assertTrue(all(v <= 4 for v in per.values()), per)

    def test_prereg_frozen(self):
        if not os.path.exists(os.path.join(RUN, "prereg.json")):
            self.skipTest("no prereg")
        pre = B.read_json(os.path.join(RUN, "prereg.json"))
        self.assertEqual(pre["thresholds"]["target_l1_na"], 0.05)
        self.assertEqual(pre["thresholds"]["k_top"], 8)


if __name__ == "__main__":
    unittest.main()
