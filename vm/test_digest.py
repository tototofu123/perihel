#!/usr/bin/env python3
"""Tests for vm/digest.py repo enumeration + section cap.

Stdlib unittest only. Run from vm/:  python3 -m unittest test_digest -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import digest


def _summary(pulls=(), issues=(), error=None):
    def item(n, title, updated):
        return {"number": n, "title": title,
                "url": f"https://github.com/x/y/{n}",
                "updated_at": updated}
    return {"pulls": [item(*p) for p in pulls],
            "issues": [item(*i) for i in issues],
            "error": error}


def _sections(repos):
    return {
        "news": {"available": True, "error": None, "items": [], "count": 0},
        "repos": {"available": True, "repos": repos},
        "plans": {"note": None, "items": [], "links": {}},
    }


def _data(repos):
    return {"date": "Mon 23 Nov 2026", "sections": _sections(repos)}


class ParseTests(unittest.TestCase):
    def test_parse_repos_keeps_valid_drops_junk(self):
        self.assertEqual(
            digest.parse_repos("a/b, c/d ,, /bad, bad/, a/b/c,  e/f "),
            ["a/b", "c/d", "e/f"])
        self.assertEqual(digest.parse_repos(""), [])
        self.assertEqual(digest.parse_repos(None), [])

    def test_parse_names(self):
        self.assertEqual(digest.parse_names("a, b ,, c "), ["a", "b", "c"])
        self.assertEqual(digest.parse_names(""), [])
        self.assertEqual(digest.parse_names(None), [])


class GetReposTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        self._org = digest.list_org_repos
        self._user = digest.list_user_repos
        for k in ("PERIHEL_DIGEST_REPOS", "PERIHEL_DIGEST_ORGS",
                  "PERIHEL_DIGEST_USERS"):
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        digest.list_org_repos = self._org
        digest.list_user_repos = self._user

    def test_combines_explicit_orgs_users_and_dedupes(self):
        os.environ["PERIHEL_DIGEST_REPOS"] = "tototofu123/agent-bus"
        digest.list_org_repos = lambda org: ["HKUST-Climbing-Society/hkust-climbing-app",
                                             "tototofu123/agent-bus"]
        digest.list_user_repos = lambda user: ["tototofu123/PERIHEL",
                                               "tototofu123/agent-bus"]
        # defaults would also add the org + user; stub them to the same
        # lists so the result is deterministic
        got, notes = digest.get_repos()
        self.assertEqual(got, ["tototofu123/agent-bus",
                               "HKUST-Climbing-Society/hkust-climbing-app",
                               "tototofu123/PERIHEL"])
        self.assertEqual(notes, [])

    def test_falls_back_to_defaults_when_everything_empty(self):
        digest.list_org_repos = lambda org: []
        digest.list_user_repos = lambda user: []
        got, notes = digest.get_repos()
        self.assertEqual(got, digest.DEFAULT_REPOS)
        self.assertEqual(len(notes), 2)
        self.assertTrue(any("HKUST-Climbing-Society" in n for n in notes))
        self.assertTrue(any("tototofu123" in n for n in notes))

    def test_note_names_an_empty_listing(self):
        digest.list_org_repos = lambda org: []
        digest.list_user_repos = lambda user: ["tototofu123/perihel"]
        got, notes = digest.get_repos()
        self.assertEqual(got, ["tototofu123/perihel"])
        self.assertEqual(len(notes), 1)
        self.assertIn("HKUST-Climbing-Society", notes[0])

    def test_env_empty_disables_a_source(self):
        os.environ["PERIHEL_DIGEST_ORGS"] = ""
        os.environ["PERIHEL_DIGEST_USERS"] = ""
        digest.list_org_repos = lambda org: (_ for _ in ()).throw(
            AssertionError("orgs disabled"))
        digest.list_user_repos = lambda user: (_ for _ in ()).throw(
            AssertionError("users disabled"))
        got, notes = digest.get_repos()
        self.assertEqual(got, digest.DEFAULT_REPOS)
        self.assertEqual(notes, [])


class FormatTests(unittest.TestCase):
    def test_caps_items_per_repo_and_orders_by_recency(self):
        repos = {
            "org/old": _summary(
                pulls=[(1, "old pr", "2026-11-01T00:00:00Z")],
                issues=[(2, "old issue", "2026-11-01T00:00:00Z")]),
            "org/new": _summary(
                pulls=[(10, "new pr", "2026-11-20T00:00:00Z"),
                       (11, "new pr 2", "2026-11-19T00:00:00Z"),
                       (12, "new pr 3", "2026-11-18T00:00:00Z"),
                       (13, "new pr 4", "2026-11-17T00:00:00Z")],
                issues=[(14, "new issue", "2026-11-21T00:00:00Z")]),
        }
        text = digest.format_digest(_data(repos))
        # newest group first
        self.assertLess(text.index("org/new"), text.index("org/old"))
        # at most 3 items shown for org/new (4 pulls + 1 issue fetched)
        self.assertIn("new issue", text)
        self.assertIn("new pr", text)
        self.assertNotIn("new pr 4", text)
        # merged ordering: newest item first inside the group
        self.assertLess(text.index("new issue"), text.index("new pr"))

    def test_section_cap_and_summary_lines(self):
        repos = {}
        for i in range(12):
            repos[f"org/r{i}"] = _summary(
                issues=[(1, f"issue {i}", f"2026-11-{10 + i:02d}T00:00:00Z")])
        repos["org/quiet"] = _summary()
        repos["org/broken"] = _summary(error="org/broken: unreachable (x)")
        text = digest.format_digest(_data(repos))
        lines = text.split("\n")
        # repos section lines (between the header and the plans header)
        start = next(i for i, l in enumerate(lines) if "Repos" in l)
        end = next(i for i, l in enumerate(lines) if "Plans" in l)
        section = lines[start:end]
        # group lines (headers + items for the org/r* repos) stay
        # within the cap; errors and summary lines sit outside it
        group_lines = [l for l in section
                       if l.startswith("<b>org/r") or l.startswith("• 🔀")
                       or l.startswith("• ❗")]
        self.assertLessEqual(len(group_lines), digest.REPO_SECTION_MAX_LINES,
                             "\n".join(section))
        self.assertTrue(any("more repo" in l and "open items" in l
                            for l in section), "\n".join(section))
        self.assertTrue(any("nothing open in 1 other repo" in l
                            for l in section), "\n".join(section))
        self.assertTrue(any("org/broken: unreachable" in l
                            for l in section), "\n".join(section))

    def test_empty_repos(self):
        text = digest.format_digest(_data({}))
        self.assertIn("no repos configured", text)


if __name__ == "__main__":
    unittest.main()
