import unittest
from fastapi.testclient import TestClient

from backend.main import app, invalidate_match_response_cache


class TestCdnAndPayloadSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        invalidate_match_response_cache(None)

    def test_full_match_deduplicated_gzip_etag_and_304(self):
        res = self.client.get(
            "/api/matches/m_aus_eng_2026_odi_1",
            headers={"Accept-Encoding": "gzip", "Origin": "https://example.com"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-encoding"), "gzip")
        self.assertEqual(res.headers.get("access-control-allow-origin"), "*")
        self.assertNotIn("origin", (res.headers.get("vary") or "").lower())
        self.assertEqual(
            res.headers.get("cache-control"),
            "public, max-age=5, s-maxage=15, stale-while-revalidate=15",
        )
        etag = res.headers.get("etag")
        self.assertTrue(etag and etag.startswith('W/"'))

        body = res.json()
        self.assertEqual(body.get("view"), "full")
        self.assertIn("score_projections", body)
        self.assertNotIn("scenario_score_projections", body)
        self.assertNotIn("scenario_score_projections", body.get("match", {}))
        concussion = body.get("concussion_info") or {}
        self.assertNotIn("team_a_bench", concussion)
        self.assertNotIn("team_b_bench", concussion)
        self.assertIn("history", concussion)

        res_304 = self.client.get(
            "/api/matches/m_aus_eng_2026_odi_1",
            headers={"If-None-Match": etag},
        )
        self.assertEqual(res_304.status_code, 304)
        self.assertEqual(res_304.content, b"")
        self.assertEqual(res_304.headers.get("etag"), etag)

    def test_live_and_static_split_endpoints(self):
        full_res = self.client.get("/api/matches/m_aus_eng_2026_odi_1")
        live_res = self.client.get("/api/matches/m_aus_eng_2026_odi_1/live")
        static_res = self.client.get("/api/matches/m_aus_eng_2026_odi_1/static")

        self.assertEqual(full_res.status_code, 200)
        self.assertEqual(live_res.status_code, 200)
        self.assertEqual(static_res.status_code, 200)

        live_body = live_res.json()
        static_body = static_res.json()

        self.assertEqual(live_body.get("view"), "live")
        self.assertNotIn("team_a_squad", live_body)
        self.assertNotIn("team_b_squad", live_body)
        self.assertNotIn("matchup_tensor", live_body)
        self.assertNotIn("concussion_info", live_body)
        self.assertNotIn("positional_comparisons", live_body.get("match", {}))
        self.assertLess(len(live_res.content), len(full_res.content) * 0.5)

        self.assertEqual(static_body.get("view"), "static")
        self.assertIn("team_a_squad", static_body)
        self.assertIn("matchup_tensor", static_body)
        self.assertEqual(
            static_res.headers.get("cache-control"),
            "public, max-age=60, s-maxage=300, stale-while-revalidate=300",
        )

    def test_system_and_post_remain_no_store(self):
        sys_res = self.client.get("/api/system/cache-status")
        self.assertEqual(sys_res.status_code, 200)
        self.assertIn("no-store", sys_res.headers.get("cache-control", ""))

        health_res = self.client.get("/health")
        self.assertEqual(health_res.status_code, 200)
        self.assertIn("no-store", health_res.headers.get("cache-control", ""))


if __name__ == "__main__":
    unittest.main()
