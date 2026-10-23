import unittest

from services.authenticity_scorer import band_for, compute_score
from services.evidence import Evidence, Finding, Status, Strength
from services.tier6_classifier import classify


def ev(cat, status, *findings, **details):
    return Evidence(cat, status, "x", findings=list(findings), details=details)


def F(code, d, s):
    return Finding(code, code, d, s)


NOT = lambda c: ev(c, Status.NOT_PERFORMED)
NA = lambda c: ev(c, Status.NOT_APPLICABLE)


def base(**over):
    e = {"registry": NOT("registry"), "cryptography": NA("cryptography"),
         "document_integrity": ev("document_integrity", Status.PASS, F("no_integrity_issues", 1, Strength.WEAK)),
         "issuer_discovery": NOT("issuer_discovery"),
         "identity_and_details": ev("identity_and_details", Status.PASS, F("details_consistent", 1, Strength.WEAK)),
         "visual_model": NOT("visual_model")}
    e.update(over)
    return list(e.values())


class Scoring(unittest.TestCase):
    def test_no_evidence_is_neutral(self):
        s = compute_score([NOT("registry"), NA("cryptography"), NOT("visual_model")])
        self.assertAlmostEqual(s.score, 50.0)

    def test_deterministic(self):
        a, b = compute_score(base()), compute_score(base())
        self.assertEqual(a.score, b.score)

    def test_heuristics_alone_stay_inconclusive(self):
        s = compute_score(base())
        self.assertTrue(40 <= s.score < 62, s.score)
        self.assertEqual(s.band["key"], "inconclusive")
        self.assertEqual(s.reliability, "low")
        many_weak = ev("document_integrity", Status.INCONCLUSIVE, *[F(f"w{i}", -1, Strength.WEAK) for i in range(10)])
        s2 = compute_score(base(document_integrity=many_weak))
        self.assertGreater(s2.score, 20)   # capped: heuristics cannot condemn a document alone

    def test_trusted_signature_reaches_verified(self):
        c = ev("cryptography", Status.PASS, F("signature_valid_trusted", 1, Strength.STRONG))
        e = base(cryptography=c)
        s = compute_score(e)
        self.assertGreaterEqual(s.score, 90)
        self.assertEqual(classify(e, s)["classification"], "VERIFIED")

    def test_untrusted_signature_not_verified(self):
        c = ev("cryptography", Status.PASS, F("signature_valid_untrusted", 1, Strength.MODERATE))
        e = base(cryptography=c)
        s = compute_score(e)
        self.assertLess(s.score, 70)
        self.assertNotEqual(classify(e, s)["classification"], "VERIFIED")

    def test_altered_signature_is_strong_negative(self):
        c = ev("cryptography", Status.FAIL, F("signed_content_altered", -1, Strength.STRONG))
        s = compute_score(base(cryptography=c))
        self.assertLess(s.score, 20)
        self.assertEqual(s.band["key"], "likely_forged")

    def test_conflict_flagged_and_shrunk(self):
        c = ev("cryptography", Status.PASS, F("signature_valid_trusted", 1, Strength.STRONG))
        v = ev("visual_model", Status.FAIL, F("visual_prediction", -1, Strength.MODERATE),
               document_p_forged=0.95, document_ood="in_distribution")
        s = compute_score(base(cryptography=c, visual_model=v))
        self.assertTrue(s.conflict)
        self.assertTrue(30 < s.score < 70, s.score)
        e = base(cryptography=c, visual_model=v)
        self.assertNotEqual(classify(e, s)["classification"], "VERIFIED")
        self.assertTrue(classify(e, s)["warnings"])

    def test_visual_model_contribution_and_ood(self):
        v = ev("visual_model", Status.PASS, document_p_forged=0.05, document_ood="in_distribution")
        self.assertGreater(compute_score(base(visual_model=v)).score, 90)
        vb = ev("visual_model", Status.PASS, document_p_forged=0.05, document_ood="borderline")
        self.assertLess(compute_score(base(visual_model=vb)).score, compute_score(base(visual_model=v)).score)
        vo = ev("visual_model", Status.INCONCLUSIVE, document_p_forged=0.01, document_ood="out_of_distribution")
        self.assertAlmostEqual(compute_score(base(visual_model=vo)).score, compute_score(base()).score)

    def test_visual_extreme_probabilities_clipped(self):
        v = ev("visual_model", Status.FAIL, document_p_forged=1.0, document_ood="in_distribution")
        s = compute_score(base(visual_model=v))
        self.assertGreater(s.score, 0)
        self.assertLess(s.score, 10)

    def test_score_bounds(self):
        c = ev("cryptography", Status.FAIL, F("signed_content_altered", -1, Strength.STRONG),
               F("signature_invalid", -1, Strength.STRONG))
        v = ev("visual_model", Status.FAIL, document_p_forged=0.999, document_ood="in_distribution")
        s = compute_score(base(cryptography=c, visual_model=v))
        self.assertTrue(0 <= s.score <= 100)

    def test_withheld_when_unreadable_or_out_of_scope(self):
        s = compute_score(base(), readable=False)
        self.assertIsNone(s.score)
        self.assertEqual(s.band["key"], "unavailable")
        s = compute_score(base(), in_scope=False)
        self.assertIsNone(s.score)
        e = base()
        self.assertEqual(classify(e, s)["classification"], "INCONCLUSIVE")

    def test_missing_signature_and_model_not_negative(self):
        s = compute_score(base())
        self.assertFalse(any(c.logit < 0 for c in s.contributions))

    def test_band_edges(self):
        self.assertEqual(band_for(90)["key"], "strong_authentic")
        self.assertEqual(band_for(89.99)["key"], "likely_authentic")
        self.assertEqual(band_for(70)["key"], "likely_authentic")
        self.assertEqual(band_for(69.9)["key"], "inconclusive")
        self.assertEqual(band_for(40)["key"], "inconclusive")
        self.assertEqual(band_for(39.9)["key"], "suspicious")
        self.assertEqual(band_for(20)["key"], "suspicious")
        self.assertEqual(band_for(19.9)["key"], "likely_forged")
        self.assertEqual(band_for(0)["key"], "likely_forged")
        self.assertEqual(band_for(None)["key"], "unavailable")

    def test_noise_finding_halved_when_model_ran(self):
        integ = ev("document_integrity", Status.INCONCLUSIVE, F("noise_inconsistency", -1, Strength.WEAK))
        v = ev("visual_model", Status.PASS, document_p_forged=0.5, document_ood="in_distribution")
        without = compute_score(base(document_integrity=integ))
        with_m = compute_score(base(document_integrity=integ, visual_model=v))
        nc = lambda s: [c.logit for c in s.contributions if c.code == "noise_inconsistency"][0]
        self.assertAlmostEqual(nc(with_m), nc(without) / 2)


if __name__ == "__main__":
    unittest.main()
