"""Synthetic contract tests; no provider, audio, network, or process execution."""
from dataclasses import FrozenInstanceError, replace
import json
import unittest

from batch_dubbing.isochrony import ActivityTimeline, SpeechInterval
from batch_dubbing.isochrony_backend import BackendCapabilities, SynthesisRequest, plan_candidates


def capabilities(**changes):
    defaults = dict(backend_id='synthetic-provider', revision='revision-a', languages=('en', 'de', 'fr'),
                    duration_conditioning=True, internal_pause_conditioning=True,
                    capability_evidence='synthetic-test-specification')
    return BackendCapabilities(**{**defaults, **changes})


def activity(**changes):
    defaults = dict(duration_seconds=5., speech=(SpeechInterval(.2, 2.), SpeechInterval(3., 4.8)),
                    provenance='synthetic', resolution_seconds=.01, evidence_id='synthetic-source-1')
    return ActivityTimeline(**{**defaults, **changes})


def request(**changes):
    defaults = dict(source_id='source-unit-1', text='Ein Beispiel. Noch ein Satz.', language='de',
                    target_duration_seconds=5., target_activity=activity())
    return SynthesisRequest(**{**defaults, **changes})


def plan(req=None, cap=None, **changes):
    return plan_candidates(cap or capabilities(), req or request(),
                           **{'seeds': (41, 42), 'max_candidates': 2, **changes})


class IsochronyBackendTests(unittest.TestCase):
    def test_plan_is_immutable_serial_no_retry_and_does_not_claim_audio(self):
        req = request()
        specs = plan(req)
        self.assertIsInstance(specs, tuple)
        self.assertEqual([spec.seed for spec in specs], [41, 42])
        for spec in specs:
            self.assertIs(spec.request, req)
            self.assertEqual(spec.max_concurrency, 1)
            self.assertEqual(spec.retries, 0)
            self.assertFalse(spec.waveform_time_stretching_allowed)
            self.assertFalse(spec.speech_trimming_allowed)
            with self.assertRaises(FrozenInstanceError):
                spec.seed = 43
            self.assertEqual(spec.to_dict()['status'], 'planned_unverified')
            self.assertTrue(spec.to_dict()['requires_content_and_listening_review'])
            json.dumps(spec.to_dict(), allow_nan=False)

    def test_unsupported_duration_cannot_silently_fall_back_to_speed_control(self):
        cap = capabilities(duration_conditioning=False, internal_pause_conditioning=False)
        with self.assertRaisesRegex(ValueError, 'duration conditioning'):
            plan(request(target_activity=None), cap)

    def test_pause_pattern_cannot_silently_become_total_duration_only(self):
        cap = capabilities(internal_pause_conditioning=False)
        with self.assertRaisesRegex(ValueError, 'internal speech/pause'):
            plan(cap=cap)
        self.assertEqual(len(plan(request(target_activity=None), cap)), 2)

    def test_undeclared_language_rejected(self):
        with self.assertRaisesRegex(ValueError, 'language'):
            plan(request(language='es'))

    def test_no_hidden_waveform_or_internal_rate_change(self):
        for field in ('playback_rate', 'speaking_rate_scale'):
            for value in (.99999999, 1.00000001, .65, 1.15, 2, True, '1', None, float('nan')):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    request(**{field: value})
        req = request(playback_rate=1, speaking_rate_scale=1)
        self.assertIs(type(req.playback_rate), float)
        self.assertEqual(req.speaking_rate_scale, 1.)

    def test_capability_fields_are_explicit_and_not_verification(self):
        for value in (None, 0, 1, 'unknown'):
            for field in ('duration_conditioning', 'internal_pause_conditioning'):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    capabilities(**{field: value})
        result = capabilities().to_dict()
        self.assertEqual(result['evidence_scope'], 'declared_capability_only')
        self.assertFalse(result['achieved_timing_verified'])
        self.assertFalse(result['naturalness_verified'])
        with self.assertRaises(TypeError):
            BackendCapabilities('id', 'revision', ('de',), True)

    def test_capability_identity_and_languages_are_validated(self):
        for changes in ({'backend_id': ''}, {'revision': ' '}, {'capability_evidence': ''},
                        {'languages': []}, {'languages': ()}, {'languages': ('de', 'de')},
                        {'languages': ('German',)}, {'languages': (None,)},
                        {'duration_conditioning': False, 'internal_pause_conditioning': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                capabilities(**changes)
        for changes in ({'capability_evidence': 'x' * 4097}, {'backend_id': 'x' * 4097},
                        {'languages': ('de-' + 'x' * 1000,)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                capabilities(**changes)

    def test_invalid_request_or_mismatched_timeline_rejected(self):
        for changes in ({'source_id': ''}, {'text': ' '}, {'language': None},
                        {'text': 'x' * 10001}, {'target_duration_seconds': 6.},
                        {'target_activity': {}}, {'target_activity': activity(speech=())}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                request(**changes)
        for duration in (0, -1, 7201, float('inf'), float('nan'), True, '5'):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                request(target_duration_seconds=duration, target_activity=None)

    def test_budget_and_seed_bounds_enforced_before_preparing_attempts(self):
        for budget in (0, -1, 9, True, 1.5, None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                plan(max_candidates=budget)
        for seeds in ((), [41], (41, 41), (True,), (-1,), (2**32,), ('41',), (None,), (1, 2, 3)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                plan(seeds=seeds)
        self.assertEqual(len(plan(seeds=tuple(range(8)), max_candidates=8)), 8)
        self.assertEqual(len(plan(seeds=(0, 2**32 - 1), max_candidates=2)), 2)

    def test_ids_bind_every_timeline_field_and_exact_text(self):
        baseline = plan()[0].candidate_id
        self.assertEqual(baseline, plan()[0].candidate_id)
        modified_requests = [request(text='Ein Beispiel. Noch ein anderer Satz.'),
                             request(source_id='source-unit-2'), request(language='fr'),
                             request(target_activity=None)]
        for timeline in (activity(speech=(SpeechInterval(.3, 2.), SpeechInterval(3., 4.8))),
                         activity(speech=(SpeechInterval(.2, 2.1), SpeechInterval(3., 4.8))),
                         activity(resolution_seconds=.02), activity(evidence_id='synthetic-source-2'),
                         activity(provenance='manual_review')):
            modified_requests.append(request(target_activity=timeline))
        modified_requests.append(request(target_duration_seconds=6., target_activity=activity(duration_seconds=6.)))
        for req in modified_requests:
            with self.subTest(request=req):
                self.assertNotEqual(baseline, plan(req)[0].candidate_id)
        self.assertNotEqual(baseline, plan(seeds=(42,), max_candidates=1)[0].candidate_id)
        self.assertNotEqual(baseline, plan(cap=capabilities(revision='revision-b'))[0].candidate_id)
        self.assertNotEqual(baseline, plan(cap=capabilities(capability_evidence='new-evidence'))[0].candidate_id)

    def test_request_keeps_exact_multilingual_text_and_timing(self):
        for language, text in (('en', 'One sentence.'), ('de', 'Grüße, äußere Änderungen.'),
                               ('fr', 'Une phrase, déjà écrite.')):
            req = request(language=language, text=text)
            before = req.to_dict()
            specs = plan(req)
            self.assertEqual(req.to_dict(), before)
            self.assertEqual(specs[0].request.text, text)
            self.assertEqual(specs[0].request.target_activity.evidence_id, 'synthetic-source-1')
            with self.assertRaises(FrozenInstanceError):
                req.text = 'different'

    def test_typed_contract_required(self):
        for cap, req in (({}, request()), (capabilities(), {})):
            with self.assertRaisesRegex(ValueError, 'Typed'):
                plan_candidates(cap, req, seeds=(41,), max_candidates=1)


if __name__ == '__main__':
    unittest.main()
