import unittest
from unittest.mock import patch
from config import Settings,CULTURES
from schemas import Attempt,State
from tests.test_core import config,started
from tests.test_provider import FakeClient
from tests.test_practice_scoring import scored,SyntheticProvider
from llm_client import LiveProvider
from orchestrator import Controller
from practice_scoring import build_rubric,focus_dimensions,practice_scores,reference
from scoring import validate_evaluation,score
from knowledge import registry
from repository import export_attempts,import_attempts
from demo_provider import DemoProvider
from diagnostics import StageError

class SimplifiedAgentTests(unittest.TestCase):
    def test_real_crewai_request_counts_for_all_levels(self):
        for level,expected in [('Beginner',4),('Intermediate',9),('Advanced',16)]:
            a=Attempt(config=config(level),mode='live');client=FakeClient(a)
            class Provider(LiveProvider):
                def run(self,stage,attempt):client.stage=stage;return super().run(stage,attempt)
            with patch('groq.Groq',return_value=client):
                c=Controller(Provider(Settings(groq_api_key='mock')));c.prepare(a);c.begin(a)
                for i in range(a.limit):c.submit(a,'A legitimate concern and option '+str(i),str(i))
            self.assertEqual(a.state,State.completed)
            self.assertEqual(len(client.payloads),expected);self.assertEqual(a.calls_used,expected)
            self.assertEqual(client.tool_stages,{'scenario_designer','conversation_partner','evaluation_coach'})
            self.assertTrue(all('tools' not in p for p in client.payloads))

    def test_exactly_two_available_dimensions_for_all_countries_and_templates(self):
        for country in CULTURES:
            for template in ['deadline','group_project']:
                cfg=config(template=template).model_copy(update={'other_culture':country})
                focus=focus_dimensions(cfg);rubric=build_rubric(cfg)
                self.assertEqual(len(set(focus)),2)
                self.assertEqual({c.dimension for c in rubric.adaptation},set(focus))
                self.assertTrue(all(reference()['countries'][country]['scores'][d] is not None for d in focus))
        self.assertEqual(focus_dimensions(config()),('Power Distance','Uncertainty Avoidance'))
        self.assertEqual(focus_dimensions(config().model_copy(update={'other_culture':'Egypt'})),('Long-Term Orientation','Indulgence'))

    def test_generator_focus_contract_rejects_other_dimensions(self):
        class Wrong(DemoProvider):
            def run(self,stage,a):
                result=super().run(stage,a)
                if stage=='scenario_designer':result.focus_dimensions=('Long-Term Orientation','Indulgence')
                return result
        a=Attempt(config=config());c=Controller(Wrong())
        with self.assertRaises(StageError):c.prepare(a)
        self.assertEqual(a.state,State.failed);self.assertIsNone(a.scenario)

    def test_sparse_exact_evidence_is_valid_and_referenced_evidence_is_required(self):
        a=scored();e=a.evaluation.model_copy(deep=True)
        used={i for rating in e.communication_ratings+e.adaptation_ratings for i in rating.evidence_turn_ids}
        e.evidence=[x for x in e.evidence if x.turn_id in used]
        self.assertLess(len(e.evidence),a.rounds)
        validate_evaluation(a,e,registry())
        e.evidence=[]
        with self.assertRaises(ValueError):validate_evaluation(a,e,registry())

    def test_previous_six_dimension_rubric_exports_remain_valid(self):
        a,c=started()
        old=build_rubric(a.config,version='practice-v1.0')
        a.scenario=a.scenario.model_copy(update={'practice_rubric':old,'relevant_dimensions':tuple(registry()['templates']['deadline']['candidate_dimensions'])})
        for i in range(a.limit):c.submit(a,'Legacy response '+str(i),str(i))
        self.assertEqual(len(a.practice_scores.adaptation.dimensions),6)
        self.assertEqual(import_attempts(export_attempts([a]))[0],a)
