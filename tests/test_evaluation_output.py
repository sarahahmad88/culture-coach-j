"""Mocked completion/contract checks; not evidence of live model output quality."""
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from groq.types.chat import ChatCompletion
from config import Settings
from schemas import Attempt,State,Evidence,AdaptationRating
from llm_client import LiveProvider,make_llm,Budget,EVALUATION_OUTPUT_TOKENS
from orchestrator import Controller
from practice_scoring import build_rubric
from repository import export_attempts,import_attempts
from tests.test_core import config
from tests.test_practice_scoring import SyntheticProvider
from diagnostics import StageError
from knowledge import registry


def pending(level='Advanced',legacy=False):
    a=Attempt(config=config(level),mode='live');c=Controller(SyntheticProvider());c.prepare(a)
    if legacy:
        rubric=build_rubric(a.config,version='practice-v1.0')
        a.scenario=a.scenario.model_copy(update={'practice_rubric':rubric,'relevant_dimensions':tuple(registry()['templates'][a.config.template_id]['candidate_dimensions']),
            'opening':' '.join(x.expectation for x in rubric.adaptation)})
    c.begin(a)
    # Include full-length turns; evaluation output need not repeat them.
    for i in range(a.limit):
        c.submit(a,'I can explain risks and propose a practical option. '+('x'*2900)+str(i),str(i))
    a.state=State.evaluating;a.evaluation=None;a.feedback=None;a.scorecard=None;a.practice_scores=None
    return a


def compact_assessment(a):
    result=SyntheticProvider().run('evaluation_coach',a)
    learner=next(t for t in a.transcript if t.role=='learner');cue=a.transcript[0]
    result.evaluation.evidence=[Evidence(turn_id=learner.id,quote=learner.text[:49],observation='Proposes an option.',
        interpretation='Offers a practical next step.',uncertainty='Synthetic fixture.',dimension_mappings=list(a.scenario.relevant_dimensions))]
    result.evaluation.adaptation_ratings=[AdaptationRating(criterion_id=x.id,value=4,evidence_turn_ids=[learner.id],
        reason='Synthetic anchor and cue fixture.',partner_cue_turn_id=cue.id,partner_cue_quote=x.expectation)
        for x in a.scenario.practice_rubric.adaptation]
    result.evaluation.uncertainty='Synthetic ratings verify contracts, not coaching accuracy.'
    result.feedback.strengths=['Offers an option.'];result.feedback.improvements=['Invite agreement on scope.']
    result.feedback.alternative='Could we agree on a limited beta and protect the testing schedule?'
    result.feedback.explanation='One possible way to negotiate scope.'
    return result


def response(content,finish='stop'):
    return ChatCompletion(id='mock',object='chat.completion',created=0,model='mock',choices=[{
        'index':0,'finish_reason':finish,'message':{'role':'assistant','content':content}}])


class EvaluationOutputTests(unittest.TestCase):
    def test_compact_complete_scorecards_all_levels_and_legacy(self):
        for level,legacy in [('Beginner',False),('Intermediate',False),('Advanced',False),('Advanced',True)]:
            with self.subTest(level=level,legacy=legacy):
                a=pending(level,legacy);expected=compact_assessment(a);raw=expected.model_dump_json();before=a.model_copy(deep=True)
                # Conservative serialized-size check, not a provider tokenizer measurement.
                self.assertLess(len(raw.encode('utf-8')),EVALUATION_OUTPUT_TOKENS)
                payloads=[]
                def create(**payload):payloads.append(payload);return response(raw)
                client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                with patch('groq.Groq',return_value=client):Controller(LiveProvider(Settings(groq_api_key='mock'))).finish(a)
                self.assertEqual(a.state,State.completed);self.assertEqual(a.evaluation,expected.evaluation)
                self.assertEqual(a.feedback,expected.feedback);self.assertEqual(a.transcript,before.transcript)
                self.assertEqual(a.practice_scores.communication.coverage,1);self.assertEqual(a.practice_scores.adaptation.coverage,1)
                self.assertEqual(import_attempts(export_attempts([a]))[0],a)
                self.assertEqual(len(payloads),1);self.assertEqual(a.output_tokens_used,EVALUATION_OUTPUT_TOKENS)
                self.assertEqual(payloads[0]['max_completion_tokens'],5000)
                prompt='\n'.join(m['content'] for m in payloads[0]['messages'])
                self.assertEqual(prompt.count('Feedback: give up to two strengths'),1)
                self.assertNotIn('Return evidence for EVERY learner turn',prompt)
                self.assertEqual(prompt.count(json.dumps(a.scenario.practice_rubric.model_dump(),ensure_ascii=False,separators=(',',':'))),1)
                self.assertIn(before.transcript[1].text,prompt)

    def run_outputs(self,outputs):
        a=pending();raw=compact_assessment(a).model_dump_json();requests=[]
        def create(**payload):
            requests.append(payload);content,finish=outputs[len(requests)-1]
            return response(raw if content=='complete' else content,finish)
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return a,requests,client

    def test_length_finish_repaired_once_even_for_parseable_json(self):
        a,requests,client=self.run_outputs([('complete','length'),('complete','stop')])
        with patch('groq.Groq',return_value=client):Controller(LiveProvider(Settings(groq_api_key='mock'))).finish(a)
        self.assertEqual(a.state,State.completed);self.assertEqual(len(requests),2)
        self.assertIn('previous response reached the output limit','\n'.join(m['content'] for m in requests[-1]['messages']))
        self.assertEqual(a.output_tokens_used,10000)

    def test_repeated_truncation_preserves_pending_attempt_without_partial_scores(self):
        a,requests,client=self.run_outputs([('{"evaluation":','length'),('complete','length')]);before=a.model_copy(deep=True)
        with patch('groq.Groq',return_value=client):
            with self.assertRaises(StageError) as error:Controller(LiveProvider(Settings(groq_api_key='mock'))).finish(a)
        self.assertEqual(error.exception.kind,'truncated output');self.assertEqual(len(requests),2)
        self.assertEqual(a.state,State.evaluating);self.assertEqual(a.transcript,before.transcript)
        self.assertIsNone(a.evaluation);self.assertIsNone(a.feedback);self.assertIsNone(a.practice_scores)

    def test_unmarked_incomplete_json_repaired_once(self):
        a,requests,client=self.run_outputs([('{"evaluation":','stop'),('complete','stop')])
        with patch('groq.Groq',return_value=client):Controller(LiveProvider(Settings(groq_api_key='mock'))).finish(a)
        self.assertEqual(a.state,State.completed);self.assertEqual(len(requests),2)

    def test_gemini_fallback_uses_same_smaller_cap(self):
        from tests.test_provider_fallback import client
        from tests.test_http_retry import HTTPError
        a=pending();settings=Settings(groq_api_key='mock',gemini_api_key='mock-google')
        first=client(HTTPError(413));google=client(compact_assessment(a).model_dump_json())
        llm=make_llm(settings,Budget(a,settings),'evaluation_coach',first,gemini_client=google)
        llm.call('authorized context')
        self.assertEqual(first.chat.completions.create.call_args.kwargs['max_completion_tokens'],5000)
        self.assertEqual(google.chat.completions.create.call_args.kwargs['max_tokens'],5000)
        self.assertEqual(a.output_tokens_used,10000)
