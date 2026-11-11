import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from modal_control.core import Store

from modal_control import billing as b


def summary(credit='-9.46'):
    return {'metered_cost': '10.15', 'billed_cost': '0', 'adjustments': {'credits': credit}}


def monitor(tmp_path):
    store = Store(tmp_path / 'state')
    rid = store.create('example', {})
    return b.BillingMonitor(store, rid, {})


def test_exact_credit_delta_separate_from_cash_and_remaining(tmp_path):
    m = monitor(tmp_path)
    with patch.object(m, 'read', side_effect=[summary(), [], summary('-9.56'), []]):
        start = m.capture('start')
        later = m.capture('progress')
    assert start['amounts']['credits_used_usd'] == '9.46'
    assert later['workspace_delta']['credits_used_usd'] == '0.10'
    assert later['amounts']['charged_usd'] == '0'
    assert later['amounts']['credits_remaining_usd'] is None
    assert later['billing_final'] is False
    assert json.loads((m.store.root / 'latest_billing.json').read_text()) == later


def test_unavailable_is_not_zero_or_a_later_fake_baseline(tmp_path):
    m = monitor(tmp_path)
    with patch.object(m, 'read', side_effect=[RuntimeError('unsupported'), [], summary(), []]):
        start = m.capture('start')
        later = m.capture('progress')
    assert start['status'] == 'unavailable'
    assert 'amounts' not in start
    assert later['workspace_delta'] is None
    assert later['baseline']['status'] == 'unavailable'


def test_missing_credit_adjustment_remains_unknown():
    data = summary()
    data['adjustments'] = {}
    assert b.normalize_summary(data)['credits_used_usd'] is None


@pytest.mark.parametrize('value', ['NaN', 'Infinity', True, 'oops', None])
def test_invalid_money_rejected(value):
    with pytest.raises(ValueError):
        b.money(value)


def test_app_cost_uses_owned_ids_not_descriptions(tmp_path):
    m = monitor(tmp_path)
    child = m.store.create('batch-step:a', {'batch_run': m.run_id})
    m.store.event(child, 'cloud_apps', {'owned_ids': ['ap-owned']})
    other = m.store.create('unrelated', {})
    m.store.event(other, 'cloud_apps', {'owned_ids': ['ap-other']})
    ids = b.owned_app_ids(m.store, m.run_id)
    assert ids == {'ap-owned'}
    rows = [{'object_id': 'ap-owned', 'description': 'same', 'cost': '.1'},
            {'object_id': 'ap-other', 'description': 'same', 'cost': '100'},
            {'App ID': 'ap-owned', 'Cost': '.2'}]
    total, selected = b.app_total(rows, ids)
    assert total == '0.3'
    assert len(selected) == 2


def test_capture_errors_redact_credentials(tmp_path):
    m = monitor(tmp_path)
    with patch.object(m, 'read', side_effect=RuntimeError('secret=should-not-be-saved')):
        row = m.capture('start')
    assert 'should-not-be-saved' not in json.dumps(row)
    assert row['owned_app_cost_usd'] is None


def test_baseline_capture_precedes_background_refresh(tmp_path):
    m = monitor(tmp_path)
    with patch.object(m, 'capture') as capture:
        m.start()
        capture.assert_called_once_with('start')
        m.finish()
        assert [call.args[0] for call in capture.call_args_list] == ['start', 'end']


def test_month_rollover_does_not_create_negative_usage(tmp_path):
    m = monitor(tmp_path)
    with patch.object(m, 'read', side_effect=[summary(), [], summary('-1.0'), []]):
        m.capture('start')
        m.baseline['period'] = '2000-01'
        row = m.capture('progress')
    assert row['workspace_delta'] is None
