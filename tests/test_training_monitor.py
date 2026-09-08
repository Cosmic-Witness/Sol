import copy
import hashlib
import math

from scripts import monitor_training as monitor


def valid_report():
    return dict(status='complete', steps=3, size=1024, backend='tpu', smoke_model=False,
                losses=[8., 7., 6.], seconds=[100., 2., 2.],
                checkpoint_reload=True, validation_forward=True,
                code_sha256=hashlib.sha256((monitor.ROOT/'kaggle/exp046_mask2_train/train.py').read_bytes()).hexdigest(),
                loss='Hungarian class+BCE+Dice')


def test_launch_gate_rejects_wrong_code_bad_losses_and_slow_training():
    good = valid_report()
    assert monitor.real_probe_passes(good)
    for change in [dict(code_sha256='stale'), dict(losses=[8., math.nan, 6.]),
                   dict(seconds=[100., 31., 2.]), dict(backend='cpu'),
                   dict(checkpoint_reload=False), dict(smoke_model=True), dict(size=256)]:
        bad = copy.deepcopy(good); bad.update(change)
        assert not monitor.real_probe_passes(bad)


def test_running_kernel_is_never_pushed_again(monkeypatch):
    monkeypatch.setattr(monitor, 'status', lambda slug: 'RUNNING')
    monkeypatch.setattr(monitor, 'log', lambda message: None)
    monkeypatch.setattr(monitor, 'command', lambda *a, **k: (_ for _ in ()).throw(AssertionError('duplicate push')))
    state = {}
    assert monitor.launch_once('training', 'unused', 1, state)
    assert state['training_launched']


def test_observation_failure_does_not_restart_job(monkeypatch):
    monkeypatch.setattr(monitor, 'status', lambda slug: 'UNKNOWN')
    monkeypatch.setattr(monitor, 'log', lambda message: None)
    monkeypatch.setattr(monitor, 'command', lambda *a, **k: (_ for _ in ()).throw(AssertionError('unverified push')))
    assert not monitor.launch_once('training', 'unused', 1, {})
