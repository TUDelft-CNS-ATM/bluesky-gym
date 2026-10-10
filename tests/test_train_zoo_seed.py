"""
Regression test: ``train_zoo.py`` builds a seeded training env.

``make_vec_env`` must seed the scenario stream (and the action space used by
the random-policy baseline), so two runs train and evaluate on the same
scenarios. The SuperSuit/SB3 vector stack cannot pass a seed down
(``venv.seed`` raises AttributeError and ``reset`` takes no seed), so the
check here is end to end: two freshly built vec envs must give the same first
observations and the same random actions.

Run from the repo root, with the repo on PYTHONPATH (or installed via
``uv sync``)::

    python tests/test_train_zoo_seed.py
    python -m pytest tests/test_train_zoo_seed.py
"""

import numpy as np

import train_zoo


def _first_reset():
    venv = train_zoo.make_vec_env()
    obs = venv.reset()
    actions = np.stack([venv.action_space.sample() for _ in range(venv.num_envs)])
    venv.close()
    return obs, actions


def test_two_runs_start_on_the_same_scenario():
    obs1, actions1 = _first_reset()
    obs2, actions2 = _first_reset()
    assert np.array_equal(obs1, obs2), "first training scenario differs between two runs"
    assert np.array_equal(actions1, actions2), "random-baseline actions differ between two runs"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, test in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            test()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)
