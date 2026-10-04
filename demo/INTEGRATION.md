# Integration referee checklist (Dev 4, task 4.7)

Rule: **`python -m src.sim` and `pytest -q` pass before ANY merge to `main`.** Dev 4 performs the merges at C1–C3.

| Gate | Command (from repo root) | Pass means |
|---|---|---|
| **C0** (H12.5) | `python -m src.sim` | everyone sees `SIM PASS` |
| **C1** (H16) | Dev 1: `python -m src.sim --real imu` · Dev 2: `--real tracker,guidance` · Dev 3: `--real brain` and `--real voice` · Dev 4: `pytest -q` + `python -m demo.audit_photos --dir demo/audit_samples` | each real class works alone against fakes (`--real brain` must print `SIM PASS`) |
| **C2** (H18) | Pair A: `python -m src.main --real imu,brain,voice` · Pair B: `python -m src.main --real tracker,guidance,auditor` | IMU-driven core talks and buzzes; camera pair tracks + audits |
| **C3** (H20) | `python -m src.main --real all --debug-keys` | one full question end to end on the real sheet |

## Merge procedure
```
git checkout main && git pull
git merge --no-ff devN-xxx
python -m src.sim && pytest -q        # must be green; if not: git merge --abort
git push
```

## Triage: symptom → owner
| Symptom | Look at | Owner |
|---|---|---|
| HUD chip `IMU ERR` / no `M,` events | serial port (`ARDUINO_PORT`), `src/arduino_link.py` | Dev 1 |
| WRITING flickers / taps missed | firmware thresholds, `tools/imu_record.py` data | Dev 1 |
| `TRA ERR`, pen jumps, wrong place on page | calibration (`data/page_calibration.json`), marker HSV | Dev 2 |
| Wrong direction / distance words, WARN flaps at box edge | `src/guidance.py` | Dev 2 |
| Box rectangles don't match the sheet | `data/layout.json` vs `demo/layout.sheet.json` | Dev 2 |
| Speaks during WRITING, wrong phase order, stuck phase | `src/state_machine.py` (`--real brain` sim shows it) | Dev 3 |
| Voice overlaps / `is_speaking` gaps / mic hears itself | `src/voice.py` | Dev 3 |
| `AUD ERR`, "could not check" every time, audit wrong | `src/audit.py`, `python -m demo.audit_photos` | Dev 4 |
| `ERROR` lines in the sim log, `ImportError` on `--real X` | contract drift: signatures in `src/contracts.py` / `src/factory.py` | Dev 4 (arbiter) |

## Contract changes
Only Dev 4 edits `src/contracts.py`. Bump `CONTRACT_VERSION`, post in chat, re-run the sim. Real classes must keep the constructors listed in `src/factory.py`.
