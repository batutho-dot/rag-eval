# The report: three scores, judge scores, agreement with hand grades

- **Does:** `python run.py` prints, per config, right passage found, right answer given,
  unanswerable declined, the judge's scores, judge vs hand-grade agreement (raw and Cohen's
  kappa) and the measured spend. Free; needs no key.
- **Reach:** `python run.py` (or `python run.py report`).
- **Verify drives it:** `report` writes a fixed fake run (all right, one false decline),
  fake judge verdicts (all correct) and fake hand grades (one incorrect) to a temp folder,
  runs the report there and checks it prints 20/20, 19/20, 10/10 and 29/30 agreement.
- **End state:** step ok.
- **Gotchas:** agreement is computed only over answers that have a hand grade.
