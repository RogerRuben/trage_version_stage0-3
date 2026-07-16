# Stage 0 data split v4

## Frozen roles

| Role | Dates | Permitted use |
|---|---|---|
| Reference fit | 2016-10-19 | Engineering distributions and reference statistics |
| Train | 2016-10-20 to 2016-10-21 | Matcher/quality development excluding manual validation judgments |
| Validation | 2016-10-22 | Threshold review and primary manual route truth |
| Development diagnostic | 2016-10-23 | Fixed 1,000-order network comparison only |
| Blind test | None available | No claim of a genuinely untouched later-date test |

The 2016-10-23 data have been inspected repeatedly and are not a blind test.
No date later than 2016-10-23 is present in the workspace. Consequently the
formal stability analysis must use retrospective rolling-origin folds and label
them as such; it must not relabel 2016-10-23 as untouched.

## Rolling-origin protocol

After manual thresholds are frozen, report the following date-level evaluations:

- fit through 2016-10-19, evaluate 2016-10-20;
- fit through 2016-10-20, evaluate 2016-10-21;
- fit through 2016-10-21, evaluate 2016-10-22;
- fit through 2016-10-22, evaluate 2016-10-23 as the development fold.

These are retrospective temporal-stability checks, not prospective blind tests.
If a later date becomes available, it supersedes this limitation and receives a
new manifest and split version before any processing begins.

## Promotion gate

Full-date processing starts only after the v4 network comparison and human truth
audit pass. Stage 1--4 remain `HOLD` until the canonical Stage 0 manifest exists.
