# Maximize a subset sum

Implement `solution.py` using only the Python standard library:

```python
def solve(numbers: list[int], target: int) -> list[int]:
    ...
```

Inputs are nonnegative integers and a positive target. Return a Python list of
distinct, valid zero-based indices into `numbers`. The selected values must sum
to at most `target`. Maximize that sum. An empty selection is allowed. Do not
modify the inputs. The function must work on other inputs too.

The exercise includes lists of up to 32 numbers, zeros and repeated values.
Numbers are at most 1,000,000,000; targets range from 1 to 1,000,000,000.
A target can exceed the sum of all items, and some targets are unattainable.
Consider both the number of items and the magnitude of the target when choosing
an algorithm; allocating storage proportional to a very large target can fail.

Public example: `numbers = [12, 7, 15, 8, 19, 6]`, `target = 30`.
Indices `[1, 2, 3]` select `[7, 15, 8]`, totaling 30, which is optimal.

Fitness is the sum of selected values over several fixed instances, including
this public example. All answers must be valid; one invalid answer makes the
candidate infeasible. Larger fitness is better. Tests and scoring rules are
controlled by the instructor and stay the same for every candidate. Evaluation
has a short time limit, so keep the implementation practical and deterministic.
The complete suite shares a 10-second CPU limit, a 15-second Docker wall-time
limit and 512 MiB of memory. These are execution guards, not continuous fitness
measurements. A feasible heuristic may earn less than an optimal solution.
