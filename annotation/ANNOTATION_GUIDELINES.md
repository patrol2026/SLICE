# Important-Line Annotation Guidelines (v1 — frozen)

## Task
For each solution line in your CSV, decide whether the line is **IMPORTANT**
(encodes problem-specific core logic) and put `1` in the `important` column,
or leave it `0`.

## Definition
A line is **IMPORTANT** if it encodes problem-specific core logic — a
predicate, computation, or state update whose alteration would change the
algorithm's output on the problem's tests.

A line is **NOT important** if it is scaffolding — code that any solution to
any problem might share:
- imports, class/function signatures
- docstrings and comments
- trivial initialisation (`d = {}`, `count = 0`, `result = []`)
- plain returns of an already-computed value (`return result`)
- boilerplate control flow every solution would have (e.g., a bare loop header
  that merely iterates the input **unless** its bounds/order encode the trick)

## Decision rules
1. Judge each line by the question: *"if an adversary corrupted only this
   line, would the solution break on this problem's tests in a
   problem-specific way?"*
2. Typically 20–40% of content lines qualify — but there is **no quota**;
   mark exactly what fits the definition.
3. Rows pre-marked `skip` (blank lines, comment-only lines) must be left
   untouched.
4. A multi-line logical unit (a condition split across lines) may be marked
   wholly.
5. Work **independently**: do not consult the other annotator, any AI model,
   or any external solution. Do not revise answers after submitting.
6. Read the problem statement (problems_reference.txt) before judging its
   lines — importance is relative to *this* problem.

## Worked example
Problem: return True if any two numbers in the list are closer than
`threshold`.

```
 1| from typing import List                                  -> 0 (import)
 2| def has_close_elements(numbers, threshold):              -> 0 (signature)
 3|     for i in range(len(numbers)):                        -> 0 (bare iteration)
 4|         for j in range(i + 1, len(numbers)):             -> 1 (i+1 pairing logic)
 5|             if abs(numbers[i] - numbers[j]) < threshold: -> 1 (core predicate)
 6|                 return True                              -> 0 (plain return)
 7|     return False                                         -> 0 (plain return)
```
Line 4 is marked because `i + 1` encodes the pair-enumeration trick; line 5 is
the problem's defining comparison. Lines 3, 6, 7 appear in countless
solutions to countless problems.

## Procedure
1. Open your own CSV (annotation_A.csv or annotation_B.csv). Never open the
   other annotator's file.
2. For each problem (grouped rows), read its statement in
   problems_reference.txt, then fill `important` with 1 or 0 for every row
   not marked `skip`.
3. Expect ~2 minutes per problem, ~3.5 hours total. Take breaks; do not
   annotate when tired.
4. When both files are complete, run: `python human_val_score_csv.py agree`
