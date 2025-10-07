System:
Score each plan on four criteria (0..1): coverage, io, simplicity, constraints.
Return per-plan scores and final raw_score = mean of the four.
No extra text.

Example target (not binding):
{
  "scores":[
    {"coverage":0.95,"io":1.0,"simplicity":0.9,"constraints":1.0},
    ...
  ]
}
