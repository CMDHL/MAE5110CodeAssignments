# Assignment 2 Report

## Sketches

![Walker sketches](assignment_2_results/walker_sketches.png)

## 1. Feedback linearization and RoA

![Region of attraction](assignment_2_results/RoA.png)

## 2. Poincare section

I used the mid-stance section theta = gamma. This section is transverse for the forward walking states because angular velocity is positive when the walker crosses it, and theta is fixed, so the return map only needs the state theta_dot_k. The step controller chooses alpha once per stance phase.

![Poincare return map](assignment_2_results/poincare_return_map.png)

## 3. Lookup table

The state grid covers theta_dot in [0, sqrt(2 g / l)]. The control grid covers alpha in [pi/8, pi/7]. I used the coarsest grid from the table below that satisfied both checks: nearest-neighbor state rounding error at most 0.010 rad/s, and alpha spacing at most 0.0015 rad.

| speed count | control count | speed spacing | control spacing | max rounding error | passes |
| --- | --- | --- | --- | --- | --- |
| 151 | 31 | 0.029530 | 0.001870 | 0.014748 | False |
| 201 | 31 | 0.022147 | 0.001870 | 0.011065 | False |
| 251 | 31 | 0.017718 | 0.001870 | 0.008857 | False |
| 251 | 41 | 0.017718 | 0.001402 | 0.008858 | True |

The 201 by 31 grid is not good enough because both the state rounding error and alpha spacing miss the criteria. The 251 by 31 grid fixes the state spacing, but the control spacing is still too coarse. The 251 by 41 grid is the first tested grid that passes both.

![Steps to standstill](assignment_2_results/steps_to_standstill.png)

For the trajectory example, I used initial section speed 3.082895 rad/s. The fewest-step policy reaches the RoA in 3 steps, with alpha values [0.4011 0.4488 0.4488]. The longest successful policy reaches the RoA in 6 steps, with alpha values [0.3927 0.3927 0.3927 0.3927 0.3927 0.3927].

![Trajectory examples](assignment_2_results/trajectory_examples.png)
