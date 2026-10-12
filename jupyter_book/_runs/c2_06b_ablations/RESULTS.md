# 06b PINN diagnostics (2026-10-11, sample 0 of coil_n100: I = 66.8 A, r_w = 14.4 mm, 100 observations, 3000 epochs unless noted, CPU)
All runs use the notebook's own losses (ProperPINNTrainer from 06b cells 1-3: wire-scale nondimensional Poisson residual,
+500 near-wire collocation points per epoch, cosine LR 1e-3 -> 1e-5, grad clip 1). NMSE on the notebook's +-600 mm grid.

| run | network inputs | hard BC | physics weight | NMSE full grid | NMSE inside box | log |
|---|---|---|---|---|---|---|
| 06a vanilla (Kaggle v2) | x/500, y/500 (tanh) | no | 0 | 15.8% | - | notebook |
| 06b proper (Kaggle v2) | 32 random Fourier (domain) + 32 (wire scale) | yes | 0.02 | 55.3% | - | notebook |
| A (1500 ep) | same as 06b | yes | 0 | 158.7% | - | c2_06b_ablation_A.log |
| B (1500 ep) | same as 06b | no | 0.02 | 69.0% | - | c2_06b_ablation_B.log |
| D | x/500, y/500 (tanh) | yes | 0 | 11.9% | - | c2_06b_ablation_D.log |
| C | x/500, y/500 (tanh) | yes | 0.02 | 5.9% | - | c2_06b_ablation_C.log |
| E | x/500, y/500 (tanh) | yes | 1 | 7.7% | - | c2_06b_ablation_E.log |
| F | x/500, y/500 (tanh) | yes | 10 | 22.1% | - | c2_06b_ablation_F.log |
| G0.02 | x/500, y/500, r/500, ln((r+1mm)/500) | yes | 0.02 | 0.24% | 0.24% | c2_06b_ablation_G002.log |
| G1 | same | yes | 1 | 0.28% | 0.28% | c2_06b_ablation_G1.log |
| H1 | analytic wire potential + correction net(x/500,y/500), soft wall BC | soft | 1 | 2.92% | 1.19% | c2_06b_ablation_H1.log |
| H0.02 | same | soft | 0.02 | 2.95% | 1.26% | c2_06b_ablation_H002.log |
| reference | analytic infinite-space wire field, zero outside the box | - | - | 0.9% | 0.9% | - |

Reading: random Fourier features at wire scale overfit the 100 points (A: data loss 2e-6, field wrong); the plain tanh net
cannot satisfy the Poisson residual at the 14 mm wire whatever the weight (C/E/F residual stays 0.015-0.03); giving the
network r and ln r (the 2-D Green's function coordinate) removes that limit: 0.24%, with physics weight 0.02 or 1 alike.
