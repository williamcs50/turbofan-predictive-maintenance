# Turbofan Predictive Maintenance v0.2 Case Study


## The setup: what was tested and why

The hypothesis going in was that distributed-SNR data with degradation signal spread across channels rather than concentrated should favor an architecture that attends across channels. That was the theoretical case for the Transformer over the linear gate. When signal is concentrated in one or two channels, a linear model captures a monotonic ramp just fine, and the Transformer adds nothing. When signal is spread thin across all five channels so no single channel suffices, attention over the full sequence should close the gap or beat the linear baseline meaningfully. The contrast sets up a clear test of whether the Transformer's added complexity is justified when the degradation signal is deliberately distributed rather than localized.

## The prediction: pre-registered, before the run

The bar was set explicitly in advance: F1 at 0.85 or higher means the Transformer earns its architectural complexity in the distributed-SNR regime. Below 0.85, it provides only marginal help without justifying the added complexity over the linear gate. This threshold and its interpretation were committed to the repository before any retraining or final evaluation.

- Pre-registration commit: [c906eaa](https://github.com/williamcs50/turbofan-predictive-maintenance/commit/c906eaa) (2026-06-04 09:49)
- Retrain / final evaluation commit: [1d3c140](https://github.com/williamcs50/turbofan-predictive-maintenance/commit/1d3c140) (2026-06-04 17:10)


## The result

The Transformer returned F1 0.82 at t* = 0.43. That lands in the 0.80 to 0.84 band. The prediction called it: helps but doesn't justify. The aggregate numbers look like a win over the gate (0.82 vs. 0.80), but the aggregate is misleading. The Transformer beat the gate on precision, not recall, and for r = 50 that's the wrong dimension to win on.


## The finding

**Per-mode recall table** 

Transformer vs gate, all five modes. Every mode regressed.

| Mode | Gate recall | Transformer recall | Change |
|---|---|---|---|
| Bearing failure | 0.7933 | 0.7883 | -0.0050 |
| Compressor stall | 0.7933 | 0.6933 | -0.1000 |
| FOD | 0.7550 | 0.7017 | -0.0533 |
| HPT wear | 0.7767 | 0.7489 | -0.0278 |
| Overheating | 0.7733 | 0.7717 | -0.0016 |

The lift came from precision (0.9124 vs 0.8233), not recall. The Transformer learned to be conservative, not to catch the hard cases.


**PR curve**

The model shows a classic flat-then-cliff precision-recall curve: precision stays near 1.0 out to roughly 0.65 recall before dropping sharply. The F1-optimal threshold (t*=0.43) sits right at the edge of that cliff. No operating point can recover additional recall without severely degrading precision. This is a fundamental property of the learned representation, not a tuning issue. I already ran a full threshold sweep; 0.43 was F1-optimal on validation.

![Precision-recall curve](../assets/precision_recall_curve.png)

**RUL scatter**

Bias is worst near failure: mean error is +25 cycles in the 0–20 band and +18 cycles in the 20–40 band, shrinking to +4 cycles in the 70–100 band. The model underestimates urgency most severely when urgency is highest. The vertical stack of predictions at 125 is the RUL cap. It arises because the targets are right-truncated while the features remain uncensored. This is a deliberate consequence of the data construction, not a modeling artifact.

![RUL scatter](../assets/rul_scatter.png)

**Confusion matrix**

The model produces 761 false negatives against 215 false positives, a 3.5:1 miss ratio. Even at its F1-optimal threshold, it sits in the wrong region of the cost landscape when a missed failure costs 50 times as much as a false alarm. In short, the F1-optimal threshold does not minimize operational cost under realistic FN/FP asymmetry.

![Confusion matrix](../assets/confusion_matrix.png)

## The mechanism

The RUL bias is worst near failure (+25 cycles in the 0–20 band) and shrinks at higher RUL (+4 cycles in the 70–100 band). The model is most wrong precisely when urgency is highest. This is not mid-stage compression. It is miscalibration near failure. The anomaly recall pattern is consistent with this: FOD and compressor stall, which have slow-onset degradation that extends into the late window, regress the most. The model appears to handle the early and middle stages adequately but loses resolution as the engine approaches the end of life. Whether that is a cross-channel integration problem or a temporal resolution problem near failure is what the future work section tests.

## Future work

### Cross-channel integration limit

The distributed signal may not be as recoverable by attention as originally hypothesized. This would point toward a different architecture.

### Temporal resolution loss

The architecture may integrate channels adequately but blur slow temporal onsets. The training objective does not appear to reward sensitivity to faint, distributed patterns. In this case the architecture could be appropriate while the loss function is the actual limitation. This would mean keeping the current architecture and instead changing the temporal encoding or the training objective.

### What to test first

Temporal resolution loss is the more actionable hypothesis and the more likely culprit. The per-mode recall table shows FOD and compressor stall (both slow-onset modes) regressing the most. The RUL bias follows the same gradient: largest near failure, where the model loses resolution as the engine approaches end of life. A cross-channel integration failure would show up as uniform degradation across modes; this isn't uniform. The next experiment should hold the architecture fixed and test whether a loss function that up-weights mid-stage windows recovers recall on those two modes before concluding the architecture itself is the problem.