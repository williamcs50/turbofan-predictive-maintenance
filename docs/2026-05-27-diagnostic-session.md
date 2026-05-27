# Diagnostic Session - May 27, 2026

## First finding: focal loss alone didn't break the collapse

The v0.1 baseline showed the anomaly head completely collapsed: Precision 0.0000, Recall 0.0000, Accuracy 0.9983, RUL RMSE 41.72 cycles. The model was predicting all-normal for every input; the RUL head was outputting a near-constant ~78 cycles regardless of input. The hypothesis going into today was that weighted cross-entropy was the cause - the loss signal from rare anomaly samples was being drowned out by the majority class. Focal loss was expected to fix this by down-weighting easy normal examples and forcing the model to attend to the hard positives.

The first focal loss run (gamma=2.0, alpha=None) produced a clean loss curve - 0.0156 at epoch 1, converging to 0.0146 by epoch 10, no thrashing. But test metrics were unchanged: Precision 0.0000, Recall 0.0000, Accuracy 0.9992, RUL RMSE 42.79 cycles. Rather than immediately tuning gamma, I went upstream. Tuning gamma without understanding why the first run failed would be guessing.

Inspecting `generate_sensors.py` revealed the actual problem. The anomaly label was defined as:

```python
anomaly = 1 if vibration > 0.05 or T50 > 1420 else 0
```

The degradation magnitudes in the simulator never crossed these thresholds. Vibration baseline is 0.010–0.015; the worst-case degradation (`vib_increase: 0.003`) adds ~0.003 at failure - nowhere near 0.05. T50 baseline is ~1400; `temp_increase: 1.0` adds ~1°C at failure - nowhere near 1420. The only failure mode that ever produced `anomaly=1` was Foreign Object Damage, which applies a single abrupt spike at one specific cycle. Every other failure mode produced zero positive labels for its entire degradation window. The class imbalance wasn't 4:1 or 10:1 - it was effectively 500:1 or higher.

The fix was to replace the threshold-based label with a RUL-proximity definition:

```python
anomaly = 1 if rul <= 30 else 0
```

This is the standard convention in predictive maintenance benchmarks (including the NASA CMAPSS dataset this project is modeled on): the last N cycles before failure are defined as the anomaly/degradation regime. It sidesteps the problem of calibrating sensor thresholds entirely and directly encodes the task the model is supposed to solve.

---

## Second finding: corrected labels gave 20% positive rate - model still collapsed

After the label fix, class weights shifted from Normal: 0.50 / Anomaly: 50.00 (100:1, capped) to Normal: 0.62 / Anomaly: 2.52 (4:1). The test set had ~3,000 anomaly samples out of ~14,662 total - about 20% positive rate. A 20% positive rate is well within the regime focal loss handles cleanly in the original paper.

Retrained on corrected data. Results: Precision 0.0000, Recall 0.0000, Accuracy 0.7954. The accuracy drop from 0.9992 to 0.7954 reflects the new class distribution - predicting all-normal now only achieves ~80% accuracy. The confusion matrix confirmed it: all 3,000 true anomalies classified as normal, zero true positives.

A 20% positive base rate with focal loss still producing a complete collapse suggested the problem was no longer the label definition or the loss function. Something was wrong with the features themselves.

I ran a sensor trace diagnostic (`scripts/diagnose_sensor_signal.py`): plotted raw (pre-scaling) and scaled (post-MinMaxScaler) sensor values for ENG_001 (High-pressure turbine wear, 299 cycles), with the RUL≤30 anomaly window marked. Three sensors: vibration, T50, and P30.

The plot showed:
- **Vibration (raw):** Flat at 0.010–0.015 throughout all 299 cycles. A slight uptick is theoretically present in the red window but it is invisible against the noise. The degradation parameter `vib_increase: 0.002` produces a maximum delta of ~0.003 at failure on a signal whose noise is of comparable magnitude.
- **T50 and P30 (raw):** Completely flat for all 299 cycles. High-pressure turbine wear only affects vibration; these sensors carry zero signal for this failure mode.
- **After MinMaxScaler:** The scaler amplifies per-cycle variance across the full trace, further burying any degradation signal in noise.

The features in the `anomaly=1` windows were statistically indistinguishable from the `anomaly=0` windows.

---

## Conclusion

The v0.1 anomaly head collapse was inevitable. No loss function - weighted cross-entropy, focal loss, any other variant - could have learned this task because the training data contained no learnable structure in the anomaly windows. The degradation magnitudes in the synthetic data generator were too small to produce a signal above the noise floor. The model was being asked to distinguish two classes that looked identical in feature space.

The meta-lesson: the bug was two levels upstream from where I was looking. I started at the loss function, found the problem was the labels, fixed the labels, and found the problem was the simulator. Each fix was necessary but not sufficient on its own.

The next session needs to redesign the simulator so degradation produces signal above the noise floor. The current `vib_increase: 0.002` produces a delta of ~0.003 against noise of comparable magnitude. An order-of-magnitude increase is one starting point, but the design exercise is choosing degradation magnitudes deliberately based on a target signal-to-noise ratio - not eyeballing a single plot. Learnability validation (confirming features are discriminative before training) should happen before any future training run.
