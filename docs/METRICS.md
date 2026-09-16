# Metric Definitions

## Detection correctness used for calibration

At IoU 0.50, predictions are sorted by confidence within each image and category. Each ground
truth may match at most one prediction. A matched prediction is a true positive; unmatched and
duplicate predictions are false positives.

## Expected Calibration Error

Predictions are assigned to equal-width confidence bins. ECE is the prediction-count-weighted
absolute difference between average confidence and empirical precision in each bin.

## Brier score

Mean squared error between calibrated probability and the binary correctness label.

## Negative log-likelihood

Binary cross-entropy after clipping probabilities away from zero and one.

## Recall at fixed false positives per image

For each score threshold, recall is matched true positives divided by the total number of ground
truth instances. False positives per image is unmatched predictions divided by the number of
evaluated images. The selected operating point must be chosen on the separate policy-tune role
and reported unchanged on frozen validation data.
