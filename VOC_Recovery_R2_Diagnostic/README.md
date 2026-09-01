# VOC Recovery R2 Diagnostic

This clean-only feasibility test removes `VOC_Room_RAW` from the model input and
uses the other 13 sensor windows as incoming edges to a masked VOC node. A small
multi-head graph aggregator reconstructs the complete 60-step VOC window.

No noise or synthetic fault is used during training. The real clean VOC window is
used only as the reconstruction target. Validation and testing follow the same
date split as the fire-classification experiments.

The report includes overall and class-wise R2, MAE, RMSE, last-point R2, window-mean
R2, and the average learned incoming edge weights.
