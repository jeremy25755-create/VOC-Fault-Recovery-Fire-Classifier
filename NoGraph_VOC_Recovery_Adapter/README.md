# No-Graph VOC Recovery Adapter

This module protects previously trained graph-free fire classifiers from
`VOC_Room_RAW` faults without training on noisy data.

- The original encoder and classifier are frozen.
- A directed recovery graph receives only the other 13 clean sensor windows.
- It reconstructs the causal low-frequency VOC trend and its uncertainty.
- Clean validation data calibrate relation and high-frequency fault scores.
- Normal windows use the original checkpoint exactly.
- Detected VOC faults use the reconstructed trend and a small fault-only adapter.
- Training uses clean Background, Fire, and Nuisance windows only. Gaussian noise
  is created only for the final test copy.
