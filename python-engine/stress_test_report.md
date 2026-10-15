# Adversarial Stress-Testing & Failure-Boundary Analysis Report
## Objective 3 Alignment

This evaluation subjects the alcohol detection pipeline to synthetic optical stressors, 
affine spatial transformations, and transmission distortions simulating real-world surveillance degradations.

### Stress-Testing Benchmark Results

| Adversarial Stress Condition | Frames Detected | Detection Rate (%) | Mean Confidence | Inference FPS |
| :--- | :---: | :---: | :---: | :---: |
| Clean Baseline | 87/87 | 100.0% | 0.773 | 22.7 |
| Affine 180° Inversion (Knocked-over) | 78/87 | 89.7% | 0.587 | 23.4 |
| Affine 90° Lateral Tilt | 53/87 | 60.9% | 0.431 | 23.3 |
| Severe Specular Glare (Glass Glint) | 85/87 | 97.7% | 0.732 | 18.5 |
| Specular Contrast Wash (Backlit) | 86/87 | 98.9% | 0.716 | 23.1 |
| Video Compression (RTSP Q=15 Blockiness) | 82/87 | 94.3% | 0.649 | 22.8 |

### Failure-Boundary Observations:
1. **Affine Inversion Recovery:** Direct upright detection drops under 180° rotation, validating the critical necessity of the 180° rotation fallback pass in `detector.py`.
2. **Specular Distortion Resilience:** Glare and contrast degradation mildly reduce average confidence, but high container contrast boundaries preserve localization.
3. **Compression Artifact Robustness:** Heavy H.264/JPEG blockiness down to Q=15 retains strong detection rates, proving robust feature extraction across compressed RTSP network streams.
