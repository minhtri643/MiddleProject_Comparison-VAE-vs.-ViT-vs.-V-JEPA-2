# Phase 3: Out-of-Distribution Adverse Weather Robustness Stress-Testing

## 1. Experimental Overview
- **Objective:** Evaluate representation stability and invariance when visual sensors face severe optical degradation.
- **Corruption Protocol:** Koschmieder's Atmospheric Scattering Model simulating dense highway fog (Level 3 severity, $\beta = 0.65$):
  $$I(x) = J(x)e^{-\beta d(x)} + A(1 - e^{-\beta d(x)})$$
- **Metrics Measured:**
  - Feature Vector Drift: $\Delta_{\text{Cos } A} = \text{Cosine}_{\text{Foggy}}(A) - \text{Cosine}_{\text{Clean}}(A)$
  - RankMe Retention Rate: $\frac{\text{RankMe}_{\text{Foggy}}}{\text{RankMe}_{\text{Clean}}} \times 100\%$

## 2. Generated Visual Artifacts & Metrics
1. `05_robustness_stress_test.png`: Side-by-side comparative panel under clean vs foggy weather.
2. `05_robustness_metrics.csv`: Tabular evaluation of RankMe retention and Cosine drift.

## 3. Key Findings
- **SD-VAE Suffers Severe Degradation:** $\Delta = +0.3808$ (38.08% drift) because its ELBO objective forces it to memorize pixel surface colors.
- **ViT and V-JEPA Exhibit Semantic Invariance:** ViT drift is $-0.0052$ and V-JEPA drift is $-0.0079$, confirming robust preservation of object geometry under adverse conditions.
