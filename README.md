# 🧪 CO₂ Absorption Prediction & Optimisation Calculator

A Streamlit web app that uses a two-stage XGBoost pipeline to:
1. **Predict** CO₂ absorption from synthesis and operating parameters
2. **Optimise** synthesis conditions to maximise CO₂ absorption

## Pipeline Architecture

```
Synthesis features (PS, AR, DR, C, H, N, AT, HR, Ht)
        │
        ▼  Stage 1 – Property Models
  BET, TPV, MPV  (predicted)
        │
        ▼  Stage 2 – Abs Model
    CO₂ Abs  (predicted)
```

Optimisation uses **Differential Evolution** with user-selectable fixed/free variables.

## Files Required (model artefacts)

Place these in the **same folder** as `app.py`:

| File | Description |
|---|---|
| `xgb_BET.json` | XGBoost model for BET surface area |
| `xgb_TPV.json` | XGBoost model for total pore volume |
| `xgb_MPV.json` | XGBoost model for micro pore volume |
| `scaler_BET.pkl` | RobustScaler for BET model |
| `scaler_TPV.pkl` | RobustScaler for TPV model |
| `scaler_MPV.pkl` | RobustScaler for MPV model |
| `xgb_stage2.json` | Stage-2 XGBoost model for Abs |
| `scaler_stage2.pkl` | RobustScaler for Stage-2 model |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

See deployment steps in the project documentation.
