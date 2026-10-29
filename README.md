# 🏏 IPL Score Predictor — Deep Learning & ANN

An educational Python project that predicts a cricket team's **final first-innings score** from the current runs, wickets, and over using a TensorFlow/Keras Artificial Neural Network (ANN).

> **Important:** The included demonstration dataset is **synthetically generated**, not historical IPL data. Do not present its evaluation metrics as real IPL prediction accuracy. To make this a genuine IPL analytics project, obtain properly licensed historical ball-by-ball IPL data and construct match snapshots with the required columns.

## ✨ Features
- ANN with two hidden layers (64 and 32 neurons)
- Preprocessing and feature scaling with scikit-learn
- Train/test split, early stopping, MAE, RMSE and R² evaluation
- Command-line prediction for an in-progress first innings
- Replaceable CSV dataset

## 🧰 Tech Stack
Python · TensorFlow · Keras · NumPy · Pandas · scikit-learn

## 🚀 Getting Started

Python 3.10–3.12 is recommended for broad TensorFlow compatibility. In VS Code's terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_demo_data.py
python train.py
python predict.py --overs 12 --balls 3 --runs 105 --wickets 3
```

On Windows, activate with `.venv\\Scripts\\activate` instead.

## 📂 Project Structure
```
ipl-score-predictor/
├── README.md
├── requirements.txt
├── generate_demo_data.py
├── train.py
├── predict.py
├── .gitignore
├── data/       # generated CSV goes here
└── models/     # saved model and scaler go here
```

## 📊 Dataset Format
Provide one **match snapshot** per row in a CSV with these headers:

| overs | balls | runs | wickets | final_score |
|---:|---:|---:|---:|---:|
| 12 | 3 | 105 | 3 | 174 |

`overs` means *completed overs* (0–19), and `balls` means balls into the current over (0–5). `final_score` is the actual final first-innings total. The table above is only an **illustrative example**, not a real IPL record.

For real data, generate snapshots from historical innings, ensuring each snapshot's `final_score` matches its innings total. Use match-level splitting instead of row-level splitting if you create multiple snapshots per match, to prevent data leakage. Consider season-based testing for realistic evaluation.

Train on your own data:
```bash
python train.py --data data/my_ipl_snapshots.csv --epochs 80
```

## ⚠️ Limitations
- Does not use team identity, venue, pitch conditions or player strength.
- Training with synthetic data is a pipeline demonstration only.
- Predictions are estimates; performance on real IPL matches has **not** been validated.
- The included random row split is intended for the demonstration dataset; real match-level data needs grouped or chronological evaluation.

## 🎯 Future Improvements
- Use real historical IPL match data with documented provenance
- Add batting/bowling teams and venue features
- Compare ANN against baseline regression models
- Build a Streamlit web app

## 📄 License and data
Code is provided for learning. Check dataset licensing and attribution before distributing real IPL data.
