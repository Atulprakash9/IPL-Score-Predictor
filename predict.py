"""Predict final first-innings score from current runs, wickets and over."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from tensorflow import keras

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overs', type=int, required=True, help='Completed overs (0-19)')
    parser.add_argument('--balls', type=int, default=0, help='Balls into current over (0-5)')
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--wickets', type=int, required=True)
    args = parser.parse_args()
    elapsed = args.overs + args.balls / 6
    if not (0 < elapsed < 20 and 0 <= args.wickets <= 10 and args.runs >= 0 and 0 <= args.balls <= 5):
        parser.error('Provide a valid first-innings match situation before 20 overs.')
    model_path = ROOT / 'models' / 'ipl_ann.keras'
    scaler_path = ROOT / 'models' / 'scaler.joblib'
    if not model_path.exists() or not scaler_path.exists():
        parser.error('Model not trained. Run python train.py first.')
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    x = np.array([[elapsed, args.runs, args.wickets, args.runs / elapsed]], dtype=np.float32)
    prediction = float(model.predict(scaler.transform(x), verbose=0)[0][0])
    print(f'Predicted final score: {round(max(args.runs, prediction))} runs')
    print('This is a statistical estimate, not a guaranteed outcome.')

if __name__ == '__main__':
    main()
