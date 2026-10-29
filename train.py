"""Train an ANN to predict final first-innings scores from match snapshots."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
FEATURES = ['overs_elapsed', 'runs', 'wickets', 'current_run_rate']

def load_data(path):
    df = pd.read_csv(path)
    required = {'overs', 'balls', 'runs', 'wickets', 'final_score'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = df.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=list(required))
    df = df[(df.overs >= 0) & (df.overs <= 19) & (df.balls.between(0, 5)) &
            (df.wickets.between(0, 10)) & (df.runs >= 0) & (df.final_score >= df.runs)]
    df['overs_elapsed'] = df.overs + df.balls / 6.0
    df = df[df.overs_elapsed > 0]
    df['current_run_rate'] = df.runs / df.overs_elapsed
    if len(df) < 30:
        raise ValueError('Need at least 30 valid match snapshots for training.')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'data' / 'demo_synthetic.csv'))
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    df = load_data(args.data)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df.final_score.to_numpy(dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    keras.utils.set_random_seed(42)
    model = keras.Sequential([
        keras.layers.Input(shape=(len(FEATURES),)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=32,
              validation_split=0.2, verbose=1,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)])
    predictions = model.predict(X_test, verbose=0).ravel()
    print(f'Test MAE: {mean_absolute_error(y_test, predictions):.2f} runs')
    print(f'Test RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f} runs')
    print(f'Test R²: {r2_score(y_test, predictions):.3f}')
    out = ROOT / 'models'
    out.mkdir(exist_ok=True)
    model.save(out / 'ipl_ann.keras')
    joblib.dump(scaler, out / 'scaler.joblib')
    print('Saved model and scaler to models/')
    if 'synthetic' in Path(args.data).name.lower():
        print('NOTE: Metrics above are on SYNTHETIC data, not real IPL performance.')

if __name__ == '__main__':
    main()


