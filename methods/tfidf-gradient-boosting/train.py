import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


METHOD_DIR = Path(__file__).resolve().parent
DATA_PATH = METHOD_DIR.parents[1] / 'data' / 'data.json'
MODEL_PATH = METHOD_DIR / 'model.pkl'
VECTORIZER_PATH = METHOD_DIR / 'vectorizer.pkl'


def normalise_code(code: str) -> str:
  """Normalise C++ code by removing comments and extra whitespace."""
  
  code = re.sub(r'//.*', '', code)
  
  code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
  
  code = re.sub(r'\s+', ' ', code)
  
  return code.strip()

def tokenise_code(code_text):
  """Tokenise C++ code into identifiers, operators, and punctuation."""

  code_text = normalise_code(code_text)

  pattern = r'[A-Za-z_]\w*|[+\-*/%=<>!&|^~]+|[;,\{\}\[\]\(\)]'

  # Replace identifiers with a unique identifier (e.g. a => "id1", b => "id2") token to reduce vocabulary size
  
  identifiers = {}
  def replace_identifier(match):
    identifier = match.group(0)
    if identifier not in identifiers:
      identifiers[identifier] = f"id{len(identifiers) + 1}"
    return identifiers[identifier]

  code_text = re.sub(r'[A-Za-z_]\w*', replace_identifier, code_text)
  
  return re.findall(pattern, code_text)


def main():
  if not DATA_PATH.exists():
    print(f'Error: {DATA_PATH} not found. Please create it with your submissions.')
    return

  with DATA_PATH.open('r', encoding='utf-8') as f:
    raw_data = json.load(f)

  df = pd.DataFrame(raw_data)
  print(f'Loaded {len(df)} total submissions from {DATA_PATH}')

  X_raw = df['code'].tolist()
  y = df['score'].values

  X_train_raw, X_test_raw, y_train, y_test = train_test_split(
      X_raw, y, test_size=0.2, random_state=42
  )

  vectorizer = TfidfVectorizer(
      tokenizer=tokenise_code,
      token_pattern=None,
      lowercase=False,
      max_features=2000,
  )

  X_train_tfidf = vectorizer.fit_transform(X_train_raw)
  X_test_tfidf = vectorizer.transform(X_test_raw)


  print('┌' + '─' * 46 + '┐')
  print('│' + ' ' * 11 + 'Learning curve evaluation' + ' '* 10 + '│')
  print('├' + '─' * 12 + '┬' + '─' * 14 + '┬' + '─' * 18 + '┤')
  print(f"│ {'Data %':<10} │ {'Train Size':<12} │ {'Test MAE (Error)'} │")
  print('└' + '─' * 12 + '┼' + '─' * 14 + '┼' + '─' * 18 + '┘')

  fractions = [p / 10 for p in range(1, 11)]  # 0.1 to 1.0 in steps of 0.1

  best_model = None
  best_mae = float('inf')

  for frac in fractions:
    subset_size = int(X_train_tfidf.shape[0] * frac)
    if subset_size == 0:
      continue

    X_subset = X_train_tfidf[:subset_size].toarray()
    y_subset = y_train[:subset_size]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_subset, y_subset)

    preds = model.predict(X_test_tfidf.toarray())
    preds = np.clip(preds, 0, 100)

    mae = mean_absolute_error(y_test, preds)
    print(f" {int(frac*100)}% {'':<7} │ {subset_size:<12} │ {mae:.4f} points")

    if mae < best_mae:
      best_mae = mae
      best_model = model

  joblib.dump(best_model, MODEL_PATH)
  joblib.dump(vectorizer, VECTORIZER_PATH)
  print("\nTraining complete.")


if __name__ == '__main__':
  main()