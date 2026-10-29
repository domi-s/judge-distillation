import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocessing import tokenise_code


METHOD_DIR = Path(__file__).resolve().parent
DATA_PATH = METHOD_DIR.parents[1] / "data" / "data.json"
MODEL_PATH = METHOD_DIR / "model.pkl"
VECTORIZER_PATH = METHOD_DIR / "vectorizer.pkl"


def main() -> None:
  if not DATA_PATH.exists():
    print(f"Error: {DATA_PATH} not found. Please create it with your submissions.")
    return

  records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
  codes = [record["code"] for record in records]
  scores = np.array([record["score"] for record in records])
  train_codes, test_codes, train_scores, test_scores = train_test_split(
      codes, scores, test_size=0.2, random_state=42, stratify=scores
  )

  vectorizer = TfidfVectorizer(
      tokenizer=tokenise_code,
      token_pattern=None,
      lowercase=False,
      ngram_range=(1, 3),
      max_features=20000,
      sublinear_tf=True,
  )
  train_features = vectorizer.fit_transform(train_codes)
  test_features = vectorizer.transform(test_codes)

  print(f"Loaded {len(records)} submissions from {DATA_PATH}")
  print('+' + '-' * 52 + '+')
  print('|' + ' ' * 12 + 'Learning curve evaluation' + ' ' * 15 + '|')
  print('+' + '-' * 12 + '+' + '-' * 14 + '+' + '-' * 24 + '+')
  print(f"| {'Data %':<10} | {'Train Size':<12} | {'Test MAE (Error)':<22} |")
  print('+' + '-' * 12 + '+' + '-' * 14 + '+' + '-' * 24 + '+')

  fractions = [percentage / 10 for percentage in range(1, 11)]
  for fraction in fractions:
    subset_size = int(train_features.shape[0] * fraction)
    model = LinearSVC(C=10, class_weight="balanced", max_iter=10000)
    model.fit(train_features[:subset_size], train_scores[:subset_size])
    predictions = model.predict(test_features)
    mae = mean_absolute_error(test_scores, predictions)
    print(
        f"| {int(fraction * 100):<10} | {subset_size:<12} | "
        f"{mae:.4f} points{' ':>9} |"
    )

  print('+' + '-' * 12 + '+' + '-' * 14 + '+' + '-' * 24 + '+')
  model = LinearSVC(C=10, class_weight="balanced", max_iter=10000)
  model.fit(train_features, train_scores)
  predictions = model.predict(test_features)
  print(f"Test accuracy: {(predictions == test_scores).mean():.4f}")
  print(f"Test MAE: {mean_absolute_error(test_scores, predictions):.4f} points")

  joblib.dump(model, MODEL_PATH)
  joblib.dump(vectorizer, VECTORIZER_PATH)
  print("Training complete.")


if __name__ == "__main__":
  main()
