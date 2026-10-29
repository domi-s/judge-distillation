# Judge distillation

This repository explores methods for distilling a programming judge into a local score predictor.

[Read the blog post here.](https://domis.at/blog/judge-distillation)

## Introduction

The purpose of this project is to explore methods for predicting the score of a C++ submission to a programming judge, without having to submit it to the judge itself.

A "judge" is a system that evaluates programming submissions, typically for competitive programming or educational purposes. It runs the submitted code against a set of test cases and assigns a score based on correctness, efficiency, and other criteria.

[InfoAs.ro](https://infoas.ro) uses a proprietary C++ and Python judge which to date has evaluated over 345,000 submissions on the main platform alone. The judge is a critical component of the platform, but it can be expensive or cumbersome to auto-scale at times of high demand.

Typically, the cost of maintaining a programming judge is not that high (ranges from $0.02 to $0.10 per hour of compute).

That being said, auto-scaling a programming judge to handle a large number of submissions can be complicated. During competitions, because of the number and size of test cases, the judge can be put under heavy load, which can lead to delays in scoring submissions. This can be frustrating for participants who are waiting for their scores to be updated. During the 2024 and 2025 editions of the Local Olympiad in Informatics, which were hosted on InfoAs, I load balanced the website onto four servers and another backup server, but the judge was still a bottleneck. During the latter part of the competition, the average time from submission to score was 20 seconds, which is not bad, but could definitely be improved, without having to hire more compute.

To mitigate this, different contest management systems employ various strategies, such as caching results for identical submissions or providing partial scoring based on a subset of test cases. However, these strategies may not always be feasible or effective: the former requires a large cache which often becomes ineffective because it is hard to decide if two submissions are identical (or argue their similarity up to some epsilon threshold); the latter is not fully indicative of the final score and can lead to unfairness in scoring (which I have also personally experienced in previous competitions).

This repository explores methods for distilling the judge into a local score predictor, which can be used to predict the score of a submission without having to submit it to the judge itself. This can be useful for participants who want to get an idea of their score before submitting, or for contest organizers who want to reduce the load on the judge during competitions. The methods explored lack the accuracy of the judge, are biased towards the training data, and are not guaranteed to be correct, but they can provide a good approximation of the score which can be used as a heuristic first pass of a submission. They require less compute than the judge, but need to be trained on a large dataset of submissions.

## Dataset

The dataset used for training the models is a collection of 10,000 C++ submissions and their corresponding scores from the InfoAs judge. The dataset is stored in `data/data.json`, which contains a list of submissions, each represented as a JSON object with the following fields:

- `code`: The C++ code of the submission.
- `score`: The score assigned by the InfoAs judge (`0 – 100`), where Compilation Error is also represented as `0`.

A sample of the dataset is provided in [`sample.md`](sample.md), which contains a few submissions with their scores.

The chosen problem is [Suma](https://infoas.ro/problema/suma), which is a simple problem requiring outputting the sum of two input numbers. The problem has five test cases weighing 20 points each. Note that this is merely a proof of concept, and due to the simplicity of the problem, the dataset contains a large number of submissions with scores of `0` and `100`.

## Methods explored

- [TF-IDF + histogram gradient boosting](#tf-idf--histogram-gradient-boosting)
- [TF-IDF + linear SVM classification](#tf-idf--linear-svm-classification)

## TF-IDF + Histogram gradient boosting

Normalises and tokenises C++ code, represents it with TF-IDF, and regresses from those features to judge scores.

Run these commands from the repository root.

```powershell
# Train and save the model
python methods/tfidf-gradient-boosting/train.py

# Start the app
streamlit run methods/tfidf-gradient-boosting/app.py
```

The app reads the model artifacts generated through training, and allows you to input C++ code to get a predicted score.

### Output

```
┌──────────────────────────────────────────────┐
│           Learning curve evaluation          │
├────────────┬──────────────┬──────────────────┤
│ Data %     │ Train Size   │ Test MAE (Error) │
└────────────┼──────────────┼──────────────────┘
 10%         │ 828          │ 19.0511 points
 20%         │ 1657         │ 16.7506 points
 30%         │ 2486         │ 16.1660 points
 40%         │ 3314         │ 16.1097 points
 50%         │ 4143         │ 15.7976 points
 60%         │ 4972         │ 15.8898 points
 70%         │ 5800         │ 15.6347 points
 80%         │ 6629         │ 15.4395 points
 90%         │ 7458         │ 15.2481 points
 100%        │ 8287         │ 15.3220 points
```

 ## TF-IDF + linear SVM classification

Since the dataset contains mostly 0 and 100 scores, classification matches the target better than continuous regression.

 ```powershell
 # Train and save the model
 python methods/tfidf-linear-svm/train.py

 # Start the app
 streamlit run methods/tfidf-linear-svm/app.py
 ```

 ### Output

 On the fixed 20% stratified test split, this method achieves approximately
 92.7% accuracy and 6.7 points MAE. Results can vary if the split or dataset
 changes.

```
┌────────────────────────────────────────────────────┐
│            Learning curve evaluation               │
├────────────┬──────────────┬────────────────────────┤
│ Data %     │ Train Size   │ Test MAE (Error)       │
└────────────┼──────────────┼────────────────────────┘
  10       % │ 828          │ 10.7819 points         
  20       % │ 1657         │ 9.4015 points         
  30       % │ 2486         │ 8.2336 points         
  40       % │ 3314         │ 8.3012 points         
  50       % │ 4143         │ 8.2143 points         
  60       % │ 4972         │ 7.8668 points         
  70       % │ 5800         │ 7.8668 points         
  80       % │ 6629         │ 7.2780 points         
  90       % │ 7458         │ 6.7954 points         
  100      % │ 8287         │ 6.7375 points
```

## Future directions and improvements

* Test on more problems, with more diverse datasets and more complex tasks and scoring.
* Explore more advanced models.
  * It must be noted that the purpose is to outweigh the cost of running the judge, so the model must be lightweight and fast to run.
