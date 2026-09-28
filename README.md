# Unmasking Review Bombing: NLP & Community Outrage on Steam

An end-to-end Machine Learning and Natural Language Processing (NLP) case study analyzing **30,000+ Turkish Steam reviews** of *Hearts of Iron IV* (AppID: `394360`) during the September 2026 Paradox Interactive community backlash.

---

## 📌 Background & Motivation
In late September 2026, a moderation dispute on Paradox Interactive's official Discord server triggered widespread community outrage, resulting in an unprecedented wave of negative user reviews on Steam.

Standard platform aggregation hides these spikes by automatically labeling them as **"Off-Topic Review Activity"** and quarantining them from the overall review score. This project bypasses API quarantines to retrieve raw historical data, extracts latent discussion themes via unsupervised clustering, evaluates supervised sentiment classification on extreme class-imbalanced data, and analyzes community virality dynamics.

---

## 🔬 Key Engineering & Analytical Findings

1. **Algorithmic Quarantine Bypass & Sentiment Distribution:**
   - Steam's standard `filter='recent'` API request quietly omits quarantined review bombing periods.
   - Injecting `filter_offtopic_activity=0` bypassed the quarantine layer, exposing the true community sentiment distribution (**96.59% Negative vs. 3.41% Positive** across 30,093 collected reviews, reduced to 26,264 clean reviews after noise filtering).

2. **Core Player Loyalty vs. Transient Trolls:**
   - Analysis of reviewer playtime revealed that negative reviews did not stem from newly created throwaway accounts.
   - Negative reviewers held an average of **231.5 hours** of in-game playtime (median **34.5 hours**), with **37.29% holding over 100+ hours**, demonstrating that the boycott was heavily driven by core, long-term players. (Positive reviewers averaged 356.6 hours with a median of 165.3 hours).

3. **NLP Theme Discovery (K-Means Clustering):**
   - Unsupervised clustering (K-Means with TF-IDF Bi-grams, $k=3$) separated the text into 3 core topics:
     - **Cluster 1 (Corporate Boycott & Protest):** Focuses on Paradox Interactive, direct boycotts, and protest terms (`atatürk`, `paradox`, `boykot`, `türk`, `oyunu`).
     - **Cluster 2 (Incident Context & Community Respect):** Details the Discord incident, moderation response, and historical respect (`mustafa kemal atatürk`, `sunucusunda atatürk`, `allah razı olsun`).
     - **Cluster 3 (Direct Score Suppression):** Short, sharp negative evaluations focused purely on lowering the game's overall rating (`oyun`, `bok`, `çöp`, `berbat`, `rezalet`, `almayın`).

4. **The Accuracy Paradox & Conjunction Failure:**
   - In a dataset with a ~97:3 class imbalance (96.59% Negative vs 3.41% Positive), an unweighted model predicting only the majority class trivially achieves ~97% accuracy while failing to detect positive sentiment.
   - Using `LogisticRegression(class_weight='balanced')` boosted **Positive Recall to ~84%**, trading off Precision (0.20) to ensure rare positive signals were caught.
   - **Bag-of-Words Limitation:** Concessive conjunctions (e.g., *"Güzel oyun ama son olaylardan dolayı param haram olsun rezalet"*) exposed linear model limits. Since positive n-grams (`güzel oyun`) counterbalanced negative terms, bag-of-words scoring struggled with semantic clause hierarchy.

---

## 📂 Project Structure

```
.
├── paradox.ipynb           # Main Jupyter Notebook (Scraping, Preprocessing, Modeling, Evaluation)
├── hoi4_steam_reviews.csv  # Raw collected Steam reviews dataset (30,093 entries)
├── sentiment_model.pkl     # Trained scikit-learn Logistic Regression model artifact
├── tfidf_vectorizer.pkl    # Fitted TF-IDF Vectorizer vocabulary artifact
└── README.md               # Project documentation
```

---

## 🛠️ Pipeline Architecture

- **Data Extraction:** Reverse engineering Steam API using Python `requests.Session` with rate-limiting, pagination cursors, and quarantine parameter handling (`filter_offtopic_activity=0`).
- **Preprocessing:** Turkish character casing normalization (`İ` $\rightarrow$ `i`, `I` $\rightarrow$ `ı`), URL/punctuation stripping via Regex, filtering short texts ($\le 2$ chars), and custom domain-specific stopword removal.
- **Vectorization:** Scikit-Learn `TfidfVectorizer` (N-gram range `(1, 2)`, `max_features=2500`, document frequency trimming `min_df=5`, `max_df=0.5`).
- **Modeling Stack:**
  - `KMeans(n_clusters=3)`: Unsupervised topic discovery.
  - `LogisticRegression(class_weight='balanced')`: Supervised sentiment classification.
  - `Ridge`: Linear regression testing character length, word count, playtime, and predicted negative probability against community helpfulness votes (`votes_up`).

---

## 📊 Evaluation Summary

### Classification Performance (Logistic Regression)
Evaluated on a 20% stratified test split ($N = 5,253$ reviews):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (0)** | 0.99 | 0.87 | 0.93 | 5059 |
| **Positive (1)** | 0.20 | 0.84 | 0.32 | 194 |
| **Overall Accuracy** | — | — | **0.87** | 5253 |
| **Macro Average** | 0.60 | 0.85 | 0.62 | 5253 |
| **Weighted Average** | 0.96 | 0.87 | 0.90 | 5253 |

### Regression Performance (Ridge on Virality `votes_up`)
- **RMSE:** `129.54`
- **R² Score:** `0.0032`
- **Feature Coefficients:**
  - `negative_prob`: `+6.4006` (Strongest positive driver for community helpfulness votes)
  - `word_count`: `+0.0414`
  - `playtime_hours`: `+0.0019`
  - `char_length`: `+0.0015`
- **Finding:** A review's virality (`votes_up`) is driven primarily by negative sentiment probability alongside platform timing dynamics rather than review length or total player hours.

---

## 🚀 Getting Started

### Prerequisites
Install the required Python packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests joblib
```

### Running the Notebook
Open and execute the main notebook:
```bash
jupyter notebook paradox.ipynb
```

### Loading Pre-trained Models
```python
import joblib

# Load saved model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')

# Classify sample text
sample = ["Oyun tarih simülasyonu olarak harika, kesinlikle tavsiye ederim."]
sample_vec = vectorizer.transform(sample)
pred = model.predict(sample_vec)
probs = model.predict_proba(sample_vec)

print("Label:", "Positive" if pred[0] == 1 else "Negative")
print(f"Probabilities -> Negative: {probs[0][0]:.1%}, Positive: {probs[0][1]:.1%}")
```
