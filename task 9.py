# ============================================================
#   NLP Task: Text Normalization
#   Source: GeeksforGeeks – NLP Tutorial
#   Author: Generated for GFG NLP Assignment
# ============================================================
#
# Text Normalization transforms raw, inconsistent text into a
# clean, standardized format before feeding it into NLP models.
#
# Techniques covered in this script:
#   1. Lowercasing (Case Folding)
#   2. Removing Punctuation
#   3. Removing Numbers
#   4. Removing Extra Whitespace
#   5. Expanding Contractions
#   6. Removing Stopwords
#   7. Stemming  (PorterStemmer)
#   8. Lemmatization (WordNetLemmatizer)
#   9. Removing URLs & HTML Tags
#  10. Full Pipeline – applying all steps together
# ============================================================

import re
import string

# ── NLTK downloads (run once) ─────────────────────────────────
import nltk
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4',     quiet=True)

from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────
# Sample raw text (messy, real-world style)
# ─────────────────────────────────────────────────────────────
raw_text = """
Hello!!! I've been working on NLP projects since 2021.
It's amazing how we can't ignore the power of AI today!
Visit https://www.geeksforgeeks.org for more info.
<b>Text Normalization</b> is KEY to good NLP pipelines...
Running, runs, & runner are all forms of the word 'run'.
We   have  extra   spaces   here.    
"""

print("=" * 60)
print("  NLP TASK: TEXT NORMALIZATION")
print("=" * 60)
print("\n[ORIGINAL TEXT]\n")
print(raw_text)

# ─────────────────────────────────────────────────────────────
# STEP 1 – Lowercasing
# ─────────────────────────────────────────────────────────────
def to_lowercase(text):
    return text.lower()

# ─────────────────────────────────────────────────────────────
# STEP 2 – Remove URLs
# ─────────────────────────────────────────────────────────────
def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

# ─────────────────────────────────────────────────────────────
# STEP 3 – Remove HTML tags
# ─────────────────────────────────────────────────────────────
def remove_html(text):
    return re.sub(r'<[^>]+>', '', text)

# ─────────────────────────────────────────────────────────────
# STEP 4 – Expand contractions
# ─────────────────────────────────────────────────────────────
CONTRACTIONS = {
    "i've":   "i have",
    "it's":   "it is",
    "can't":  "cannot",
    "we're":  "we are",
    "they're":"they are",
    "you're": "you are",
    "isn't":  "is not",
    "aren't": "are not",
    "won't":  "will not",
    "didn't": "did not",
    "don't":  "do not",
    "doesn't":"does not",
    "i'm":    "i am",
    "i'd":    "i would",
    "he's":   "he is",
    "she's":  "she is",
    "that's": "that is",
    "there's":"there is",
    "we've":  "we have",
    "they've":"they have",
    "would've":"would have",
    "could've":"could have",
    "should've":"should have",
}

def expand_contractions(text):
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    return text

# ─────────────────────────────────────────────────────────────
# STEP 5 – Remove punctuation
# ─────────────────────────────────────────────────────────────
def remove_punctuation(text):
    # Keep apostrophes handled already; remove rest
    return text.translate(str.maketrans('', '', string.punctuation))

# ─────────────────────────────────────────────────────────────
# STEP 6 – Remove numbers
# ─────────────────────────────────────────────────────────────
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# ─────────────────────────────────────────────────────────────
# STEP 7 – Remove extra whitespace
# ─────────────────────────────────────────────────────────────
def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

# ─────────────────────────────────────────────────────────────
# STEP 8 – Remove stopwords
# ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(filtered)

# ─────────────────────────────────────────────────────────────
# STEP 9 – Stemming
# ─────────────────────────────────────────────────────────────
stemmer = PorterStemmer()

def apply_stemming(text):
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(w) for w in tokens]
    return ' '.join(stemmed)

# ─────────────────────────────────────────────────────────────
# STEP 10 – Lemmatization
# ─────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

def apply_lemmatization(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in tokens]  # pos='v' for verbs
    return ' '.join(lemmatized)

# ─────────────────────────────────────────────────────────────
# FULL PIPELINE – show each step
# ─────────────────────────────────────────────────────────────
def normalize_text(text, verbose=True):
    steps = [
        ("Step 1 – Lowercase",           to_lowercase),
        ("Step 2 – Remove URLs",          remove_urls),
        ("Step 3 – Remove HTML Tags",     remove_html),
        ("Step 4 – Expand Contractions",  expand_contractions),
        ("Step 5 – Remove Punctuation",   remove_punctuation),
        ("Step 6 – Remove Numbers",       remove_numbers),
        ("Step 7 – Remove Extra Spaces",  remove_extra_spaces),
        ("Step 8 – Remove Stopwords",     remove_stopwords),
    ]

    current = text
    for label, func in steps:
        current = func(current)
        if verbose:
            print(f"\n[{label}]\n  → {current.strip()[:120]}")

    # Show Stemming & Lemmatization side-by-side on the cleaned text
    print("\n" + "─" * 60)
    print("[Step 9 – Stemming (PorterStemmer)]")
    print("  →", apply_stemming(current))

    print("\n[Step 10 – Lemmatization (WordNetLemmatizer)]")
    print("  →", apply_lemmatization(current))

    return current

# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("APPLYING NORMALIZATION PIPELINE STEP BY STEP")
print("─" * 60)

final_output = normalize_text(raw_text, verbose=True)

print("\n" + "=" * 60)
print("FINAL NORMALIZED OUTPUT")
print("=" * 60)
print(final_output)
print("\n" + "=" * 60)
print("✅  Text Normalization complete!")
print("=" * 60)
