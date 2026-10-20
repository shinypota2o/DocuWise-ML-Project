# DocuWise — Defense Guide

Everything you need to explain, demo and defend the project. Slide numbers refer to the professor's
**Lecture 5** (Supervised Learning: Regression & Classification) and **Lecture 6** (SVM, KNN, Naive Bayes, Ensembles, Model Selection) decks and the **CS-312 Software Testing** lesson PDF.

---

## 1. The 60-second pitch

> "DocuWise is a study-aid extractor. A student uploads a PowerPoint, PDF, Word file or screenshot and gets back a study sheet.
> It reads the file **while keeping its structure** – titles, bullets, indentation, tables and even PowerPoint equations – and cuts it into *study units*.
> A **supervised text classifier (TF-IDF + Multinomial Naive Bayes)**, trained when the app starts, sorts each unit into **Identification/Terminologies**, **Enumeration** (a label with its *words* numbered under it) or **Formula**. A label followed by full sentences is kept as a *term card* (label + the source's own statements) under Terminologies. Rules then add **Acronyms** and **Comparisons** (tables).
> The text is **extractive**: we copy the original wording, we never paraphrase. The result can be searched, copied, and exported to PDF."

Why it is "AI": the sorting is a *trained* model (supervised classification, Lecture 5/6), not a pile of if-statements.
Why it is trustworthy: nothing is generated, so nothing can be made up.

---

## 2. What the user sees → what runs behind it

| Step | What happens | File |
|---|---|---|
| 1. Attach files | Browser shows a chip per file (name, size, type, ✕ remove, "ready to analyze"); button says "Analyze N files"; progress bar while uploading | `static/js/main.js`, `templates/index.html` |
| 2. Upload | `POST /api/process` (multipart, many files + optional pasted text) | `app.py` |
| 3. Read | One reader per format. Output = *lines with level, bullet/number and frame*, plus tables and pictures | `utils/pptx_reader.py`, `utils/extractors.py`, `utils/image_extractors.py` |
| 4. Understand structure | Lines → indentation tree → **study units** (definition / list / table / equation) | `utils/structure.py`, `utils/pipeline.py` |
| 5. Classify | Each unit → shape tokens + words → TF-IDF → Multinomial NB → label + confidence | `classifier/features.py`, `classifier/model.py` |
| 6. Show | Terminologies cards (term + definition), Enumeration cards (label + 1. 2. 3. words), Formula cards, Comparisons, Figures | `static/js/main.js` |
| 6b. Organize | `utils/reviewer.py` merges duplicate terms, finds acronyms (`Full Name (ABBR)`), moves tables to Comparisons, removes repeated animation slides and junk labels | `utils/reviewer.py` |
| 7. Export | Two-column reviewer in the required order (Terminologies → Enumeration → Acronyms → Formulas → Comparisons): summary strip, then per part: term + its source text, label + numbered words, boxed formulas, comparison tables, figures | `utils/pdf_export.py` |

### Step 3 in more detail – "reading" is where most files go wrong
* **PPTX**: we read the slide XML directly. Many decks hide text inside `mc:AlternateContent`, and equations are `a14:m / m:oMath` objects that normal libraries skip – `utils/omml.py` converts those to readable text such as `Accuracy = (TP + TN)/(TP + TN + FP + FN)`.
  Indent level comes from `lvl` and `marL`; bullets from `buChar` / `buAutoNum`.
* **PDF**: `pdfplumber` gives every word with its x/y/font size. We cluster words into rows, detect two-column pages, re-join wrapped lines, take **titles from font size** (or a centred first line), and get nesting from the bullet/number markers.
* **DOCX**: Title/Heading styles → sections; list styles and `numPr` → bullets/levels; tables and equations included.
* **Screenshots**: Tesseract OCR (`image_to_data`) – keeps the left offset of each line so indentation survives. Without Tesseract the picture is still shown under *Figures*.

### Step 4 – how a "study unit" is found (rules, explained simply)
1. **Titles/section headings are context only** (shown as "Slide 5 › Regression"), never listed as terms.
2. **Terminologies (Identification)** = one of
   * a defining sentence: "*X is / refers to / is called / stands for / is defined as …*"
   * a term line with a sentence under it (`Root Node` / `The first decision made by the tree.`)
   * a `Term – meaning` or `Term: meaning` line.
3. **Enumeration** = an *intro or label* followed by bullets/numbers/short fragments, e.g. `Types of Supervised Learning` → 1. Regression 2. Classification. Only the **words** are listed. If an item is `Term – description`, the list keeps the term and the description becomes its own Terminologies card. Nested items are numbered 1.1, 1.2. Tables are kept as tables.
3b. A *term card* = a label followed by full *sentences* (e.g. "Characteristics", "How KNN works"). The classifier learns this difference from item length (`__sentence_items__`, `__avgwords_n__` tokens).
3c. **Terms found in more ways**: bold lead-ins (`**Snooping** means …`), `Term:` headings with details underneath (shown as a term card with its details numbered), `X (ABBR) is …`, `X stands for …`. Headings such as "Lesson 3: …" are never treated as terms.
4. **Formula** = equations (including PowerPoint equation objects) – trivial settings like `K = 5` are ignored.
5. Everything else (plain explanatory sentences) is skipped and counted ("86 plain sentences left out").

### Step 5 – the machine-learning part
* **Training data** (`classifier/training_data.py`): ~325 hand-written *units* in the same shapes the extractor produces (term+definition, label+items, tables, equations) from many subjects.
* **Features** (`classifier/features.py`): the unit's words **plus shape tokens** such as `__shape_pair__`, `__items_4__`, `__long_items__`, `__has_math__`. To TF-IDF these are ordinary words, so the model *learns* that "pair + defining words" → Identification, "group + several short items" → Enumeration, "math symbols" → Formula.
* **Model**: `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` → `MultinomialNB(alpha=0.3)` in a scikit-learn `Pipeline`. **Trained once at startup** (`get_pipeline()`).
* **Decision**: the NB label is final. If the probability is under 0.5 the unit falls back to the label implied by its shape; the card records `decided_by` (`naive-bayes` or `shape-fallback`) and `confidence`.

---

## 3. What from the professor's slides is used (concept → where in DocuWise)

| Lecture concept (slide) | How DocuWise applies it |
|---|---|
| **Supervised learning**: learn from labelled examples (L5 s2) | Our classifier learns from labelled units: *features* = unit text + shape tokens, *label* = Identification / Enumeration / Formula |
| **Classification vs regression** (L5 s4, s30) | Our target is a discrete category → it is a **classification** problem (multi-class, 4 classes). No regression. |
| **Features and target / label** (L5 s2) | `unit_features()` builds the features; the category is the label |
| **Training set / testing set, 80/20 split** (L5 s14) | `classifier/evaluate.py` uses `train_test_split(test_size=0.2, stratify=y)` |
| **Confusion matrix, TP/TN/FP/FN** (L5 s47–49) | Printed by `evaluate.py` and `/api/model-info` (4×4 matrix for 4 classes) |
| **Accuracy, Precision, Recall, F1** (L5 s50–57) | Reported overall and **per class** by `evaluate.py` |
| **Overfitting vs underfitting** (L5 s62–63) | Small training set → risk of memorising. We test on held-out data **and** on a separate hand-labelled set (`classifier/external_test.py`); `alpha` smoothing and sub-linear TF reduce over-confidence |
| **Naive Bayes, Bayes' theorem** (L5 s46, L6 s28) | The classifier is Multinomial Naive Bayes: P(class | words) ∝ P(class) · ∏ P(wordᵢ | class) |
| **Why "naive": features assumed conditionally independent** (L6 s30) | We treat each word/token as independent evidence. Said honestly: words are not independent, but NB still works well for text |
| **Types: Gaussian / Multinomial / Bernoulli** (L6 s32–34) | We chose **Multinomial** – the slide says it is for **text / word counts / document classification** (s33). Gaussian = continuous features, Bernoulli = binary present/absent |
| **Advantages & limitations of NB** (L6 s35) | Fast, simple, good with high-dimensional text, little training data ✔ (why we picked it). Independence assumption, probability estimates not well calibrated ✔ (why we added the confidence floor and shape fallback) |
| **Comparing SVM, KNN, NB** (L6 s36) | `evaluate.py` compares them with the same cross-validation (results below) |
| **Cross-validation** (L6 s58–59) | 5-fold stratified CV in `evaluate.py` |
| **Hyperparameters / tuning** (L6 s60–61) | `alpha` (NB), n-gram range, and the 0.5 confidence floor are our hyperparameters; GridSearch would be the next step |
| **Model selection – "the best model depends on dataset size…"** (L6 s56–57, s63) | Small dataset + text features + need for fast start-up training → Naive Bayes (Dataset size, Speed, Interpretability) |
| **Ensemble ideas (bagging, Random Forest)** (L6 s39–55) | Not used – mentioned as future work; Decision Tree comparison shows what a tree-based model does on the same data |
| **ML workflow: split → train → evaluate → tune → pick the model** (L5 s60–61 Regression/Classification workflow, L6 s62 General Model Selection Workflow) | Collect/label units → 80/20 split → fit pipeline → metrics → tune alpha → Flask app trains on start-up |
| **Software testing** (CS-312 lesson) | `tests/test_pipeline.py` = **unit/integration tests**; the sample files are *black-box* checks (input file → expected output); `evaluate.py` is *system-level* validation of the classifier |

> **Be honest in the defense:** **TF-IDF** and **Laplace/additive smoothing (`alpha`)** are *not* in the professor's slides. They are standard, simple add-ons: TF-IDF is a better version of the word counts the lecture mentions for Multinomial NB (it down-weights words that appear everywhere); `alpha` solves the "zero-frequency problem" (a word never seen in a class would make the whole product zero).

---

## 4. Numbers to quote (from `python -m classifier.evaluate` or `GET /api/model-info`)

Training data: **651 labelled units** = 325 hand-written examples (`classifier/training_data.py`) + **326 units from 9 real
course files** labelled by us in `dataset/label_sheet.xlsx` (403 Identification / 96 Enumeration / 90 Concepts / 62 Formula).

**The main result: whole files the model has never seen** (`python tools/evaluate.py`)

| Test | Result |
|---|---|
| Leave-one-file-out, 6 development files (281 units): classifier | **92.9%** (F1: Identification 96.6, Enumeration 38.5, Concepts 28.6, Formula 85.7) |
| Same files, whole system (file read → classified → reviewer section) | **96.0%** of units end up in the right section |
| Junk units (labelled "skip") still shown to the student | **1 of 14** (was 19 of 20 before the v16 fixes) |
| **Locked test set: 3 files never used for tuning, scored once** | **84.4%** classifier, **83.3%** whole system |

Classic Lecture 5/6 numbers (`python -m classifier.evaluate`, all 651 units):
* 80/20 stratified hold-out: **90.1%**; 5-fold cross-validation: **88.8% ± 1.3**
* Unseen hand-labelled lecture units (`classifier/external_test.py`): **96.7%** (59/61) model alone, 95.1% full app logic
* Model comparison (same CV): Naive Bayes 88.8%, Linear SVM 93.7%, Decision Tree 92.5%, KNN 88.2%.
  On leave-one-file-out the gap almost disappears: NB 93.2%, LinearSVC 94.0%, Logistic Regression 93.2% (1-2 units of 281).

**How the evaluation was done (say this):** *"Every lecture file is tested by a model trained without it (leave-one-file-out),
and a slide that repeats in another file is removed from training, so the model never sees the file it is tested on.
Only our own labels count as truth. Three files were locked away before any tuning and scored exactly once at the end:
83%, compared with 94-96% on the development files, and we report the lower number as the honest estimate for new files.
Each rule fix was kept only if no file got worse; six reference lectures and 43 automated tests check nothing regressed."*

**How to say it honestly:** *"Naive Bayes is not the single best model on our small set – SVM and trees score a few points higher
in cross-validation – but on whole unseen files they are within 1-2 units of each other, NB trains instantly at start-up, needs
little data, gives real probabilities for the confidence fallback, and fits the text-count setting from Lecture 6 (s33–35)."*

**Why Enumeration/Concepts F1 is low:** only 17 Enumeration and 12 Concepts labels come from real files (one unit moves F1 by ~8
points), and many Concepts units are term cards that end up in the same Terminologies section either way.

**Limits you should state yourself (examiners respect this):**
1. Training data is small (9 real files, one of them 60% of the labels), so scores on new files are lower: 83% on the locked test set.
2. Because we add *shape tokens*, the classifier relies partly on structure – that is intended (a list *looks* different from a definition), but it means the reader/structure step matters as much as the model.
3. Scanned PDFs and screenshots need Tesseract OCR; OCR mistakes carry through.
4. Unusual slide designs (text in pictures, heavy diagrams) are skipped, not guessed.

---

## 5. Live demo script (3 minutes)

1. Open the app → drop **two files at once** (e.g. Lecture-6.pptx + the CS-312 PDF). Point at the green chips: name, size, type, ✕, "ready to analyze", button changes to **"Analyze 2 files"**; progress bar runs.
2. **Identification** tab → show `Unit Testing` + definition, `Root Node` + definition. "These are terms with their definitions, copied word-for-word; slide title is only the grey context line."
3. **Enumeration** tab → `Types of Integration Testing` with 1–4 items; `Advantages` / `Limitations` of Naive Bayes; the SVM/KNN comparison table.
4. **Formula** tab → `Accuracy = (TP + TN)/(TP + TN + FP + FN)` read from a PowerPoint equation object.
5. **Figures** tab → pictures from the slides; click to enlarge.
6. Type "spam" in the filter; click copy on one card; **Download as PDF**.
7. Open `/api/model-info` to show accuracy, per-class metrics and the confusion matrix.

---

## 6. Likely questions and answers

**Why Naive Bayes and not a neural network / SVM?** Small labelled set, text-like features, instant training at start-up; Lecture 6 s33–35 – Multinomial NB for text, fast, simple, little data. We did compare SVM/KNN/Tree (section 4).

**Why is it called "naive"?** It assumes features are independent given the class (L6 s30). False for real language, but NB's *ranking* of classes is usually still right.

**What is TF-IDF?** Term Frequency × Inverse Document Frequency. Words that are frequent in one unit but rare overall get higher weight. We use it instead of raw counts. (Not in the slides – extension.)

**What does `alpha` do?** Additive (Laplace) smoothing: adds a small count so unseen words don't give zero probability.

**Is the extraction AI?** Two layers: (1) deterministic structure reading (rules about indentation, bullets, definitions), (2) learned classifier deciding the category. We say both clearly.

**Why not just use rules for everything?** Rules find the *units*; the category boundary (is "Accuracy: (TP+TN)/…" a definition or a formula? is "Characteristics" a list or a definition?) is fuzzy and is learned from examples, and it can be improved by adding labelled data – not by editing code.

**How do you avoid rewording / hallucination?** Extractive only: readers may join a sentence broken over two lines, nothing else. No text generation anywhere.

**How did you evaluate?** Lecture-5 workflow: 80/20 stratified split, confusion matrix, accuracy/precision/recall/F1, plus 5-fold CV and a hand-labelled set that was never trained on.

**What are false positives here?** E.g. predicting *Enumeration* for something that is really *Identification* – shown in the confusion matrix off-diagonal cells.

**Overfitting?** Small data is the risk. Held-out and external tests; smoothing; no deep features.

**How would you improve it?** More labelled data from real classes; GridSearch over `alpha`/n-grams; calibrated probabilities; user feedback button ("wrong category") to grow the training set.

**Where is the "unit test" you learned in CS-312?** `tests/test_pipeline.py` – one test per input type (text, DOCX, PDF, screenshot) asserting expected terms/lists/formulas.

---

## 7. Run it

```
pip install -r requirements.txt
# optional, for screenshots and scanned PDFs:
#   Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki  (add to PATH)
#   macOS:   brew install tesseract      Ubuntu: sudo apt install tesseract-ocr
python app.py          # http://127.0.0.1:5000
python -m classifier.evaluate   # the numbers for the defense
python tests/test_pipeline.py   # tests
```

---

## 8. The reviewer (final scope)

The reviewer focuses on what is most reliable to extract and most useful for exams:

1. **Terminologies (Identification)** – term in bold, its original definition / description as bullets. Same term from several slides is merged (different meanings kept, duplicates removed). A label followed by statements ("Characteristics", "Advantages", "How KNN works") is also shown here as a term card.
2. **Enumeration** – title + the *words* numbered in source order.
3. **Acronyms** – `ABBR – Full meaning`, only when the source really contains it.
4. **Formulas** and 5. **Comparisons** (tables) – copied as written.
6. Figures (optional in the PDF).

*Important Concepts* and *Exam Keypoints* were removed: they were the least reliable parts and duplicated other sections.
The PDF is a two-column reviewer with page numbers (`utils/pdf_export.py`).

**Why some terms can still be missing (and what "more training" can and cannot do):**
the classifier only *labels* units that the reader has already found. Missing terms are almost always a *reading* problem (a sentence broken over two lines, a heading style the rules haven't seen), so the fix is testing on more real files, not just adding more training rows. More labelled examples help the *category* decision (term vs list vs formula).


## 9. What the four real files taught us (v6)

Running real lecture files (a Flutter PPTX, a Flutter DOCX, two PDFs) exposed *reading* problems, not training problems. Each fix is a rule you can explain:

| Problem seen | Rule added |
|---|---|
| `Term ➢ definition` rows whose second line sits beside the next bullet | PDF reader moves a hanging line to the row whose arrow it continues (`_split_hanging`) |
| Big slide title overlapping the first numbered item | `_split_sizes` separates the two font sizes into two rows |
| `Term → meaning`, `Term ➢ meaning` | `split_item` keeps the *term* in the enumeration and makes the meaning a terminology |
| "There are two kinds" / "Responsible for" without a subject | `_assign_parents` attaches the term defined just before (shown as `Abstraction ›`) |
| Same list repeated on animation slides | duplicates dropped (`_dedupe_enums`, repeated-slide merge) |
| Footer "Lecture 5", "Lecture 6" | digits ignored when finding repeated footers |
| `✅ a ✅ b` in one paragraph | exploded into separate bullets |
| `↓ │ ┌─┐` diagram art, code lines, `A → B → C` flows | filtered (they are not study text) |
| `watch()` + code + `Meaning: "..."` | paired as a term and its source meaning |
| Slides such as References / Thank you, labels like Objectives / Instructions | skipped |
