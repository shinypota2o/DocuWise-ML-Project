# DocuWise — AI-Powered Study Aid Extractor

Upload slides, PDFs, Word files or screenshots (several at once) – or paste notes – and get a study sheet:

* **Terminologies** – each term with its definition
* **Enumeration** – a label with its *words* numbered underneath (and tables kept as tables)
* **Acronyms and Comparisons** – the PDF is a two-column reviewer in the order Terminologies → Enumeration → Acronyms → Formulas → Comparisons
* **Formula** – equations, including PowerPoint equation objects
* **People** – every person named in the files, with the lines the file says about them (their idea, theory or role)
* **Figures** – pictures from the files, plus screenshots you upload

Text is copied exactly as written (extractive – never reworded). Results can be filtered, copied and exported to PDF.

**Stack:** Flask · python-pptx / pdfplumber / python-docx (+ direct slide-XML reading) · Tesseract OCR (optional) ·
scikit-learn `TfidfVectorizer` + `MultinomialNB` (trained at start-up) · ReportLab · Tailwind (CDN).

## Run

```
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:5000
python tests/test_pipeline.py # tests
python -m classifier.evaluate # accuracy, confusion matrix, CV, model comparison
```

Screenshots and scanned PDFs need the free **Tesseract** program (Windows: UB-Mannheim installer, add to PATH ·
macOS: `brew install tesseract` · Ubuntu: `sudo apt install tesseract-ocr`). Without it everything else still works.

**Defending the project?** Read [`DEFENSE_GUIDE.md`](DEFENSE_GUIDE.md).

## v9 additions

* **Summary strip**: unique terminologies, number of enumerations and their total items, acronyms, formulas, comparisons, figures.
* **Structure tab**: term > list > items tree (an item that is also a term shows its meaning). Counts are in the copy text and the PDF header.
* **Download .json**: summary, structure tree, terminologies, enumerations, acronyms, formulas, comparisons.
* **Text inside pictures (OCR)**: needs Tesseract. Only clean lists and definitions are kept, marked "from picture (OCR)";
  the full text read from each figure is under Figures. Turn off with `DOCUWISE_IMAGE_OCR=0`.
* `python -m classifier.evidence` re-runs the "why Naive Bayes" experiments.

## Layout

```
app.py                      Flask routes (/api/process, /api/export-pdf, /api/model-info)
utils/pptx_reader.py        slide XML reader (hidden text, equations, tables, groups)
utils/omml.py               PowerPoint/Word equation -> readable text
utils/extractors.py         text, PDF (columns, titles), DOCX, screenshot readers
utils/structure.py          lines -> study units (definition / list / table / equation)
utils/pipeline.py, service.py  glue
utils/people.py             people named in the source + the source's lines about them (rules, no model)
classifier/                 features, training data, model, evaluation, hand-labelled test set
dataset/                    your own files (files/) and their label sheets (*.xlsx) used for training
tools/                      make_label_sheet.py (files -> sheet to label), train.py (retrain + save model.pkl)
templates/, static/         UI
tests/                      sample files + tests
```

## Adding your own files to the dataset

The built-in examples in `classifier/training_data.py` are hand-written. To grow the dataset with real course material:

```
1. copy your PDF / PPTX / DOCX files into   dataset/files/
2. python tools/make_label_sheet.py         # writes dataset/label_sheet.xlsx, one row per study unit
3. open the sheet, fill the yellow `label` column (Identification / Enumeration / Concepts / Formula / skip)
4. python tools/train.py                    # retrains on built-in + your rows, saves classifier/model.pkl
5. python -m classifier.evaluate            # new accuracy, plus `own_files_held_out`
```

Add more files any time and repeat from step 2: labels already filled in are kept. `/api/model-info` shows
`from_own_files`. If the dataset has grown since `model.pkl` was saved, the app ignores it and trains at start-up.

Two rules that keep the scores honest: label every row yourself (`model_guess` is only a hint; copying it teaches
the model nothing), and do not put the lectures used in `classifier/external_test.py` into `dataset/files/`,
because that test set must stay unseen.

## Training in Google Colab (optional)

`classifier/model.pkl` is optional. If it exists (and was made with the same scikit-learn version), the app loads it;
otherwise it trains itself at start-up from `classifier/training_data.py`. See `DocuWise_Colab_Training_Guide.pdf`
for the step-by-step process of training and saving it in Colab. `/api/model-info` shows which one is in use (`model_source`).
