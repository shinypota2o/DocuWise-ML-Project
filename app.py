"""
DocuWise - AI-powered Study Aid Extractor.

Flow of one request:
  upload(s) / pasted text
    -> utils.service      read the file, keep its structure (indentation, bullets, tables, equations)
    -> utils.structure    find study units: term+definition, label+items, tables, formulas
    -> classifier.model   TF-IDF + Multinomial Naive Bayes decides Identification / Enumeration / Formula
    -> JSON for the dashboard  (and /api/export-pdf for the PDF)
"""

import os
import threading

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from classifier.model import LABELS, classify_units, get_pipeline, model_info
from utils.extractors import SUPPORTED_EXTENSIONS
from utils.pdf_export import generate_pdf
from utils.reviewer import ORDER, TITLES, build_reviewer
from utils.service import process_file, process_text, read_error_message
from utils.summary import build_tree, summarize

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB per request (all files together)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0          # always serve the newest CSS/JS (stale cache = unstyled results)


@app.after_request
def _no_cache(resp):
    if request.path.startswith("/static/") or request.path == "/":
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@app.context_processor
def _asset_version():
    def v(name):
        try:
            return int(os.path.getmtime(os.path.join(app.static_folder, name)))
        except OSError:
            return 0
    return {"asset_v": v}


_eval_cache = {}
_eval_lock = threading.Lock()


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "The files are too large (40 MB maximum in total)."}), 413


@app.route("/")
def index():
    return render_template("index.html", supported=sorted(SUPPORTED_EXTENSIONS))


@app.route("/api/process", methods=["POST"])
def process():
    """
    multipart/form-data with one or more 'files' fields (.pdf .docx .pptx .png .jpg),
    and/or a 'text' field with pasted notes.
    """
    try:
        files = [f for f in request.files.getlist("files") if f and f.filename]
        text = (request.form.get("text") or (request.get_json(silent=True) or {}).get("text") or "").strip()
        if not files and not text:
            return jsonify({"error": "Please attach a file or paste some text first."}), 400

        all_units, images, notices, sources, skipped, texts = [], [], [], [], 0, []
        results = []

        for f in files:
            name = secure_filename(f.filename) or f.filename
            ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in SUPPORTED_EXTENSIONS:
                notices.append(f"{f.filename}: unsupported type '{ext or 'unknown'}'. Use PDF, DOCX, PPTX, PNG or JPG.")
                continue
            data = f.read()
            try:
                r = process_file(name, data)
            except Exception as exc:  # noqa: BLE001
                app.logger.exception("Could not read %s", name)
                notices.append(read_error_message(f.filename, data, exc))
                continue
            results.append((name, r))
        if text:
            results.append(("Pasted Text", process_text(text)))

        for name, r in results:
            sources.append(name)
            for u in r["units"]:
                u["source"] = name
            for im in r["images"]:
                im["source"] = name
            all_units.extend(r["units"])
            images.extend(r["images"])
            notices.extend(r["notices"])
            skipped += r["skipped"]
            texts.append(r.get("text", ""))

        if not all_units and not images:
            msg = ("None of the files could be read." if not results else
                   "No definitions, lists or formulas were found. If the file is scanned or made of pictures, "
                   "its text can only be read when OCR (Tesseract) is installed.")
            return jsonify({"error": msg, "notices": notices}), 422

        classified = classify_units(all_units)
        categories = build_reviewer(classified, "\n".join(texts))
        counts = {k: len(categories[k]) for k in ORDER}
        counts["Figures"] = len(images)
        summary = summarize(categories, images, sources)
        tree = build_tree(categories)
        return jsonify({
            "sources": sources,
            "counts": counts,
            "titles": TITLES,
            "total": sum(counts[k] for k in ORDER if k != "Exam"),
            "categories": categories,
            "summary": summary,
            "tree": tree,
            "images": images,
            "notices": notices,
            "skipped": skipped,
        })
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Processing failed")
        return jsonify({"error": f"Something went wrong while processing your input: {exc}"}), 500


@app.route("/api/export-pdf", methods=["POST"])
def export_pdf():
    try:
        payload = request.get_json(silent=True) or {}
        categories = payload.get("categories", {})
        if not any(categories.get(k) for k in ORDER) and not payload.get("images"):
            return jsonify({"error": "Nothing to export."}), 400
        buf = generate_pdf(categories, images=payload.get("images", []),
                           document_title=payload.get("source") or "DocuWise Study Summary",
                           summary=payload.get("summary"))
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                         download_name="DocuWise_Study_Summary.pdf")
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("PDF export failed")
        return jsonify({"error": f"Could not generate PDF: {exc}"}), 500


@app.route("/api/model-info")
def api_model_info():
    """The numbers to quote at the defense: training set, split, accuracy, confusion matrix."""
    from classifier.evaluate import evaluate
    with _eval_lock:
        if "r" not in _eval_cache:
            _eval_cache["r"] = evaluate()
    return jsonify({"model": model_info(), "evaluation": _eval_cache["r"]})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    get_pipeline()  # train once at start-up
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "1") == "1")
