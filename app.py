"""
app.py — Makeup AI Web Application
Interactive makeup application with live camera and photo upload.
"""

from flask import Flask, render_template, jsonify
import sys

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "python": sys.version})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
