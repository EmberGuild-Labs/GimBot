# GimBot

A Python bot that reads your screen, recognizes the current Gimkit question using OCR, and automatically clicks the correct answer based on a Q&A file you configure in advance.

## How it works

1. You fill in a text file with your questions and answers before starting.
2. GimBot takes a screenshot every few seconds and runs OCR on it.
3. It fuzzy-matches the on-screen question against your answer file (so exact wording doesn't need to match).
4. It moves the mouse to the correct answer and clicks it.

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (install via Homebrew on macOS)
- A browser with Gimkit open

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/GimBot.git
cd GimBot

# 2. Install Tesseract (macOS)
brew install tesseract

# 3. Create a virtual environment and install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Windows/Linux:** Download Tesseract from the [official wiki](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) or run `sudo apt install tesseract-ocr` (Linux).

## Configuration

Edit `answers.txt` before running. Each line is a question and its answer separated by `|`. Lines starting with `#` are ignored.

```
# Format: Question | Answer

What is the capital of France? | Paris
What is 2 + 2? | 4
Who wrote Romeo and Juliet? | Shakespeare
```

The question text does not need to be a perfect match — GimBot uses fuzzy matching, so minor OCR errors or slightly different phrasing will still work.

## Usage

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run with default settings
python gimbot.py

# Preview what would be clicked without actually clicking
python gimbot.py --dry-run --verbose

# Watch only a specific region of the screen (e.g. just the browser window)
python gimbot.py --region

# Use a different answers file
python gimbot.py --answers my_quiz.txt

# Adjust the check interval and cooldown
python gimbot.py --interval 2.0 --cooldown 4.0
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--answers` | `answers.txt` | Path to your Q&A file |
| `--interval` | `1.5` | Seconds between screen checks |
| `--cooldown` | `3.0` | Seconds to wait after clicking before checking again |
| `--dry-run` | off | Print what would be clicked without clicking |
| `--region` | off | Interactively select which part of the screen to watch |
| `--q-threshold` | `0.55` | Min similarity score (0–1) to match a question |
| `--a-threshold` | `0.50` | Min similarity score (0–1) to locate the answer on screen |
| `--verbose` / `-v` | off | Show extra debug output |

## macOS permissions

The first time you run GimBot, macOS will ask for two permissions:

- **Screen Recording** — needed to take screenshots
- **Accessibility** — needed to move the mouse and click

Grant both in **System Settings → Privacy & Security**.

## Tips

- **Use `--region`** to restrict GimBot to just the browser window. This reduces noise from other open apps and improves accuracy.
- **Use `--dry-run --verbose`** first to verify questions are being detected correctly before letting it click.
- If matching is too aggressive (clicking wrong answers), raise `--q-threshold` toward `0.8`. If it's missing questions, lower it toward `0.4`.
- Questions in `answers.txt` can be written in natural language — they don't need to match the Gimkit wording character-for-character.

## Project structure

```
GimBot/
├── gimbot.py        # Main program
├── answers.txt      # Your Q&A pairs (edit this)
├── requirements.txt # Python dependencies
└── README.md
```
