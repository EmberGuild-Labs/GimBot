#!/usr/bin/env python3
"""
GimBot — Automatic Gimkit answer selector.

Usage:
    python gimbot.py                        # Run with default answers.txt
    python gimbot.py --answers my_qa.txt    # Use a different answers file
    python gimbot.py --dry-run              # Preview without clicking
    python gimbot.py --interval 2.0         # Check screen every 2 seconds
    python gimbot.py --region               # Select screen region interactively
"""

import time
import sys
import argparse
from pathlib import Path
from difflib import SequenceMatcher

try:
    import pytesseract
    from PIL import Image, ImageGrab, ImageEnhance, ImageFilter
    import pyautogui
    import numpy as np
    import cv2
except ImportError as e:
    print(f"\nMissing dependency: {e}")
    print("Install all dependencies with:\n")
    print("    pip install pytesseract pillow pyautogui opencv-python numpy\n")
    print("Also install Tesseract OCR engine:")
    print("    macOS:  brew install tesseract")
    print("    Linux:  sudo apt install tesseract-ocr")
    print("    Windows: https://github.com/UB-Mannheim/tesseract/wiki\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_ANSWERS_FILE = "answers.txt"
DEFAULT_INTERVAL = 1.5       # seconds between screen checks
DEFAULT_COOLDOWN = 3.0       # seconds to wait after clicking before checking again
DEFAULT_Q_THRESHOLD = 0.55   # minimum similarity to match a question
DEFAULT_A_THRESHOLD = 0.50   # minimum similarity to find the answer on screen


# ---------------------------------------------------------------------------
# Load Q&A pairs
# ---------------------------------------------------------------------------

def load_answers(filepath: str) -> dict[str, str]:
    """Load question→answer pairs from a text file."""
    path = Path(filepath)

    if not path.exists():
        print(f"[!] Answers file '{filepath}' not found — creating a sample one.")
        sample = (
            "# GimBot Answers File\n"
            "# Format:  Question | Answer\n"
            "# Lines starting with # are ignored.\n"
            "#\n"
            "What is the capital of France? | Paris\n"
            "What is 2 + 2? | 4\n"
        )
        path.write_text(sample)
        print(f"    Sample created at: {filepath}\n")
        return {}

    answers: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                print(f"[!] Line {lineno}: missing '|' separator, skipping → {line!r}")
                continue
            question, _, answer = line.partition("|")
            question = question.strip()
            answer = answer.strip()
            if question and answer:
                answers[question] = answer

    print(f"[+] Loaded {len(answers)} Q&A pair(s) from '{filepath}'")
    return answers


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def find_best_question_match(
    screen_text: str,
    answers: dict[str, str],
    threshold: float = DEFAULT_Q_THRESHOLD,
) -> tuple[str | None, float]:
    """Return (correct_answer, score) for the best matching question."""
    best_answer = None
    best_score = 0.0

    norm_screen = normalize(screen_text)

    for question, answer in answers.items():
        norm_q = normalize(question)
        score = SequenceMatcher(None, norm_screen, norm_q).ratio()

        # Bonus: if most words of the question appear in the screen text
        words = [w for w in norm_q.split() if len(w) > 3]
        if words:
            found = sum(1 for w in words if w in norm_screen)
            word_ratio = found / len(words)
            score = max(score, word_ratio * 0.9)

        if score > best_score:
            best_score = score
            best_answer = answer

    if best_answer and best_score >= threshold:
        return best_answer, best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Screen capture & OCR
# ---------------------------------------------------------------------------

def capture_screen(region: tuple | None = None) -> Image.Image:
    """Capture the full screen (or a region) as a PIL Image."""
    return ImageGrab.grab(bbox=region, all_screens=True)


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Enhance an image to improve OCR accuracy."""
    # Scale up — Tesseract works better at higher resolutions
    w, h = image.size
    image = image.resize((w * 2, h * 2), Image.LANCZOS)

    # Convert to grayscale
    image = image.convert("L")

    # Increase contrast
    image = ImageEnhance.Contrast(image).enhance(2.0)

    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)

    return image


def ocr_full(image: Image.Image) -> list[dict]:
    """
    Run Tesseract on the image and return a list of text blocks with positions.
    Each block: { text, x, y, cx, cy, w, h }
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n = len(data["text"])

    # Group words by block_num
    blocks: dict[int, dict] = {}
    for i in range(n):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        block_num = data["block_num"][i]

        if conf < 20 or not word:
            continue

        if block_num not in blocks:
            blocks[block_num] = {
                "words": [],
                "x1": data["left"][i],
                "y1": data["top"][i],
                "x2": data["left"][i] + data["width"][i],
                "y2": data["top"][i] + data["height"][i],
            }
        else:
            blocks[block_num]["x1"] = min(blocks[block_num]["x1"], data["left"][i])
            blocks[block_num]["y1"] = min(blocks[block_num]["y1"], data["top"][i])
            blocks[block_num]["x2"] = max(
                blocks[block_num]["x2"], data["left"][i] + data["width"][i]
            )
            blocks[block_num]["y2"] = max(
                blocks[block_num]["y2"], data["top"][i] + data["height"][i]
            )
        blocks[block_num]["words"].append(word)

    result = []
    for block in blocks.values():
        text = " ".join(block["words"])
        cx = (block["x1"] + block["x2"]) // 2
        cy = (block["y1"] + block["y2"]) // 2
        result.append(
            {
                "text": text,
                "x": block["x1"],
                "y": block["y1"],
                "cx": cx,
                "cy": cy,
                "w": block["x2"] - block["x1"],
                "h": block["y2"] - block["y1"],
            }
        )

    return result


# ---------------------------------------------------------------------------
# Locate question and answer on screen
# ---------------------------------------------------------------------------

def find_question_block(blocks: list[dict], screen_h: int) -> dict | None:
    """
    Heuristic: Gimkit shows the question in the upper ~50% of the screen.
    Among those blocks, prefer the longest text.
    """
    upper = [b for b in blocks if b["y"] < screen_h * 0.55 and len(b["text"]) > 5]
    if not upper:
        upper = [b for b in blocks if len(b["text"]) > 5]
    if not upper:
        return None
    upper.sort(key=lambda b: len(b["text"]), reverse=True)
    return upper[0]


def find_answer_block(
    answer: str,
    blocks: list[dict],
    screen_h: int,
    threshold: float = DEFAULT_A_THRESHOLD,
) -> tuple[dict | None, float]:
    """
    Find which on-screen text block best matches the expected answer.
    Gimkit answers are typically in the lower ~70% of the screen.
    """
    candidates = [b for b in blocks if b["y"] > screen_h * 0.2]
    if not candidates:
        candidates = blocks

    best_block = None
    best_score = 0.0

    norm_ans = normalize(answer)

    for block in candidates:
        norm_block = normalize(block["text"])

        # Direct containment check (handles partial text in a larger block)
        if norm_ans in norm_block or norm_block in norm_ans:
            score = 0.85
        else:
            score = SequenceMatcher(None, norm_ans, norm_block).ratio()

        if score > best_score:
            best_score = score
            best_block = block

    if best_block and best_score >= threshold:
        return best_block, best_score
    return None, best_score


# ---------------------------------------------------------------------------
# Click helper
# ---------------------------------------------------------------------------

def click_at(x: int, y: int, scale: float = 1.0) -> None:
    """Click at (x, y), adjusting for any image scaling applied before OCR."""
    real_x = int(x / scale)
    real_y = int(y / scale)
    pyautogui.moveTo(real_x, real_y, duration=0.25)
    time.sleep(0.05)
    pyautogui.click()


# ---------------------------------------------------------------------------
# Interactive region selector
# ---------------------------------------------------------------------------

def select_region() -> tuple | None:
    """
    Let the user drag a rectangle on the screen to select the Gimkit window region.
    Returns (x1, y1, x2, y2) or None to use the full screen.
    """
    print("\nRegion selection:")
    print("  You'll have 3 seconds to switch to your Gimkit window.")
    print("  Then draw a rectangle around it by clicking top-left then bottom-right.")
    print("  Press Enter to skip and use the full screen instead.\n")
    choice = input("Select region? [y/N]: ").strip().lower()
    if choice != "y":
        return None

    print("Switching in 3 seconds…")
    time.sleep(3)

    print("Click the TOP-LEFT corner of the Gimkit window…")
    pos1 = pyautogui.position()
    while True:
        new_pos = pyautogui.position()
        if pyautogui.mouseInfo() or True:  # just wait for a click
            break
        time.sleep(0.1)

    # Simpler: ask user to click twice via input pauses
    input("Move mouse to the TOP-LEFT corner then press Enter…")
    pos1 = pyautogui.position()
    print(f"  Top-left: {pos1}")

    input("Move mouse to the BOTTOM-RIGHT corner then press Enter…")
    pos2 = pyautogui.position()
    print(f"  Bottom-right: {pos2}")

    x1 = min(pos1.x, pos2.x)
    y1 = min(pos1.y, pos2.y)
    x2 = max(pos1.x, pos2.x)
    y2 = max(pos1.y, pos2.y)

    print(f"  Region set to: ({x1}, {y1}, {x2}, {y2})\n")
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    answers_file: str = DEFAULT_ANSWERS_FILE,
    interval: float = DEFAULT_INTERVAL,
    cooldown: float = DEFAULT_COOLDOWN,
    dry_run: bool = False,
    region: tuple | None = None,
    q_threshold: float = DEFAULT_Q_THRESHOLD,
    a_threshold: float = DEFAULT_A_THRESHOLD,
    verbose: bool = False,
) -> None:
    answers = load_answers(answers_file)
    if not answers:
        print("[!] No answers loaded. Edit answers.txt and restart.")
        return

    print("\n" + "=" * 60)
    print("  GimBot is running!")
    if dry_run:
        print("  MODE: DRY RUN — will NOT actually click")
    print(f"  Checking screen every {interval}s  |  Cooldown: {cooldown}s")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # macOS reports coordinates at logical resolution; OCR image is 2x scaled.
    # We capture at logical res so scale factor for click correction = 2.0
    ocr_scale = 2.0

    last_answered_question = None
    last_click_time = 0.0

    while True:
        try:
            screenshot = capture_screen(region=region)
            screen_w, screen_h = screenshot.size

            enhanced = preprocess_for_ocr(screenshot)
            blocks = ocr_full(enhanced)

            if not blocks:
                if verbose:
                    print("[~] No text detected on screen.")
                time.sleep(interval)
                continue

            # Build full-screen text for question matching
            full_text = " ".join(b["text"] for b in blocks)

            # Try to isolate the question from the upper part of the screen
            q_block = find_question_block(blocks, screen_h * ocr_scale)
            question_text = q_block["text"] if q_block else full_text

            if verbose:
                print(f"[~] Detected text: {question_text!r}")

            # Skip if we already answered this question
            if normalize(question_text) == normalize(last_answered_question or ""):
                time.sleep(interval)
                continue

            # Match to our answer database
            answer, q_score = find_best_question_match(
                full_text, answers, threshold=q_threshold
            )

            if not answer:
                if verbose:
                    print(
                        f"[~] No question match (best score {q_score:.2f} < {q_threshold})"
                    )
                time.sleep(interval)
                continue

            print(f"[?] Question: {question_text!r}")
            print(f"[✓] Answer:   {answer!r}  (match score: {q_score:.2f})")

            # Find where the answer text appears on screen
            a_block, a_score = find_answer_block(
                answer, blocks, screen_h * ocr_scale, threshold=a_threshold
            )

            if not a_block:
                print(f"[!] Could not locate answer on screen (best score: {a_score:.2f})")
                time.sleep(interval)
                continue

            print(
                f"[+] Found at pixel ({a_block['cx']}, {a_block['cy']}) "
                f" (match score: {a_score:.2f})"
            )

            # Respect cooldown
            now = time.time()
            if now - last_click_time < cooldown:
                remaining = cooldown - (now - last_click_time)
                print(f"[~] Cooldown — waiting {remaining:.1f}s before clicking…")
                time.sleep(remaining)

            if dry_run:
                print(
                    f"[DRY RUN] Would click at "
                    f"({int(a_block['cx'] / ocr_scale)}, "
                    f"{int(a_block['cy'] / ocr_scale)})\n"
                )
            else:
                click_at(a_block["cx"], a_block["cy"], scale=ocr_scale)
                print("[+] Clicked!\n")

            last_answered_question = question_text
            last_click_time = time.time()

        except KeyboardInterrupt:
            print("\n\nGimBot stopped. Goodbye!")
            break
        except Exception as exc:
            print(f"[!] Error: {exc}")
            if verbose:
                import traceback
                traceback.print_exc()

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GimBot — Automatically select the correct Gimkit answer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gimbot.py
  python gimbot.py --answers my_class.txt --interval 2
  python gimbot.py --dry-run --verbose
  python gimbot.py --region
""",
    )
    parser.add_argument(
        "--answers",
        default=DEFAULT_ANSWERS_FILE,
        help=f"Path to Q&A text file (default: {DEFAULT_ANSWERS_FILE})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between screen checks (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN,
        help=f"Seconds to wait after clicking before checking again (default: {DEFAULT_COOLDOWN})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be clicked without actually clicking",
    )
    parser.add_argument(
        "--region",
        action="store_true",
        help="Interactively select which screen region to watch",
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=DEFAULT_Q_THRESHOLD,
        help=f"Min similarity to match a question (0–1, default: {DEFAULT_Q_THRESHOLD})",
    )
    parser.add_argument(
        "--a-threshold",
        type=float,
        default=DEFAULT_A_THRESHOLD,
        help=f"Min similarity to find an answer on screen (0–1, default: {DEFAULT_A_THRESHOLD})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show extra debug output",
    )

    args = parser.parse_args()

    region = None
    if args.region:
        region = select_region()

    run(
        answers_file=args.answers,
        interval=args.interval,
        cooldown=args.cooldown,
        dry_run=args.dry_run,
        region=region,
        q_threshold=args.q_threshold,
        a_threshold=args.a_threshold,
        verbose=args.verbose,
    )
