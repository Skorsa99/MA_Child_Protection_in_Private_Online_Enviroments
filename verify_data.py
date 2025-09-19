from flask import Flask, send_from_directory, jsonify, request
import os
from pathlib import Path

from custom_logging import log_data_filtering, image_tally

app = Flask(__name__)
MODE = "unsafe"
SUBREDDIT = "ToplessInPublic"
IMAGE_DIR = Path(f"data/reddit_pics/{MODE}/{SUBREDDIT}")  # oder dein Bildpfad
image_list = sorted([f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
TOTAL_AT_START = len(image_list)
current_index = 0
last_deleted = []

@app.route('/')
def frontend():
    return send_from_directory("src", "ui/verify_data.html") # FinderTag

@app.route('/next')
def get_next_image():
    global current_index
    if current_index >= len(image_list):
        return jsonify({"done": True})

    filename = image_list[current_index].name
    return jsonify({"filename": filename})

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/progress')
def get_progress():
    total_baseline = TOTAL_AT_START
    total_remaining = len(image_list)
    # images processed = passed (current_index) + deleted so far (baseline - remaining)
    checked = current_index + max(0, total_baseline - total_remaining)
    if total_baseline <= 0:
        percent = 100.0
    else:
        percent = (checked / float(total_baseline)) * 100.0
    return jsonify({
        "checked": int(checked),
        "total": int(total_baseline),
        "percent": float(percent)
    })

@app.route('/delete', methods=["POST"])
def delete_image():
    global current_index
    if current_index >= len(image_list):
        return jsonify({"error": "No image to delete"})

    path = image_list[current_index]
    try:
        # Move the image to deleted_pics directory instead of deleting
        orig_path = image_list[current_index]
        new_path = Path(str(orig_path).replace("reddit_pics", "deleted_pics/varification"))
        new_path.parent.mkdir(parents=True, exist_ok=True)
        orig_path.rename(new_path)
        # Store for undo
        last_deleted.append((orig_path, new_path, current_index))
        print(f"Moved: {orig_path} to {new_path}")
        image_list.pop(current_index)
        if MODE != "TESTS": # Only tally the images when we are in an actually importantly folder
            image_tally(-1, MODE)
            log_data_filtering(f"Deleted: {orig_path} to {new_path}")
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pass', methods=["POST"])
def pass_image():
    global current_index
    if current_index < len(image_list):
        current_index += 1
    return jsonify({"passed": True})

@app.route('/undo', methods=["POST"])
def undo_delete():
    if not last_deleted:
        return jsonify({"error": "Nothing to undo"}), 400
    orig_path, deleted_path, idx = last_deleted.pop()
    deleted_path.rename(orig_path)
    image_list.insert(idx, orig_path)
    global current_index
    current_index = idx
    if MODE != "TESTS": # Only tally the images when we are in an actually importantly folder
        image_tally(1, MODE)
        log_data_filtering(f"Restored: {deleted_path} to {orig_path}")
    return jsonify({"restored": True, "filename": orig_path.name})

@app.route('/previous', methods=["POST"])
def previous_image():
    global current_index
    if current_index <= 0:
        return jsonify({"error": "No previous image"}), 400
    current_index -= 1
    filename = image_list[current_index].name
    return jsonify({"filename": filename})

if __name__ == "__main__":
    # launch by: 'python verify_data.py'
    app.run(debug=True)
