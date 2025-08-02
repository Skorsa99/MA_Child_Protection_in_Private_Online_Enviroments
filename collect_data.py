import os
import cv2
import time
import dotenv
import random
import requests
from html import unescape
from urllib.parse import urlparse


# Variables
dotenv.load_dotenv("src/private/private.env")
reddit_username = os.getenv("reddit_username")
reddit_password = os.getenv("reddit_password")
reddit_client_id = os.getenv("reddit_client_id")
reddit_api_key = os.getenv("reddit_api_key")
reddit_user_agent = 'script:online_child_protection:v1.0 (by u/Practical_Year_7524)'


# === Ziel-Subreddit und Speicherort ===
CATEGORY = 'explicit' # explicit | safe | empty
SUBREDDIT = 'Nudes'
MAX_ITEMS = 1000  # Maximal zu ladende Beiträge (paginierend)
SAVE_DIR = f'data/reddit_pics/{CATEGORY}/{SUBREDDIT}'

def collect_data():
    # Validate required credentials early to fail fast with a clear message
    if not all([reddit_username, reddit_password, reddit_client_id, reddit_api_key]):
        raise RuntimeError("Missing Reddit credentials in src/private/private.env")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # === Schritt 1: Authentifiziere dich via OAuth2 ===
    auth = requests.auth.HTTPBasicAuth(reddit_client_id, reddit_api_key)
    token_data = {
        'grant_type': 'password',
        'username': reddit_username,
        'password': reddit_password
    }
    headers = {'User-Agent': reddit_user_agent}


    res = requests.post(
        'https://www.reddit.com/api/v1/access_token',
        auth=auth,
        data=token_data,
        headers=headers,
        timeout=30
    )
    res.raise_for_status()
    TOKEN = res.json()['access_token']
    headers['Authorization'] = f'bearer {TOKEN}'

    # === Schritt 2: Lade die neuesten Beiträge aus dem Subreddit (paginierend bis ~1000) ===
    posts = []
    after = None

    while len(posts) < MAX_ITEMS:
        size = min(100, MAX_ITEMS - len(posts))
        url = f'https://oauth.reddit.com/r/{SUBREDDIT}/new?limit={size}&raw_json=1' + (f'&after={after}' if after else '')
        res = requests.get(url, headers=headers, timeout=30)

        if res.status_code in (429, 500, 502, 503, 504):
            _, _, reset = _parse_rate_headers(res.headers)
            backoff = (reset if res.status_code == 429 and reset > 0 else 2.0)
            time.sleep(backoff + random.uniform(0.25, 0.75))
            continue

        res.raise_for_status()
        payload = res.json()
        children = payload.get('data', {}).get('children', [])
        posts.extend(children)
        after = payload.get('data', {}).get('after')

        _adaptive_sleep(res.headers)

        if not after:
            break

    print(f"Gesamt geladene Posts: {len(posts)} (max {MAX_ITEMS})")

    # --- Helpers to extract media from different post types (single image, gallery, video, preview) ---


    # === Schritt 3: Filtere Medien (Einzelbild, Galerie, Video) und lade sie herunter ===
    allowed_ext = {'.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov'}
    downloaded_urls = set()

    post_counter = 0
    image_counter = 0

    for post in posts:
        data = post.get('data', {})
        post_counter += 1
        for media_url in extract_media_urls(data):
            if not media_url or media_url in downloaded_urls:
                continue
            downloaded_urls.add(media_url)

            path = urlparse(media_url).path
            ext = os.path.splitext(path)[-1].lower()
            if ext not in allowed_ext:
                # viele Preview-Links haben Query-Strings; wenn kein valider Suffix, versuche heuristisch zuzulassen
                # Skip falls unbekanntes Format
                continue

            try:
                video_exts = {'.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v', '.3gp', '.mpeg', '.mpg', '.ogv'}
                media = requests.get(media_url, stream=True, timeout=30)
                media.raise_for_status()

                if ext in video_exts:
                    basename = filename.split('.')[0]
                    print(basename)
                    random_suffix = random.randint(000000, 999999)
                    filename = f"{basename}_{random_suffix}.{ext.lstrip('.')}"
                    print(filename)
                else:
                    filename = os.path.join(SAVE_DIR, os.path.basename(path))

                with open(filename, 'wb') as f:
                    for chunk in media.iter_content(chunk_size=8192):
                        f.write(chunk)

                image_counter += 1
                print(f"Post: {str(post_counter)} | Media: {str(image_counter)}")
            except Exception as e:
                print(f"Failed: {media_url} ({e})")

    print(f"Completed download of {post_counter} posts with {image_counter} media files.")

def _parse_rate_headers(hdrs):
    def _to_float(val, default):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    used = _to_float(hdrs.get('x-ratelimit-used'), 0.0)
    remaining = _to_float(hdrs.get('x-ratelimit-remaining'), 60.0)
    reset = _to_float(hdrs.get('x-ratelimit-reset'), 60.0)
    return used, remaining, reset


def _adaptive_sleep(hdrs):
    """Sleep based on Reddit rate-limit headers to avoid 429 and maximize throughput.
    Rule of thumb: wait at least reset/remaining seconds between requests.
    If headers are missing, fall back to a conservative small delay with jitter.
    """
    used, remaining, reset = _parse_rate_headers(hdrs)
    if remaining <= 1 and reset > 0:
        time.sleep(reset + random.uniform(0.25, 0.75))
        return
    if remaining > 0 and reset > 0:
        base_delay = max(reset / remaining, 0.15)
    else:
        base_delay = 0.4
    time.sleep(base_delay + random.uniform(0.05, 0.15))

def extract_media_urls(data):
    urls = set()

    # 1) Direct URL overrides (typischer Bild-/Video-Post)
    direct = data.get('url_overridden_by_dest') or data.get('url')
    if direct:
        urls.add(direct)

    # 2) Reddit Gallery (mehrere Bilder in einem Post)
    if data.get('is_gallery') and data.get('media_metadata'):
        gallery_items = (data.get('gallery_data') or {}).get('items', [])
        for it in gallery_items:
            media_id = it.get('media_id') or it.get('id')
            meta = data['media_metadata'].get(media_id, {}) if media_id else {}
            if not meta:
                continue
            # Bevorzugt die volle Auflösung unter 's'
            s = meta.get('s') if isinstance(meta.get('s'), dict) else None
            if s:
                u = s.get('u') or s.get('gif') or s.get('mp4')
                if u:
                    urls.add(unescape(u).replace('&amp;', '&'))

    # 3) Reddit-Video (fallback MP4)
    rv = (data.get('secure_media') or {}).get('reddit_video')
    if not rv:
        rv = (data.get('preview') or {}).get('reddit_video_preview')
    if rv and isinstance(rv, dict):
        fbu = rv.get('fallback_url')
        if fbu:
            urls.add(fbu)

    # 4) Preview-Bilder (auch wenn kein sauberer Dateiname in der URL steht)
    preview = data.get('preview') or {}
    images = preview.get('images') or []
    for img in images:
        src = (img.get('source') or {}).get('url')
        if src:
            urls.add(unescape(src).replace('&amp;', '&'))

    return list(urls)

def convert_videos(folder_path):
    """
    Convert ONLY video-like files (videos + GIFs) in a folder to JPGs.
    - For videos and animated GIFs: extract the middle frame when possible (fallback to first frame).
    - Static images (jpg/png/webp/...) are **ignored**.
    Output JPGs are written to the same folder; existing names won't be overwritten.
    """
    if not os.path.isdir(folder_path):
        print(f"Ordner nicht gefunden: {folder_path}")
        return

    video_like_exts = {'.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v', '.3gp', '.mpeg', '.mpg', '.ogv', '.gif'}

    def unique_jpg_path(directory: str, stem: str) -> str:
        candidate = os.path.join(directory, f"{stem}.jpg")
        i = 1
        while os.path.exists(candidate):
            candidate = os.path.join(directory, f"{stem}_{i}.jpg")
            i += 1
        return candidate

    saved = 0
    for entry in sorted(os.listdir(folder_path)):
        src_path = os.path.join(folder_path, entry)
        if not os.path.isfile(src_path):
            continue

        ext = os.path.splitext(entry)[1].lower()
        stem = os.path.splitext(entry)[0]

        # Skip anything that's not video-like (including normal images)
        if ext not in video_like_exts:
            continue

        try:
            cap = cv2.VideoCapture(src_path)
            if not cap.isOpened():
                # Fallback für GIFs, die OpenCV nicht als Video öffnen kann
                if ext == '.gif':
                    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                    out_path = os.path.join(folder_path, f"{stem}.jpg")
                    if img is not None:
                        if os.path.exists(out_path):
                            print(f"Bild existiert bereits: {out_path}")
                            # GIF löschen und überspringen
                            continue
                        cv2.imwrite(out_path, img)
                        print(f"Gespeichert (GIF-Frame): {out_path}")
                        saved += 1
                    else:
                        print(f"Konnte GIF nicht öffnen: {src_path}")
                else:
                    print(f"Konnte Video nicht öffnen: {src_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_idx = frame_count // 2 if frame_count > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            cap.release()

            out_path = os.path.join(folder_path, f"{stem}.jpg")
            if ret and frame is not None:
                if os.path.exists(out_path):
                    print(f"Bild existiert bereits: {out_path}")
                    # Video löschen und überspringen
                    continue
                cv2.imwrite(out_path, frame)
                print(f"Gespeichert (Video/GIF-Frame): {out_path}")
                saved += 1
            else:
                print(f"Fehler beim Auslesen: {src_path}")
        finally:
            # Datei nach Verarbeitung oder Fehler löschen
            os.remove(src_path)

    print(f"Fertig. {saved} JPG-Dateien in '{folder_path}' erzeugt.")

if __name__ == "__main__":
    collect_data()
    convert_videos(SAVE_DIR)