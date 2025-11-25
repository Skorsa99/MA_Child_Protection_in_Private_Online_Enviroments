import os
import cv2
import time
import dotenv
import random
import requests
from PIL import Image
from pathlib import Path
from html import unescape
from urllib.parse import urlparse

from custom_logging import log_data_collection, image_tally_v2

# --- Allowed media types (centralized) ---
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v', '.3gp', '.mpeg', '.mpg', '.ogv'}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS

# Variables
dotenv.load_dotenv("src/private/private.env")
reddit_username = os.getenv("reddit_username")
reddit_password = os.getenv("reddit_password")
reddit_client_id = os.getenv("reddit_client_id")
reddit_api_key = os.getenv("reddit_api_key")
reddit_user_agent = 'script:online_child_protection:v1.0 (by u/Practical_Year_7524)'

def collect_data(SUBREDDIT, SAVE_DIR):
    MAX_ITEMS = 10000  # Maximal zu ladende Beiträge (paginierend)

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
    allowed_ext = ALLOWED_EXTS
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
                video_exts = ALLOWED_VIDEO_EXTS
                media = requests.get(
                    media_url,
                    stream=True,
                    timeout=30,
                    headers={"User-Agent": reddit_user_agent, "Referer": "https://www.reddit.com/"},
                )
                media.raise_for_status()

                filename = build_filename(media_url, SAVE_DIR)
                stem, _ = os.path.splitext(os.path.basename(filename))
                jpg_target = os.path.join(SAVE_DIR, f"{stem}.jpg")

                # Skip download if this media (original or its converted JPG) is already on disk
                if os.path.exists(filename) or os.path.exists(jpg_target):
                    continue

                with open(filename, 'wb') as f:
                    for chunk in media.iter_content(chunk_size=8192):
                        f.write(chunk)

                    image_counter += 1
                    
                print(f"Subreddit: {SUBREDDIT} | Post: {str(post_counter)} | Media: {str(image_counter)}")
            except Exception as e:
                print(f"Failed: {media_url} ({e})")

    return image_counter, post_counter

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
    from html import unescape
    from urllib.parse import urlparse
    import os

    def _dedupe_key(u: str):
        p = urlparse(u)
        # de-dupe by host+path only (ignore query), but KEEP query for the actual download
        return (p.netloc.lower(), p.path)

    urls, seen = [], set()

    def _add(u):
        if not u:
            return
        raw = unescape(u).replace('&amp;', '&')  # keep full URL incl. query
        key = _dedupe_key(raw)
        if key not in seen:
            seen.add(key)
            urls.append(raw)

    # 1) Reddit video → return only the MP4 (no preview)
    rv = (data.get('secure_media') or {}).get('reddit_video')
    if not rv:
        rv = (data.get('preview') or {}).get('reddit_video_preview')
    if isinstance(rv, dict) and rv.get('fallback_url'):
        _add(rv['fallback_url'])
        return urls

    # 2) Gallery → return gallery items (full-res)
    if data.get('is_gallery') and data.get('media_metadata'):
        items = (data.get('gallery_data') or {}).get('items', [])
        for it in items:
            mid = it.get('media_id') or it.get('id')
            meta = data['media_metadata'].get(mid, {}) if mid else {}
            s = meta.get('s') if isinstance(meta.get('s'), dict) else None
            _add(s.get('u') or s.get('gif') or s.get('mp4') if s else None)
        return urls

    # 3) Single direct media (prefer actual media extensions)
    direct = data.get('url_overridden_by_dest') or data.get('url')
    if direct:
        path_ext = os.path.splitext(urlparse(direct).path)[1].lower()
        allowed_exts = {'.jpg', '.jpeg', '.png', '.gif',
                        '.mp4', '.mov', '.webm', '.mkv', '.avi',
                        '.m4v', '.3gp', '.mpeg', '.mpg', '.ogv'}
        if path_ext in allowed_exts:
            _add(direct)
            return urls

    # 4) Fallback: use preview only if nothing better was found
    preview = data.get('preview') or {}
    for img in (preview.get('images') or []):
        _add((img.get('source') or {}).get('url'))

    return urls

def build_filename(media_url: str, save_dir: str) -> str:
    p = urlparse(media_url)
    host = p.netloc.lower()
    base = os.path.basename(p.path)        # e.g. "DASH_720.mp4" oder "abc123.jpg"
    stem, ext = os.path.splitext(base)

    if 'v.redd.it' in host:
        # /<asset_id>/DASH_720.mp4  -> asset_id = erster Pfadteil
        parts = [seg for seg in p.path.split('/') if seg]
        asset_id = parts[0] if parts else 'vid'
        # name = f"{asset_id}_{stem}{ext}"
        name = f"{asset_id}{ext}" # keine Duplikate mit verscheidenen video-Qualitäten
    elif 'i.redd.it' in host:
        # i.redd.it ist bereits eindeutig
        name = f"{stem}{ext}"
    else:
        # Fallback für externe Hosts
        host_key = host.replace('.', '-')
        name = f"{host_key}_{stem}{ext}"

    return os.path.join(save_dir, name)

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
    error_rate = 0

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
                    error_rate += 1
                    # Video löschen und überspringen
                    continue
                cv2.imwrite(out_path, frame)
                print(f"Gespeichert (Video/GIF-Frame): {out_path}")
                saved += 1
            else:
                print(f"Fehler beim Auslesen: {src_path}")
                error_rate += 1
        finally:
            # Datei nach Verarbeitung oder Fehler löschen
            os.remove(src_path)

    print(f"Fertig. {saved} JPG-Dateien in '{folder_path}' erzeugt.")

    return saved, error_rate

def convert_jpg_to_jpeg(folder_path, delete_original=False):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a valid directory")
    
    counter_c = 0
    counter_d = 0
    counter_ed = 0
    counter_e = 0

    for jpg_file in folder.glob("*.jpg"):
        jpeg_path = jpg_file.with_suffix(".jpeg")

        try:
            with Image.open(jpg_file) as img:
                rgb_img = img.convert("RGB")  # Ensure standard RGB format
                rgb_img.save(jpeg_path, "JPEG")
            # print(f"Converted: {jpg_file} -> {jpeg_path}")
            counter_c += 1

            if delete_original:
                jpg_file.unlink()
                # print(f"Deleted original: {jpg_file}")
                counter_d += 1

        except Exception as e:
            print(f"Error converting {jpg_file}: {e}")
            try:
                jpg_file.unlink()
                counter_ed += 1
            except Exception as ex:
                counter_e += 1

    return counter_c, counter_d, counter_ed, counter_e

if __name__ == "__main__":
    # CATEGORY = unsafe | safe | empty
    subreddits = [
        # --- unsafe ---
        ["Nudes", "unsafe"], # Cleaned (1,01% error rate)
        ["Nudes_Heaven", "unsafe"], # Cleaned (19,77% error rate) # (maybe not that great, has a lot of text in the images)
        ["NudeSports", "unsafe"], # Cleaned (4,41% error rate [only 227 images])
        ["Nudeshoots", "unsafe"], # Cleaned (9,59% error rate [only 146 images])
        ["ChangingRooms", "unsafe"], # Cleaned (10,05% error rate)
        ["Flashing", "unsafe"], # Cleaned (8,97% error rate)
        ["Flashingmilfs", "unsafe"], # Cleaned (4,42% error rate)
        ["FlashingAndFlaunting", "unsafe"], # Cleaned (6,14% error rate)
        ["FlashingGirls", "unsafe"], # Cleaned (7,06% error rate)
        ["malesnude", "unsafe"], # Cleaned (0,38% error rate)
        ["MalenudesEU", "unsafe"], # Cleaned (0,11% error rate)
        ["MaleNudesInspiration", "unsafe"], # Cleaned (0,00% error rate [only 20 images])
        ["Bulges", "unsafe"], # Cleaned (14,83% error rate) # this was awful, so many issues, dont run again
        ["Musk4Musk", "unsafe"], # Cleaned (7,18% error rate)
        ["GaybrosGoneWild", "unsafe"], # Cleaned (4,72% error rate)
        ["DarkAngels", "unsafe"], # Cleaned (2,34% error rate)
        ["blackmale", "unsafe"], # Cleaned (2,61% error rate)
        ["blackcock", "unsafe"], # Cleaned (2,69% error rate)
        ["sexyselfie", "unsafe"], # Cleaned (a lot of errors, since this kinda defines the edge of what is acceptable [14,56%])
        ["AsianNSFW", "unsafe"], # Cleaned (2,48% error rate)
        ["BigAsianCock", "unsafe"], # Cleaned (1,16% error rate)
        ["topless", "unsafe"], # Partly Checked (first 1000 images with 0.4% failure rate)
        ["ToplessInPublic", "unsafe"], # Cleaned (1,72% error rate)
        ["MaleUnderwear", "unsafe"], # Cleaned (2,39% error rate)
        ["OldGuysRule", "unsafe"], # Cleaned (1,04% error rate)
        ["GrannyOldWomen", "unsafe"], # Cleaned (13,34% error rate)
        ["pornstarsinswimwear", "unsafe"], # Partly Checked (first 1000 images with 0.0% failure rate)
        ["swimbrief", "unsafe"], # Cleaned (0,88% error rate)
        ["Femalenudesonly", "unsafe"], # Cleaned (9,60% error rate)

        ["beach_bodies", "unsafe"], # Cleaned (0,61% error rate)
        ["NudeMaleArt", "unsafe"], # Cleaned (6,22% error rate)
        ["FreedomToFlaunt", "unsafe"], # Cleaned (0,00% error rate)
        ["Shirtless_Men", "unsafe"], # Cleaned (0,00% error rate)
        ["shirtlessguys", "unsafe"], # Cleaned (11,76% error rate [only 34 images])
        ["ShirtlessCelebMen", "unsafe"], # Cleaned (4,51% error rate)
        ["shirtlessinpublic", "unsafe"], # Cleaned (7,21% error rate)
        ["Brogress", "unsafe"], # Cleaned (13,02% error rate)
        ["Beardsandboners", "unsafe"], # Cleaned (4,85% error rate)
        ["gonewild", "unsafe"], # Cleaned (2,85% error rate)
        ["ladybonersgonemild", "unsafe"], # Cleaned (37,29% error rate)

        ["SwimmingCostume", "unsafe"], # Cleaned (0,75% error rate)
        ["BodybuildingCycle", "unsafe"], # Cleaned (45,59% error rate [only 68 images])
        ["bodybuildingpics", "unsafe"], # Cleaned (20,17% error rate)
        ["bodybuilding", "unsafe"], # Cleaned (1,99% error rate)
        ["Dick", "unsafe"], # Cleaned (1,03% error rate)
        ["dickgirls", "unsafe"], # Cleaned (4,54% error rate)                                                                            # Idk just found it and thought why not, maybe coupeling dicks and breasts is not a bad idea, but not sure
        ["DicksOnBabes", "unsafe"], # Cleaned (4,46% error rate)                                                                         # Idk just found it and thought why not, maybe coupeling dicks and breasts is not a bad idea, but not sure
        ["DickSlips", "unsafe"], # Cleaned (4,64% error rate)
        ["cock", "unsafe"], # Partly Checked (first 1000 images with 0.7% failure rate)
        ["cockheadlovers", "unsafe"], # Cleaned (1,16% error rate)
        ["cockcompareing", "unsafe"], # Cleaned (7,24% error rate)
        ["CockOutline", "unsafe"], # Cleaned (47,97% error rate)
        ["schwanzshow", "unsafe"], # Cleaned (2,75% error rate)
        ["SchwanzgeileTiroler", "unsafe"], # Cleaned (8,08% error rate)
        ["bigblackcocks", "unsafe"], # Cleaned (3,32% error rate)
        ["BlackGirlsCentral", "unsafe"], # Cleaned (4,65% error rate)

        ["theratio", "unsafe"], # Cleaned (7,29% error rate)
        ["CuteLittleButts", "unsafe"], # Cleaned (5,67% error rate)
        ["CuteGirlsinPanties", "unsafe"], # Partly Checked (first 1000 images with 0.5% failure rate)

        # --- safe ---
        ["selfies", "safe"],
        ["outfitoftheday", "safe"],
        ["malehairadvice", "safe"],
        ["selfie", "safe"],
        ["Outfits", "safe"],
        ["TodayIWore", "safe"],
        ["MakeupAddiction", "safe"],
        ["malefashion", "safe"],
        ["malegrooming", "safe"],
        ["beards", "safe"],
        ["FreeCompliments", "safe"],
        ["over60selfies", "safe"],
        ["50something", "safe"],
        ["40something", "safe"],
        ["blackladies", "safe"],
        ["FreeAIHeadshots", "safe"],
        ["headshots", "safe"],
        ["medicalillustration", "safe"],
        ["anatomicalart", "safe"],
        ["Homescreens", "safe"],
        
        ["UI_Design", "safe"],
        ["portraits", "safe"],
        ["whitetanktops", "safe"],

        ["SelfieOver25", "safe"],

        # ["cute_face", "safe"], # Likely banned

        # --- empty ---
        ["InteriorDesign", "empty"],
        ["AmateurRoomPorn", "empty"],
        ["Workspaces", "empty"],
        ["EarthPorn", "empty"],
        ["SkyPorn", "empty"],
        ["MyRoom", "empty"],
    ]

    safe_only_subreddits = [
        # --- safe ---
        ["selfies", "safe"], # Good
        ["outfitoftheday", "safe"], # Good
        ["malehairadvice", "safe"], # Bad-ish
        ["selfie", "safe"], # Good
        ["Outfits", "safe"], # Good
        ["TodayIWore", "safe"], # Good
        ["MakeupAddiction", "safe"], # Bad
        ["malefashion", "safe"], # Good
        ["malegrooming", "safe"], # Good-ish
        ["beards", "safe"], # Good
        ["FreeCompliments", "safe"], # Good
        ["over60selfies", "safe"], # Good
        ["50something", "safe"], # Good-ish
        ["40something", "safe"], # Good
        ["blackladies", "safe"], # Bad-ish
        ["FreeAIHeadshots", "safe"], # Good
        ["headshots", "safe"], # Good
        ["medicalillustration", "safe"], # Different
        ["anatomicalart", "safe"], # Different
        ["Homescreens", "safe"], # Different
        
        ["UI_Design", "safe"], # Different
        ["portraits", "safe"], # Good-ish 
        ["whitetanktops", "safe"], # Mid

        ["SelfieOver25", "safe"], # Good

        # ["cute_face", "safe"], # Likely banned

        # --- empty ---
        ["InteriorDesign", "empty"],
        ["AmateurRoomPorn", "empty"],
        ["Workspaces", "empty"],
        ["EarthPorn", "empty"],
        ["SkyPorn", "empty"],
        ["MyRoom", "empty"],
    ]

    run_counter = 0 # Counts how many images where stored in this run

    for SUBREDDIT, CATEGORY in safe_only_subreddits:
        SAVE_DIR = f'data/reddit_pics/{CATEGORY}/{SUBREDDIT}'

        image_counter, post_counter = collect_data(SUBREDDIT, SAVE_DIR)
        saved, error_rate = convert_videos(SAVE_DIR)
        counter_c, counter_d, counter_ed, counter_e = convert_jpg_to_jpeg(SAVE_DIR, delete_original=True)

        image_counter = image_counter - (error_rate + counter_ed + counter_e)

        image_tally_variable = image_tally_v2(f'data/reddit_pics/{CATEGORY}', CATEGORY)
        if counter_e > 0:
            end_message = f"Completed download of {post_counter} posts for '{SUBREDDIT}', and stored {image_counter} media files [with {counter_e} unhandled errors]. Updated tally: {image_tally_variable}"
        else:
            end_message = f"Completed download of {post_counter} posts for '{SUBREDDIT}', and stored {image_counter} media files. Updated tally: {image_tally_variable}"
        print(end_message)

        run_counter += image_counter

        if CATEGORY != "TESTS":
            log_data_collection(end_message)

    print(f"------------------------------------------------------------------------")
    print(f"Total images collected in this run: {run_counter}")
    print(f"------------------------------------------------------------------------")
