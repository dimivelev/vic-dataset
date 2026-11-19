#!/usr/bin/env python3
"""
Run the whole pipeline on a FREE Render web service.

  * A small web server answers on $PORT:  /  status page,  /healthz,
    /api/status (JSON),  /download/<file>?key=DOWNLOAD_KEY
  * A background thread scrapes the sites one by one with scrape.py, mirrors every
    joke and the scraper's resume state into Postgres every 20 s, so a restart or
    spin-down loses at most a few pages.
  * While scraping, the app requests its own public URL every 10 minutes so the
    free instance is not spun down for lack of inbound traffic. When everything is
    finished it stops doing that and Render lets it sleep; a download wakes it.
  * After the last site: clean.py + build_dataset.py run and the outputs are stored
    gzipped in Postgres for download.

Locally (no DATABASE_URL) it uses a SQLite file, which is how it is tested.
"""
from __future__ import annotations

import gzip
import hmac
import html
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scrape  # noqa: E402

DATABASE_URL = os.environ.get("DATABASE_URL", "")
DOWNLOAD_KEY = os.environ.get("DOWNLOAD_KEY", "")
SITES = [s.strip() for s in os.environ.get(
    "SITES", "fun_dir_bg,vicove_com,zaigravka_bg,vicove_biz").split(",") if s.strip()]
DELAY = float(os.environ.get("SCRAPE_DELAY", "1.5"))
PUBLIC_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
WORK = os.environ.get("WORK_DIR", os.path.join(HERE, "work"))
PORT = int(os.environ.get("PORT", "10000"))

FILES = {  # download name -> (local path relative to WORK, description)
    "recipe_seed.jsonl": ("dataset_all/recipe_seed.jsonl", "Unsloth Studio recipe seed (all jokes)"),
    "recipe_seed_sfw.jsonl": ("dataset_sfw/recipe_seed.jsonl", "Recipe seed without NSFW/sensitive jokes"),
    "train.jsonl": ("dataset_chat/train.jsonl", "Chat-format training set"),
    "val.jsonl": ("dataset_chat/val.jsonl", "Chat-format validation set"),
    "jokes.jsonl": ("clean/jokes.jsonl", "Cleaned, deduplicated jokes with categories and flags"),
    "stats.json": ("clean/stats.json", "Cleaning statistics"),
}


def log(*a):
    print(datetime.now(timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- #
# Database (Postgres on Render, SQLite locally)
# --------------------------------------------------------------------------- #
class DB:
    def __init__(self, url: str):
        self.pg = url.startswith(("postgres://", "postgresql://"))
        self.url = url
        self.lock = threading.Lock()
        self.conn = None

    def _connect(self):
        if self.pg:
            import psycopg
            self.conn = psycopg.connect(self.url, autocommit=False, connect_timeout=20)
        else:
            import sqlite3
            self.conn = sqlite3.connect(os.path.join(WORK, "local.db"), check_same_thread=False)

    def _sql(self, q: str) -> str:
        if self.pg:
            q = q.replace("?", "%s").replace("BLOB", "BYTEA")
        return q

    def run(self, q: str, params=(), many=False, fetch=False):
        """Execute with one reconnect+retry (free Postgres may drop idle connections)."""
        with self.lock:
            for attempt in (1, 2):
                try:
                    if self.conn is None:
                        self._connect()
                    cur = self.conn.cursor()
                    if many:
                        cur.executemany(self._sql(q), params)
                    else:
                        cur.execute(self._sql(q), params)
                    rows = cur.fetchall() if fetch else None
                    self.conn.commit()
                    return rows
                except Exception as e:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                    self.conn = None
                    if attempt == 2:
                        raise
                    log(f"db error ({e.__class__.__name__}: {e}); reconnecting")
                    time.sleep(3)

    def init(self):
        self.run("CREATE TABLE IF NOT EXISTS jokes (id TEXT PRIMARY KEY, site TEXT, url TEXT, text TEXT)")
        self.run("CREATE TABLE IF NOT EXISTS joke_cats (id TEXT, category TEXT, PRIMARY KEY (id, category))")
        self.run("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.run("CREATE TABLE IF NOT EXISTS files (name TEXT PRIMARY KEY, data BLOB, size INTEGER, created TEXT)")

    def get(self, k, default=None):
        rows = self.run("SELECT v FROM kv WHERE k = ?", (k,), fetch=True)
        return rows[0][0] if rows else default

    def set(self, k, v):
        self.run("INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT (k) DO UPDATE SET v = excluded.v", (k, str(v)))

    def add_jokes(self, recs: list[dict]):
        if not recs:
            return
        self.run("INSERT INTO jokes (id, site, url, text) VALUES (?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                 [(r["id"], r["site"], r.get("url", ""), r["text"]) for r in recs], many=True)
        cats = [(r["id"], r["category"]) for r in recs if r.get("category")]
        if cats:
            self.run("INSERT INTO joke_cats (id, category) VALUES (?, ?) ON CONFLICT DO NOTHING", cats, many=True)

    def count(self, site=None) -> int:
        if site:
            return self.run("SELECT COUNT(*) FROM jokes WHERE site = ?", (site,), fetch=True)[0][0]
        return self.run("SELECT COUNT(*) FROM jokes", fetch=True)[0][0]

    def put_file(self, name: str, path: str):
        buf = io.BytesIO()
        with open(path, "rb") as src, gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
            shutil.copyfileobj(src, gz)
        data = buf.getvalue()
        self.run("INSERT INTO files (name, data, size, created) VALUES (?, ?, ?, ?) "
                 "ON CONFLICT (name) DO UPDATE SET data = excluded.data, size = excluded.size, created = excluded.created",
                 (name, data, os.path.getsize(path), datetime.now(timezone.utc).isoformat(timespec="seconds")))

    def get_file(self, name: str):
        rows = self.run("SELECT data FROM files WHERE name = ?", (name,), fetch=True)
        return bytes(rows[0][0]) if rows else None

    def list_files(self):
        return self.run("SELECT name, size, created FROM files ORDER BY name", fetch=True)

    def export_jokes(self, path: str) -> int:
        """Write one JSON line per joke with all its categories (input for clean.py)."""
        n = 0
        last = ""
        with open(path, "w", encoding="utf-8") as fh:
            while True:  # keyset pagination keeps memory flat
                rows = self.run("SELECT id, site, url, text FROM jokes WHERE id > ? ORDER BY id LIMIT 5000",
                                (last,), fetch=True)
                if not rows:
                    break
                ids = [r[0] for r in rows]
                marks = ",".join("?" * len(ids))
                cats: dict[str, list] = {}
                for jid, cat in self.run(f"SELECT id, category FROM joke_cats WHERE id IN ({marks})",
                                         ids, fetch=True):
                    cats.setdefault(jid, []).append(cat)
                for jid, site, url, text in rows:
                    fh.write(json.dumps({"id": jid, "site": site, "url": url, "text": text,
                                         "categories": cats.get(jid, [])}, ensure_ascii=False) + "\n")
                    n += 1
                last = ids[-1]
        return n


db = DB(DATABASE_URL)
STATUS = {"phase": "starting", "site": None, "error": None, "probe": {}}


# --------------------------------------------------------------------------- #
# Syncing scraper output -> DB
# --------------------------------------------------------------------------- #
class Syncer(threading.Thread):
    """Mirror data/raw/<site>.jsonl (appended by scrape.py) and its state file into the DB.
    State is read BEFORE the jokes, so the saved state never claims pages whose jokes
    are not yet in the DB (scrape.py flushes jokes before it writes state)."""

    def __init__(self, site: str, raw_dir: str):
        super().__init__(daemon=True)
        self.site = site
        self.jsonl = os.path.join(raw_dir, f"{site}.jsonl")
        self.state = os.path.join(raw_dir, "_state", f"{site}.json")
        self.offset = 0
        self.stop_evt = threading.Event()

    def sync_once(self):
        state_txt = None
        if os.path.exists(self.state):
            try:
                with open(self.state, encoding="utf-8") as fh:
                    state_txt = fh.read()
                json.loads(state_txt)
            except (OSError, json.JSONDecodeError):
                state_txt = None  # being written; next round
        recs = []
        if os.path.exists(self.jsonl):
            with open(self.jsonl, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
            end = chunk.rfind(b"\n") + 1  # only complete lines
            for line in chunk[:end].splitlines():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            self.offset += end
        for i in range(0, len(recs), 1000):
            db.add_jokes(recs[i:i + 1000])
        if state_txt:
            db.set(f"state:{self.site}", state_txt)
        return len(recs)

    def run(self):
        while not self.stop_evt.wait(20):
            try:
                self.sync_once()
            except Exception as e:
                log(f"sync error: {e}")

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=5)
        self.sync_once()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def probe_sites():
    """Fetch one category page per site and log what the extractor finds (sanity check)."""
    f = scrape.Fetcher(delay=DELAY)
    for site in SITES:
        try:
            cfg = scrape.SITES[site]
            ex = scrape.Extractor(cfg)
            cats = scrape.discover(site, f)
            if not cats:
                STATUS["probe"][site] = "no categories found"
                log(f"PROBE {site}: no categories found")
                continue
            cat = sorted(cats.values(), key=lambda c: c["slug"])[min(1, len(cats) - 1)]
            r = f.get(cat["url"])
            jokes = ex.extract(r, cat["url"]) if r is not None else []
            sample = [j["text"].replace("\n", " / ")[:140] for j in jokes[:2]]
            STATUS["probe"][site] = f"{len(cats)} categories; {len(jokes)} jokes on {cat['url']}"
            log(f"PROBE {site}: {STATUS['probe'][site]}")
            for s in sample:
                log(f"PROBE {site}:   {s}")
        except Exception as e:
            STATUS["probe"][site] = f"error: {e}"
            log(f"PROBE {site} error: {e}")


def scrape_site(site: str):
    raw = os.path.join(WORK, "raw")
    os.makedirs(os.path.join(raw, "_state"), exist_ok=True)
    jsonl = os.path.join(raw, f"{site}.jsonl")
    if os.path.exists(jsonl):
        os.remove(jsonl)  # contents are already in the DB; start a fresh tail
    saved = db.get(f"state:{site}")
    if saved:
        with open(os.path.join(raw, "_state", f"{site}.json"), "w", encoding="utf-8") as fh:
            fh.write(saved)
        log(f"{site}: resuming from saved state")
    syncer = Syncer(site, raw)
    syncer.start()
    try:
        scrape.cmd_run(types.SimpleNamespace(site=site, out_dir=raw, delay=DELAY,
                                             max_pages=3000, only=None))
    finally:
        syncer.stop()
    state = json.loads(db.get(f"state:{site}") or "{}")
    return state


def build_outputs():
    attempts = int(db.get("build_attempts", "0")) + 1
    db.set("build_attempts", attempts)
    STATUS["phase"] = "building dataset"
    os.makedirs(os.path.join(WORK, "clean"), exist_ok=True)
    allraw = os.path.join(WORK, "all_raw.jsonl")
    n = db.export_jokes(allraw)
    log(f"exported {n} jokes for cleaning (build attempt {attempts})")
    py = sys.executable
    clean_cmd = [py, os.path.join(HERE, "clean.py"), "--raw", allraw,
                 "--out", os.path.join(WORK, "clean/jokes.jsonl")]
    if attempts > 1:  # a previous attempt died (likely out of memory on the free instance)
        clean_cmd.append("--no-near-dedup")
        log("previous build did not finish -> skipping near-duplicate pass to save memory")
    subprocess.run(clean_cmd, check=True)
    os.remove(allraw)
    inp = os.path.join(WORK, "clean/jokes.jsonl")
    b = os.path.join(HERE, "build_dataset.py")
    subprocess.run([py, b, "--inp", inp, "--format", "seed", "--out-dir", os.path.join(WORK, "dataset_all")], check=True)
    subprocess.run([py, b, "--inp", inp, "--format", "seed", "--exclude-nsfw", "--exclude-sensitive",
                    "--out-dir", os.path.join(WORK, "dataset_sfw")], check=True)
    subprocess.run([py, b, "--inp", inp, "--format", "chat", "--out-dir", os.path.join(WORK, "dataset_chat")], check=True)
    for name, (rel, _) in FILES.items():
        path = os.path.join(WORK, rel)
        if os.path.exists(path):
            db.put_file(name, path)
            log(f"stored {name} ({os.path.getsize(path) / 1e6:.1f} MB)")
    db.set("finished", datetime.now(timezone.utc).isoformat(timespec="seconds"))


def worker():
    try:
        db.init()
        if db.get("finished"):
            STATUS["phase"] = "finished"
            log("pipeline already finished; serving downloads only")
            return
        STATUS["phase"] = "probing sites"
        probe_sites()
        for site in SITES:
            while not db.get(f"site_done:{site}"):
                STATUS["phase"], STATUS["site"] = "scraping", site
                log(f"=== scraping {site} ===")
                try:
                    state = scrape_site(site)
                except Exception as e:
                    STATUS["error"] = f"{site}: {e}"
                    log(f"{site} failed: {e}\n{traceback.format_exc()}")
                    time.sleep(120)
                    continue
                cats = state.get("categories") or {}
                if not cats:
                    tries = int(db.get(f"discover_tries:{site}", "0")) + 1
                    db.set(f"discover_tries:{site}", tries)
                    if tries < 3:
                        log(f"{site}: no categories found (try {tries}/3); retrying in 5 min")
                        time.sleep(300)
                        continue
                    db.set(f"site_status:{site}", "no categories found (site down or blocked?)")
                    log(f"{site}: no categories found, skipping")
                elif len(state.get("done", [])) < len(cats):
                    log(f"{site}: run ended early; retrying in 60 s")
                    time.sleep(60)
                    continue
                db.set(f"site_done:{site}", "1")
                log(f"{site}: done, {db.count(site)} jokes in DB")
        build_outputs()
        STATUS["phase"] = "finished"
        log("ALL DONE - outputs ready for download")
    except Exception as e:
        STATUS["phase"], STATUS["error"] = "error", str(e)
        log(f"worker crashed: {e}\n{traceback.format_exc()}")


def keep_alive():
    """Ping our own public URL while work is in progress so the free instance stays up."""
    if not PUBLIC_URL:
        return
    while STATUS["phase"] not in ("finished", "error"):
        time.sleep(600)
        try:
            urllib.request.urlopen(PUBLIC_URL.rstrip("/") + "/healthz", timeout=30).read()
        except Exception as e:
            log(f"keep-alive ping failed: {e}")


# --------------------------------------------------------------------------- #
# Web
# --------------------------------------------------------------------------- #
def status_json():
    out = {"phase": STATUS["phase"], "current_site": STATUS["site"], "error": STATUS["error"],
           "probe": STATUS["probe"], "sites": {}}
    try:
        out["total_jokes"] = db.count()
        for site in SITES:
            st = json.loads(db.get(f"state:{site}") or "{}")
            out["sites"][site] = {
                "jokes": db.count(site),
                "categories_done": len(st.get("done", [])),
                "categories_total": len(st.get("categories") or {}),
                "done": bool(db.get(f"site_done:{site}")),
                "note": db.get(f"site_status:{site}"),
            }
        out["files"] = [{"name": n, "bytes": s, "created": c} for n, s, c in db.list_files()]
        out["finished"] = db.get("finished")
    except Exception as e:
        out["db_error"] = str(e)
    return out


def status_html():
    s = status_json()
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['jokes']:,}</td><td>{v['categories_done']}/{v['categories_total']}</td>"
        f"<td>{'✅' if v['done'] else ('⏳' if s['current_site'] == k else '')}</td>"
        f"<td>{html.escape(v['note'] or '')}</td></tr>"
        for k, v in s["sites"].items())
    files = "".join(
        f"<li><code>{html.escape(f['name'])}</code> – {f['bytes'] / 1e6:.1f} MB "
        f"(<code>/download/{html.escape(f['name'])}?key=…</code>)</li>" for f in s.get("files", []))
    probe = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in s["probe"].items())
    if files:
        downloads = (f"<ul>{files}</ul><p>Files are gzip-compressed. Add <code>?key=</code> with the "
                     "DOWNLOAD_KEY from the service's Environment tab in the Render dashboard.</p>")
    else:
        downloads = "<p>Available when scraping and cleaning are finished.</p>"
    refresh = '<meta http-equiv="refresh" content="300">' if s["phase"] != "finished" else ""
    return f"""<!doctype html><html lang="bg"><head><meta charset="utf-8">{refresh}
<title>vic-dataset</title><style>body{{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}</style></head>
<body><h1>Bulgarian jokes scraper</h1>
<p>Phase: <b>{html.escape(s['phase'])}</b>{' – ' + html.escape(str(s['current_site'])) if s['current_site'] else ''}
&nbsp;·&nbsp; jokes collected: <b>{s.get('total_jokes', 0):,}</b></p>
{f"<p style='color:#b00'>Last error: {html.escape(s['error'])}</p>" if s['error'] else ''}
<table><tr><th>Site</th><th>Jokes</th><th>Categories</th><th></th><th></th></tr>{rows}</table>
<h3>Downloads</h3>{downloads}
<h3>Startup check</h3><ul>{probe or '<li>running…</li>'}</ul></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep logs for the pipeline

    def _send(self, code, body: bytes, ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/healthz":
            return self._send(200, b"ok")
        if u.path == "/api/status":
            return self._send(200, json.dumps(status_json(), ensure_ascii=False, indent=1).encode(),
                              "application/json; charset=utf-8")
        if u.path == "/":
            return self._send(200, status_html().encode(), "text/html; charset=utf-8")
        if u.path.startswith("/download/"):
            name = u.path[len("/download/"):]
            key = urllib.parse.parse_qs(u.query).get("key", [""])[0]
            if not DOWNLOAD_KEY or not hmac.compare_digest(key.encode(), DOWNLOAD_KEY.encode()):
                return self._send(403, b"wrong or missing key")
            if name not in FILES:
                return self._send(404, b"unknown file")
            data = db.get_file(name)
            if data is None:
                return self._send(404, b"not ready yet")
            return self._send(200, data, "application/gzip",
                              {"Content-Disposition": f'attachment; filename="{name}.gz"'})
        return self._send(404, b"not found")


def main():
    os.makedirs(WORK, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    log(f"listening on :{PORT}; sites={SITES}; delay={DELAY}s; db={'postgres' if db.pg else 'sqlite'}")
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=keep_alive, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
