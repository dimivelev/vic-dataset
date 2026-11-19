#!/usr/bin/env python3
"""
Polite, resumable scraper for Bulgarian joke (виц) websites.

How it works
------------
1. Discover category pages by collecting links that match a site's category pattern.
2. Walk each category page by page (?page=N or /page/N/) until a page brings
   (almost) no new jokes — robust to windowed pagers and "last page repeats".
3. On each listing page, find links to individual joke permalinks. For every
   joke link, climb the DOM to the largest element that contains *only that
   joke's* links: that element is the joke card. Metadata lines (views, likes,
   "Добавен преди...", "Преглед", ...) are stripped.
   If the site lays jokes out flat (no wrapper per joke), the text between
   consecutive permalink links is used instead.
   You can pin an exact CSS selector per site with `item_selector`.

Output: one JSON line per (joke, category) in data/raw/<site>.jsonl.
A joke listed in several categories appears several times; clean.py merges them.

Usage
-----
  python scrape.py list                         # show configured sites
  python scrape.py categories vicove_biz        # discover + print categories
  python scrape.py probe vicove_biz https://www.vicove.biz/bai-ganio
  python scrape.py run vicove_biz               # full scrape (resumable)
  python scrape.py run fun_dir_bg --max-pages 2 --only ivancho,blondinki   # quick test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.robotparser
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urlencode, parse_qsl, urlunparse, unquote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# --------------------------------------------------------------------------- #
# Site configuration
# --------------------------------------------------------------------------- #
# category_re / joke_link_re are matched against ABSOLUTE urls.
# joke_link_re must have one capture group = the stable joke id.
SITES: dict[str, dict] = {
    # ~199k jokes in ~580 categories (site's own counter, Nov 2026).
    "vicove_biz": {
        "start_urls": ["https://www.vicove.biz/categories"],
        "category_re": r"^https?://(?:www\.)?vicove\.biz/([a-z0-9][a-z0-9-]*)/?$",
        "category_exclude": {
            "categories", "novi", "top", "nai-dobri", "kontakti", "contact",
            "biskvitki", "cookies", "citati", "mobile", "login", "register",
            "search", "tarsene", "dobavi", "add",
        },
        "joke_link_re": r"^https?://(?:www\.)?vicove\.biz/([0-9a-f]{32})/?$",
        "pagination": "query",          # ?page=N
        "item_selector": None,
        # Verified on the live site (Nov 2026): jokes are flat <p> blocks inside
        # <article>, each followed by a meta line ending with the 'Преглед' permalink.
        "content_selector": "article",
        "flat_link_position": "after",
    },
    # ~20k+ user-submitted jokes; listing 18/page, category/{slug}?page=N
    "fun_dir_bg": {
        "start_urls": ["https://fun.dir.bg/category/razni"],
        "category_re": r"^https?://fun\.dir\.bg/category/([a-z0-9][a-z0-9-]*)/?$",
        "category_exclude": set(),
        "joke_link_re": r"^https?://fun\.dir\.bg/fun/(\d+)(?:/[^/?#]*)?/?$",
        "pagination": "query",
        "item_selector": None,
        "content_selector": None,
        "flat_link_position": "after",
    },
    # Same "?h=<hash>" engine as vicove.biz -> heavy overlap; dedup handles it.
    "zaigravka_bg": {
        "start_urls": ["https://www.zaigravka.bg/vicove.php"],
        "category_re": r"^https?://(?:www\.)?zaigravka\.bg/(vicove-[a-z0-9-]+)\.php$",
        "category_exclude": set(),
        "joke_link_re": r"^https?://(?:www\.)?zaigravka\.bg/vicove\.php\?h=([0-9a-f]{32})$",
        "pagination": "query",
        "item_selector": None,
        "content_selector": None,
        "flat_link_position": "after",  # 'Виж този виц' link follows the text
    },
    # Small WordPress site (~400 posts). Skips its AI-generated category.
    "vicove_com": {
        "start_urls": ["https://vicove.com/"],
        "category_re": r"^https?://(?:www\.)?vicove\.com/category/([^/?#]+)/?$",
        "category_exclude": {"ai-истории"},
        "joke_link_re": r"^https?://(?:www\.)?vicove\.com/(?!category/|page/|tag/|author/|wp-)([^/?#]+/[^/?#]+)/?$",
        "pagination": "path",           # /page/N/
        "item_selector": "article",
        "content_selector": None,
        "flat_link_position": "before",
    },
}

USER_AGENT = (
    "Mozilla/5.0 (compatible; vic-dataset-research/1.0; "
    "personal non-commercial LLM fine-tuning project)"
)

# Lines that are UI chrome, not joke text.
META_LINE_RE = re.compile(
    r"""^(
        \d+([.,]\d+)?\s*(к|k)?                       # bare counters
      | (Преглед|Прочети|Виж\s+този\s+виц|към\s+вица|към\s+вицa|Още\s+яки\s+вицове.*|Още\s+вицове.*)
      | (Добавен|Публикуван|Добавил|Автор|Категори[яи]|Тагове|Етикети|Оценка|Рейтинг|Статус|Гласове|Харесвания|Коментари?|Сподели|Харесай|Следващ|Предишен)\b.*
      | [«»<>|•·\-–—\s]+                              # separators only
      | (Следваща|Предишна|Напред|Назад)\s*[»›]*
      | \d+\s*(коментара?|гласа?|харесвания|прегледа?)
      | \d[\d\s]*\s+вица?\b.*                          # "40 вица, разпределени в 4 страници"
      | В\s+тази\s+категория\s+има.*
      | \(\d{4}\)                                        # "(2003)" year stamp (fun.dir.bg)
      | (към\s+)?вица                                   # split "към вица" link text
      | (Вицове|Още\s+вицове)\s+(за|с|от)\s+[^.!?]{1,40}  # category headings
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "tr", "table", "section", "article", "header", "footer", "pre",
}
DROP_TAGS = ["script", "style", "noscript", "iframe", "svg", "form", "button",
             "nav", "header", "footer", "aside", "select", "input", "img"]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Fetcher:
    def __init__(self, delay: float = 1.5, timeout: int = 30, max_retries: int = 5):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "bg,en;q=0.5"})
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._last = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _robots_for(self, url: str):
        p = urlparse(url)
        root = f"{p.scheme}://{p.netloc}"
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.s.get(root + "/robots.txt", timeout=self.timeout)
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                else:
                    rp = None  # no robots.txt -> allowed
            except requests.RequestException:
                rp = None
            self._robots[root] = rp
            if rp is not None:
                cd = rp.crawl_delay(USER_AGENT) or rp.crawl_delay("*")
                if cd and float(cd) > self.delay:
                    print(f"  robots.txt asks for crawl-delay {cd}s on {root}", file=sys.stderr)
                    self.delay = float(cd)
        return self._robots[root]

    def allowed(self, url: str) -> bool:
        rp = self._robots_for(url)
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> requests.Response | None:
        if not self.allowed(url):
            print(f"  [robots.txt disallows] {url}", file=sys.stderr)
            return None
        for attempt in range(self.max_retries):
            wait = self.delay * random.uniform(0.8, 1.4) - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                back = min(120, 5 * 2 ** attempt)
                print(f"  network error ({e.__class__.__name__}), retry in {back}s", file=sys.stderr)
                time.sleep(back)
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (404, 410):
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                back = int(ra) if ra and ra.isdigit() else min(300, 10 * 2 ** attempt)
                print(f"  HTTP {r.status_code}, backing off {back}s", file=sys.stderr)
                time.sleep(back)
                self.delay = min(self.delay * 1.5, 10)  # slow down for good
                continue
            print(f"  HTTP {r.status_code} for {url}", file=sys.stderr)
            return None
        return None


def soup_of(resp_or_html) -> BeautifulSoup:
    content = resp_or_html.content if hasattr(resp_or_html, "content") else resp_or_html
    # Passing bytes lets bs4 honour <meta charset> (some BG sites are windows-1251).
    try:
        return BeautifulSoup(content, "lxml")
    except Exception:
        return BeautifulSoup(content, "html.parser")


# --------------------------------------------------------------------------- #
# Text extraction helpers
# --------------------------------------------------------------------------- #
def node_text(el) -> str:
    """Text with line breaks at <br> and block elements, but not inside inline tags."""
    parts: list[str] = []

    def walk(n):
        if isinstance(n, NavigableString):
            if n.__class__.__name__ in ("Comment", "Doctype", "Declaration", "ProcessingInstruction"):
                return
            parts.append(str(n))
            return
        if not isinstance(n, Tag):
            return
        if n.name == "br":
            parts.append("\n")
            return
        block = n.name in BLOCK_TAGS
        if block:
            parts.append("\n")
        for c in n.children:
            walk(c)
        if block:
            parts.append("\n")

    walk(el)
    return "".join(parts)


def clean_lines(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\r", "")
    out = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if META_LINE_RE.match(line):
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    while out and out[0] == "":
        out.pop(0)
    # drop truncated title line duplicated by the body ("Срещнали се ... и...")
    if len(out) >= 2:
        first = out[0].rstrip()
        if first.endswith(("...", "…")):
            stem = first.rstrip(".… ").strip()
            rest = " ".join(l for l in out[1:] if l)
            if stem and rest.startswith(stem):
                out = out[1:]
                while out and out[0] == "":
                    out.pop(0)
    # collapse blank lines: jokes are short; keep single newlines
    return "\n".join(l for l in out if l != "")


# --------------------------------------------------------------------------- #
# Joke extraction from a listing page
# --------------------------------------------------------------------------- #
class Extractor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.joke_re = re.compile(cfg["joke_link_re"])
        self.cat_re = re.compile(cfg["category_re"])

    def joke_id(self, href: str, base: str) -> tuple[str, str] | None:
        if not href or href.startswith(("#", "javascript:", "mailto:", "viber:", "tel:")):
            return None
        absu = urljoin(base, href).split("#")[0]
        m = self.joke_re.match(absu)
        return (unquote(m.group(1)), absu) if m else None

    def _prepare(self, soup: BeautifulSoup):
        root = soup
        if self.cfg.get("content_selector"):
            sel = soup.select_one(self.cfg["content_selector"])
            if sel is not None:
                root = sel
        for t in root.find_all(DROP_TAGS):
            t.decompose()
        return root

    def _strip_links(self, el: Tag, base: str):
        """Remove short UI links (permalink 'Преглед', category, user, share)."""
        for a in el.find_all("a"):
            txt = a.get_text(" ", strip=True)
            href = a.get("href") or ""
            is_joke = self.joke_id(href, base) is not None
            if len(txt) <= 40 or (is_joke and txt.endswith(("...", "…"))):
                # keep if removing would leave almost nothing (whole joke is a link)
                rest = el.get_text(" ", strip=True).replace(txt, "", 1)
                if len(rest.strip()) >= 20:
                    a.decompose()

    def extract(self, html_or_resp, page_url: str) -> list[dict]:
        soup = soup_of(html_or_resp)
        root = self._prepare(soup)

        if self.cfg.get("item_selector"):
            return self._extract_by_selector(root, page_url)

        # map joke id -> first anchor; record order
        anchors: dict[str, tuple[Tag, str]] = {}
        for a in root.find_all("a", href=True):
            ji = self.joke_id(a["href"], page_url)
            if ji and ji[0] not in anchors:
                anchors[ji[0]] = (a, ji[1])
        if not anchors:
            return []

        def ids_in(el: Tag) -> set[str]:
            s = set()
            for a in el.find_all("a", href=True):
                ji = self.joke_id(a["href"], page_url)
                if ji:
                    s.add(ji[0])
            if el.name == "a" and el.get("href"):
                ji = self.joke_id(el["href"], page_url)
                if ji:
                    s.add(ji[0])
            return s

        results, flat_ids = [], []
        for jid, (a, url) in anchors.items():
            container = a
            while container.parent is not None and isinstance(container.parent, Tag) \
                    and container.parent.name not in ("body", "html", "[document]"):
                if len(ids_in(container.parent)) > 1:
                    break
                container = container.parent
            frag = BeautifulSoup(str(container), "lxml")
            body = frag.body or frag
            self._strip_links(body, page_url)
            text = clean_lines(node_text(body))
            if len(text) >= 15:
                results.append({"jid": jid, "url": url, "text": text,
                                "_sig": _sig(container)})
            else:
                flat_ids.append(jid)

        if flat_ids:
            results.extend(self._extract_flat(root, page_url, set(flat_ids)))
        return results

    def _extract_by_selector(self, root, page_url):
        out = []
        for item in root.select(self.cfg["item_selector"]):
            jid, url = None, None
            for a in item.find_all("a", href=True):
                ji = self.joke_id(a["href"], page_url)
                if ji:
                    jid, url = ji
                    break
            frag = BeautifulSoup(str(item), "lxml")
            body = frag.body or frag
            for h in body.find_all(["h1", "h2", "h3"]):  # post titles, not joke text
                rest = body.get_text(" ", strip=True).replace(h.get_text(" ", strip=True), "", 1)
                if len(rest.strip()) >= 20:
                    h.decompose()
            self._strip_links(body, page_url)
            text = clean_lines(node_text(body))
            if not text:
                continue
            if jid is None:  # no permalink: use content hash as id
                jid = "h" + hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
                url = page_url
            out.append({"jid": jid, "url": url, "text": text, "_sig": self.cfg["item_selector"]})
        return out

    def _extract_flat(self, root, page_url, wanted: set[str]):
        """Jokes laid out as siblings: text ... <a permalink> text ... <a permalink>."""
        pos = self.cfg.get("flat_link_position", "after")
        # Linearise the document into a token stream of text and joke-boundaries.
        stream: list[tuple[str, str]] = []

        def walk(n):
            if isinstance(n, NavigableString):
                if n.__class__.__name__ == "NavigableString":
                    stream.append(("t", str(n)))
                return
            if not isinstance(n, Tag):
                return
            if n.name == "a" and n.get("href"):
                ji = self.joke_id(n["href"], page_url)
                if ji:
                    stream.append(("j", ji[0] + "\x00" + ji[1]))
                    txt = n.get_text(" ", strip=True)
                    if len(txt) > 40 and not txt.endswith(("...", "…")):
                        stream.append(("t", txt))  # the link IS the joke text
                    return
            if n.name == "br":
                stream.append(("t", "\n"))
                return
            if n.name in ("h1", "h2", "h3"):  # page / section titles in flat layouts
                stream.append(("t", "\n"))
                return
            block = n.name in BLOCK_TAGS
            if block:
                stream.append(("t", "\n"))
            for c in n.children:
                walk(c)
            if block:
                stream.append(("t", "\n"))

        walk(root)
        # split into segments between boundaries
        segments, buf = [], []
        bounds = []
        for kind, val in stream:
            if kind == "j":
                segments.append("".join(buf))
                buf = []
                bounds.append(val)
            else:
                buf.append(val)
        segments.append("".join(buf))
        out, seen = [], set()
        for i, b in enumerate(bounds):
            jid, url = b.split("\x00", 1)
            if jid in seen or jid not in wanted:
                continue
            seen.add(jid)
            raw = segments[i] if pos == "after" else segments[i + 1]
            text = clean_lines(raw)
            if len(text) >= 15:
                out.append({"jid": jid, "url": url, "text": text, "_sig": "flat"})
        return out

    # ---- categories ------------------------------------------------------- #
    def categories(self, html_or_resp, page_url: str) -> dict[str, dict]:
        soup = soup_of(html_or_resp)
        cats = {}
        excl = {e.lower() for e in self.cfg.get("category_exclude", set())}
        for a in soup.find_all("a", href=True):
            full = urljoin(page_url, a["href"]).split("#")[0]
            if "?" in full and re.search(r"[?&](page|p|start|offset)=", full):
                continue  # pagination link, not a category
            absu = full.split("?")[0]
            if self.joke_re.match(absu):
                continue
            m = self.cat_re.match(absu)
            if not m:
                continue
            slug = unquote(m.group(1)).strip("/").lower()
            if slug in excl:
                continue
            name = a.get_text(" ", strip=True)
            name = re.sub(r"\s*\(\s*[\d\s.,]+\)\s*$", "", name).strip()  # "Иванчо (905)"
            name = re.sub(r"^(Вицове\s+за\s+|Вицове\s+)", "", name, flags=re.I).strip() or slug
            if re.fullmatch(r"[\d\s«»‹›<>.…|-]*|(Следваща|Предишна|Напред|Назад|Първа|Последна|Още)\b.*", name, re.I):
                name = slug  # pager / arrow text
            if slug not in cats or (len(name) > 1 and len(cats[slug]["name"]) <= 1):
                cats[slug] = {"slug": slug, "name": name, "url": absu}
        return cats


def _sig(el: Tag) -> str:
    cls = ".".join(el.get("class", [])[:2]) if isinstance(el, Tag) else ""
    return f"{el.name}{'.' + cls if cls else ''}"


def page_url(base: str, n: int, style: str) -> str:
    if n <= 1:
        return base
    if style == "path":
        return base.rstrip("/") + f"/page/{n}/"
    p = urlparse(base)
    q = dict(parse_qsl(p.query))
    q["page"] = str(n)
    return urlunparse(p._replace(query=urlencode(q)))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def discover(site: str, fetcher: Fetcher) -> dict[str, dict]:
    cfg = SITES[site]
    ex = Extractor(cfg)
    cats: dict[str, dict] = {}
    for u in cfg["start_urls"]:
        r = fetcher.get(u)
        if r is None:
            print(f"  could not fetch {u}", file=sys.stderr)
            continue
        cats.update(ex.categories(r, u))
    return cats


def cmd_categories(args):
    f = Fetcher(delay=args.delay)
    cats = discover(args.site, f)
    for c in sorted(cats.values(), key=lambda c: c["slug"]):
        print(f"{c['slug']:35s} {c['name']:30s} {c['url']}")
    print(f"\n{len(cats)} categories", file=sys.stderr)


def cmd_probe(args):
    """Fetch one listing page and show what the extractor finds — use this to verify a site."""
    cfg = SITES[args.site]
    ex = Extractor(cfg)
    if args.file:
        html = open(args.file, "rb").read()
        jokes = ex.extract(html, args.url)
    else:
        r = Fetcher(delay=args.delay).get(args.url)
        if r is None:
            sys.exit("fetch failed")
        jokes = ex.extract(r, args.url)
    sigs = Counter(j["_sig"] for j in jokes)
    print(f"found {len(jokes)} jokes; container types: {dict(sigs)}")
    if sigs:
        top = sigs.most_common(1)[0][0]
        if top not in ("flat", "a") and "." in top:
            print(f"tip: if these look right you can pin  item_selector = '{top}'")
    for j in jokes[: args.show]:
        t = j["text"].replace("\n", " ⏎ ")
        print(f"\n[{j['jid']}] {j['url']}\n  {t[:300]}{'…' if len(t) > 300 else ''}")


def cmd_run(args):
    site = args.site
    cfg = SITES[site]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "_state"), exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{site}.jsonl")
    state_path = os.path.join(args.out_dir, "_state", f"{site}.json")
    state = {"done": [], "partial": {}, "categories": None}
    if os.path.exists(state_path):
        state = json.load(open(state_path, encoding="utf-8"))

    f = Fetcher(delay=args.delay)
    ex = Extractor(cfg)

    if not state.get("categories"):
        print(f"discovering categories for {site} ...")
        state["categories"] = discover(site, f)
        json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cats = state["categories"]
    if args.only:
        wanted = {s.strip().lower() for s in args.only.split(",")}
        cats = {k: v for k, v in cats.items() if k in wanted}
    print(f"{len(cats)} categories to scrape; {len(state['done'])} already done")

    total_new = 0
    seen_global: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    seen_global.add(r["id"] + "|" + r["category"])
                except Exception:
                    pass

    out = open(out_path, "a", encoding="utf-8")
    try:
        for ci, (slug, cat) in enumerate(sorted(cats.items()), 1):
            if slug in state["done"]:
                continue
            start = state["partial"].get(slug, 0) + 1
            print(f"[{ci}/{len(cats)}] {cat['name']} ({slug}) from page {start}")
            seen_cat: set[str] = set()
            n = start
            empty_streak = 0
            while n <= args.max_pages:
                url = page_url(cat["url"], n, cfg["pagination"])
                r = f.get(url)
                if r is None:
                    break
                # some sites redirect past-the-end pages to page 1
                if n > 1 and r.url.rstrip("/") == cat["url"].rstrip("/"):
                    break
                jokes = ex.extract(r, url)
                new = [j for j in jokes if j["jid"] not in seen_cat]
                for j in jokes:
                    seen_cat.add(j["jid"])
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for j in new:
                    rid = f"{site}:{j['jid']}"
                    if rid + "|" + cat["name"] in seen_global:
                        continue
                    seen_global.add(rid + "|" + cat["name"])
                    out.write(json.dumps({
                        "id": rid, "site": site, "url": j["url"], "text": j["text"],
                        "category": cat["name"], "category_slug": slug,
                        "page_url": url, "scraped_at": now,
                    }, ensure_ascii=False) + "\n")
                    total_new += 1
                out.flush()
                print(f"    p{n}: {len(jokes)} jokes, {len(new)} new", end="\r")
                state["partial"][slug] = n
                if n % 10 == 0:
                    json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                # stop: no jokes, or mostly repeats (past the end / random sidebar only)
                if not jokes or len(new) < max(1, len(jokes) // 2):
                    empty_streak += 1
                    if empty_streak >= 1:
                        break
                else:
                    empty_streak = 0
                n += 1
            print()
            state["done"].append(slug)
            state["partial"].pop(slug, None)
            json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved, re-run the same command to resume")
    finally:
        out.close()
        json.dump(state, open(state_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"done: {total_new} new records -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    p = sub.add_parser("categories"); p.add_argument("site", choices=SITES)
    p.add_argument("--delay", type=float, default=1.5)

    p = sub.add_parser("probe"); p.add_argument("site", choices=SITES); p.add_argument("url")
    p.add_argument("--file", help="parse a saved HTML file instead of fetching")
    p.add_argument("--show", type=int, default=5)
    p.add_argument("--delay", type=float, default=1.5)

    p = sub.add_parser("run"); p.add_argument("site", choices=SITES)
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--delay", type=float, default=1.5, help="seconds between requests (default 1.5)")
    p.add_argument("--max-pages", type=int, default=3000, help="max pages per category")
    p.add_argument("--only", help="comma-separated category slugs to scrape")

    args = ap.parse_args()
    if args.cmd == "list":
        for k, v in SITES.items():
            print(f"{k:14s} {v['start_urls'][0]}")
    elif args.cmd == "categories":
        cmd_categories(args)
    elif args.cmd == "probe":
        cmd_probe(args)
    elif args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
