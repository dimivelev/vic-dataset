#!/usr/bin/env python3
"""
Turn cleaned jokes into a fine-tuning dataset.

  python build_dataset.py                                 # chat format, everything
  python build_dataset.py --exclude-nsfw --exclude-sensitive
  python build_dataset.py --format text                   # raw text (continued pre-training)
  python build_dataset.py --format seed                   # seed file for the Unsloth Studio recipe

Output (data/dataset/):
  train.jsonl, val.jsonl   {"messages":[{"role":"user",...},{"role":"assistant",...}], "topic":..., "id":...}
  topics.json              topic -> count (use it to pick prompts at inference time)

Each joke becomes ONE example (no repeats -> less memorisation). The user prompt asks
for a joke on the joke's most specific topic, phrased in one of several Bulgarian
templates; ~15% of prompts are topic-less ("Разкажи ми виц.") so the model also
learns to pick a topic by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

# Category names that say nothing about the topic.
GENERIC = {
    "разни", "нови", "нови вицове", "вицове", "всички", "топ", "най-добрите", "най-нови",
    "кратки", "кратки вицове", "лафове", "въпрос-отговор", "истории", "весели истории",
    "други", "смешни", "яки", "подбрани", "любопитно", "настроение", "усмивки", "тъпизми",
    "надписи", "обяви", "бисери", "още бисери", "още вицове", "графити", "лафоризми",
    "вицоблог", "sms", "герои", "професии", "градове", "национални", "приказни",
}

# Curated canonical topics: category name (lowercase) -> noun phrase for the prompt.
# The phrase must fit "Разкажи ми {np}." and "Знаеш ли някой {np}?"
CANON = {
    # characters
    "иванчо": "виц за Иванчо", "иванчо и марийка": "виц за Иванчо и Марийка",
    "марийка": "виц за Иванчо и Марийка", "бай ганьо": "виц за Бай Ганьо",
    "чък норис": "виц за Чък Норис", "радио ереван": "виц за Радио Ереван",
    "хитър петър": "виц за Хитър Петър", "щирлиц": "виц за Щирлиц",
    "поручик ржевски": "виц за поручик Ржевски", "чапай и петка": "виц за Чапай и Петка",
    "крикор и гарабед": "виц за Крикор и Гарабед", "нане и вуте": "виц за Нане и Вуте",
    "хасан и айшето": "виц за Хасан и Айшето", "златната рибка": "виц за златната рибка",
    "шерлок холмс": "виц за Шерлок Холмс", "мечо пух": "виц за Мечо Пух",
    "червената шапчица": "виц за Червената шапчица", "снежанка": "виц за Снежанка",
    "свети петър": "виц за Свети Петър", "адам и ева": "виц за Адам и Ева",
    "тодор живков": "виц за Тодор Живков", "бойко борисов": "виц за Бойко Борисов",
    "петърчо": "виц за Петърчо", "генко": "виц за Генко", "пешо": "виц за Пешо",
    # places / regional
    "габровски": "габровски виц", "габровци": "габровски виц", "габрово": "габровски виц",
    "перничани": "виц за перничани", "перник": "виц за перничани",
    "шопи": "виц за шопи", "враца": "виц за Враца", "софия": "виц за София",
    "пловдив": "виц за Пловдив", "варна": "виц за Варна", "бургас": "виц за Бургас",
    "българия": "виц за България", "русия": "виц за Русия",
    # adjectives -> "<adj> виц"
    "мръсни": "мръсен виц", "мръсни вицове +18": "мръсен виц", "с*кс": "мръсен виц",
    "семейни": "семеен виц", "семейството": "семеен виц",
    "пиянски": "пиянски виц", "пияници": "виц за пияници", "алкохол": "виц за алкохол",
    "политически": "политически виц", "политика": "политически виц", "политици": "политически виц",
    "ученически": "ученически виц", "студентски": "студентски виц", "студенти": "студентски виц",
    "медицински": "виц за лекари", "лекари": "виц за лекари", "доктори": "виц за лекари",
    "лекарски": "виц за лекари", "черен хумор": "виц с черен хумор",
    "компютърни": "виц за компютри", "програмисти": "виц за програмисти",
    "военни": "войнишки виц", "войници": "войнишки виц", "казарма": "войнишки виц",
    "спорт": "виц за спорт", "футбол": "виц за футбол",
    "празнични": "празничен виц", "президентски": "виц за президенти",
    # people / groups
    "жени": "виц за жени", "жените": "виц за жени", "мъже": "виц за мъже", "мъжете": "виц за мъже",
    "блондинки": "виц за блондинки", "тъщи": "виц за тъщата", "баби": "виц за баби",
    "деца": "виц за деца", "полицаи": "виц за полицаи", "катаджии": "виц за катаджии",
    "мутри": "виц за мутри", "борци": "виц за мутри", "животни": "виц за животни",
    "работа": "виц за работата", "шефове": "виц за шефове", "началници": "виц за шефове",
    "изневери": "виц за изневери", "свалки": "виц за свалки", "младоженци": "виц за младоженци",
    "луди": "виц за луди", "лудница": "виц за луди", "евреи": "виц за евреи",
    "цигани": "виц за цигани", "арменци": "виц за арменци", "лов и риболов": "виц за ловци и рибари",
    "ловци": "виц за ловци", "риболов": "виц за рибари", "коронавирус": "виц за коронавируса",
}
CANON_NPS = set(CANON.values())

TEMPLATES = [
    "Разкажи ми {np}.",
    "Разкажи {np}.",
    "Знаеш ли някой {np}?",
    "Искам да чуя {np}.",
    "Сподели един {np}.",
    "Дай ми {np}, моля.",
    "Хайде, един {np}!",
    "Сети ли се за някой {np}?",
    "Можеш ли да ми разкажеш {np}?",
    "Развесели ме с {np_instr}.",
]
GENERIC_NP = ["виц", "смешен виц", "хубав виц", "някой виц"]


def to_instrumental(np_: str) -> str:
    # "виц за X" -> "виц за X" works after "с" too ("с виц за X"); adjectives agree as-is
    return np_


def topic_np(cat: str) -> str | None:
    k = cat.strip().lower()
    if not k or k in GENERIC:
        return None
    if k in CANON:
        return CANON[k]
    # fallback: "за <category>" — keep capitals for multi-word proper names
    words = cat.strip().split()
    if len(words) >= 2 and all(w[:1].isupper() for w in words):
        name = cat.strip()
    elif re.search(r"[A-Za-z]", cat):
        name = cat.strip()        # brands: BMW, Google ...
    else:
        name = cat.strip()[:1].lower() + cat.strip()[1:]
    return f"виц за {name}"


def stable_frac(s: str) -> float:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", default="data/clean/jokes.jsonl")
    ap.add_argument("--out-dir", default="data/dataset")
    ap.add_argument("--format", choices=["chat", "text", "seed"], default="chat")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--generic-frac", type=float, default=0.15, help="share of topic-less prompts")
    ap.add_argument("--exclude-nsfw", action="store_true")
    ap.add_argument("--exclude-sensitive", action="store_true")
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--min-topic-count", type=int, default=20,
                    help="uncurated categories with fewer jokes are used only as a last resort")
    ap.add_argument("--max-per-topic", type=int, default=0, help="cap examples per topic (0 = no cap)")
    ap.add_argument("--system", default="", help="optional system prompt added to every example")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    jokes = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]

    # category frequencies -> choose the most specific (rarest) meaningful topic per joke
    cat_freq = Counter()
    for j in jokes:
        nps = {topic_np(c) for c in j["categories"]} - {None}
        cat_freq.update(nps)

    skipped = Counter()
    per_topic = Counter()
    seed_rows: list[dict] = []
    train, val = [], []
    rng.shuffle(jokes)
    for j in jokes:
        if args.exclude_nsfw and j.get("nsfw"):
            skipped["nsfw"] += 1; continue
        if args.exclude_sensitive and j.get("sensitive"):
            skipped["sensitive"] += 1; continue
        if not (args.min_chars <= len(j["text"]) <= args.max_chars):
            skipped["length"] += 1; continue

        # prefer curated topics, then topics with enough examples, then the most specific
        nps = sorted({topic_np(c) for c in j["categories"]} - {None},
                     key=lambda n: (n not in CANON_NPS,
                                    cat_freq[n] < args.min_topic_count,
                                    cat_freq[n], n))
        np_ = nps[0] if nps else None
        topic = np_ or "(общ)"
        if args.max_per_topic and per_topic[topic] >= args.max_per_topic:
            skipped["topic_cap"] += 1; continue
        per_topic[topic] += 1

        if args.format == "seed":
            # Unsloth Studio Data Recipe seed: prompts are built inside the recipe.
            # Topic-less jokes get the generic noun phrase so `topic` is never empty.
            seed_rows.append({"id": j["id"], "output": j["text"], "topic": np_ or "виц"})
            continue
        if args.format == "text":
            ex = {"id": j["id"], "text": j["text"], "topic": topic}
        else:
            if np_ is None or rng.random() < args.generic_frac:
                gnp = "кратък виц" if len(j["text"]) < 140 and rng.random() < 0.3 else rng.choice(GENERIC_NP)
                tpl = rng.choice(TEMPLATES)
                if gnp == "някой виц" and "някой" in tpl:
                    gnp = "виц"
                prompt = tpl.format(np=gnp, np_instr=gnp)
            else:
                prompt = rng.choice(TEMPLATES).format(np=np_, np_instr=to_instrumental(np_))
            prompt = prompt[0].upper() + prompt[1:]
            msgs = []
            if args.system:
                msgs.append({"role": "system", "content": args.system})
            msgs += [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": j["text"]}]
            ex = {"id": j["id"], "messages": msgs, "topic": topic}

        (val if stable_frac(j["id"]) < args.val_frac else train).append(ex)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.format == "seed":
        path = os.path.join(args.out_dir, "recipe_seed.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in seed_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"seed rows: {len(seed_rows)} -> {path}   skipped: {dict(skipped)}")
        print(f"In Unsloth Studio set the run's row count to {len(seed_rows)} (one row per joke).")
        return
    for name, rows in (("train", train), ("val", val)):
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "topics.json"), "w", encoding="utf-8") as fh:
        json.dump(per_topic.most_common(), fh, ensure_ascii=False, indent=1)

    print(f"train: {len(train)}  val: {len(val)}  skipped: {dict(skipped)}")
    print("top topics:", ", ".join(f"{t} ({n})" for t, n in per_topic.most_common(12)))


if __name__ == "__main__":
    main()
