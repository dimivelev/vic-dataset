# Bulgarian jokes (вицове) dataset for LLM fine-tuning

Three steps: **scrape → clean → build dataset**. Everything runs on your own computer
with Python 3.9+.

```bash
pip install -r requirements.txt
```

## Sources (checked Nov 2026)

| Site key | Site | Size | Notes |
|---|---|---|---|
| `vicove_biz` | vicove.biz | ~199,000 jokes, ~580 categories (site's own counter) | Largest by far. Category pages use `?page=N`. Joke IDs are 32-char hashes. |
| `fun_dir_bg` | fun.dir.bg | ~25,000+ (e.g. Разни 7,345, Семейството 2,479, Жените 1,755, Иванчо 905) | User-submitted, active. 18 jokes per page, `category/{slug}?page=N`. |
| `zaigravka_bg` | zaigravka.bg | a few thousand | Same engine as vicove.biz, so expect heavy overlap. Clean-up removes the duplicates. |
| `vicove_com` | vicove.com | ~400 posts | Small WordPress site. Its "AI истории" category is skipped. |

Other sites worth adding later: haha.bg (themed articles), vic.bg / vicovete.bg /
vic-bg.com, 2leva.com (~4,000), and forum threads on bg-mamma.com. Most are smaller
and overlap with the sites above.

After deduplication, expect roughly **60–120k unique jokes**. vicove.biz lists many
jokes in several categories and has many retellings. The real number shows up after
`clean.py`.

## 1. Scrape

**Probe first.** The scraper finds jokes by their permalink links, not by CSS class
names, so it adapts to most layouts. Still, check each site before a long run:

```bash
python scrape.py probe vicove_biz https://www.vicove.biz/bai-ganio
python scrape.py probe fun_dir_bg https://fun.dir.bg/category/ivancho
python scrape.py probe zaigravka_bg https://www.zaigravka.bg/vicove-bai-ganio.php
python scrape.py probe vicove_com https://vicove.com/
```

Each probe prints how many jokes it found and the first few texts. Look for complete
jokes with no "Преглед", likes or "Добавен преди…" text. If it prints a tip such as
`item_selector = 'div.xyz'`, you can paste that into `SITES[...]` in `scrape.py` to
fix the layout.

**Quick test run** (2 pages of 2 categories):

```bash
python scrape.py run fun_dir_bg --only ivancho,blondinki --max-pages 2
```

**Full runs.** These are resumable: press Ctrl-C at any time and run the same command
again to continue.

```bash
python scrape.py run fun_dir_bg      # ~1.5k requests, about 45 min
python scrape.py run vicove_biz      # many thousands of requests, leave it overnight
python scrape.py run zaigravka_bg
python scrape.py run vicove_com
```

Output goes to `data/raw/<site>.jsonl`, one line per (joke, category). Progress is
saved in `data/raw/_state/`.

What the scraper does to avoid overloading the sites:
- It follows `robots.txt`, including any crawl-delay it sets.
- It waits about 1.5 s (with random jitter) between requests. Change this with `--delay`.
- It backs off on HTTP 429/5xx and slows down for the rest of the run.
- It stops a category as soon as a page brings mostly already-seen jokes, so it
  never loops past the end.

## 2. Clean and deduplicate

```bash
python clean.py
```

This command:
- normalises the text: Unicode, quotes, dialogue dashes (`– `), and Latin
  look-alike letters inside Cyrillic words (`пoлицай` → `полицай`)
- removes junk: very short or very long texts, non-Bulgarian text (English or
  Russian), URLs and ads, all-caps posts
- merges the same joke across categories and sites, keeping every category as a tag
- removes exact duplicates (ignoring punctuation and case) and **near-duplicates**
  (MinHash on character 5-grams, Jaccard ≥ 0.7, adjustable with `--near-threshold`)
- flags jokes as `nsfw` (dirty or sexual) or `sensitive` (ethnic or religious
  targets), so you can choose what goes into training

Output is `data/clean/jokes.jsonl`, plus `data/clean/stats.json` with rejection
reasons, flag counts, length percentiles and top categories. About 200k records
take about a minute.

**Before training, read 100–200 random jokes yourself:**
```bash
shuf -n 150 data/clean/jokes.jsonl | python -c "import sys,json;[print(json.loads(l)['text'],'\n---') for l in sys.stdin]" | less
```

## 3. Build the fine-tuning dataset

```bash
python build_dataset.py                                    # everything
python build_dataset.py --exclude-nsfw --exclude-sensitive # family-friendly model
python build_dataset.py --max-per-topic 3000               # stop "разни"/"жени" dominating
python build_dataset.py --format text                      # raw text for continued pre-training
```

Output goes to `data/dataset/train.jsonl`, `val.jsonl` (5%) and `topics.json`.

Each joke becomes one chat example:
```json
{"messages": [
  {"role": "user", "content": "Знаеш ли някой габровски виц?"},
  {"role": "assistant", "content": "Един габровец си купил часовник без стрелки.\n– Защо без стрелки? – питат го.\n– Да не се хаби времето, бе!"}
], "topic": "габровски виц", "id": "…"}
```

- Topics come from the site categories and are mapped to natural Bulgarian phrases
  (`виц за Иванчо`, `мръсен виц`, `габровски виц`, `виц с черен хумор`…). To add or
  merge topics, edit the `CANON` table in `build_dataset.py`.
- About 15% of prompts have no topic ("Разкажи ми виц."), so the model also learns
  to pick its own topic.
- There are 10 prompt templates, so the model doesn't overfit to a single phrasing.

This format works directly with Hugging Face TRL `SFTTrainer`, Unsloth, Axolotl and
LLaMA-Factory.

## 3b. Alternative: build the dataset in Unsloth Studio (Data Recipe)

`unsloth/vic-jokes-sft.recipe.json` does the same job as step 3, but inside
Unsloth Studio's Data Recipes, so the dataset lands straight in Studio's dataset
picker for fine-tuning.

Data Recipes can't fetch websites or remove rows, so scraping, cleaning, topic
mapping and the NSFW/sensitive/length filters stay in Python. You export a seed file:

```bash
python build_dataset.py --format seed --exclude-nsfw     # add/remove filters as you like
# -> data/dataset/recipe_seed.jsonl   (columns: id, output, topic) and prints the row count
```

Then in Unsloth Studio:

1. Go to **Data Recipes**, choose **Import**, and select `unsloth/vic-jokes-sft.recipe.json`.
2. Click the **seed** block and upload `data/dataset/recipe_seed.jsonl` (local file).
   Load the columns. `id` and `topic` are already marked to be dropped from the output.
3. Click **Validate**, then **Preview** (10 rows). Check that the `instruction` values read naturally.
4. Run the full build with **rows = the number `build_dataset.py` printed**, so each
   joke is used once. Setting more rows than that repeats jokes.
5. In **Train**, pick the generated dataset. Its `instruction` + `output` columns
   are detected as Alpaca format. The download also includes a chat-format copy
   (`messages`) from the `messages_format` processor.

What the recipe does:

| Block | Type | Purpose |
|---|---|---|
| `seed` | local JSONL, shuffled | the jokes (`output`) and their topic phrase (`topic`) |
| `prompt_template` | category sampler | one of 10 Bulgarian requests with a `ТЕМА` placeholder |
| `use_topic` | bernoulli (p = 0.85) | 85% of prompts name the topic, 15% are generic |
| `generic_topic` | category sampler | `виц` / `смешен виц` / `хубав виц` for generic prompts |
| `instruction` | Jinja expression | fills `ТЕМА` with the topic or the generic phrase |
| `drop_seed_columns` | processor | removes `id`, `topic` |
| `messages_format` | schema_transform | adds a `messages` chat version |

The recipe uses no LLM, so it runs on CPU in seconds. To get more varied prompts,
you could add an LLM Text block that writes the user request, or an LLM Judge block
that scores how funny each joke is. Either one needs a model provider (any
OpenAI-compatible endpoint, e.g. llama.cpp serving BgGPT).

## Running everything on Render (free)

`render.yaml` sets up a **free** web service and a **free** Postgres database
in Frankfurt. `render_app.py` then runs the whole pipeline with no computer of
yours involved:

- It scrapes the sites one by one, saving every joke and its progress to
  Postgres every 20 seconds, so a restart costs at most a few pages.
- While scraping, it requests its own address every 10 minutes so the free
  instance isn't put to sleep. When finished, it stops, and Render lets it sleep.
- It then runs `clean.py` and `build_dataset.py` and stores the outputs
  (gzipped) in Postgres.
- The service URL shows a progress page. Downloads are at
  `/download/<file>?key=<DOWNLOAD_KEY>`. The key is generated by Render: find it
  on the service's **Environment** tab.

Files available to download: `recipe_seed.jsonl`, `recipe_seed_sfw.jsonl`,
`train.jsonl`, `val.jsonl`, `jokes.jsonl`, `stats.json`.

Free-tier limits to know about:
- The free Postgres database expires 30 days after creation, so download your files before then.
- Render can restart a free service at any time. The app resumes from Postgres when that happens.
- One always-on free service uses about 24 of the workspace's 750 monthly free
  instance hours per day.

## Tests

```bash
python tests/test_scrape.py
```

These run the extractor offline on HTML that copies each site's layout.

## Troubleshooting

- **The probe finds 0 jokes.** Either the site changed its permalink format or it
  loads jokes with JavaScript. Open a joke in your browser, copy its URL and update
  `joke_link_re` for that site.
- **Joke texts include extra lines** such as a rating or date. Add the pattern to
  `META_LINE_RE` in `scrape.py`, or pin `item_selector`.
- **HTTP 403 / 429.** Raise `--delay` (e.g. `--delay 4`) and resume later.

## Responsible use

Jokes are mostly folklore, but each site's collection and wording belong to its
owners. Use the data for personal or research fine-tuning, and don't republish the
scraped dataset or the sites' content without permission. Check each site's terms of
use before you scrape it.
