"""Offline tests: run with  python -m pytest tests  or  python tests/test_scrape.py"""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))
from scrape import SITES, Extractor, page_url  # noqa: E402


def load(name):
    return open(os.path.join(HERE, "fixtures", name), "rb").read()


def test_vicove_biz():
    ex = Extractor(SITES["vicove_biz"])
    j = ex.extract(load("vicove_biz_listing.html"), "https://www.vicove.biz/bai-ganio")
    assert [x["jid"] for x in j] == ["0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"]
    assert j[0]["text"].startswith("Бай Ганьо отишъл в Париж")
    assert "Преглед" not in j[0]["text"] and "Добавен" not in j[0]["text"]
    assert "много!" in j[0]["text"]          # inline <b> kept on the same line
    assert all("страничната" not in x["text"] for x in j)  # sidebar ignored
    cats = ex.categories(load("vicove_biz_listing.html"), "https://www.vicove.biz/bai-ganio")
    assert set(cats) == {"ivancho", "bai-ganio"}  # /categories excluded, joke hashes excluded


def test_vicove_biz_real_layout():
    """Layout copied from the live site (Nov 2026): flat <p> + meta line with permalink after."""
    ex = Extractor(SITES["vicove_biz"])
    j = ex.extract(load("vicove_biz_real.html"), "https://www.vicove.biz/afganistan")
    assert [x["jid"][:6] for x in j] == ["db6f8b", "74336c", "f7fc59"]
    assert j[0]["text"].startswith("Първи виц") and j[0]["text"].endswith("първия виц!")
    assert j[1]["text"].startswith("Втори виц") and j[2]["text"].startswith("Трети виц")
    assert all("Оценка" not in x["text"] and "Добавен" not in x["text"] for x in j)


def test_zaigravka_flat_cp1251():
    ex = Extractor(SITES["zaigravka_bg"])
    j = ex.extract(load("zaigravka_listing.html"), "https://www.zaigravka.bg/vicove-bai-ganio.php")
    assert len(j) == 2
    assert j[0]["text"].startswith("Габровец")   # header + rating stripped, cp1251 decoded
    assert "Добавен" not in j[0]["text"]
    assert j[1]["text"].startswith("Срещат се двама шопи")


def test_fun_dir_bg():
    ex = Extractor(SITES["fun_dir_bg"])
    j = ex.extract(load("fun_dir_listing.html"), "https://fun.dir.bg/category/ivancho")
    assert [x["jid"] for x in j] == ["48838", "48835"]
    t = j[0]["text"]
    assert t.startswith("Иванчо, защо закъсня?") and "Добавил" not in t and "3382" not in t
    cats = ex.categories(load("fun_dir_listing.html"), "https://fun.dir.bg/category/ivancho")
    assert cats["ivancho"]["name"] == "Иванчо"     # "(905)" count stripped


def test_vicove_com():
    ex = Extractor(SITES["vicove_com"])
    html = load("vicove_com_listing.html")
    j = ex.extract(html, "https://vicove.com/")
    assert len(j) == 2 and j[0]["text"].startswith("Рибар хванал")
    cats = ex.categories(html, "https://vicove.com/")
    assert "разни" in cats and "ai-истории" not in cats   # AI-generated category skipped


def test_page_url():
    assert page_url("https://x.bg/cat", 1, "query") == "https://x.bg/cat"
    assert page_url("https://x.bg/cat", 3, "query") == "https://x.bg/cat?page=3"
    assert page_url("https://x.bg/cat/", 2, "path") == "https://x.bg/cat/page/2/"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
