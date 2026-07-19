# -*- coding: utf-8 -*-
"""美股数据: 15只科技/半导体龙头

- 行情: 新浪 ak.stock_us_daily (前复权, 全历史, 缓存 data/us/{T}.csv)
- 新闻: Google News RSS (无需 key, 缓存 data/us_news/{T}.json)
- 估值: 百度股市通美股接口国内不可用, PE/PB 暂缺 (网页显示 —)
"""
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

DATA_DIR = Path("data/us")
US_NEWS_DIR = Path("data/us_news")

US_STOCKS = [
    ("NVDA", "英伟达"), ("AAPL", "苹果"), ("MSFT", "微软"), ("GOOGL", "谷歌"),
    ("AMZN", "亚马逊"), ("META", "Meta"), ("TSLA", "特斯拉"), ("AVGO", "博通"),
    ("AMD", "超威半导体"), ("MU", "美光科技"), ("SMCI", "超微电脑"), ("PLTR", "Palantir"),
    ("ARM", "ARM控股"), ("INTC", "英特尔"), ("QCOM", "高通"),
]

YEARS = 10  # 只保留最近10年数据, 控制网页体积


def fetch_us_daily(ticker: str) -> pd.DataFrame:
    """新浪美股日线(前复权), 无日期参数, 每次全量拉取覆盖缓存."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    f = DATA_DIR / f"{ticker}.csv"
    if f.exists() and pd.Timestamp.today() - pd.Timestamp(f.stat().st_mtime, unit="s") < pd.Timedelta(days=1):
        df = pd.read_csv(f, parse_dates=["date"])
    else:
        df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.to_csv(f, index=False)
        time.sleep(0.5)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=int(YEARS * 365.25))
    return df[df["date"] >= cutoff].set_index("date")


def fetch_us_all() -> dict:
    out = {}
    for ticker, name in US_STOCKS:
        try:
            out[ticker] = {"name": name, "df": fetch_us_daily(ticker)}
        except Exception as e:
            print(f"  {ticker} {name} 行情获取失败: {type(e).__name__}")
    return out


def fetch_us_news(ticker: str, name: str, limit: int = 12) -> list:
    """Google News RSS 个股新闻, 缓存合并(按链接去重), 返回 [{time,title,link,source,title_zh,score,...}]"""
    US_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    f = US_NEWS_DIR / f"{ticker}.json"
    cached = json.loads(f.read_text()) if f.exists() else []
    by_link = {n["link"]: n for n in cached}
    try:
        q = urllib.parse.quote(f"{ticker} stock")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for it in root.findall(".//item")[:limit]:
            link = it.find("link").text
            if link not in by_link:
                pub = it.find("pubDate").text  # RFC822
                t = pd.Timestamp(pub).strftime("%Y-%m-%d %H:%M:%S")
                by_link[link] = {
                    "time": t, "title": it.find("title").text, "link": link,
                    "source": "新闻", "title_zh": None,
                    "score": None, "label": None, "reason": None,
                }
        time.sleep(0.3)
    except Exception as e:
        print(f"  {ticker} {name} 新闻拉取失败: {type(e).__name__}: {str(e)[:80]}")
    news = sorted(by_link.values(), key=lambda n: n["time"], reverse=True)[:15]
    f.write_text(json.dumps(news, ensure_ascii=False, indent=1))
    return news


if __name__ == "__main__":
    stocks = fetch_us_all()
    print(f"行情: {len(stocks)}/{len(US_STOCKS)} 只")
    for t, v in stocks.items():
        print(f"  {t} {v['name']}: {v['df'].index[0].date()} ~ {v['df'].index[-1].date()}, {len(v['df'])} 条")
    n_ok = 0
    for ticker, name in US_STOCKS:
        news = fetch_us_news(ticker, name)
        n_ok += bool(news)
    print(f"新闻: {n_ok}/{len(US_STOCKS)} 只")
