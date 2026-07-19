# -*- coding: utf-8 -*-
"""成分股新闻情绪分析

- 新闻源: 东方财富个股新闻 (ak.stock_news_em), 缓存 data/news/{code}.json
- 情绪打分: DeepSeek API (OpenAI 兼容接口, deepseek-chat)
  key 读取顺序: 环境变量 DEEPSEEK_API_KEY > 项目根目录 deepseek_key.txt
  打分结果缓存在同一 json 里, 已打分的新闻不会重复计费
- 无 key 时: 正常拉取新闻, 情绪字段留空, 网页提示"未配置 API Key"
"""
import json
import os
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import requests

NEWS_DIR = Path("data/news")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

PROMPT = """你是A股金融新闻分析师。下面是股票「{name}({code})」的新闻标题列表。
逐条判断该新闻对该股票短期股价的影响, 输出 JSON 数组, 每条包含:
- "i": 新闻序号(整数, 从0开始)
- "score": 情绪分数, -1(明显利空)到+1(明显利好), 中性为0, 保留1位小数
- "label": "利好"/"利空"/"中性"
- "impact": 影响类型, 三选一:
  "直接" = 新闻主体是该公司本身(业绩/订单/公告/处罚等; 标题带【公告】的是公司正式披露, 带【快讯】的是财经快讯, 通常为直接)
  "间接" = 通过产业链传导(客户/供应商/同行/上下游的事件), score 绝对值不超过0.5
  "情绪" = 仅市场关注度或板块联动, 无实质基本面传导, score 绝对值不超过0.3
- "reason": 不超过15个字的理由
只输出 JSON 数组, 不要输出其他内容。

新闻列表:
{titles}"""

IMPACT_WEIGHT = {"直接": 1.0, "间接": 0.5, "情绪": 0.25}

US_PROMPT = """你是美股金融新闻分析师。下面是美股「{name}({code})」的英文新闻标题列表。
逐条分析, 输出 JSON 数组, 每条包含:
- "i": 新闻序号(整数, 从0开始)
- "title_zh": 标题的中文翻译(准确简洁, 不超过30字)
- "score": 情绪分数, -1(明显利空)到+1(明显利好), 中性为0, 保留1位小数
- "label": "利好"/"利空"/"中性"
- "impact": 影响类型, 三选一:
  "直接" = 新闻主体是该公司本身(财报/产品/并购/监管/高管等)
  "间接" = 通过产业链传导(客户/供应商/同行/上下游的事件), score 绝对值不超过0.5
  "情绪" = 仅市场关注度或板块联动, 无实质基本面传导, score 绝对值不超过0.3
- "reason": 不超过15个字的中文理由
只输出 JSON 数组, 不要输出其他内容。

新闻列表:
{titles}"""


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key and Path("deepseek_key.txt").exists():
        key = Path("deepseek_key.txt").read_text().strip()
    return key


def fetch_telegraph() -> list:
    """东财财经快讯流(全局), 返回 [{title,summary,time,link}]"""
    try:
        df = ak.stock_info_global_em()
        return [{"title": str(r["标题"]), "summary": str(r["摘要"]),
                 "time": str(r["发布时间"]), "link": str(r["链接"])}
                for _, r in df.iterrows()]
    except Exception as e:
        print(f"  东财快讯拉取失败: {type(e).__name__}")
        return []


def fetch_news(code: str, name: str, limit: int = 15, telegraph: list = None) -> list:
    """拉取个股新闻+公告+快讯, 与本地缓存合并(按链接去重).

    来源: 东方财富个股新闻(媒体相关报道) + 东方财富公告大全(公司正式披露)
          + 东财财经快讯(按股票名过滤, 时效性强)
    返回 [{time,title,link,source,score,label,impact,reason}], source ∈ {新闻, 公告, 快讯}
    """
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    f = NEWS_DIR / f"{code}.json"
    cached = json.loads(f.read_text()) if f.exists() else []
    for n in cached:  # 兼容旧缓存: 无 source 字段的视为新闻
        n.setdefault("source", "新闻")
    by_link = {(n["link"] or "tel:" + n["title"]): n for n in cached}

    try:
        df = ak.stock_news_em(symbol=code)
        for _, row in df.head(limit).iterrows():
            link = str(row["新闻链接"])
            if link not in by_link:
                by_link[link] = {
                    "time": str(row["发布时间"]), "title": str(row["新闻标题"]),
                    "link": link, "source": "新闻",
                    "score": None, "label": None, "reason": None,
                }
    except Exception as e:
        print(f"  {code} {name} 新闻拉取失败: {type(e).__name__}")

    try:
        start = (pd.Timestamp.today() - pd.Timedelta(days=45)).strftime("%Y%m%d")
        end = pd.Timestamp.today().strftime("%Y%m%d")
        df = ak.stock_individual_notice_report(security=code, symbol="全部",
                                               begin_date=start, end_date=end)
        for _, row in df.head(10).iterrows():
            link = str(row["网址"])
            if link not in by_link:
                by_link[link] = {
                    "time": str(row["公告日期"]), "title": str(row["公告标题"]),
                    "link": link, "source": "公告",
                    "score": None, "label": None, "reason": None,
                }
        time.sleep(0.3)
    except Exception as e:
        print(f"  {code} {name} 公告拉取失败: {type(e).__name__}")

    if telegraph:  # 快讯按股票名过滤, 最多5条
        cnt = 0
        for it in telegraph:
            if cnt >= 5:
                break
            if name in it["title"] or name in it["summary"]:
                key = it["link"] or ("tel:" + it["title"])
                if key not in by_link:
                    by_link[key] = {
                        "time": it["time"], "title": it["title"],
                        "link": it["link"] or "", "source": "快讯",
                        "score": None, "label": None, "reason": None,
                    }
                    cnt += 1

    news = sorted(by_link.values(), key=lambda n: n["time"], reverse=True)[:25]
    f.write_text(json.dumps(news, ensure_ascii=False, indent=1))
    return news


def deepseek_score(code: str, name: str, news: list, us: bool = False) -> int:
    """对未打分的新闻调 DeepSeek, 返回新打分条数. us=True 时附带标题中文翻译"""
    todo = [(i, n) for i, n in enumerate(news) if n.get("score") is None]
    if not todo:
        return 0
    key = get_api_key()
    if not key:
        return 0
    titles = "\n".join(f"{i}. 【{n.get('source', '新闻')}】{n['title']}" for i, n in todo)
    prompt = (US_PROMPT if us else PROMPT).format(name=name, code=code, titles=titles)
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "response_format": {"type": "json_object"}},
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(content)
    items = data if isinstance(data, list) else next(iter(data.values()))
    idx2news = dict(todo)
    count = 0
    for item in items:
        n = idx2news.get(int(item["i"]))
        if n is not None:
            n["score"] = float(item["score"])
            n["label"] = str(item["label"])
            n["impact"] = str(item.get("impact", "直接"))
            n["reason"] = str(item["reason"])[:20]
            if us and item.get("title_zh"):
                n["title_zh"] = str(item["title_zh"])[:40]
            count += 1
    return count


def aggregate(news: list) -> dict:
    """按影响类型加权: 直接×1.0, 间接×0.5, 情绪×0.25"""
    scored = [n for n in news if n.get("score") is not None]
    if not scored:
        return {"均分": None, "倾向": "未分析"}
    ws = [IMPACT_WEIGHT.get(n.get("impact"), 1.0) for n in scored]
    avg = sum(n["score"] * w for n, w in zip(scored, ws)) / sum(ws)
    label = "偏多" if avg > 0.15 else ("偏空" if avg < -0.15 else "中性")
    return {"均分": round(avg, 2), "倾向": label, "已分析": len(scored)}


def main():
    from cons_data import get_constituents
    cons = get_constituents()
    has_key = bool(get_api_key())
    print(f"DeepSeek API Key: {'已配置' if has_key else '未配置 (只拉新闻不打分)'}")
    summary = {}
    telegraph = fetch_telegraph()
    print(f"东财快讯流: {len(telegraph)} 条")
    for _, row in cons.iterrows():
        code, name = str(row["code"]), row["name"]
        news = fetch_news(code, name, telegraph=telegraph)
        try:
            n = deepseek_score(code, name, news)
        except Exception as e:
            n = 0
            print(f"  {code} {name} 打分失败: {type(e).__name__}: {e}")
        if n:
            # 打分后写回缓存
            (NEWS_DIR / f"{code}.json").write_text(
                json.dumps(news, ensure_ascii=False, indent=1))
        agg = aggregate(news)
        summary[code] = {"name": name, **agg}
        print(f"{code} {name}: 新闻{len(news)}条, 新打分{n}条, {agg}")
        time.sleep(0.3)
    # 美股/韩股: Google News RSS + DeepSeek 翻译/打分
    try:
        from us_data import US_NEWS_DIR, overseas_stocks, fetch_us_news
        for ticker, name, query in overseas_stocks():
            news = fetch_us_news(ticker, name, query)
            try:
                n = deepseek_score(ticker, name, news, us=True)
            except Exception as e:
                n = 0
                print(f"  {ticker} {name} 打分失败: {type(e).__name__}: {e}")
            if n:
                (US_NEWS_DIR / f"{ticker}.json").write_text(
                    json.dumps(news, ensure_ascii=False, indent=1))
            agg = aggregate(news)
            print(f"{ticker} {name}: 新闻{len(news)}条, 新打分{n}条, {agg}")
            time.sleep(0.3)
    except Exception as e:
        print(f"美股情绪模块跳过: {type(e).__name__}: {e}")

    (NEWS_DIR / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n完成, 结果在 {NEWS_DIR}/")


if __name__ == "__main__":
    main()
