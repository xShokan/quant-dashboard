# -*- coding: utf-8 -*-
"""生成科创50分析的静态预览网站: site/index.html

- 个股面板: K线(蜡烛图) + 成交量 + 均线 + 预测概率条 + 新闻情绪
- 指数面板: 指数K线 + 择时净值对比 + 回撤 + 年度收益
- 预测: predict.py 的 walk-forward 样本外概率 (data/predict/)
- 情绪: sentiment.py 的 DeepSeek 新闻打分 (data/news/)
"""
import json
from pathlib import Path

import pandas as pd

from kc50_analysis import fetch_kc50, ma_timing, perf_stats
from cons_data import fetch_all, stock_stats, load_valuation

WINDOWS = (20, 60, 120)
PRED_DIR = Path("data/predict")
NEWS_DIR = Path("data/news")


def build_index_data():
    df = fetch_kc50()
    close = df["close"]
    ret = close.pct_change()

    curves = {"买入持有": (1 + ret.fillna(0)).cumprod()}
    stats = [perf_stats(ret, "买入持有")]
    for w in WINDOWS:
        s = ma_timing(close, w)
        curves[f"MA{w}择时"] = (1 + s).cumprod()
        stats.append(perf_stats(s, f"MA{w}择时"))

    bench = curves["买入持有"]
    dd = bench / bench.cummax() - 1
    yearly = ((1 + ret).resample("Y").prod() - 1)

    dates = [d.strftime("%Y-%m-%d") for d in close.index]
    return {
        "dates": dates,
        "range": f"{close.index[0].date()} ~ {close.index[-1].date()} ({len(close)} 个交易日)",
        "kline": [[round(float(r.open), 2), round(float(r.close), 2),
                   round(float(r.low), 2), round(float(r.high), 2)] for _, r in df.iterrows()],
        "vol": [int(v) for v in df["volume"]],
        "series": {name: [round(float(v), 4) for v in nav.values] for name, nav in curves.items()},
        "drawdown": [round(float(v) * 100, 2) for v in dd.values],
        "yearly": {"years": [str(d.year) for d in yearly.index],
                   "values": [round(float(v) * 100, 1) for v in yearly.values]},
        "stats": stats,
    }


def build_stock_data():
    pred_summary = {}
    if (PRED_DIR / "_summary.json").exists():
        pred_summary = json.loads((PRED_DIR / "_summary.json").read_text())

    stocks = fetch_all()
    out = []
    for code, v in stocks.items():
        df = v["df"]
        close = df["close"]
        ret = close.pct_change()
        val = load_valuation(code)

        # 估值/涨幅速查表
        info = {}
        for d, c in close.items():
            ds = d.strftime("%Y-%m-%d")
            pe, pb = val.get(ds, [None, None])
            pct = ret.get(d)
            info[ds] = [round(float(pct) * 100, 2) if pd.notna(pct) else None, pe, pb]

        # 预测概率
        probs, pred = [], pred_summary.get(code, {})
        pf = PRED_DIR / f"{code}.csv"
        if pf.exists():
            p = pd.read_csv(pf)
            probs = [[str(r["date"]), round(float(r["prob"]), 3)] for _, r in p.iterrows()]

        # 新闻情绪
        news = []
        nf = NEWS_DIR / f"{code}.json"
        if nf.exists():
            news = json.loads(nf.read_text())

        out.append({
            "code": code, "name": v["name"],
            "stats": stock_stats(df),
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "kline": [[round(float(r.open), 2), round(float(r.close), 2),
                       round(float(r.low), 2), round(float(r.high), 2)] for _, r in df.iterrows()],
            "vol": [int(x) for x in df["volume"]],
            "info": info,
            "probs": probs,
            "pred": {k: pred[k] for k in ("样本数", "模型胜率", "基准胜率", "最新概率") if k in pred},
            "news": news,
        })
    out.sort(key=lambda s: float(s["stats"]["累计收益"].strip("+%")) if s["stats"] else 0, reverse=True)
    return out


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>科创50 量化分析</title>
<script src="echarts.min.js"></script>
<style>
  :root { --bg:#0f1420; --card:#1a2233; --fg:#e8ecf4; --sub:#8b95ab; --line:#2a3550; }
  * { box-sizing: border-box; margin: 0; }
  body { background: var(--bg); color: var(--fg); font-family: -apple-system, "PingFang SC", sans-serif; padding: 24px; }
  .wrap { max-width: 1280px; margin: 0 auto; }
  h1 { font-size: 22px; } .sub { color: var(--sub); font-size: 13px; margin: 6px 0 20px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 12px; margin-bottom: 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  .card .k { color: var(--sub); font-size: 12px; } .card .v { font-size: 20px; margin-top: 6px; font-weight: 600; }
  .pos { color: #f6465d; } .neg { color: #0ecb81; }   /* A股配色: 红涨绿跌 */
  .panel { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px; margin-bottom: 20px; }
  .panel h2 { font-size: 15px; margin-bottom: 8px; }
  #nav { height: 400px; } #dd { height: 180px; } #yearly { height: 260px; } #ikline { height: 440px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  th { color: var(--sub); font-weight: 500; }
  .note { color: var(--sub); font-size: 12px; line-height: 1.8; }
  .tbl-wrap { max-height: 420px; overflow-y: auto; }
  #cons tbody tr { cursor: pointer; }
  #cons tbody tr:hover { background: #243050; }
  #cons tbody tr.sel { background: #2a3a5f; }
  #cons th[data-k] { cursor: pointer; user-select: none; }
  #cons th[data-k]:hover { color: #e8ecf4; }

  /* 个股面板: 侧栏 + 图表 */
  .stock-flex { display: flex; gap: 14px; }
  .side { width: 230px; flex-shrink: 0; display: flex; flex-direction: column; }
  .side input { background: #0f1420; border: 1px solid var(--line); color: var(--fg);
    border-radius: 8px; padding: 8px 10px; font-size: 13px; outline: none; margin-bottom: 8px; }
  .side input:focus { border-color: #5b8ff9; }
  .side-list { overflow-y: auto; height: 640px; border: 1px solid var(--line); border-radius: 8px; }
  .side-item { display: flex; justify-content: space-between; align-items: center;
    padding: 8px 10px; cursor: pointer; border-bottom: 1px solid #202a42; font-size: 13px; }
  .side-item:hover { background: #243050; }
  .side-item.sel { background: #2a3a5f; border-left: 3px solid #5b8ff9; }
  .side-item .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .side-item .cd { color: var(--sub); font-size: 11px; }
  .side-item .rt { font-size: 12px; margin-left: 6px; }
  .stock-main { flex: 1; min-width: 0; }
  .stock-title { font-size: 14px; color: var(--sub); margin-bottom: 6px; }
  .stock-title b { color: var(--fg); font-size: 17px; }
  .badge { display: inline-block; border-radius: 6px; padding: 2px 8px; font-size: 12px; margin-left: 6px; }
  .badge.pred { background: #22304f; color: #7ea6ff; }
  .badge.sent-pos { background: #3d1f2a; color: #f6465d; }
  .badge.sent-neg { background: #14332a; color: #0ecb81; }
  .badge.sent-mid { background: #2a3550; color: #8b95ab; }
  #stock { height: 640px; }
  .news-box { margin-top: 10px; border-top: 1px solid var(--line); padding-top: 10px; }
  .news-item { display: flex; gap: 8px; padding: 6px 4px; font-size: 13px; border-bottom: 1px dashed #202a42; align-items: baseline; }
  .news-item .t { color: var(--sub); font-size: 11px; white-space: nowrap; }
  .news-item a { color: var(--fg); text-decoration: none; flex: 1; }
  .news-item a:hover { color: #7ea6ff; }
  .news-item .lb { font-size: 11px; border-radius: 4px; padding: 1px 6px; white-space: nowrap; }
  .lb.利好 { background: #3d1f2a; color: #f6465d; } .lb.利空 { background: #14332a; color: #0ecb81; }
  .lb.中性 { background: #2a3550; color: #8b95ab; }
  .lb.imp { background: #22304f; color: #7ea6ff; }
  .news-item .rs { color: var(--sub); font-size: 11px; white-space: nowrap; }
  @media (max-width: 800px) { .stock-flex { flex-direction: column; } .side { width: 100%; } .side-list { height: 220px; } }

  .tt { font-size: 12px; line-height: 1.7; }
  .tt .d { color: #8b95ab; }
</style>
</head>
<body>
<div class="wrap">
  <h1>科创50 (000688) 量化分析</h1>
  <div class="sub" id="range"></div>

  <div class="cards" id="cards"></div>

  <div class="panel">
    <h2>成分股个股 <span class="note">(K线前复权; 悬浮查看涨幅/市盈率/市净率/预测概率)</span></h2>
    <div class="stock-flex">
      <div class="side">
        <input id="search" placeholder="搜索代码或名称...">
        <div class="side-list" id="side-list"></div>
      </div>
      <div class="stock-main">
        <div class="stock-title" id="stock-title"></div>
        <div id="stock"></div>
        <div class="news-box" id="news-box"></div>
      </div>
    </div>
  </div>

  <div class="panel"><h2>科创50指数 K线</h2><div id="ikline"></div></div>
  <div class="panel"><h2>择时净值对比: 买入持有 vs 均线择时</h2><div id="nav"></div></div>
  <div class="panel"><h2>买入持有回撤</h2><div id="dd"></div></div>
  <div class="panel"><h2>分年度收益 (买入持有)</h2><div id="yearly"></div></div>

  <div class="panel">
    <h2>指数择时绩效对比 (含万5单边成本, 信号次日执行)</h2>
    <table id="tbl"></table>
    <p class="note" style="margin-top:10px">
      说明: 均线择时信号 T 日收盘产生、T+1 执行, 避免前视偏差; 收益为指数口径, 未含现金端利息。
      MA20 为三参数中最优, 存在事后选参的偏差, 稳健结论仅为"趋势过滤可显著降低回撤"。
    </p>
  </div>

  <div class="panel">
    <h2>成分股统计 <span class="note">(点击表头排序, 点击行联动上方个股图; 当前名单, 有幸存者偏差)</span></h2>
    <div class="tbl-wrap"><table id="cons"></table></div>
  </div>

  <div class="panel">
    <h2>关于"预测"的说明</h2>
    <p class="note">
      个股页的概率来自逻辑回归模型(特征: 动量/均线偏离/波动率/量比, 目标: 未来5日方向),
      全部预测为 walk-forward 样本外结果, 无未来函数。<b>样本内外的胜率显示该模型仅略优于抛硬币、多数股票跑不赢"永远猜多数方向"的基准</b>——
      这与学术结论一致: 个股短期方向基本不可预测。概率可作为情绪/拥挤度的参考指标, 不构成买卖建议。
      量化长期期望收益应来自因子分散、趋势过滤与风控, 而非单点预测。
    </p>
  </div>
</div>

<script>
const DATA = __DATA__;
const STOCKS = __STOCKS__;
const SENT_OK = __SENT_OK__;

document.getElementById('range').textContent = '数据区间: ' + DATA.range + ' | 数据源: 新浪财经/中证指数/百度股市通/东方财富';

const cls = v => String(v).startsWith('+') ? 'pos' : (String(v).startsWith('-') ? 'neg' : '');
const fmtPct = v => v == null ? '—' : (v > 0 ? '+' : '') + v + '%';
const UP = '#f6465d', DOWN = '#0ecb81';

// ---------- 顶部卡片 ----------
const bh = DATA.stats[0];
const cards = [['累计收益', bh['累计收益']], ['年化收益', bh['年化收益']],
               ['年化波动', bh['年化波动']], ['夏普', bh['夏普']],
               ['最大回撤', bh['最大回撤']], ['日胜率', bh['日胜率']]];
document.getElementById('cards').innerHTML = cards.map(([k, v]) =>
  `<div class="card"><div class="k">${k} (买入持有)</div><div class="v ${cls(v)}">${v}</div></div>`
).join('');

const axisStyle = { axisLine: {lineStyle:{color:'#2a3550'}}, axisLabel:{color:'#8b95ab'} };
const colors = ['#5b8ff9', '#f6bd16', '#0ecb81', '#945fb9'];

function calcMA(n, kline) {
  return kline.map((_, i) => {
    if (i < n - 1) return null;
    let s = 0; for (let j = i - n + 1; j <= i; j++) s += kline[j][1];
    return +(s / n).toFixed(2);
  });
}

// ---------- 个股 K线 ----------
const stockChart = echarts.init(document.getElementById('stock'));
let selCode = null;

function stockTooltip(params) {
  const s = STOCKS.find(x => x.code === selCode);
  const k = params.find(p => p.seriesType === 'candlestick');
  if (!k) return '';
  const date = s.dates[k.dataIndex];
  const [o, c, l, h] = k.value.slice(1);
  const inf = s.info[date] || [null, null, null];
  const prob = s.probs.length && k.dataIndex < s.dates.length ?
    (s.probs.find(p => p[0] === date) || [])[1] : undefined;
  const pctC = inf[0] > 0 ? UP : (inf[0] < 0 ? DOWN : '#8b95ab');
  let html = `<div class="tt"><span class="d">${date}</span><br>` +
    `开 <b>${o}</b> 收 <b style="color:${c>=o?UP:DOWN}">${c}</b> 低 <b>${l}</b> 高 <b>${h}</b><br>` +
    `涨幅: <b style="color:${pctC}">${fmtPct(inf[0])}</b><br>` +
    `市盈率(TTM): <b>${inf[1] ?? '—'}</b> &nbsp; 市净率: <b>${inf[2] ?? '—'}</b>`;
  if (prob != null) html += `<br>模型5日上涨概率: <b style="color:#7ea6ff">${Math.round(prob*100)}%</b>`;
  return html + '</div>';
}

function renderStock(s) {
  if (!s) return;
  selCode = s.code;
  const p = s.pred || {};
  const predBadge = p['最新概率'] != null
    ? `<span class="badge pred">5日上涨概率 ${Math.round(p['最新概率']*100)}% | 历史胜率 ${Math.round((p['模型胜率']??0)*100)}% vs 基准 ${Math.round((p['基准胜率']??0)*100)}%</span>`
    : `<span class="badge pred">预测数据不足</span>`;
  document.getElementById('stock-title').innerHTML =
    `<b>${s.name} (${s.code})</b>${predBadge}<br>起始 ${s.stats['起始日期']} | ` +
    `累计 <span class="${cls(s.stats['累计收益'])}">${s.stats['累计收益']}</span> | ` +
    `年化 ${s.stats['年化收益']} | 最大回撤 ${s.stats['最大回撤']}`;

  const lastProb = s.probs.length ? s.probs[s.probs.length - 1] : null;
  const probSeries = s.dates.map(d => {
    const it = s.probs.find(x => x[0] === d);
    return it ? it[1] : null;
  });
  const peSeries = s.dates.map(d => (s.info[d] || [])[1] ?? null);
  const pbSeries = s.dates.map(d => (s.info[d] || [])[2] ?? null);

  const series = [
    { name: 'K线', type: 'candlestick', xAxisIndex: 0, yAxisIndex: 0, data: s.kline,
      itemStyle: { color: UP, color0: DOWN, borderColor: UP, borderColor0: DOWN } },
    ...[5, 10, 20, 60].map((n, i) => ({
      name: 'MA' + n, type: 'line', xAxisIndex: 0, yAxisIndex: 0,
      data: calcMA(n, s.kline), showSymbol: false, smooth: true,
      lineStyle: { width: 1 }, itemStyle: { color: ['#f6bd16','#945fb9','#5b8ff9','#8b95ab'][i] } })),
    { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
      data: s.vol.map((v, i) => ({ value: v, itemStyle: { color: s.kline[i][1] >= s.kline[i][0] ? UP : DOWN } })) },
    { name: '市盈率TTM', type: 'line', xAxisIndex: 2, yAxisIndex: 2, data: peSeries,
      showSymbol: false, connectNulls: true, lineStyle: { width: 1.2 }, itemStyle: { color: '#f6bd16' } },
    { name: '市净率', type: 'line', xAxisIndex: 2, yAxisIndex: 3, data: pbSeries,
      showSymbol: false, connectNulls: true, lineStyle: { width: 1.2 }, itemStyle: { color: '#945fb9' } },
    { name: '5日上涨概率', type: 'line', xAxisIndex: 3, yAxisIndex: 4, data: probSeries,
      showSymbol: false, lineStyle: { width: 1.2 }, itemStyle: { color: '#7ea6ff' },
      areaStyle: { opacity: 0.15 }, connectNulls: true,
      markLine: { silent: true, symbol: 'none', data: [{ yAxis: 0.5 }],
        lineStyle: { color: '#8b95ab', type: 'dashed', width: 1 }, label: { show: false } } },
  ];
  if (lastProb) {
    series[0].markPoint = { symbol: 'pin', symbolSize: 44,
      itemStyle: { color: '#7ea6ff' },
      label: { color: '#fff', fontSize: 10, formatter: Math.round(lastProb[1]*100) + '%' },
      data: [{ coord: [s.dates.length - 1, s.kline[s.kline.length - 1][3]] }] };
  }

  stockChart.setOption({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' }, formatter: stockTooltip },
    legend: { textStyle: {color:'#e8ecf4'}, data: ['K线','MA5','MA10','MA20','MA60','成交量','市盈率TTM','市净率','5日上涨概率'] },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    grid: [
      { left: 60, right: 55, top: 36, height: '44%' },
      { left: 60, right: 55, top: '56%', height: '8%' },
      { left: 60, right: 55, top: '68%', height: '12%' },
      { left: 60, right: 55, top: '84%', height: '11%' },
    ],
    xAxis: [0, 1, 2, 3].map(g => ({ type: 'category', gridIndex: g, data: s.dates, ...axisStyle,
      axisLabel: g === 3 ? axisStyle.axisLabel : { show: false } })),
    yAxis: [
      { gridIndex: 0, scale: true, ...axisStyle, splitLine:{lineStyle:{color:'#222c42'}} },
      { gridIndex: 1, ...axisStyle, axisLabel: { show: false }, splitLine: { show: false } },
      { gridIndex: 2, scale: true, ...axisStyle, splitLine:{lineStyle:{color:'#222c42'}}, axisLabel:{color:'#f6bd16'} },
      { gridIndex: 2, scale: true, position: 'right', ...axisStyle, splitLine:{show:false}, axisLabel:{color:'#945fb9'} },
      { gridIndex: 3, min: 0, max: 1, ...axisStyle, splitLine:{lineStyle:{color:'#222c42'}},
        axisLabel: { color: '#8b95ab', formatter: v => Math.round(v*100) + '%' } },
    ],
    dataZoom: [{ type: 'inside', xAxisIndex: [0,1,2,3] }, { type: 'slider', bottom: 4, xAxisIndex: [0,1,2,3] }],
    series,
  }, true);

  renderNews(s);
  document.querySelectorAll('.side-item').forEach(el =>
    el.classList.toggle('sel', el.dataset.code === s.code));
  document.querySelectorAll('#cons tbody tr').forEach(el =>
    el.classList.toggle('sel', el.dataset.code === s.code));
}

// ---------- 新闻情绪 ----------
function renderNews(s) {
  const box = document.getElementById('news-box');
  if (!s.news.length) { box.innerHTML = '<div class="note">暂无新闻数据</div>'; return; }
  const scored = s.news.filter(n => n.score != null);
  let head;
  if (!SENT_OK) {
    head = `<div class="note">最新新闻 (配置 DeepSeek API Key 后显示情绪分析: 环境变量 DEEPSEEK_API_KEY 或项目根目录 deepseek_key.txt, 然后重跑 sentiment.py 和 build_site.py)</div>`;
  } else if (scored.length) {
    const W = {'直接': 1, '间接': 0.5, '情绪': 0.25};
    const ws = scored.reduce((a, n) => a + (W[n.impact] ?? 1), 0);
    const avg = scored.reduce((a, n) => a + n.score * (W[n.impact] ?? 1), 0) / ws;
    const label = avg > 0.15 ? 'sent-pos">偏多' : (avg < -0.15 ? 'sent-neg">偏空' : 'sent-mid">中性');
    head = `<div class="note">新闻情绪 (DeepSeek 分析, 近${scored.length}条, 按影响类型加权): <span class="badge ${label}</span> 加权均分 ${avg.toFixed(2)}</div>`;
  } else {
    head = `<div class="note">最新新闻 (等待 DeepSeek 打分)</div>`;
  }
  box.innerHTML = head + s.news.slice(0, 10).map(n => {
    const lb = n.label ? `<span class="lb ${n.label}">${n.label}</span>` : '';
    const imp = n.impact ? `<span class="lb imp">${n.impact}</span>` : '';
    const rs = n.reason ? `<span class="rs">${n.reason}</span>` : '';
    return `<div class="news-item"><span class="t">${(n.time||'').slice(0,16)}</span>${lb}${imp}<a href="${n.link}" target="_blank">${n.title}</a>${rs}</div>`;
  }).join('');
}

// ---------- 侧栏选择器 ----------
function renderSide(filter = '') {
  const kw = filter.trim();
  document.getElementById('side-list').innerHTML = STOCKS
    .filter(s => !kw || s.code.includes(kw) || s.name.includes(kw))
    .map(s => `<div class="side-item ${s.code === selCode ? 'sel' : ''}" data-code="${s.code}">
      <span class="nm">${s.name}<br><span class="cd">${s.code}</span></span>
      <span class="rt ${cls(s.stats['累计收益'])}">${s.stats['累计收益']}</span></div>`)
    .join('');
  document.querySelectorAll('.side-item').forEach(el =>
    el.onclick = () => renderStock(STOCKS.find(s => s.code === el.dataset.code)));
}
document.getElementById('search').oninput = e => renderSide(e.target.value);

// ---------- 指数 K线 ----------
const iChart = echarts.init(document.getElementById('ikline'));
iChart.setOption({
  backgroundColor: 'transparent',
  tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
  legend: { textStyle: {color:'#e8ecf4'}, data: ['K线','MA5','MA10','MA20','MA60'] },
  axisPointer: { link: [{ xAxisIndex: 'all' }] },
  grid: [
    { left: 60, right: 20, top: 36, height: '62%' },
    { left: 60, right: 20, top: '76%', height: '14%' },
  ],
  xAxis: [0, 1].map(g => ({ type: 'category', gridIndex: g, data: DATA.dates, ...axisStyle,
    axisLabel: g === 1 ? axisStyle.axisLabel : { show: false } })),
  yAxis: [
    { gridIndex: 0, scale: true, ...axisStyle, splitLine:{lineStyle:{color:'#222c42'}} },
    { gridIndex: 1, ...axisStyle, axisLabel: { show: false }, splitLine: { show: false } },
  ],
  dataZoom: [{ type: 'inside', xAxisIndex: [0,1] }, { type: 'slider', bottom: 4, xAxisIndex: [0,1] }],
  series: [
    { name: 'K线', type: 'candlestick', xAxisIndex: 0, yAxisIndex: 0, data: DATA.kline,
      itemStyle: { color: UP, color0: DOWN, borderColor: UP, borderColor0: DOWN } },
    ...[5, 10, 20, 60].map((n, i) => ({
      name: 'MA' + n, type: 'line', xAxisIndex: 0, yAxisIndex: 0,
      data: calcMA(n, DATA.kline), showSymbol: false, smooth: true,
      lineStyle: { width: 1 }, itemStyle: { color: ['#f6bd16','#945fb9','#5b8ff9','#8b95ab'][i] } })),
    { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1,
      data: DATA.vol.map((v, i) => ({ value: v, itemStyle: { color: DATA.kline[i][1] >= DATA.kline[i][0] ? UP : DOWN } })) },
  ],
});

// ---------- 择时净值对比 ----------
const navChart = echarts.init(document.getElementById('nav'));
navChart.setOption({
  backgroundColor: 'transparent', color: colors,
  tooltip: { trigger: 'axis', formatter: params => {
    let html = `<div class="tt"><span class="d">${params[0].axisValue}</span><br>`;
    params.forEach(p => {
      const i = p.dataIndex, v = p.value, prev = i > 0 ? p.seriesName && navChart.getOption().series.find(s => s.name === p.seriesName).data[i-1] : null;
      const pct = prev ? (v / prev - 1) * 100 : null;
      const c = pct > 0 ? UP : (pct < 0 ? DOWN : '#8b95ab');
      html += `${p.marker}${p.seriesName}: <b>${v}</b> <span style="color:${c}">${fmtPct(pct)}</span><br>`;
    });
    return html + '</div>';
  }},
  legend: { textStyle: {color:'#e8ecf4'} },
  grid: { left: 55, right: 20, top: 40, bottom: 60 },
  xAxis: { type: 'category', data: DATA.dates, ...axisStyle },
  yAxis: { type: 'value', scale: true, ...axisStyle, splitLine:{lineStyle:{color:'#222c42'}} },
  dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 10 }],
  series: Object.entries(DATA.series).map(([name, vals]) => ({
    name, type: 'line', data: vals, showSymbol: false, lineStyle: { width: 1.5 }
  }))
});

// ---------- 回撤 ----------
const ddChart = echarts.init(document.getElementById('dd'));
ddChart.setOption({
  backgroundColor: 'transparent',
  tooltip: { trigger: 'axis', valueFormatter: v => v + '%' },
  grid: { left: 55, right: 20, top: 10, bottom: 25 },
  xAxis: { type: 'category', data: DATA.dates, ...axisStyle },
  yAxis: { type: 'value', ...axisStyle, splitLine:{lineStyle:{color:'#222c42'}}, axisLabel:{color:'#8b95ab', formatter:'{value}%'} },
  series: [{ type: 'line', data: DATA.drawdown, showSymbol: false,
             areaStyle: {opacity: 0.4}, lineStyle: {width: 1}, itemStyle: {color: UP} }]
});

// ---------- 年度收益 ----------
const yChart = echarts.init(document.getElementById('yearly'));
yChart.setOption({
  backgroundColor: 'transparent',
  tooltip: { trigger: 'axis', valueFormatter: v => v + '%' },
  grid: { left: 55, right: 20, top: 20, bottom: 25 },
  xAxis: { type: 'category', data: DATA.yearly.years, ...axisStyle },
  yAxis: { type: 'value', ...axisStyle, splitLine:{lineStyle:{color:'#222c42'}}, axisLabel:{color:'#8b95ab', formatter:'{value}%'} },
  series: [{ type: 'bar', data: DATA.yearly.values.map(v => ({
      value: v, itemStyle: { color: v >= 0 ? UP : DOWN }
    })), label: { show: true, position: 'top', color: '#8b95ab', formatter: '{c}%' } }]
});

// ---------- 指数择时绩效表 ----------
const keys = Object.keys(DATA.stats[0]);
document.getElementById('tbl').innerHTML =
  '<tr>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>' +
  DATA.stats.map(r => '<tr>' + keys.map(k => `<td>${r[k]}</td>`).join('') + '</tr>').join('');

// ---------- 成分股统计表 ----------
const pct2num = s => parseFloat(String(s).replace('%','').replace('+','')) || 0;
const COLS = [
  ['code', '代码', null], ['name', '名称', null], ['起始日期', '起始日期', null],
  ['累计收益', '累计收益', pct2num], ['年化收益', '年化收益', pct2num],
  ['年化波动', '年化波动', pct2num], ['夏普', '夏普', parseFloat],
  ['最大回撤', '最大回撤', pct2num], ['日胜率', '日胜率', pct2num],
];
let sortKey = '累计收益', sortAsc = false;

function renderCons() {
  const num = COLS.find(c => c[0] === sortKey)[2];
  const rows = [...STOCKS].sort((a, b) => {
    const va = num ? num(a.stats[sortKey] ?? a[sortKey]) : (a.stats[sortKey] ?? a[sortKey]);
    const vb = num ? num(b.stats[sortKey] ?? b[sortKey]) : (b.stats[sortKey] ?? b[sortKey]);
    return (va > vb ? 1 : va < vb ? -1 : 0) * (sortAsc ? 1 : -1);
  });
  document.getElementById('cons').innerHTML =
    '<thead><tr>' + COLS.map(([k, label]) =>
      `<th data-k="${k}">${label}${k === sortKey ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>`).join('') + '</tr></thead>' +
    '<tbody>' + rows.map(s => {
      const cells = COLS.map(([k]) => {
        const v = k === 'code' || k === 'name' ? s[k] : (s.stats[k] ?? '-');
        return `<td class="${cls(v)}">${v}</td>`;
      }).join('');
      return `<tr data-code="${s.code}" class="${s.code === selCode ? 'sel' : ''}">${cells}</tr>`;
    }).join('') + '</tbody>';
  document.querySelectorAll('#cons th[data-k]').forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (k === sortKey) sortAsc = !sortAsc; else { sortKey = k; sortAsc = false; }
    renderCons();
  });
  document.querySelectorAll('#cons tbody tr').forEach(tr => tr.onclick = () => {
    renderStock(STOCKS.find(s => s.code === tr.dataset.code));
    renderCons();
    document.getElementById('stock').scrollIntoView({behavior: 'smooth', block: 'center'});
  });
}

renderSide();
renderCons();
renderStock(STOCKS[0]);

window.addEventListener('resize', () => [iChart, navChart, ddChart, yChart, stockChart].forEach(c => c.resize()));
</script>
</body>
</html>
"""


def main():
    data = build_index_data()
    stocks = build_stock_data()
    sent_ok = any(
        any(n.get("score") is not None for n in s["news"]) for s in stocks
    )
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__STOCKS__", json.dumps(stocks, ensure_ascii=False))
            .replace("__SENT_OK__", "true" if sent_ok else "false"))
    with open("site/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成 site/index.html (含 {len(stocks)} 只成分股, 情绪打分: {'有' if sent_ok else '无'})")


if __name__ == "__main__":
    main()
