# -*- coding: utf-8 -*-
"""
บอทส่งซิกทอง + Forex เข้า Discord (ฟรี รันบน GitHub Actions)

โหมดการรัน (ตัวแปร RUN_MODE):
  live      สแกนตลาดแล้วส่งซิกใหม่เข้า Discord (ค่าเริ่มต้น)
  test      ส่งข้อความทดสอบว่า Discord เชื่อมต่อได้
  backtest  ทดสอบกลยุทธ์ย้อนหลัง แล้วส่งสรุปวินเรทเข้า Discord
  pa        ทดสอบกลยุทธ์ Price Action บน 1H (Breakout, Sweep, Retest) แยก train/test + จำลองพอร์ต
  research  ทดสอบหลายชุดตัวกรอง แยก train/test เพื่อหาชุดที่ไม่ใช่แค่โชคดีย้อนหลัง
  pa        ทดสอบกลยุทธ์ Price Action บน 1H (Breakout/Retest/Sweep) แยก train/test

หลักการ: เทรดตามเทรนด์ใหญ่ รอราคาย่อมาโซนมูลค่า (EMA21) แล้วมีแท่งเทียนยืนยัน
ต้องผ่านหลายเงื่อนไขพร้อมกัน (RSI, MACD, ADX, EMA200) ใช้เฉพาะแท่งที่ปิดแล้ว (ไม่รีเพนต์)
SL คิดจาก ATR + swing, TP ตาม Risk:Reward
"""
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "").strip()
RUN_MODE = os.environ.get("RUN_MODE", "live").strip().lower()
STATE_FILE = "state.json"

# ชื่อที่แสดง -> สัญลักษณ์ใน Yahoo Finance
# (ทองใช้ GC=F ฟิวเจอร์ส ราคาใกล้เคียง XAUUSD แต่อาจต่างจากโบรกเกอร์เล็กน้อย)
SYMBOLS = {
    "XAUUSD": "GC=F",
    "GBPJPY": "GBPJPY=X",
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "EURJPY": "EURJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}

TF_MIN = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}

# need = จำนวนเงื่อนไขเสริม (จาก 4 ข้อ) ที่ต้องผ่าน, sweep = ต้องมีการกวาดสภาพคล่อง
MODES = {
    "SNIPER": dict(entry="15m", trend="4h", t2="1d", need=4, sweep=True, rr1=1.5, rr2=3.0, session=True),
    "DAY": dict(entry="15m", trend="1h", t2="4h", need=3, sweep=False, rr1=1.5, rr2=2.5, session=True),
    "SWING": dict(entry="4h", trend="1d", t2=None, need=3, sweep=False, rr1=2.0, rr2=3.5, session=False),
}
MODE_TH = {"SNIPER": "Sniper", "DAY": "Day Trade", "SWING": "Swing"}

WARMUP = 210


# ----------------------------------------------------------------- ข้อมูล
def _clean(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


def download(ticker, interval, period):
    import yfinance as yf
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         progress=False, auto_adjust=False, threads=False)
    except Exception as e:  # noqa
        print("download error", ticker, interval, e)
        return None
    return _clean(df)


def resample(df, rule):
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def drop_open_bar(df, tf):
    """ตัดแท่งที่ยังไม่ปิดทิ้ง เพื่อไม่ให้สัญญาณรีเพนต์"""
    now = datetime.now(timezone.utc)
    end = df.index + pd.Timedelta(minutes=TF_MIN[tf])
    return df[end <= now]


def get_frames(ticker, deep=False, with15=True):
    d15 = download(ticker, "15m", "60d") if with15 else None
    h1 = download(ticker, "1h", "730d" if deep else "120d")
    d1 = download(ticker, "1d", "5y" if deep else "2y")
    if (with15 and d15 is None) or h1 is None or d1 is None:
        return None
    h1c = drop_open_bar(h1, "1h")
    return {
        "15m": drop_open_bar(d15, "15m") if with15 else h1c.iloc[0:0],
        "1h": h1c,
        "4h": drop_open_bar(resample(h1c, "4h"), "4h"),
        "1d": drop_open_bar(d1, "1d"),
    }


# -------------------------------------------------------------- อินดิเคเตอร์
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1 / n, adjust=False).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_ind(df):
    d = df.copy()
    c, h, l = d["close"], d["high"], d["low"]
    d["ema21"], d["ema50"], d["ema200"] = ema(c, 21), ema(c, 50), ema(c, 200)
    d["rsi"] = rsi(c)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    d["pdi"] = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / d["atr"]
    d["mdi"] = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / d["atr"]
    dx = 100 * (d["pdi"] - d["mdi"]).abs() / (d["pdi"] + d["mdi"]).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    macd = ema(c, 12) - ema(c, 26)
    d["hist"] = macd - ema(macd, 9)
    d["body"] = (c - d["open"]).abs()
    d["uw"] = h - np.maximum(c, d["open"])
    d["lw"] = np.minimum(c, d["open"]) - l
    return d.fillna(0.0)


def trend_series(df):
    """+1 ขาขึ้น / -1 ขาลง / 0 ไม่ชัด (ใช้ EMA50, EMA200)"""
    e50, e200 = ema(df["close"], 50), ema(df["close"], 200)
    bull = (df["close"] > e50) & (e50 > e200) & (e50 > e50.shift(5))
    bear = (df["close"] < e50) & (e50 < e200) & (e50 < e50.shift(5))
    t = pd.Series(0, index=df.index)
    t[bull] = 1
    t[bear] = -1
    t.iloc[:WARMUP] = 0
    return t


def htf_at(trend, tf, times):
    """เทรนด์ของกรอบใหญ่ ณ เวลาที่แท่งเล็กปิด (ใช้เฉพาะแท่งใหญ่ที่ปิดแล้ว)"""
    t = pd.Series(trend.values, index=trend.index + pd.Timedelta(minutes=TF_MIN[tf]))
    return t.reindex(times, method="ffill").fillna(0).astype(int)


# --------------------------------------------------------------- กลยุทธ์
def build(d):
    """สร้างข้อมูลมุมมอง 'ฝั่งซื้อ' และ 'ฝั่งขาย' (ฝั่งขายคือกลับด้านราคา) ให้ใช้ตรรกะเดียวกัน"""
    g = lambda k: d[k].to_numpy(float)  # noqa
    up = dict(o=g("open"), h=g("high"), l=g("low"), c=g("close"), ema21=g("ema21"),
              ema50=g("ema50"), ema200=g("ema200"), rsi=g("rsi"), atr=g("atr"),
              adx=g("adx"), pdi=g("pdi"), mdi=g("mdi"), hist=g("hist"),
              body=g("body"), uw=g("uw"), lw=g("lw"))
    dn = dict(o=-g("open"), h=-g("low"), l=-g("high"), c=-g("close"), ema21=-g("ema21"),
              ema50=-g("ema50"), ema200=-g("ema200"), rsi=100 - g("rsi"), atr=g("atr"),
              adx=g("adx"), pdi=g("mdi"), mdi=g("pdi"), hist=-g("hist"),
              body=g("body"), uw=g("lw"), lw=g("uw"))
    return {1: up, -1: dn}


def check(X, i, cfg):
    """ตรวจว่าแท่ง i มีสัญญาณฝั่งซื้อ (ในมุมมอง X) หรือไม่ คืน dict หรือ None"""
    if i < WARMUP:
        return None
    c, o, h, l = X["c"][i], X["o"][i], X["h"][i], X["l"][i]
    atr = X["atr"][i]
    if not atr > 0:
        return None

    # บังคับ 1: ย่อมาแตะโซนมูลค่า EMA21 ภายใน 6 แท่ง และยังยืนเหนือ EMA50
    zone = cfg.get("zone", "ema21")
    pull = X["l"][i - 5:i + 1].min() <= X[zone][i] + 0.25 * atr and c > X["ema50"][i]
    # บังคับ 2: แท่งยืนยัน (ปิดเหนือ high แท่งก่อน หรือ pin bar หางล่างยาว)
    rng = h - l
    pin = rng > 0 and X["lw"][i] >= 2 * max(X["body"][i], 1e-12) and (c - l) / rng >= 0.66
    trig = (c > o and c > X["h"][i - 1]) or pin
    if not (pull and trig):
        return None
    adx_min = cfg.get("adx_min", 0)
    if adx_min and not (X["adx"][i] >= adx_min and X["pdi"][i] > X["mdi"][i]):
        return None

    reasons = ["ย่อมาแตะโซน EMA21 ตามเทรนด์ใหญ่", "แท่งเทียนยืนยันการกลับตัว"]
    pts = []
    if 40 <= X["rsi"][i] <= 65 and X["rsi"][i] > X["rsi"][i - 1]:
        pts.append("RSI หันตามทิศทาง (ไม่ตึงเกินไป)")
    if X["hist"][i] > X["hist"][i - 1]:
        pts.append("MACD โมเมนตัมเริ่มเร่งตามทิศทาง")
    if X["adx"][i] >= 20 and X["pdi"][i] > X["mdi"][i]:
        pts.append("ADX ≥ 20 เทรนด์มีแรง")
    if c > X["ema200"][i]:
        pts.append("ราคาอยู่ฝั่งเดียวกับ EMA200")
    if len(pts) < cfg["need"]:
        return None
    reasons += pts

    sweep = False
    if cfg["sweep"]:
        ref = X["l"][i - 24:i - 3].min()
        sweep = X["l"][i - 3:i + 1].min() < ref and c > ref
        if not sweep:
            return None
        reasons.append("กวาดสภาพคล่องแล้วดึงกลับ (Sniper)")

    entry = c
    low_ref = X["l"][i - 3:i + 1].min() if sweep else X["l"][i - 4:i + 1].min()
    sl = low_ref - 0.2 * atr
    risk = entry - sl
    if risk < 0.8 * atr:
        sl = entry - 0.8 * atr
        risk = 0.8 * atr
    if risk > 2.5 * atr:  # SL กว้างเกิน ไม่คุ้ม
        return None
    return dict(entry=entry, sl=sl, risk=risk,
                tp1=entry + cfg["rr1"] * risk, tp2=entry + cfg["rr2"] * risk,
                reasons=reasons, score=2 + len(pts) + (1 if sweep else 0),
                max_score=6 + (1 if cfg["sweep"] else 0))


def in_session(ts, hours=(7, 20)):
    """ช่วงลอนดอน+นิวยอร์ก (UTC 07-20) เลี่ยงเช้ามืดสภาพคล่องต่ำ และเย็นวันศุกร์"""
    if ts.weekday() >= 5:
        return False
    if ts.weekday() == 4 and ts.hour >= 18:
        return False
    return hours[0] <= ts.hour < hours[1]


def prep(fr):
    ind = {tf: add_ind(df) for tf, df in fr.items()}
    trd = {tf: trend_series(df) for tf, df in fr.items()}
    views = {tf: build(d) for tf, d in ind.items()}
    return trd, views


def mode_inputs(fr, trd, cfg):
    ent = fr[cfg["entry"]]
    times = ent.index + pd.Timedelta(minutes=TF_MIN[cfg["entry"]])
    tarr = htf_at(trd[cfg["trend"]], cfg["trend"], times).to_numpy()
    t2 = cfg.get("trend2")
    if t2:  # ต้องเทรนด์ตรงกันทั้งสองกรอบใหญ่
        tarr2 = htf_at(trd[t2], t2, times).to_numpy()
        tarr = np.where(tarr == tarr2, tarr, 0)
    return times, tarr


# ------------------------------------------------------------------ Discord
def send(embed=None, content=None):
    if not WEBHOOK:
        print("ไม่พบ DISCORD_WEBHOOK (ยังไม่ได้ตั้ง Secret)")
        return False
    import requests
    payload = {"username": "Signal Bot"}
    if content:
        payload["content"] = content
    if embed:
        payload["embeds"] = [embed]
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=20)
        print("discord status", r.status_code)
        return r.status_code in (200, 204)
    except Exception as e:  # noqa
        print("discord error", type(e).__name__)
        return False


def fmt(name, x):
    if name == "XAUUSD":
        return f"{x:,.2f}"
    if "JPY" in name:
        return f"{x:.3f}"
    return f"{x:.5f}"


def signal_embed(name, mode, s, r, close_time):
    cfg = MODES[mode]
    side = "BUY 🟢" if s == 1 else "SELL 🔴"
    p = lambda v: fmt(name, s * v)  # noqa  (แปลงกลับจากมุมมองฝั่งขาย)
    return {
        "title": f"{side}  {name}  |  {MODE_TH[mode]}",
        "color": 0x2ECC71 if s == 1 else 0xE74C3C,
        "fields": [
            {"name": "Entry (ราคาปิดแท่งล่าสุด)", "value": p(r["entry"]), "inline": True},
            {"name": "Stop Loss", "value": p(r["sl"]), "inline": True},
            {"name": "\u200b", "value": "\u200b", "inline": True},
            {"name": f"TP1 ({cfg['rr1']}R)", "value": p(r["tp1"]), "inline": True},
            {"name": f"TP2 ({cfg['rr2']}R)", "value": p(r["tp2"]), "inline": True},
            {"name": "คะแนนเงื่อนไข", "value": f"{r['score']}/{r['max_score']}", "inline": True},
            {"name": "เหตุผล", "value": "\n".join("• " + x for x in r["reasons"])},
            {"name": "แนวทาง", "value": "เสี่ยงไม่เกิน 1% ของพอร์ตต่อไม้ • ปิดครึ่งที่ TP1 แล้วเลื่อน SL มาจุดเข้า • เลี่ยงช่วงข่าวแรง"},
        ],
        "footer": {"text": "สัญญาณช่วยคัดกรองเท่านั้น ไม่ใช่คำแนะนำการลงทุน ผลไม่รับประกัน"},
        "timestamp": close_time.isoformat(),
    }


# --------------------------------------------------------------------- live
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:  # noqa
        return []


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state[-600:], f, ensure_ascii=False)


def scan(name, fr, now=None):
    now = now or datetime.now(timezone.utc)
    trd, views = prep(fr)
    found, seen = [], set()
    for mode in ("SNIPER", "DAY", "SWING"):  # Sniper มาก่อน เพื่อไม่ส่งซ้ำกับ Day
        cfg = MODES[mode]
        ent = fr[cfg["entry"]]
        if len(ent) < WARMUP + 50:
            continue
        times, tarr = mode_inputs(fr, trd, cfg)
        i = len(ent) - 1
        ct = times[i]
        age = now - ct.to_pydatetime()
        if age.total_seconds() > (2 * TF_MIN[cfg["entry"]] + 5) * 60:
            continue  # ข้อมูลเก่า/ตลาดปิด
        if cfg["session"] and not in_session(ct):
            continue
        s = int(tarr[i])
        if s == 0:
            continue
        r = check(views[cfg["entry"]][s], i, cfg)
        if not r:
            continue
        if cfg["entry"] == "15m":
            if (ct, s) in seen:
                continue
            seen.add((ct, s))
        found.append((mode, s, r, ct))
    return found


def run_live():
    state = load_state()
    sent = 0
    for name, ticker in SYMBOLS.items():
        try:
            fr = get_frames(ticker)
            if not fr:
                print(name, "ไม่มีข้อมูล")
                continue
            sigs = scan(name, fr)
        except Exception as e:  # noqa
            print("error", name, type(e).__name__, e)
            continue
        for mode, s, r, ct in sigs:
            key = f"{name}|{mode}|{ct:%Y%m%d%H%M}|{s}"
            if key in state:
                continue
            if send(embed=signal_embed(name, mode, s, r, ct)):
                state.append(key)
                sent += 1
    save_state(state)
    print("ส่งซิกใหม่", sent, "รายการ")


# ----------------------------------------------------------------- backtest
def simulate(X, i, r, rr, n, cost=0.1):
    """จำลองเทรด: ถ้าแท่งเดียวชนทั้ง SL และ TP ถือว่า SL ก่อน (เข้มงวด) หัก cost R ต่อไม้"""
    entry, sl, risk = r["entry"], r["sl"], r["risk"]
    tp = entry + rr * risk
    end = min(n - 1, i + 100)
    for j in range(i + 1, end + 1):
        if X["l"][j] <= sl:
            return -1.0 - cost, j
        if X["h"][j] >= tp:
            return rr - cost, j
    return (X["c"][end] - entry) / risk - cost, end


def backtest_mode(fr, trd, views, cfg):
    ent = fr[cfg["entry"]]
    out = {"tp1": [], "tp2": []}
    if len(ent) < WARMUP + 50:
        return out
    times, tarr = mode_inputs(fr, trd, cfg)
    V = views[cfg["entry"]]
    n, i = len(ent), WARMUP
    while i < n - 1:
        s = int(tarr[i])
        if s != 0 and (not cfg["session"] or in_session(times[i])):
            r = check(V[s], i, cfg)
            if r:
                r1, j1 = simulate(V[s], i, r, cfg["rr1"], n)
                r2, j2 = simulate(V[s], i, r, cfg["rr2"], n)
                out["tp1"].append(r1)
                out["tp2"].append(r2)
                i = max(j1, j2) + 1
                continue
        i += 1
    return out


def stats(rs):
    if not rs:
        return None
    a = np.array(rs)
    gp, gl = a[a > 0].sum(), -a[a < 0].sum()
    pf = gp / gl if gl > 0 else float("inf")
    eq = np.cumsum(a)
    dd = (np.maximum.accumulate(eq) - eq).max()
    return len(a), (a > 0).mean() * 100, a.mean(), pf, dd


def stat_line(label, rs):
    s = stats(rs)
    if not s:
        return f"{label}: ไม่มีเทรด"
    n, wr, ex, pf, dd = s
    return f"{label}: {n} ไม้ | ชนะ {wr:.0f}% | เฉลี่ย {ex:+.2f}R | PF {pf:.2f} | DD {dd:.1f}R"


def run_backtest():
    agg = {}
    for name, ticker in SYMBOLS.items():
        fr = get_frames(ticker, deep=True)
        if not fr:
            print(name, "ไม่มีข้อมูล")
            continue
        trd, views = prep(fr)
        for mode, cfg in MODES.items():
            res = backtest_mode(fr, trd, views, cfg)
            for k, v in res.items():
                agg.setdefault((mode, k), []).extend(v)
                print(stat_line(f"{name} {mode} {k}", v))
    lines = ["**สรุป Backtest** (หักต้นทุน 0.1R/ไม้, ถ้าชน SL/TP แท่งเดียวกันนับ SL ก่อน)"]
    for mode, cfg in MODES.items():
        lines.append(f"\n**{MODE_TH[mode]}** ({cfg['entry']} / เทรนด์ {cfg['trend']})")
        lines.append(stat_line(f"ออกที่ TP1 {cfg['rr1']}R", agg.get((mode, 'tp1'), [])))
        lines.append(stat_line(f"ออกที่ TP2 {cfg['rr2']}R", agg.get((mode, 'tp2'), [])))
    lines.append("\nข้อมูลย้อนหลัง: 15m ~60 วัน, 4h ~2 ปี, ตัวอย่างน้อยอาจคลาดเคลื่อนได้ ผลในอดีตไม่รับประกันอนาคต")
    msg = "\n".join(lines)
    print(msg)
    send(content=msg[:1990])



# ----------------------------------------------------------------- research
VARIANTS = {
    "base": {},
    "adx25": {"adx_min": 25},
    "dual": {"dual": True},
    "deep": {"zone": "ema50"},
    "adx25+dual": {"adx_min": 25, "dual": True},
    "adx25+dual+overlap": {"adx_min": 25, "dual": True, "hours": (12, 17)},
}
XRR = (1.5, 3.0)
TRAIN_FRAC = 0.6


def resolve(mode, var):
    cfg = dict(MODES[mode])
    cfg.update({k: v for k, v in var.items() if k != "dual"})
    if var.get("dual"):
        cfg["trend2"] = MODES[mode].get("t2")
    return cfg


def short(rs):
    st = stats(rs)
    return "ไม่มีเทรด" if not st else f"{st[0]} ไม้ PF {st[3]:.2f} ({st[2]:+.2f}R)"


def send_long(text):
    chunk = ""
    for line in text.split("\n"):
        if len(chunk) + len(line) + 1 > 1900:
            send(content=chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        send(content=chunk)


def run_research():
    rows = {}  # (mode, variant, rr) -> list of (symbol, is_test, R)
    for name, ticker in SYMBOLS.items():
        fr = get_frames(ticker, deep=True)
        if not fr:
            print(name, "ไม่มีข้อมูล")
            continue
        trd, views = prep(fr)
        for mode in MODES:
            for vname, var in VARIANTS.items():
                cfg = resolve(mode, var)
                if var.get("dual") and not cfg.get("trend2"):
                    continue  # โหมดนี้ไม่มีกรอบที่สอง
                ent = fr[cfg["entry"]]
                n = len(ent)
                if n < WARMUP + 50:
                    continue
                times, tarr = mode_inputs(fr, trd, cfg)
                V = views[cfg["entry"]]
                split = int(n * TRAIN_FRAC)
                hours = cfg.get("hours", (7, 20))
                for rr in XRR:
                    out = rows.setdefault((mode, vname, rr), [])
                    i = WARMUP
                    while i < n - 1:
                        s_ = int(tarr[i])
                        if s_ and (not cfg["session"] or in_session(times[i], hours)):
                            r = check(V[s_], i, cfg)
                            if r:
                                R, j = simulate(V[s_], i, r, rr, n)
                                out.append((name, i >= split, R))
                                i = j + 1
                                continue
                        i += 1

    for mode in MODES:
        lines = [f"**Research: {MODE_TH[mode]}**  (train 60% เก่า | test 40% ล่าสุด ที่ไม่เคยใช้ปรับ)"]
        best, best_pf = None, 0
        passed = []
        for (m, vname, rr), tr in rows.items():
            if m != mode:
                continue
            a = [x[2] for x in tr if not x[1]]
            b = [x[2] for x in tr if x[1]]
            lines.append(f"{vname} TP{rr}R\n  train: {short(a)}\n  test: {short(b)}")
            sa, sb = stats(a), stats(b)
            if sa and sa[0] >= 40 and sa[3] > best_pf:
                best, best_pf = (m, vname, rr), sa[3]
            if sa and sb and sa[0] >= 40 and sb[0] >= 30 and sa[3] >= 1.15 and sb[3] >= 1.15:
                passed.append(f"{vname} TP{rr}R")
        lines.append("")
        lines.append("ผ่านเกณฑ์ (PF ≥ 1.15 ทั้ง train และ test): " + (", ".join(passed) if passed else "ไม่มี"))
        if best:
            lines.append(f"\nรายคู่ของชุดที่ train ดีสุด ({best[1]} TP{best[2]}R):")
            tr = rows[best]
            for sym in SYMBOLS:
                a = [x[2] for x in tr if x[0] == sym and not x[1]]
                b = [x[2] for x in tr if x[0] == sym and x[1]]
                lines.append(f"{sym}: train {short(a)} | test {short(b)}")
        text = "\n".join(lines)
        print(text)
        send_long(text)
    send(content="หมายเหตุ: ไม่ควรเชื่อชุดที่ดีแค่ train หรือมีไม้น้อย ให้เชื่อเฉพาะชุดที่ผ่านทั้ง train และ test และยังต้องทดสอบบนเดโมต่อ")


# ------------------------------------------------- price action (1H) research
def pivots(h, l, k=3):
    """swing high/low แบบ fractal ยืนยันหลังผ่านไป k แท่ง (ไม่มองอนาคต)
    คืนค่า: ระดับ swing high/low ล่าสุดที่ยืนยันแล้ว ณ แต่ละแท่ง"""
    n = len(h)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    last_h = last_l = np.nan
    for i in range(n):
        j = i - k
        if j >= k:
            if h[j] == h[j - k:i + 1].max():
                last_h = h[j]
            if l[j] == l[j - k:i + 1].min():
                last_l = l[j]
        sh[i], sl[i] = last_h, last_l
    return sh, sl


def pa_finish(X, i, sl, atr):
    c = X["c"][i]
    risk = c - sl
    if risk < 1.0 * atr:
        sl = c - 1.0 * atr
        risk = 1.0 * atr
    if risk > 2.5 * atr:  # SL กว้างเกินไป ไม่คุ้ม
        return None
    return dict(entry=c, sl=sl, risk=risk)


def pa_check(X, i, kind):
    """สัญญาณ Price Action ฝั่งซื้อ (ฝั่งขายใช้มุมมองกลับด้าน) ใช้เฉพาะแท่งที่ปิดแล้ว"""
    if i < WARMUP:
        return None
    c, o, h, l = X["c"][i], X["o"][i], X["h"][i], X["l"][i]
    atr = X["atr"][i]
    rng = h - l
    if not (atr > 0 and rng > 0):
        return None

    if kind in ("BRK20", "BRK40"):
        # เบรกกรอบสูงสุด N แท่ง ด้วยแท่งแข็งแรง ADX ผ่าน และไม่ไล่ราคาไกลเกิน
        n = 20 if kind == "BRK20" else 40
        top = X["h"][i - n:i].max()
        if not (c > top and c > o and (c - o) >= 0.5 * rng):
            return None
        if X["adx"][i] < 20 or c - top > 1.5 * atr:
            return None
        return pa_finish(X, i, X["l"][i - 4:i + 1].min() - 0.2 * atr, atr)

    if kind == "RETEST":
        # เบรก swing high แล้วย่อกลับมาทดสอบแนวเดิม (แนวต้านกลายเป็นแนวรับ) + แท่งเขียวยืนยัน
        k = 12
        lvl = X["sh"][i - k]
        if np.isnan(lvl):
            return None
        if not (X["c"][i - k + 1:i].max() > lvl):
            return None
        if not (l <= lvl + 0.3 * atr and c > lvl and c > o):
            return None
        if X["l"][i - k + 1:i + 1].min() < lvl - 1.0 * atr:
            return None
        return pa_finish(X, i, min(l, lvl) - 0.3 * atr, atr)

    if kind == "SWEEP":
        # กวาด stop ใต้ swing low แล้วปิดกลับเหนือแนว (stop hunt) ตามเทรนด์ขึ้น
        lvl = X["sl"][i - 1]
        if np.isnan(lvl):
            return None
        if not (l < lvl and c > lvl and c > o):
            return None
        if (c - l) / rng < 0.6 or (lvl - l) > 1.5 * atr:
            return None
        return pa_finish(X, i, l - 0.2 * atr, atr)
    return None


def simulate_trail(X, i, r, k, n, cost=0.1, maxbars=200):
    """ออกด้วย trailing stop (ตาม high สูงสุด - k*ATR) ปล่อยกำไรวิ่ง"""
    entry, sl, risk = r["entry"], r["sl"], r["risk"]
    hh = X["h"][i]
    end = min(n - 1, i + maxbars)
    for j in range(i + 1, end + 1):
        if X["l"][j] <= sl:
            return (sl - entry) / risk - cost, j
        hh = max(hh, X["h"][j])
        sl = max(sl, hh - k * X["atr"][j])
    return (X["c"][end] - entry) / risk - cost, end


PA_KINDS = ("BRK20", "BRK40", "RETEST", "SWEEP")
PA_TRENDS = ("4h", "4h+1d")
PA_EXITS = (("rr", 2.0), ("rr", 3.0), ("trail", 3.0))


def monte_carlo(rs, risk, target=10.0, floor=0.2, trades=200, paths=3000, seed=1):
    rng = np.random.default_rng(seed)
    rs = np.array(rs)
    hit = ruin = 0
    finals = []
    for _ in range(paths):
        eq = 1.0
        for r in rng.choice(rs, trades):
            eq *= 1 + risk * r
            if eq >= target:
                hit += 1
                break
            if eq <= floor:
                ruin += 1
                break
        finals.append(eq)
    return hit / paths, ruin / paths, float(np.median(finals))


def run_pa_research():
    rows = {}  # (kind, trend, exit_label) -> list of (symbol, is_test, R)
    for name, ticker in SYMBOLS.items():
        fr = get_frames(ticker, deep=True)
        if not fr:
            print(name, "ไม่มีข้อมูล")
            continue
        trd, views = prep(fr)
        ent = fr["1h"]
        n = len(ent)
        if n < WARMUP + 100:
            continue
        V = views["1h"]
        for sgn in (1, -1):
            V[sgn]["sh"], V[sgn]["sl"] = pivots(V[sgn]["h"], V[sgn]["l"])
        times = ent.index + pd.Timedelta(minutes=60)
        t4 = htf_at(trd["4h"], "4h", times).to_numpy()
        t1d = htf_at(trd["1d"], "1d", times).to_numpy()
        tar = {"4h": t4, "4h+1d": np.where(t4 == t1d, t4, 0)}
        split = int(n * TRAIN_FRAC)
        for kind in PA_KINDS:
            for tname in PA_TRENDS:
                tarr = tar[tname]
                for ex in PA_EXITS:
                    label = f"{ex[0]}{ex[1]:g}"
                    out = rows.setdefault((kind, tname, label), [])
                    i = WARMUP
                    while i < n - 1:
                        sg = int(tarr[i])
                        if sg and in_session(times[i], (0, 24)):
                            r = pa_check(V[sg], i, kind)
                            if r:
                                if ex[0] == "rr":
                                    R, j = simulate(V[sg], i, r, ex[1], n)
                                else:
                                    R, j = simulate_trail(V[sg], i, r, ex[1], n)
                                out.append((name, i >= split, R))
                                i = j + 1
                                continue
                        i += 1
        print(name, "เสร็จ")

    lines = ["**Price Action 1H Research** (train 60% เก่า | test 40% ล่าสุด)"]
    passed = []
    nsym = len(SYMBOLS)
    for key, tr in rows.items():
        a = [x[2] for x in tr if not x[1]]
        b = [x[2] for x in tr if x[1]]
        sa, sb = stats(a), stats(b)
        pos = 0
        for sym in SYMBOLS:
            bs = [x[2] for x in tr if x[0] == sym and x[1]]
            st = stats(bs)
            if st and st[0] >= 3 and st[3] > 1:
                pos += 1
        label = f"{key[0]}|{key[1]}|{key[2]}"
        if sa and sb:
            lines.append(f"{label}: train {sa[0]} ไม้ PF {sa[3]:.2f} | test {sb[0]} ไม้ PF {sb[3]:.2f} ({sb[2]:+.2f}R) | คู่บวก {pos}/{nsym}")
            if sa[0] >= 60 and sb[0] >= 50 and sa[3] >= 1.2 and sb[3] >= 1.2 and pos >= 6:
                passed.append((min(sa[3], sb[3]), key, label, b, sa, sb))
        else:
            lines.append(f"{label}: ไม้ไม่พอ")
    lines.append("")
    if passed:
        passed.sort(reverse=True, key=lambda x: x[0])
        lines.append("**ผ่านเกณฑ์** (PF ≥ 1.2 ทั้งสองช่วง, test ≥ 50 ไม้, คู่บวก ≥ 6):")
        for _, _, label, _, sa, sb in passed:
            lines.append(f"✅ {label}")
    else:
        lines.append("**ไม่มีชุดไหนผ่านเกณฑ์** (PF ≥ 1.2 ทั้งสองช่วง, test ≥ 50 ไม้, คู่บวก ≥ 6)")
    send_long("\n".join(lines))
    print("\n".join(lines))

    if passed:
        _, key, label, b, sa, sb = passed[0]
        years = 730 / 365
        per_year = len(rows[key]) / years
        msg = [f"**จำลองการโตของพอร์ต** ด้วยชุด {label} (สุ่มจากผลไม้ช่วง test {len(b)} ไม้)",
               f"ประมาณ {per_year:.0f} ไม้/ปี รวมทุกคู่ | เป้า 10 เท่า, ถือว่าพังถ้าเหลือ 20% ของทุน | 200 ไม้"]
        for risk in (0.02, 0.05, 0.10):
            hit, ruin, med = monte_carlo(b, risk)
            msg.append(f"เสี่ยง {risk*100:.0f}%/ไม้: ถึง 10 เท่า {hit*100:.0f}% | พัง {ruin*100:.0f}% | ทุนกลาง {med:.1f}x")
        msg.append("สมมติว่าอนาคตเหมือนช่วง test ซึ่งไม่แน่นอน และไม้แพ้ติดกันจริงอาจแย่กว่านี้")
        send_long("\n".join(msg))


# ------------------------------------------------- Price Action (1H) research
# สเปรด+สลิปเพจโดยประมาณ (หน่วยราคา) ใช้คิดต้นทุนจริงต่อไม้ ตามขนาด SL
SPREAD = {"XAUUSD": 0.30, "GBPJPY": 0.025, "GBPUSD": 0.00012, "EURUSD": 0.00010,
          "USDJPY": 0.010, "EURJPY": 0.015, "AUDUSD": 0.00012, "USDCAD": 0.00015,
          "NZDUSD": 0.00015}
COST_MULT = 1.5  # สเปรด + สลิปเพจ ไป-กลับ


def pivots(h, l, k):
    """Swing high/low ที่ยืนยันแล้ว (ต้องรอ k แท่งหลังจุดนั้นปิด จึงไม่มองอนาคต)"""
    n = len(h)
    sh1 = np.full(n, np.nan); sh2 = np.full(n, np.nan)
    sl1 = np.full(n, np.nan); sl2 = np.full(n, np.nan)
    shi = np.full(n, -1); sli = np.full(n, -1)
    hs, ls = [], []
    for t in range(n):
        p = t - k
        if p >= k:
            if h[p] == h[p - k:p + k + 1].max():
                hs.append((p, h[p]))
            if l[p] == l[p - k:p + k + 1].min():
                ls.append((p, l[p]))
        if hs:
            sh1[t], shi[t] = hs[-1][1], hs[-1][0]
            if len(hs) > 1:
                sh2[t] = hs[-2][1]
        if ls:
            sl1[t], sli[t] = ls[-1][1], ls[-1][0]
            if len(ls) > 1:
                sl2[t] = ls[-2][1]
    return sh1, sh2, sl1, sl2, shi, sli


def struct_trend(df, k=2):
    """โครงสร้างราคา: HH+HL = +1, LH+LL = -1, อื่นๆ = 0"""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    sh1, sh2, sl1, sl2, _, _ = pivots(h, l, k)
    t = np.zeros(len(df), dtype=int)
    t[(sh1 > sh2) & (sl1 > sl2)] = 1
    t[(sh1 < sh2) & (sl1 < sl2)] = -1
    return pd.Series(t, index=df.index)


def atr_np(h, l, c, n=14):
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    tr = pd.concat([hs - ls, (hs - cs.shift()).abs(), (ls - cs.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().fillna(0).to_numpy()


def pa_view(o, h, l, c, a):
    sh1, sh2, sl1, sl2, shi, sli = pivots(h, l, 3)
    return dict(o=o, h=h, l=l, c=c, atr=a, sh1=sh1, sl1=sl1, shi=shi, sli=sli)


def sig_breakout(X, N):
    """ทะลุกรอบสูงสุด N แท่ง ด้วยแท่งแรงปิดใกล้ high"""
    c, o, h, l, a = X["c"], X["o"], X["h"], X["l"], X["atr"]
    hh = pd.Series(h).rolling(N).max().shift(1).to_numpy()
    rng = h - l
    strong = ((c - o) > 0.5 * a) & ((c - l) >= 0.7 * np.where(rng > 0, rng, np.nan))
    brk = c > hh
    first = brk & ~np.r_[False, brk[:-1]]
    out = []
    for i in np.where(first & strong)[0]:
        if i < 60:
            continue
        entry = c[i]
        sl = l[i - 7:i + 1].min() - 0.1 * a[i]
        risk = entry - sl
        if risk > 3 * a[i]:
            continue
        if risk < a[i]:
            sl = entry - a[i]
        out.append((int(i), entry, sl))
    return out


def sig_sweep(X):
    """กวาด swing low (ล่า stop) แล้วปิดกลับเหนือระดับด้วยแท่งกลับตัว"""
    c, o, h, l, a = X["c"], X["o"], X["h"], X["l"], X["atr"]
    sl1, sli = X["sl1"], X["sli"]
    out = []
    for i in range(70, len(c)):
        lv, li = sl1[i - 1], sli[i - 1]
        if np.isnan(lv) or li < 0 or (i - 1 - li) > 80:
            continue
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        depth = lv - l[i]
        if (l[i] < lv and c[i] > lv and c[i] > o[i] and (c[i] - l[i]) / rng >= 0.6
                and 0.05 * a[i] <= depth <= 1.5 * a[i]):
            entry = c[i]
            sl = l[i] - 0.15 * a[i]
            risk = entry - sl
            if risk > 2.5 * a[i]:
                continue
            if risk < 0.7 * a[i]:
                sl = entry - 0.7 * a[i]
            out.append((i, entry, sl))
    return out


def sig_retest(X):
    """ทะลุ swing high แล้วย่อกลับมาทดสอบระดับเดิมภายใน 12 แท่ง ได้แท่งยืนยัน"""
    c, o, h, l, a = X["c"], X["o"], X["h"], X["l"], X["atr"]
    sh1 = X["sh1"]
    out, lvl, bidx = [], None, -1
    for i in range(70, len(c)):
        if lvl is not None:
            rng = h[i] - l[i]
            if i - bidx > 12 or c[i] < lvl - 0.5 * a[i]:
                lvl = None
            elif (i > bidx and rng > 0 and l[i] <= lvl + 0.25 * a[i] and c[i] > lvl
                  and c[i] > o[i] and (c[i] - l[i]) / rng >= 0.55):
                entry = c[i]
                sl = min(l[i], lvl) - 0.25 * a[i]
                risk = entry - sl
                lvl = None
                if risk <= 2.5 * a[i]:
                    if risk < 0.7 * a[i]:
                        sl = entry - 0.7 * a[i]
                    out.append((i, entry, sl))
                continue
        L = sh1[i - 1]
        if not np.isnan(L) and c[i] > L and c[i - 1] <= L and c[i] - L >= 0.3 * a[i]:
            lvl, bidx = L, i
    return out


def sim_fixed(X, i, entry, sl, rr, cost_px, maxb=150):
    risk = entry - sl
    tp = entry + rr * risk
    n = len(X["c"])
    end = min(n - 1, i + maxb)
    for j in range(i + 1, end + 1):
        if X["l"][j] <= sl:
            px = X["o"][j] if X["o"][j] < sl else sl
            return (px - entry - cost_px) / risk, j
        if X["h"][j] >= tp:
            return (tp - entry - cost_px) / risk, j
    return (X["c"][end] - entry - cost_px) / risk, end


def sim_trail(X, i, entry, sl, cost_px, M=10, maxb=300):
    risk = entry - sl
    stop = sl
    n = len(X["c"])
    end = min(n - 1, i + maxb)
    for j in range(i + 1, end + 1):
        if X["l"][j] <= stop:
            px = X["o"][j] if X["o"][j] < stop else stop
            return (px - entry - cost_px) / risk, j
        stop = max(stop, X["l"][j - M + 1:j + 1].min())
    return (X["c"][end] - entry - cost_px) / risk, end


PA_STRATS = ("BRK20", "BRK40", "SWEEP", "RETEST")
PA_FILTERS = ("none", "4h", "1d")
PA_EXITS = ("2R", "3R", "TRAIL")


def pf_of(rs):
    a = np.array(rs)
    if len(a) == 0:
        return 0.0
    gl = -a[a < 0].sum()
    return float("inf") if gl == 0 else a[a > 0].sum() / gl


def pa_line(rs):
    if not rs:
        return "-"
    return f"{len(rs)}/{pf_of(rs):.2f}"


def mc_growth(rs, risk, trades=200, sims=2000, target=10.0, ruin=0.3):
    rng = np.random.default_rng(1)
    a = np.array(rs)
    hit = dead = 0
    finals = []
    for _ in range(sims):
        eq = 1.0
        for r in rng.choice(a, trades):
            eq *= 1 + risk * r
            if eq >= target:
                hit += 1
                break
            if eq <= ruin:
                dead += 1
                break
        finals.append(eq)
    return hit / sims * 100, dead / sims * 100, float(np.median(finals))


def run_pa_research():
    rows = {}
    for name, ticker in SYMBOLS.items():
        fr = get_frames(ticker, deep=True, with15=False)
        if not fr or len(fr["1h"]) < 500:
            print(name, "ไม่มีข้อมูลพอ")
            continue
        h1 = fr["1h"]
        n = len(h1)
        times = h1.index + pd.Timedelta(minutes=60)
        o, h, l, c = (h1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        a = atr_np(h, l, c)
        filt = {
            "none": None,
            "4h": htf_at(struct_trend(fr["4h"]), "4h", times).to_numpy(),
            "1d": htf_at(struct_trend(fr["1d"]), "1d", times).to_numpy(),
        }
        views = {1: pa_view(o, h, l, c, a), -1: pa_view(-o, -l, -h, -c, a)}
        cost_px = SPREAD.get(name, 0) * COST_MULT
        for strat in PA_STRATS:
            sigs = {}
            for s_, X in views.items():
                if strat == "BRK20":
                    sigs[s_] = sig_breakout(X, 20)
                elif strat == "BRK40":
                    sigs[s_] = sig_breakout(X, 40)
                elif strat == "SWEEP":
                    sigs[s_] = sig_sweep(X)
                else:
                    sigs[s_] = sig_retest(X)
            merged = sorted([(i, e, sl, s_) for s_ in sigs for (i, e, sl) in sigs[s_]])
            for fname, F in filt.items():
                for ex in PA_EXITS:
                    out = rows.setdefault((strat, fname, ex), [])
                    free = 0
                    for i, e, sl, s_ in merged:
                        if i < free:
                            continue
                        if F is not None and F[i] != s_:
                            continue
                        X = views[s_]
                        if ex == "TRAIL":
                            R, j = sim_trail(X, i, e, sl, cost_px)
                        else:
                            R, j = sim_fixed(X, i, e, sl, float(ex[0]), cost_px)
                        out.append((name, i / n, R))
                        free = j + 1
        print(name, "เสร็จ")

    lines = ["**Price Action 1H** (train 60% เก่า | test 40% ล่าสุด) รูปแบบ ไม้/PF", "✅ = PF ≥ 1.2 ทั้งสองช่วง, train ≥ 100 ไม้, test ≥ 60 ไม้"]
    ranked = []
    for key, tr in rows.items():
        a_ = [x[2] for x in tr if x[1] < TRAIN_FRAC]
        b_ = [x[2] for x in tr if x[1] >= TRAIN_FRAC]
        ok = len(a_) >= 100 and len(b_) >= 60 and pf_of(a_) >= 1.2 and pf_of(b_) >= 1.2
        ranked.append((pf_of(a_) if len(a_) >= 100 else -1, key, a_, b_, ok))
    ranked.sort(key=lambda x: -x[0])
    for _, key, a_, b_, ok in ranked:
        avg = f" {np.mean(b_):+.2f}R" if b_ else ""
        lines.append(f"{'✅' if ok else '•'} {key[0]}|{key[1]}|{key[2]}: train {pa_line(a_)} | test {pa_line(b_)}{avg}")
    passed = [r for r in ranked if r[4]]
    lines.append("")
    lines.append(f"ผ่านเกณฑ์: {len(passed)} จาก {len(ranked)} ชุด (ทดสอบหลายชุด จึงมีโอกาสผ่านเพราะโชคได้)")
    best = next((r for r in ranked if r[0] > 0), None)
    if best:
        _, key, a_, b_, ok = best
        lines.append(f"\nชุดที่เลือกจาก train เท่านั้น: {key[0]}|{key[1]}|{key[2]}")
        tr = rows[key]
        for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
            seg = [x[2] for x in tr if lo <= x[1] < hi]
            lines.append(f"ช่วงเวลา {int(lo * 3) + 1}/3: {pa_line(seg)}" + (f" เฉลี่ย {np.mean(seg):+.2f}R" if seg else ""))
        lines.append("รายคู่ (test):")
        for sym in SYMBOLS:
            seg = [x[2] for x in tr if x[0] == sym and x[1] >= TRAIN_FRAC]
            lines.append(f"{sym}: {pa_line(seg)}")
        if len(b_) >= 30 and pf_of(b_) > 1:
            lines.append("\nจำลองการเติบโตของพอร์ต 200 ไม้ (สุ่มจากผล test) เป้า 10 เท่า / ถือว่าพังเมื่อพอร์ตเหลือ 30%:")
            for risk in (0.02, 0.05, 0.10):
                hit, dead, med = mc_growth(b_, risk)
                lines.append(f"เสี่ยง {int(risk * 100)}%/ไม้: ถึงเป้า {hit:.0f}% | พัง {dead:.0f}% | มัธยฐาน {med:.2f} เท่า")
        else:
            lines.append("\nชุดนี้ช่วง test ไม่ได้กำไร จึงไม่จำลองการเติบโตของพอร์ต")
    text = "\n".join(lines)
    print(text)
    send_long(text)

# --------------------------------------------------------------------- main
def main():
    if RUN_MODE == "test":
        send(content="✅ ทดสอบสำเร็จ: บอทเชื่อมต่อ Discord ได้แล้ว")
    elif RUN_MODE == "backtest":
        run_backtest()
    elif RUN_MODE == "research":
        run_research()
    elif RUN_MODE == "pa":
        run_pa_research()
    elif RUN_MODE == "pa":
        run_pa_research()
    else:
        run_live()


if __name__ == "__main__":
    main()
