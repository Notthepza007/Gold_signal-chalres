# -*- coding: utf-8 -*-
"""
บอทส่งซิกทอง + Forex เข้า Discord (ฟรี รันบน GitHub Actions)

โหมดการรัน (ตัวแปร RUN_MODE):
  live      สแกนตลาดแล้วส่งซิกใหม่เข้า Discord (ค่าเริ่มต้น)
  test      ส่งข้อความทดสอบว่า Discord เชื่อมต่อได้
  backtest  ทดสอบกลยุทธ์ย้อนหลัง แล้วส่งสรุปวินเรทเข้า Discord
  research  ทดสอบหลายชุดตัวกรอง แยก train/test เพื่อหาชุดที่ไม่ใช่แค่โชคดีย้อนหลัง

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


def get_frames(ticker, deep=False):
    d15 = download(ticker, "15m", "60d")
    h1 = download(ticker, "1h", "730d" if deep else "120d")
    d1 = download(ticker, "1d", "5y" if deep else "2y")
    if d15 is None or h1 is None or d1 is None:
        return None
    h1c = drop_open_bar(h1, "1h")
    return {
        "15m": drop_open_bar(d15, "15m"),
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

# --------------------------------------------------------------------- main
def main():
    if RUN_MODE == "test":
        send(content="✅ ทดสอบสำเร็จ: บอทเชื่อมต่อ Discord ได้แล้ว")
    elif RUN_MODE == "backtest":
        run_backtest()
    elif RUN_MODE == "research":
        run_research()
    else:
        run_live()


if __name__ == "__main__":
    main()
