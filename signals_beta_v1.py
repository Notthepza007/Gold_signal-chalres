# -*- coding: utf-8 -*-
"""
GOLD BOT  —  BETA v1.0   (ทองคำรายวัน -> Discord)

กฎการเทรด (ไม่เปลี่ยนจากชุดที่ผ่านการทดสอบ train/test บนทอง 26 ปี):
  ซื้อ: ปิดรายวันเหนือจุดสูงสุด 55 วันก่อนหน้า | ขาย: ปิดต่ำกว่าจุดต่ำสุด 55 วันก่อนหน้า
  เข้าที่ราคาเปิดแท่งถัดไป, SL เริ่มต้น = 2 ATR(20), เลื่อน SL ตาม Chandelier 3 ATR, ไม่มี TP ตายตัว

สิ่งใหม่ใน BETA v1.0
  1) เตือนเมื่อดึงข้อมูลไม่ได้ / ข้อมูลไม่อัปเดต (กันบอทเงียบแล้วเข้าใจผิดว่า "ไม่มีซิก")
  2) บันทึกอัตโนมัติ: gold_log.csv (ทุกแท่งที่ตรวจ) และ gold_journal.json (ไม้ที่ปิด)
  3) โหมด export ส่งไฟล์ CSV เข้า Discord ให้ดาวน์โหลดเก็บ
  4) คำนวณล็อต/ความเสี่ยงต่อไม้ (ถ้าตั้ง Secret ACCOUNT_USD) และเตือนเมื่อ 0.01 ล็อตเสี่ยงเกินพอร์ต
  5) ติดป้ายเวอร์ชันในทุกข้อความ

โหมด (RUN_MODE): daily | daily_status | journal | export | test | version
ตัวเลือก (ตั้งใน GitHub Secrets/Variables ถ้าต้องการ): ACCOUNT_USD, RISK_PCT (ค่าเริ่มต้น 1)
"""
import os
import io
import csv
import json
import time
import math

import numpy as np
import pandas as pd

VERSION = "BETA v1.0"
WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "").strip()
RUN_MODE = os.environ.get("RUN_MODE", "daily").strip().lower()


def _env_float(name):
    try:
        v = float(os.environ.get(name, "").strip())
        return v if v > 0 else None
    except Exception:  # noqa
        return None


ACCOUNT_USD = _env_float("ACCOUNT_USD")        # ไม่บังคับ
RISK_PCT = _env_float("RISK_PCT") or 1.0       # % ของพอร์ตต่อไม้

TICKER = "GC=F"
N = 55
SL_ATR = 2.0
TRAIL_ATR = 3.0
STALE_DAYS = 5
OZ_PER_LOT = 100.0

GOLD_STATE = "gold_state.json"
JOURNAL = "gold_journal.json"
GOLD_LOG = "gold_log.csv"
BT_EXPECT = "Backtest คาดไว้: ชนะ ~41% | เฉลี่ย +0.27R | แพ้ติดสูงสุด 5 ไม้ | DD ราว 6R"
FOOT = {"text": f"Gold Bot {VERSION} • ตั้ง SL จริงที่โบรกเกอร์ทุกครั้ง บอทแจ้งหลังแท่งรายวันปิดเท่านั้น • ไม่ใช่คำแนะนำการลงทุน ผลไม่รับประกัน"}


# ----------------------------------------------------------------- Discord
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


def send_file(filename, data, text=""):
    if not WEBHOOK:
        print("ไม่พบ DISCORD_WEBHOOK (ยังไม่ได้ตั้ง Secret)")
        return False
    import requests
    try:
        r = requests.post(WEBHOOK, data={"username": "Signal Bot", "content": text},
                          files={"file": (filename, data, "text/csv")}, timeout=30)
        print("discord file status", r.status_code)
        return r.status_code in (200, 204)
    except Exception as e:  # noqa
        print("discord file error", type(e).__name__)
        return False


# --------------------------------------------------------------------- ข้อมูล
def _clean(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.empty:
        return None
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df.sort_index()


def fetch_daily():
    """ดึงทองรายวัน ลองซ้ำ 3 ครั้ง (คืน None ถ้าไม่สำเร็จ)"""
    import yfinance as yf
    for attempt in range(3):
        try:
            df = _clean(yf.download(TICKER, interval="1d", period="2y", progress=False,
                                    auto_adjust=False, threads=False))
            if df is not None and len(df) >= 260:
                return df
        except Exception as e:  # noqa
            print("download error", type(e).__name__)
        time.sleep(4 * (attempt + 1))
    return None


def drop_unclosed(df):
    """แท่งรายวันของทองปิดราว 21-22 UTC ถือว่าปิดแล้วหลัง 22:30 UTC ของวันนั้น"""
    now = pd.Timestamp.now(tz="UTC")
    closed = df.index.normalize() + pd.Timedelta(hours=22, minutes=30)
    return df[closed <= now]


def atr_np(h, l, c, n=20):
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    tr = pd.concat([hs - ls, (hs - cs.shift()).abs(), (ls - cs.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().fillna(0).to_numpy()


# ---------------------------------------------------------------- ไฟล์สถานะ/บันทึก
def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa
        return default


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def journal_load():
    j = _load_json(JOURNAL, [])
    return j if isinstance(j, list) else []


def journal_add(rec):
    j = journal_load()
    j.append(rec)
    _save_json(JOURNAL, j[-500:])


def pf_of(rs):
    a = np.array(rs, dtype=float)
    if len(a) == 0:
        return 0.0
    gl = -a[a < 0].sum()
    return float("inf") if gl == 0 else float(a[a > 0].sum() / gl)


def journal_stats(rs):
    a = np.array(rs, dtype=float)
    best = cur = 0
    for r in a:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    eq = np.cumsum(a)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(a), win=float((a > 0).mean() * 100), avg=float(a.mean()), pf=pf_of(a),
                cur=cur, best=best, dd=dd)


def journal_lines():
    rs = [t.get("R", 0) for t in journal_load()]
    if not rs:
        return ["สมุดบันทึก: ยังไม่มีไม้ที่ปิด"], []
    st = journal_stats(rs)
    lines = [f"สมุดบันทึก {st['n']} ไม้ | ชนะ {st['win']:.0f}% | เฉลี่ย {st['avg']:+.2f}R | PF {st['pf']:.2f} | "
             f"แพ้ติดกัน (ปัจจุบัน/สูงสุด) {st['cur']}/{st['best']} | DD {st['dd']:.1f}R", BT_EXPECT]
    warns = []
    if st["cur"] >= 6:
        warns.append("⚠️ แพ้ติดกัน ≥ 6 ไม้ เกินสถิติ Backtest (5) ควรทบทวนและลดความเสี่ยง")
    if st["dd"] >= 12:
        warns.append("🛑 DD ≥ 12R เป็น 2 เท่าของที่ Backtest คาด ควรหยุดเทรดจริงและตรวจสอบ")
    elif st["dd"] >= 8:
        warns.append("⚠️ DD ≥ 8R เริ่มเกินช่วงปกติของ Backtest (ราว 6R)")
    if st["n"] >= 20 and st["avg"] < -0.1:
        warns.append("⚠️ ผลเฉลี่ยติดลบหลัง 20 ไม้ ผลจริงเริ่มแย่กว่า Backtest")
    return lines, warns


def journal_csv_bytes():
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["signal_date", "exit_date", "side", "entry", "exit_price", "R"])
    for t in journal_load():
        w.writerow([t.get("sig"), t.get("exit"), "BUY" if t.get("side") == 1 else "SELL",
                    t.get("entry"), t.get("exit_px"), t.get("R")])
    return ("\ufeff" + out.getvalue()).encode("utf-8")


def log_append(date, close, atr, buy, sell, action):
    new = not os.path.exists(GOLD_LOG)
    with open(GOLD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "close", "atr20", "buy_above", "sell_below", "action", "version"])
        w.writerow([date, round(close, 2), round(atr, 2), round(buy, 2), round(sell, 2), action or "-", VERSION])


# ------------------------------------------------------------------ ขนาดไม้
def gp(x):
    return f"{x:,.2f}"


def size_lines(dist):
    """dist = ระยะ SL ต่อออนซ์ (ดอลลาร์) | 1 ล็อตมาตรฐาน = 100 ออนซ์ | 0.01 ล็อต = 1 ออนซ์"""
    min_risk = dist * OZ_PER_LOT * 0.01
    if not ACCOUNT_USD:
        return [f"0.01 ล็อต (1 ออนซ์) เสี่ยงราว ${min_risk:,.0f} ต่อไม้ (ตั้ง Secret ACCOUNT_USD เพื่อให้บอทคำนวณล็อตให้)"]
    risk_usd = ACCOUNT_USD * RISK_PCT / 100
    lots = math.floor(risk_usd / (dist * OZ_PER_LOT) * 100) / 100
    if lots >= 0.01:
        real = lots * dist * OZ_PER_LOT
        return [f"ล็อตแนะนำ ≈ {lots:.2f} (เสี่ยง ≈ ${real:,.0f} = {real / ACCOUNT_USD * 100:.1f}% ของพอร์ต)"]
    return [f"⚠️ 0.01 ล็อตเสี่ยง ${min_risk:,.0f} = {min_risk / ACCOUNT_USD * 100:.0f}% ของพอร์ต เกินเพดาน {RISK_PCT:g}% "
            f"→ ไม่แนะนำเทรดจริงไม้นี้ (ใช้เดโมหรือบัญชีเซ็นต์)"]


# ------------------------------------------------------------------ จำลองสถานะ
def replay(X, a, n, i, s):
    """จำลองตั้งแต่แท่งสัญญาณ i -> (stop, entry, exit, sl0) เป็นราคาจริง"""
    e = X["o"][i + 1] if i + 1 < n else X["c"][i]
    sl = e - SL_ATR * a[i]
    stop, top = sl, e
    for j in range(i + 1, n):
        if X["l"][j] <= stop:
            px = X["o"][j] if X["o"][j] < stop else stop
            return s * stop, s * e, (j, s * px), s * sl
        top = max(top, X["h"][j])
        stop = max(stop, top - TRAIL_ATR * a[j])
    return s * stop, s * e, None, s * sl


# --------------------------------------------------------------------- บอทหลัก
def run_daily(status_only=False):
    df = fetch_daily()
    if df is None:
        send(embed={"title": "⚠️ บอททอง: ดึงข้อมูลไม่สำเร็จ", "color": 0xE67E22,
                    "description": "ลองดึงข้อมูลราคา 3 ครั้งแล้วไม่ได้ วันนี้บอทยังไม่ได้ตรวจซิก\nถ้าเกิดซ้ำหลายวัน ให้ตรวจแท็บ Actions แล้วแจ้งผู้ดูแลบอท",
                    "footer": FOOT})
        return
    df = drop_unclosed(df)
    n = len(df)
    if n < 260:
        print("ข้อมูลไม่พอ")
        return
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr_np(h, l, c, 20)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    last = dates[-1]
    age = (pd.Timestamp.now(tz="UTC") - df.index[-1]).days
    if age > STALE_DAYS and not status_only:
        send(embed={"title": "⚠️ บอททอง: ข้อมูลราคาไม่อัปเดต", "color": 0xE67E22,
                    "description": f"แท่งล่าสุดคือ {last} (เก่า {age} วัน) อาจเป็นช่วงตลาดปิดยาวหรือแหล่งข้อมูลมีปัญหา",
                    "footer": FOOT})
    views = {1: dict(o=o, h=h, l=l, c=c), -1: dict(o=-o, h=-l, l=-h, c=-c)}
    st = _load_json(GOLD_STATE, {"pos": None, "last_bar": ""})
    if not isinstance(st, dict):
        st = {"pos": None, "last_bar": ""}
    if not status_only and st.get("last_bar") == last:
        print("ประมวลผลแท่งนี้แล้ว", last)
        return

    i = n - 1
    hh, ll = h[i - N:i].max(), l[i - N:i].min()
    pos = st.get("pos")
    actions, exited_now = [], False

    if pos:
        idx = [k for k, d in enumerate(dates) if d == pos["sig_date"]]
        if not idx:
            pos = None
        else:
            s_ = pos["side"]
            stop, e, ex, sl0 = replay(views[s_], a, n, idx[0], s_)
            risk = abs(e - sl0)
            name = "BUY 🟢" if s_ == 1 else "SELL 🔴"
            if ex and not status_only:
                R = s_ * (ex[1] - e) / risk
                journal_add({"side": s_, "sig": pos["sig_date"], "exit": dates[ex[0]],
                             "entry": round(e, 2), "exit_px": round(ex[1], 2), "R": round(R, 2)})
                jl, warns = journal_lines()
                extra = ("\n" + "\n".join(warns)) if warns else ""
                send(embed={"title": f"🔔 ออกจากไม้ {name} ทอง", "color": 0xF1C40F,
                            "description": (f"ราคาแตะ SL ที่ประมาณ {gp(ex[1])} (เข้าอ้างอิง {gp(e)}) ผลประมาณ {R:+.1f}R "
                                            f"(≈ {R * risk:+,.1f} ดอลลาร์ต่อ 1 ออนซ์)\n{jl[0]}{extra}\n"
                                            f"รอสัญญาณใหม่ (บอทไม่เข้าซ้ำในแท่งเดียวกัน)"),
                            "footer": FOOT})
                actions.append("EXIT")
                pos = None
                exited_now = (ex[0] == n - 1)
            elif not ex and s_ * (stop - pos["last_stop"]) > 0.3 * a[i] and not status_only:
                R_lock = s_ * (stop - e) / risk
                send(embed={"title": f"📈 เลื่อน SL ไม้ {name} ทอง", "color": 0x3498DB,
                            "description": f"เลื่อน SL ไปที่ **{gp(stop)}** (ถ้าโดนจะได้ประมาณ {R_lock:+.1f}R)\nราคาปิดล่าสุด {gp(c[i])}",
                            "footer": FOOT})
                pos["last_stop"] = stop
                actions.append("SL_UPDATE")

    if pos is None and not exited_now and not status_only:
        s_ = 1 if c[i] > hh else (-1 if c[i] < ll else 0)
        if s_:
            dist = SL_ATR * a[i]
            sl = c[i] - s_ * dist
            name = "BUY 🟢" if s_ == 1 else "SELL 🔴"
            lvl = hh if s_ == 1 else ll
            sz = "\n".join(size_lines(dist))
            send(embed={"title": f"{name} ทอง (XAUUSD) | เทรนด์ฟอลโลว์รายวัน",
                        "color": 0x2ECC71 if s_ == 1 else 0xE74C3C,
                        "description": (f"ราคาปิด {gp(c[i])} ทะลุกรอบ {'สูง' if s_ == 1 else 'ต่ำ'} {N} วัน ({gp(lvl)})\n"
                                        f"**เข้า:** ราคาตลาด (อ้างอิง {gp(c[i])})\n"
                                        f"**SL เริ่มต้น:** {gp(sl)} (ห่าง {gp(dist)} = {SL_ATR:g} ATR)\n"
                                        f"**ออก:** ไม่มี TP ตายตัว ใช้ SL เลื่อนตามแนวโน้ม บอทจะแจ้งทุกครั้งที่ควรเลื่อน\n"
                                        f"**ขนาดไม้:** เสี่ยงไม่เกิน 1-2% ของพอร์ต\n{sz}"),
                        "footer": FOOT})
            pos = {"side": s_, "sig_date": last, "last_stop": sl}
            actions.append("ENTRY_BUY" if s_ == 1 else "ENTRY_SELL")

    hh2, ll2 = h[n - N:n].max(), l[n - N:n].min()
    if status_only or df.index[-1].weekday() == 4:
        ptxt = "ไม่มีสถานะเปิด"
        if pos:
            ptxt = f"มีไม้ {'ซื้อ' if pos['side'] == 1 else 'ขาย'} เปิดอยู่ (สัญญาณ {pos['sig_date']}) SL ล่าสุด {gp(pos['last_stop'])}"
        jl, warns = journal_lines()
        send(embed={"title": f"สถานะบอททองรายวัน ({VERSION})", "color": 0x95A5A6,
                    "description": (f"แท่งล่าสุด {last} ปิด {gp(c[i])} | ATR20 {gp(a[i])}\n"
                                    f"**ซื้อ** เมื่อปิดเหนือ {gp(hh2)} | **ขาย** เมื่อปิดต่ำกว่า {gp(ll2)} (กรอบ {N} วัน)\n{ptxt}\n\n"
                                    + "\n".join(jl + warns)),
                    "footer": FOOT})
    if not status_only:
        log_append(last, c[i], a[i], hh2, ll2, "+".join(actions))
        st["pos"] = pos
        st["last_bar"] = last
        _save_json(GOLD_STATE, st)


def run_export():
    send_file("gold_journal.csv", journal_csv_bytes(), f"สมุดบันทึกเทรดบอททอง ({VERSION})")
    if os.path.exists(GOLD_LOG):
        with open(GOLD_LOG, "rb") as f:
            send_file("gold_log.csv", f.read(), "บันทึกการตรวจรายวัน")
    else:
        send(content="ยังไม่มีไฟล์บันทึกการตรวจรายวัน (จะเริ่มสร้างเมื่อบอทรันโหมด daily)")


def main():
    if RUN_MODE == "test":
        send(content=f"✅ ทดสอบสำเร็จ: บอทเชื่อมต่อ Discord ได้แล้ว ({VERSION})")
    elif RUN_MODE == "version":
        send(content=f"Gold Bot {VERSION} | กรอบ {N} วัน | SL {SL_ATR:g} ATR | Chandelier {TRAIL_ATR:g} ATR")
    elif RUN_MODE == "daily_status":
        run_daily(status_only=True)
    elif RUN_MODE == "journal":
        jl, warns = journal_lines()
        send(content="\n".join([f"**สมุดบันทึกบอททอง ({VERSION})**"] + jl + warns))
    elif RUN_MODE == "export":
        run_export()
    elif RUN_MODE == "daily":
        run_daily()
    else:
        print("ไม่มีโหมดนี้:", RUN_MODE)
        send(content=f"โหมด `{RUN_MODE}` ไม่มีในฉบับนี้ ใช้ได้: daily, daily_status, journal, export, test, version")


if __name__ == "__main__":
    main()
