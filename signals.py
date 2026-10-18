# -*- coding: utf-8 -*-
"""
บอททองคำรายวัน (ฉบับหลัก) -> Discord  | ชุดที่ผ่านการทดสอบ: ทะลุกรอบสูง/ต่ำ 55 วัน (ซื้อ+ขาย)
เข้าที่ราคาเปิดแท่งถัดไป SL เริ่มต้น 2 ATR(20) แล้วเลื่อนตาม Chandelier 3 ATR (ไม่มี TP ตายตัว)

โหมด (ตัวแปร RUN_MODE):
  daily         ตรวจแท่งรายวันที่ปิดแล้ว ส่งซิกเข้า/เลื่อน SL/ออก (ตั้งเวลารันวันละครั้ง)
  daily_status  ส่งสถานะตอนนี้ (ระดับทะลุ ATR สถานะไม้) เพื่อเช็กว่าบอททำงาน
  journal       ส่งรายงานสมุดบันทึกเทรด + เช็กสุขภาพกลยุทธ์
  test          ส่งข้อความทดสอบว่า Discord เชื่อมต่อได้
"""
import os
import json

import numpy as np
import pandas as pd

WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "").strip()
RUN_MODE = os.environ.get("RUN_MODE", "daily").strip().lower()

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


def atr_np(h, l, c, n=14):
    hs, ls, cs = pd.Series(h), pd.Series(l), pd.Series(c)
    tr = pd.concat([hs - ls, (hs - cs.shift()).abs(), (ls - cs.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().fillna(0).to_numpy()


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


def pf_of(rs):
    a = np.array(rs)
    if len(a) == 0:
        return 0.0
    gl = -a[a < 0].sum()
    return float("inf") if gl == 0 else a[a > 0].sum() / gl


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa
        return default


def gp(x):
    return f"{x:,.2f}"


def drop_unclosed_daily(df):
    """แท่งรายวันของทองปิดราว 21-22 UTC ถือว่าปิดแล้วหลัง 22:30 UTC ของวันนั้น"""
    now = pd.Timestamp.now(tz="UTC")
    closed = df.index.normalize() + pd.Timedelta(hours=22, minutes=30)
    return df[closed <= now]


G_N = 55          # ทะลุกรอบสูง/ต่ำ 55 วัน
G_SL_ATR = 2.0    # SL เริ่มต้น 2 ATR
G_TRAIL_ATR = 3.0 # Chandelier 3 ATR
GOLD_STATE = "gold_state.json"
GOLD_FOOT = {"text": "ตั้ง SL จริงที่โบรกเกอร์ทุกครั้ง บอทแจ้งหลังแท่งรายวันปิดเท่านั้น • ไม่ใช่คำแนะนำการลงทุน ผลไม่รับประกัน"}
JOURNAL = "gold_journal.json"
BT_EXPECT = "Backtest คาดไว้: ชนะ ~41% | เฉลี่ย +0.27R | แพ้ติดสูงสุด 5 ไม้ | DD ราว 6R"


def replay_gold(X, a, n, i, s):
    """จำลองสถานะตั้งแต่แท่งสัญญาณ i: คืน (stop, entry, exit) ในหน่วยราคาจริง"""
    e = X["o"][i + 1] if i + 1 < n else X["c"][i]
    sl = e - G_SL_ATR * a[i]
    stop, top = sl, e
    for j in range(i + 1, n):
        if X["l"][j] <= stop:
            px = X["o"][j] if X["o"][j] < stop else stop
            return s * stop, s * e, (j, s * px), s * sl
        top = max(top, X["h"][j])
        stop = max(stop, top - G_TRAIL_ATR * a[j])
    return s * stop, s * e, None, s * sl


def journal_load():
    j = _load_json(JOURNAL, [])
    return j if isinstance(j, list) else []


def journal_add(rec):
    j = journal_load()
    j.append(rec)
    with open(JOURNAL, "w", encoding="utf-8") as f:
        json.dump(j[-500:], f, ensure_ascii=False)


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


def run_gold_daily(status_only=False):
    df = download("GC=F", "1d", "2y")
    if df is None or len(df) < 260:
        print("ข้อมูลทองไม่พอ")
        return
    df = drop_unclosed_daily(df)
    n = len(df)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    a = atr_np(h, l, c, 20)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    last = dates[-1]
    views = {1: dict(o=o, h=h, l=l, c=c), -1: dict(o=-o, h=-l, l=-h, c=-c)}
    st = _load_json(GOLD_STATE, {"pos": None, "last_bar": ""})
    if not status_only and st.get("last_bar") == last:
        print("ประมวลผลแท่งนี้แล้ว", last)
        return
    i = n - 1
    hh = h[i - G_N:i].max()
    ll = l[i - G_N:i].min()
    pos = st.get("pos")
    exited_now = False

    if pos:
        idx = [k for k, d in enumerate(dates) if d == pos["sig_date"]]
        if not idx:
            pos = None
        else:
            s_ = pos["side"]
            stop, e, ex, sl0 = replay_gold(views[s_], a, n, idx[0], s_)
            risk = abs(e - sl0)
            name = "BUY 🟢" if s_ == 1 else "SELL 🔴"
            if ex:
                R = s_ * (ex[1] - e) / risk
                if not status_only:
                    journal_add({"side": s_, "sig": pos["sig_date"], "exit": dates[ex[0]],
                                 "entry": round(e, 2), "exit_px": round(ex[1], 2), "R": round(R, 2)})
                    jl, warns = journal_lines()
                    extra = ("\n" + "\n".join(warns)) if warns else ""
                    send(embed={"title": f"🔔 ออกจากไม้ {name} ทอง", "color": 0xF1C40F,
                                "description": (f"ราคาแตะ SL ที่ประมาณ {gp(ex[1])} (เข้าอ้างอิง {gp(e)}) ผลประมาณ {R:+.1f}R\n"
                                                f"{jl[0]}{extra}\nรอสัญญาณใหม่ (บอทไม่เข้าซ้ำในแท่งเดียวกัน)"),
                                "footer": GOLD_FOOT})
                    pos = None
                    exited_now = (ex[0] == n - 1)
            elif s_ * (stop - pos["last_stop"]) > 0.3 * a[i] and not status_only:
                R_lock = s_ * (stop - e) / risk
                send(embed={"title": f"📈 เลื่อน SL ไม้ {name} ทอง", "color": 0x3498DB,
                            "description": f"เลื่อน SL ไปที่ **{gp(stop)}** (ถ้าโดนจะได้ประมาณ {R_lock:+.1f}R)\nราคาปิดล่าสุด {gp(c[i])}",
                            "footer": GOLD_FOOT})
                pos["last_stop"] = stop

    if pos is None and not exited_now and not status_only:
        s_ = 1 if c[i] > hh else (-1 if c[i] < ll else 0)
        if s_:
            sl = c[i] - s_ * G_SL_ATR * a[i]
            name = "BUY 🟢" if s_ == 1 else "SELL 🔴"
            lvl = hh if s_ == 1 else ll
            send(embed={"title": f"{name} ทอง (XAUUSD) | เทรนด์ฟอลโลว์รายวัน",
                        "color": 0x2ECC71 if s_ == 1 else 0xE74C3C,
                        "description": (f"ราคาปิด {gp(c[i])} ทะลุกรอบ {'สูง' if s_ == 1 else 'ต่ำ'} 55 วัน ({gp(lvl)})\n"
                                        f"**เข้า:** ราคาตลาด (อ้างอิง {gp(c[i])})\n"
                                        f"**SL เริ่มต้น:** {gp(sl)} (ห่าง {gp(G_SL_ATR * a[i])} = 2 ATR)\n"
                                        f"**ออก:** ไม่มี TP ตายตัว ใช้ SL เลื่อนตามแนวโน้ม บอทจะแจ้งทุกครั้งที่ควรเลื่อน\n"
                                        f"**ขนาดไม้:** เสี่ยงไม่เกิน 1-2% ของพอร์ต ล็อต ≈ เงินที่ยอมเสีย ÷ (ระยะ SL × 100) (1 ล็อตมาตรฐาน = 100 ออนซ์)"),
                        "footer": GOLD_FOOT})
            pos = {"side": s_, "sig_date": last, "last_stop": sl}

    if status_only or df.index[-1].weekday() == 4:
        hh2 = h[n - G_N:n].max()
        ll2 = l[n - G_N:n].min()
        ptxt = "ไม่มีสถานะเปิด"
        if pos:
            ptxt = f"มีไม้ {'ซื้อ' if pos['side'] == 1 else 'ขาย'} เปิดอยู่ (สัญญาณ {pos['sig_date']}) SL ล่าสุด {gp(pos['last_stop'])}"
        jl, warns = journal_lines()
        send(embed={"title": "สถานะบอททองรายวัน", "color": 0x95A5A6,
                    "description": (f"แท่งล่าสุด {last} ปิด {gp(c[i])} | ATR20 {gp(a[i])}\n"
                                    f"**ซื้อ** เมื่อปิดเหนือ {gp(hh2)} | **ขาย** เมื่อปิดต่ำกว่า {gp(ll2)} (กรอบ 55 วัน)\n{ptxt}\n\n"
                                    + "\n".join(jl + warns)),
                    "footer": GOLD_FOOT})
    if not status_only:
        st["pos"] = pos
        st["last_bar"] = last
        with open(GOLD_STATE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)


def main():
    if RUN_MODE == "test":
        send(content="✅ ทดสอบสำเร็จ: บอทเชื่อมต่อ Discord ได้แล้ว")
    elif RUN_MODE == "daily_status":
        run_gold_daily(status_only=True)
    elif RUN_MODE == "journal":
        jl, warns = journal_lines()
        send(content="\n".join(["**สมุดบันทึกบอททอง**"] + jl + warns))
    elif RUN_MODE == "daily":
        run_gold_daily()
    else:
        print("ไม่มีโหมดนี้ในฉบับหลัก:", RUN_MODE)
        send(content=f"โหมด `{RUN_MODE}` ไม่มีในไฟล์ฉบับหลัก ใช้ได้: daily, daily_status, journal, test")


if __name__ == "__main__":
    main()
