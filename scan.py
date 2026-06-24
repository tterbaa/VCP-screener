#!/usr/bin/env python3
"""
VCP Screener — Minervini Volatility Contraction Pattern scanner.

Runs server-side (e.g. in GitHub Actions) where yfinance works without
the CORS / rate-limit problems that affect browsers. Scans the S&P 400
(mid-cap) + S&P 600 (small-cap) universe and writes docs/results.json.
"""

import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────
# Universe: S&P 400 + S&P 600 small/mid-cap tickers
# ──────────────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAON","ABCB","ABR","ACAD","ACLS","ACM","ADMA","ADNT","ADUS","AEO",
    "AGO","AIN","AIT","ALGT","ALRM","AMPH","AMRC","AMN","AMNB","AMWD",
    "ANDE","ANF","APG","APLS","APPF","ARCB","ARES","ARI","ARKO","AROC",
    "ASB","ASGN","ATI","ATKR","AVAV","AVNS","AVNT","AVT","AWI","AXS",
    "AXTA","AYI","AZZ","BANF","BANR","BC","BCO","BDC","BG","BHF",
    "BHLB","BJ","BJRI","BKH","BLKB","BLD","BLMN","BMI","BOOT","BRC",
    "BRKL","BRP","BTU","CABO","CADE","CAL","CALM","CALX","CBT","CCOI",
    "CCRN","CDNA","CDRE","CELH","CENT","CENX","CHEF","CHGG","CHRS","CHUY",
    "CIR","CIVI","CKH","CLC","CLF","CLFD","CNK","CNM","CNO","CNX",
    "CODI","COHU","COLB","COLL","COLM","CONN","CPRX","CRC","CRGY","CRL",
    "CRSP","CRVL","CSL","CSWI","CTBI","CTRE","CUTR","CVBF","CVET","CWT",
    "CYTK","DAR","DAVA","DCO","DCOM","DDS","DENN","DFH","DGII","DLB",
    "DLTH","DLX","DNB","DNOW","DOCS","DORM","DRH","DRQ","DRVN","DSP",
    "DV","DVAX","EAT","EBC","EFC","EGAN","EGO","EIG","ELF","ENPH",
    "ENS","ENVA","EPAC","EPRT","ESAB","ESNT","ETSY","EVH","EVTC","EXEL",
    "EXLS","EXP","EXPI","EXPO","EXTR","EYE","EZPW","FARO","FBP","FCF",
    "FCFS","FDP","FELE","FFIN","FIVN","FIZZ","FLNC","FLOW","FLS","FMBH",
    "FMC","FMNB","FORM","FORR","FOUR","FOXF","FRME","FRPT","FSS","FTDR",
    "FULT","G","GBX","GEF","GENI","GFF","GHM","GKOS","GLPI","GNRC",
    "GNW","GOLF","GPI","GPOR","GRPN","GT","GTES","HAFC","HAIN","HALO",
    "HAYW","HCI","HCSG","HHC","HI","HIW","HLMN","HLX","HMST","HOMB",
    "HOPE","HOV","HPP","HQY","HRMY","HSII","HURN","HWC","HWKN","IART",
    "IBP","IDCC","IDYA","IEX","IGT","IIPR","INDB","INFN","INMD","INSP",
    "IOSP","IPAR","IPGP","IRTC","ITGR","ITRI","IVT","JACK","JBGS","JELD",
    "JOBY","JOUT","KAI","KALU","KAR","KBH","KFRC","KFY","KMT","KNX",
    "KOP","KRC","KREF","KRG","KRYS","KSS","KW","KYMR","LADR","LAUR",
    "LCII","LDI","LGIH","LGND","LHCG","LII","LIVN","LKFN","LNTH","LOB",
    "LPG","LRN","LSCC","LSTR","LTC","LXP","MAC","MASI","MATW","MATX",
    "MBI","MBUU","MCY","MD","MDGL","MDU","MEDP","MGEE","MIDD","MIRM",
    "MKSI","MLAB","MLI","MMSI","MODV","MPB","MRCY","MRTN","MSA","MSEX",
    "MSM","MTH","MTN","MUR","NARI","NATH","NAVI","NBR","NBTB","NEO",
    "NEOG","NGVT","NNN","NOG","NOVA","NPO","NSP","NTB","NTCT","NTGR",
    "NUS","NVCR","NVEE","NVR","NWE","NX","NXRT","OFG","OFIX","OGS",
    "OHI","OII","OMCL","OMI","OPCH","OPI","ORGO","ORI","ORLY","OXM",
    "PAAS","PACB","PATK","PAYC","PCRX","PDM","PEBO","PFBC","PFC","PFGC",
    "PFIS","PGNY","PIPR","PKOH","PLAY","PLCE","PLMR","PLNT","PMT","PODD",
    "POOL","POR","POWI","PRK","PRMW","PRO","PSA","PSMT","PTC","PTCT",
    "PTGX","PUMP","PVH","QCR","QDEL","QNST","QTWO","RAMP","RCII","RCKT",
    "RCM","RES","REVG","RGP","RH","RHP","RICK","RIG","RLJ","RMBS",
    "RMR","ROIC","RPM","RPRX","RRC","RSG","RXO","RYAM","SASR","SBCF",
    "SBGI","SBSI","SCHL","SCSC","SEE","SF","SFBS","SFNC","SGMO","SHAK",
    "SHOO","SIG","SITC","SITM","SKT","SLG","SLVM","SMBC","SMCI","SMG",
    "SMMF","SMPL","SNEX","SNV","SONO","SPSC","SPT","SPXC","SRC","SRDX",
    "SRI","SSB","SSTK","STAA","STBA","STC","STEP","STLD","STNG","STRL",
    "SUM","SWI","SXC","SXI","SXT","SYBT","SYM","TALO","TBI","TCBK",
    "TCMD","TDOC","TDW","TGLS","TGTX","THC","THRM","THS","TMDX","TMHC",
    "TMST","TNET","TOWN","TPH","TRHC","TRNO","TRST","TRUP","TTEC","TTGT",
    "TTM","TVTX","TXRH","UCBI","UFI","UFPI","UGI","UMBF","UNFI","UNIT",
    "UNM","UPBD","URBN","USLM","USNA","UVSP","VAC","VCEL","VECO","VIRT",
    "VIST","VMD","VNOM","VSH","VSCO","VSTO","VTLE","VVV","WAFD","WASH",
    "WD","WDFC","WERN","WGO","WHD","WLDN","WLY","WMS","WOLF","WOR",
    "WOW","WPC","WRBY","WSBC","WSFS","WTBA","WTFC","WTTR","WWD","WWW",
    "XPEL","YELP","ZD","ZI","ZION","ZUO",
]


def analyze(ticker: str, df: pd.DataFrame, true_rs: int | None = None) -> dict | None:
    """Run VCP analysis on a single stock's OHLCV dataframe.

    true_rs: market-relative RS percentile (1-99) computed across the whole
    universe in pass 1. If None, falls back to the internal return proxy.
    """
    if df is None or len(df) < 150:
        return None

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)
    n = len(close)
    price = close[-1]

    def sma(arr, p):
        return arr[-p:].mean() if len(arr) >= p else None

    sma50 = sma(close, 50)
    sma150 = sma(close, 150)
    sma200 = sma(close, 200)
    if sma150 is None or sma200 is None:
        return None

    # 200d MA trending up: compare to 20 sessions ago
    sma200_prev = close[-220:-20].mean() if n >= 220 else None
    ma200_rising = (sma200 > sma200_prev * 0.995) if sma200_prev else True

    win = min(252, n)
    high52 = close[-win:].max()
    low52 = close[-win:].min()

    # ── Volatility contraction: TRUE WEEKLY candles ───────────────────────
    # Resample daily OHLC into weekly bars (week ending Friday), then measure
    # the high-low range of each of the last several weeks. A valid VCP shows
    # successively tighter weekly ranges (each contraction smaller than the
    # prior) — this is the classic weekly view Minervini teaches.
    weekly = df.resample("W-FRI").agg(
        {"High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    # Drop the current in-progress week. The latest resampled bar usually
    # represents an incomplete week (only a few sessions), whose artificially
    # small range would otherwise pass the tightness check and mask a big
    # prior-week breakout move. Judge structure on completed weeks only.
    today = df.index[-1]
    last_week_end = weekly.index[-1]
    # If the final weekly bar's Friday is on/after the last daily bar's date,
    # that week hasn't closed yet — drop it.
    if last_week_end.normalize() >= today.normalize():
        weekly = weekly.iloc[:-1]

    wk_high = weekly["High"].to_numpy(dtype=float)
    wk_low = weekly["Low"].to_numpy(dtype=float)
    wk_range = np.where(wk_high > 0, (wk_high - wk_low) / wk_high, np.nan)

    # Most recent 6 completed weeks, oldest -> newest
    recent_weeks = wk_range[-6:]
    recent_weeks = recent_weeks[~np.isnan(recent_weeks)]

    # Count contractions: each week tighter than the one before it
    contractions = 0
    for i in range(1, len(recent_weeks)):
        if recent_weeks[i] < recent_weeks[i - 1] * 0.90:
            contractions += 1

    # Last 4 weekly ranges for the chart, newest -> oldest
    windows = [round(float(r), 4) for r in recent_weeks[-4:][::-1]]
    while len(windows) < 4:
        windows.append(None)

    # Most recent completed week's range (absolute tightness of the base)
    latest_wk_range = float(recent_weeks[-1]) if len(recent_weeks) else 1.0

    # ── Volume dry-up ──────────────────────────────────────────────────────
    recent_vol = vol[-15:].mean()
    prior_vol = vol[-45:-15].mean() if n >= 45 else vol[:-15].mean()
    vol_ratio = recent_vol / prior_vol if prior_vol else 1.0

    # ── Relative strength ────────────────────────────────────────────────────
    # Prefer the true market-relative percentile from pass 1; fall back to a
    # return-based proxy only if it wasn't supplied.
    ret6m = (price / close[-126] - 1) * 100 if n >= 126 else 0
    ret3m = (price / close[-63] - 1) * 100 if n >= 63 else 0
    if true_rs is not None:
        rs = int(true_rs)
    else:
        rs = int(min(99, max(0, round(50 + ret6m * 1.0 + ret3m * 0.5))))

    # ── Base depth & length ──────────────────────────────────────────────────
    # Find the most recent significant swing high, then measure how long the
    # stock has been basing since (weeks) and how deep the pullback ran
    # (peak-to-lowest-low, as a %). A real VCP base is several weeks long with
    # a controlled depth; a 1-2 week dip after a vertical run is not a base.
    lookback = min(60, n)  # ~3 months of daily bars
    seg_high_idx = int(np.argmax(high[-lookback:]))
    bars_since_high = (lookback - 1) - seg_high_idx
    base_weeks = round(bars_since_high / 5.0, 1)
    base_peak = high[-lookback:][seg_high_idx]
    base_trough = low[-(lookback - seg_high_idx):].min() if seg_high_idx < lookback else low[-1]
    base_depth = (base_peak - base_trough) / base_peak * 100 if base_peak else 0

    # A valid base: at least ~3 weeks long, depth not deeper than ~35%,
    # and not a runaway shallow blip (depth at least ~5% so it's a real pause)
    base_ok = (base_weeks >= 3.0) and (5.0 <= base_depth <= 35.0)

    # Proper MA stacking: 50 > 150 > 200, all in correct Stage-2 order
    ma_stacked = bool(
        sma50 is not None
        and sma50 > sma150 > sma200
    )

    # Extension guard: a proper VCP buy point sits in a tight base, NOT far
    # above the 50d MA. If price is stretched well above the 50d, the breakout
    # has likely already happened and the stock is extended (e.g. a vertical
    # spike on huge volume). Reject anything more than 12% above its 50d.
    pct_above_50 = (price / sma50 - 1) * 100 if sma50 else 0
    not_extended = pct_above_50 <= 12.0

    checks = {
        "MA Stack 50>150>200": ma_stacked,
        "Not Extended (<12% > 50d)": bool(not_extended),
        "Within 15% of High": bool(price >= high52 * 0.85),
        "RS > 80": bool(rs >= 80),
        "VCP (3+ tightenings)": bool(contractions >= 3),
        "Base Is Tight (<10%)": bool(latest_wk_range < 0.10),
        "Valid Base (3wk+, 5-35%)": bool(base_ok),
        "Volume Dry-Up": bool(vol_ratio < 0.80),
    }

    # ── Weighted score ────────────────────────────────────────────────────────
    # Each criterion contributes its weight to the score. The structural
    # essentials (uptrend, near highs, RS, contraction) carry more weight than
    # the finer-grain checks, so a stock missing one minor criterion can still
    # score well and surface — rather than being silently dropped.
    weights = {
        "MA Stack 50>150>200": 18,
        "Within 15% of High": 16,
        "RS > 80": 16,
        "VCP (3+ tightenings)": 16,
        "Valid Base (3wk+, 5-35%)": 12,
        "Base Is Tight (<10%)": 10,
        "Volume Dry-Up": 8,
        "Not Extended (<12% > 50d)": 4,
    }
    vcp_score = sum(w for k, w in weights.items() if checks[k])
    pass_count = sum(checks.values())

    # ── Minimal hard gate ─────────────────────────────────────────────────────
    # Only the things that define a valid candidate at all. Everything else is
    # handled by the weighted score. A stock must be in a Stage 2 uptrend and
    # genuinely near its highs — these are non-negotiable. The rest is scored.
    essential = (
        ma_stacked                          # established Stage 2 uptrend
        and price > sma50                   # holding above the 50d
        and checks["Within 15% of High"]    # genuinely near highs (kills recoveries)
    )
    if not essential or vcp_score < 70:
        return None

    change_pct = (close[-1] / close[-2] - 1) * 100 if n >= 2 else 0
    pivot = high[-15:].max()  # recent resistance as pivot estimate

    return {
        "ticker": ticker,
        "price": round(float(price), 2),
        "changePct": round(float(change_pct), 2),
        "vcpScore": vcp_score,
        "rs": rs,
        "contractions": contractions,
        "volRatio": round(float(vol_ratio), 2),
        "pivot": round(float(pivot), 2),
        "distFromHigh": round(float((high52 - price) / high52 * 100), 1),
        "sma50": round(float(sma50), 2) if sma50 else None,
        "sma150": round(float(sma150), 2),
        "sma200": round(float(sma200), 2),
        "passCount": pass_count,
        "checks": checks,
        "ranges": [round(w * 100, 1) if w else None for w in windows],
        "baseWeeks": base_weeks,
        "baseDepth": round(float(base_depth), 1),
        "ret6m": round(float(ret6m), 1),
    }


def download_all() -> dict:
    """Pass 1a: download daily data for the whole universe, chunked."""
    frames = {}
    CHUNK = 40
    for i in range(0, len(UNIVERSE), CHUNK):
        chunk = UNIVERSE[i : i + CHUNK]
        try:
            data = yf.download(
                chunk,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"  chunk {i} download error: {e}", flush=True)
            continue

        for ticker in chunk:
            try:
                df = data if len(chunk) == 1 else data[ticker]
                df = df.dropna()
                if not df.empty and len(df) >= 150:
                    frames[ticker] = df
            except Exception:
                continue

        print(f"  downloaded {min(i+CHUNK, len(UNIVERSE))}/{len(UNIVERSE)} · {len(frames)} usable", flush=True)
        time.sleep(1)
    return frames


def compute_rs_ranks(frames: dict) -> dict:
    """Pass 1b: compute each stock's true RS percentile (1-99).

    Uses a Minervini-style weighted performance score (more weight on recent
    quarters) and ranks every stock against the whole universe, so RS is
    genuinely market-relative rather than an absolute return.
    """
    perf = {}
    for ticker, df in frames.items():
        close = df["Close"].to_numpy(dtype=float)
        n = len(close)
        if n < 126:
            continue
        # Weighted: 40% last quarter, then 20/20/20 for the prior three
        c0 = close[-1]
        q1 = close[-63] if n >= 63 else close[0]
        q2 = close[-126] if n >= 126 else close[0]
        q3 = close[-189] if n >= 189 else close[0]
        q4 = close[-252] if n >= 252 else close[0]
        score = (
            0.40 * (c0 / q1 - 1)
            + 0.20 * (c0 / q2 - 1)
            + 0.20 * (c0 / q3 - 1)
            + 0.20 * (c0 / q4 - 1)
        )
        perf[ticker] = score

    # Rank into 1-99 percentiles
    ranked = sorted(perf.items(), key=lambda kv: kv[1])
    total = len(ranked)
    rs_map = {}
    for idx, (ticker, _) in enumerate(ranked):
        pct = round((idx + 1) / total * 99) if total > 1 else 50
        rs_map[ticker] = max(1, min(99, pct))
    return rs_map


def main():
    print(f"VCP scan starting — {len(UNIVERSE)} tickers", flush=True)

    # ── Pass 1: download everything, compute true market-relative RS ─────────
    print("Pass 1: downloading universe…", flush=True)
    frames = download_all()
    print(f"Pass 1: computing RS ranks for {len(frames)} stocks…", flush=True)
    rs_map = compute_rs_ranks(frames)

    # ── Pass 2: run strict VCP analysis using true RS ───────────────────────
    print("Pass 2: analyzing for VCP setups…", flush=True)
    results = []
    for ticker, df in frames.items():
        try:
            r = analyze(ticker, df, true_rs=rs_map.get(ticker))
            if r:
                results.append(r)
        except Exception:
            continue

    results.sort(key=lambda x: (x["vcpScore"], x["rs"]), reverse=True)

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "universeSize": len(UNIVERSE),
        "scanned": len(frames),
        "failed": len(UNIVERSE) - len(frames),
        "hitCount": len(results),
        "results": results,
    }

    with open("docs/results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done. {len(results)} VCP setups found. Wrote docs/results.json", flush=True)


if __name__ == "__main__":
    main()
