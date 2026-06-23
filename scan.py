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


def analyze(ticker: str, df: pd.DataFrame) -> dict | None:
    """Run VCP analysis on a single stock's OHLCV dataframe."""
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

    # ── Volatility contraction: rolling 10-day range windows ──────────────
    def range_window(end_idx, length=10):
        s = max(0, end_idx - length)
        if end_idx - s < 6:
            return None
        hi = high[s:end_idx].max()
        lo = low[s:end_idx].min()
        return (hi - lo) / hi if hi else None

    windows = [
        range_window(n),
        range_window(n - 10),
        range_window(n - 20),
        range_window(n - 35),
    ]
    valid = [w for w in windows if w is not None]
    contractions = 0
    for i in range(len(valid) - 1):
        if valid[i] < valid[i + 1] * 0.90:
            contractions += 1

    # ── Volume dry-up ──────────────────────────────────────────────────────
    recent_vol = vol[-15:].mean()
    prior_vol = vol[-45:-15].mean() if n >= 45 else vol[:-15].mean()
    vol_ratio = recent_vol / prior_vol if prior_vol else 1.0

    # ── Relative strength proxy ──────────────────────────────────────────────
    ret6m = (price / close[-126] - 1) * 100 if n >= 126 else 0
    ret3m = (price / close[-63] - 1) * 100 if n >= 63 else 0
    rs = int(min(99, max(0, round(50 + ret6m * 1.0 + ret3m * 0.5))))

    checks = {
        "Above 150d MA": bool(price > sma150),
        "Above 200d MA": bool(price > sma200),
        "200d Rising": bool(ma200_rising),
        "Within 25% of High": bool(price >= high52 * 0.75),
        "30% Above Low": bool(price > low52 * 1.30),
        "RS > 70": bool(rs >= 70),
        "VCP Pattern": bool(contractions >= 2),
        "Volume Dry-Up": bool(vol_ratio < 0.80),
    }
    pass_count = sum(checks.values())
    vcp_score = round(pass_count / len(checks) * 100)

    # Require the core structure
    core = checks["Above 150d MA"] and checks["Above 200d MA"] and checks["VCP Pattern"]
    if not core or vcp_score < 50:
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
    }


def main():
    print(f"VCP scan starting — {len(UNIVERSE)} tickers", flush=True)
    results = []
    failed = 0

    # yfinance can batch-download; chunk to be gentle and resilient
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
            failed += len(chunk)
            continue

        for ticker in chunk:
            try:
                if len(chunk) == 1:
                    df = data
                else:
                    df = data[ticker]
                df = df.dropna()
                if df.empty:
                    failed += 1
                    continue
                r = analyze(ticker, df)
                if r:
                    results.append(r)
            except Exception:
                failed += 1
                continue

        print(f"  processed {min(i+CHUNK, len(UNIVERSE))}/{len(UNIVERSE)} · {len(results)} hits", flush=True)
        time.sleep(1)

    results.sort(key=lambda x: x["vcpScore"], reverse=True)

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "universeSize": len(UNIVERSE),
        "scanned": len(UNIVERSE) - failed,
        "failed": failed,
        "hitCount": len(results),
        "results": results,
    }

    with open("docs/results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Done. {len(results)} VCP setups found. Wrote docs/results.json", flush=True)


if __name__ == "__main__":
    main()
