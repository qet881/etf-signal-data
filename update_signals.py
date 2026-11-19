import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path

import pandas as pd
import pytz

try:
    import FinanceDataReader as fdr
    _FDR = True
except ImportError:
    _FDR = False

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False

ASSETS = [
    {
        "id": "kr_stock",
        "name": "국내주식 (코스피200 TR)",
        "desc": "코스피200 TR 지수형",
        "ticker": "294400",
        "tax_group": "국내주식형",
    },
    {
        "id": "us_stock",
        "name": "미국주식 (나스닥100 · 환노출)",
        "desc": "미국나스닥100 환노출형",
        "ticker": "133690",
        "tax_group": "해외·채권형",
    },
    {
        "id": "kr_bond",
        "name": "한국장기채 (국고채30년)",
        "desc": "한국 국고채30년 액티브형",
        "ticker": "439870",
        "tax_group": "해외·채권형",
    },
    {
        "id": "us_bond",
        "name": "미국장기채 (미국채30년 · 환노출)",
        "desc": "미국 30년국채 환노출형",
        "ticker": "476760",
        "tax_group": "해외·채권형",
    },
    {
        "id": "gold",
        "name": "금 (금현물 · 환노출)",
        "desc": "KRX금현물 ETF",
        "ticker": "411060",
        "alt_ticker": "0064K0",
        "tax_group": "금현물형",
    },
]

CASH_ASSET = {
    "id": "cash",
    "name": "현금성 대기 (단기채/MMF)",
    "desc": "초단기 머니마켓(MMF) ETF",
    "ticker": "455890",
}

SLOT_WEIGHT = 0.20
THRESHOLD = 0.05


def get_kst_now() -> datetime:
    return datetime.now(pytz.timezone("Asia/Seoul")).replace(tzinfo=None)


def get_month_end(dt: datetime) -> datetime:
    nxt = dt.replace(day=28) + timedelta(days=4)
    return (nxt - timedelta(days=nxt.day)).replace(hour=0, minute=0, second=0, microsecond=0)


def _clean_df(raw: pd.DataFrame | None) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            next((p for p in col if p in ("Adj Close", "Close")), col[0])
            if isinstance(col, tuple) else col
            for col in df.columns
        ]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if "Close" not in df.columns:
        return None
    out = pd.DataFrame(index=pd.to_datetime(df.index).tz_localize(None))
    out["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    if "Adj Close" in df.columns:
        adj = pd.to_numeric(df["Adj Close"], errors="coerce")
        out["Price"] = adj.where(adj > 0, out["Close"])
    else:
        out["Price"] = out["Close"]
    out = out.dropna(subset=["Price"])
    return out if not out.empty else None


def fetch_series(ticker: str, start_str: str, end_str: str) -> pd.DataFrame | None:
    if _FDR and ticker.isdigit():
        try:
            df = _clean_df(fdr.DataReader(ticker, start_str, end_str))
            if df is not None and len(df) >= 10:
                return df
        except Exception:
            pass
    if _YF:
        try:
            end_plus = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            raw = yf.download(f"{ticker}.KS", start=start_str, end=end_plus, progress=False, auto_adjust=False, timeout=15)
            df = _clean_df(raw)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    if _FDR:
        try:
            end_plus = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df = _clean_df(fdr.DataReader(f"{ticker}.KS", start_str, end_plus))
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return None


def price_asof(df: pd.DataFrame | None, dt: datetime) -> float | None:
    if df is None or df.empty:
        return None
    val = df["Price"].asof(pd.Timestamp(dt))
    return float(val) if pd.notna(val) and float(val) > 0 else None


def calc_rule(df: pd.DataFrame | None, as_of: datetime) -> dict | None:
    curr = price_asof(df, as_of)
    if curr is None:
        return None
    pts = [curr]
    for m in range(1, 12):
        me = get_month_end(as_of - relativedelta(months=m))
        p = price_asof(df, me)
        if p is not None:
            pts.append(p)
    if len(pts) < 6:
        return None
    high_12m = float(max(pts))
    cutoff = high_12m * (1.0 - THRESHOLD)
    dd = (curr / high_12m) - 1.0
    is_active = bool(curr >= cutoff)
    return {
        "current_price": round(curr),
        "high_12m": round(high_12m),
        "cutoff_price": round(cutoff),
        "drawdown_pct": round(dd * 100, 2),
        "is_active": is_active,
        "weight_pct": 20 if is_active else 0,
        "points_used": len(pts),
    }


def main():
    now_kst = get_kst_now()
    today = now_kst.replace(hour=0, minute=0, second=0, microsecond=0)
    start_str = (today - relativedelta(months=18)).strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    items = []
    for item in ASSETS:
        df_main = fetch_series(item["ticker"], start_str, end_str)
        res_main = calc_rule(df_main, today)
        if res_main is None:
            raise RuntimeError(f"시세 조회 실패: {item['ticker']}")

        entry = {
            "id": item["id"],
            "name": item["name"],
            "desc": item["desc"],
            "ticker": item["ticker"],
            "tax_group": item["tax_group"],
            "standard": res_main,
        }

        if item.get("alt_ticker"):
            df_alt = fetch_series(item["alt_ticker"], start_str, end_str)
            res_alt = calc_rule(df_alt, today)
            if res_alt and res_alt["points_used"] >= 10:
                entry["alt_signal"] = {
                    "ticker": item["alt_ticker"],
                    **res_alt,
                }

        items.append(entry)

    df_cash = fetch_series(CASH_ASSET["ticker"], start_str, end_str)
    cash_px = price_asof(df_cash, today) or 55895.0

    payload = {
        "updated_at": now_kst.strftime("%Y-%m-%d %H:%M"),
        "base_date": today.strftime("%Y-%m-%d"),
        "assets": items,
        "cash": {
            "id": CASH_ASSET["id"],
            "name": CASH_ASSET["name"],
            "desc": CASH_ASSET["desc"],
            "ticker": CASH_ASSET["ticker"],
            "current_price": round(cash_px),
        },
    }

    Path("signals.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("signals.json 업데이트 완료:", payload["updated_at"])


if __name__ == "__main__":
    main()
