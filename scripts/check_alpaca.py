"""Verify Alpaca paper account + options access.

Read-only by default:
    uv run scripts/check_alpaca.py

With --test-order, also places a 1-contract BUY limit at $0.01 (won't fill)
on a far-OTM SPY call in the PAPER account, then cancels it:
    uv run scripts/check_alpaca.py --test-order
"""

import argparse
import os
import sys
from datetime import date, timedelta

from dotenv import load_dotenv
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-order", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("FAIL: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
        return 1

    # paper=True is hardcoded on purpose: this repo never talks to live.
    tc = TradingClient(key, secret, paper=True)
    acct = tc.get_account()
    print(f"account     {acct.account_number}  status={acct.status}")
    print(f"equity      {acct.equity}   buying_power={acct.buying_power}")
    print(f"options     approved_level={acct.options_approved_level}  "
          f"trading_level={acct.options_trading_level}  "
          f"options_bp={acct.options_buying_power}")

    req = GetOptionContractsRequest(
        underlying_symbols=["SPY"],
        type=ContractType.CALL,
        expiration_date_gte=date.today() + timedelta(days=7),
        expiration_date_lte=date.today() + timedelta(days=45),
        limit=500,
    )
    contracts = tc.get_option_contracts(req).option_contracts
    if not contracts:
        print("FAIL: no SPY option contracts returned")
        return 1
    print(f"contracts   {len(contracts)} SPY calls listed (7-45 DTE)")

    # Pick a liquid-ish contract (bid $0.50-$5) in the nearest expiry so the
    # test limit can sit far BELOW the bid. A $0.01 limit on a far-OTM call
    # can fill when the ask is itself $0.01 (happened 2026-11-23).
    nearest = min(c.expiration_date for c in contracts)
    syms = [c.symbol for c in contracts if c.expiration_date == nearest]
    md = OptionHistoricalDataClient(key, secret)
    quotes = {}
    for i in range(0, len(syms), 100):  # API caps symbols per request
        quotes.update(md.get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=syms[i:i + 100])))
    print(f"quotes      {len(quotes)}/{len(syms)} returned for {nearest} expiry")
    candidates = [(s, q) for s, q in quotes.items() if 0.50 <= q.bid_price <= 5.0]
    if not candidates:
        print("FAIL: no contract with bid in $0.50-$5 (market closed or feed issue?)")
        return 1
    sym, quote = min(candidates, key=lambda sq: abs(sq[1].bid_price - 2.0))
    print(f"sample      {sym}  bid={quote.bid_price} ask={quote.ask_price} at {quote.timestamp}")

    if args.test_order:
        limit = max(0.01, round(quote.bid_price * 0.25, 2))  # far below bid: should rest, not fill
        order = tc.submit_order(LimitOrderRequest(
            symbol=sym, qty=1, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, limit_price=limit,
        ))
        print(f"order       submitted id={order.id} limit={limit} status={order.status}")
        try:
            tc.cancel_order_by_id(order.id)
        except Exception as e:
            final = tc.get_order_by_id(order.id)
            print(f"FAIL: cancel failed ({e}); order status={final.status} "
                  f"filled_avg={final.filled_avg_price}. Check positions.")
            return 1
        print(f"order       cancelled, final status={tc.get_order_by_id(order.id).status}")

    print("OK: paper account reachable, options listed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
