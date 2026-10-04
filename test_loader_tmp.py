import sys
sys.path.insert(0, ".")
import research.backtest as bt

bars, dates = bt.load_ticker("AAPL")
print("bars:", len(bars), "first date:", dates[0], "last date:", dates[-1])
print("first close:", bars[0].close, "last close:", bars[-1].close)
print("dates[0]:", dates[0], "dates[60]:", dates[60])
