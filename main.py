import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_data, compute_performance_metrics, print_results
from strategies.longshort import run_backtest_longshort
from strategies.mean_variance import run_backtest_meanvar
from strategies.buy_hold import *

def main():
    df, stock_data, rf = get_data(start_date='2012-01-01', end_date='2021-12-31')
    
    print("\n=== Mean-Variance Strategy Results ===")
    mv_metrics, mv_daily = run_backtest_meanvar(stock_data, rf, frequency="yearly")
    mv_results = compute_performance_metrics(mv_metrics, mv_daily, rf)
    print_results(mv_results)
    
    print("\n===== Buy & Hold Strategy =====")
    bh_period_metrics, bh_daily_series = run_backtest_buyhold(stock_data, df, rf, frequency="yearly")
    bh_results = compute_performance_metrics(bh_period_metrics, bh_daily_series, rf)
    print_results(bh_results)
    
    print("\n===== Long/Short Strategy =====")
    ls_period_metrics, ls_daily_series = run_backtest_longshort(stock_data, df, rf, frequency="yearly")
    ls_results = compute_performance_metrics(ls_period_metrics, ls_daily_series, rf)
    print_results(ls_results)

if __name__ == "__main__":
    main()