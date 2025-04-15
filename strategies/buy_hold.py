import pandas as pd
import numpy as np
from utils import apply_weight_strategy, compute_portfolio_daily_returns, get_rebalancing_dates

def run_backtest_buyhold(stock_data, df, rf, frequency="yearly"):
    """
    Buy&Hold 백테스트 실행
    """
    # 각 주식마다 첫 번째 날짜의 데이터를 사용하여 포트폴리오 생성
    portfolio_df = df.sort_values('date').groupby('ticker').first().reset_index()

    # yfinance 데이터에 있는 티커만 사용
    available_tickers = set(stock_data.columns)
    portfolio_df = portfolio_df[portfolio_df['ticker'].isin(available_tickers)] # 유효한 티커만 필터링

    tickers = portfolio_df['ticker'].unique().tolist() # 포트폴리오에 있는 고유 티커 목록을 가져옴
    stock_data_subset = stock_data[tickers] # 주식 데이터에서 해당 티커의 데이터만 추출
    portfolio_df = portfolio_df.drop_duplicates(subset=['ticker']) # 중복 티커 제거

    portfolio_types = ["equal", "value"]
    
    # 각 전략별 포트폴리오 계산 (equal, value)
    portfolios = {}
    for ptype in portfolio_types:
        portfolios[ptype] = apply_weight_strategy(portfolio_df, weight_strategy=ptype)

    daily_data = stock_data_subset.pct_change().iloc[1:]  # 일별 수익률 계산
    daily_returns = {ptype: compute_portfolio_daily_returns(daily_data, portfolios[ptype], sign=1) for ptype in portfolio_types} # 각 포트폴리오의 일별 수익률 계산
    rebalancing_dates = get_rebalancing_dates(stock_data, frequency) # 리밸런싱 날짜 계산(sharpe ratio 계산을 위해)

    period_metrics = {p: {"return": [], "std": [], "sharpe": []} for p in portfolio_types}  # 리밸런싱 구간별 성과 기록
    daily_returns_dict = {p: [] for p in portfolio_types}  # 일별 수익률 딕셔너리

    # 구간별 성과 계산
    for port_type in portfolio_types:
        seg_returns = []
        seg_std = []
        seg_sharpe = []
        
        # 리밸런싱 구간별 성과 계산
        for i in range(len(rebalancing_dates) - 1):
            start_date = rebalancing_dates[i] # 리밸런싱 시작일
            end_date = rebalancing_dates[i + 1] # 리밸런싱 종료일
            
            segment = daily_returns[port_type].loc[start_date:end_date] # 해당 구간의 일별 수익률
        
            ann_ret = np.mean(segment) * 252  # 연간화 수익률
            ann_std = np.std(segment) * np.sqrt(252)  # 연간화 표준편차
            ann_sharpe = (ann_ret - rf) / ann_std  # 샤프 비율
            
            seg_returns.append(ann_ret)
            seg_std.append(ann_std)
            seg_sharpe.append(ann_sharpe)
        
        # 구간별 성과를 period_metrics에 저장
        period_metrics[port_type]["return"].extend(seg_returns)
        period_metrics[port_type]["std"].extend(seg_std)
        period_metrics[port_type]["sharpe"].extend(seg_sharpe)

        daily_returns_dict[port_type] = daily_returns[port_type]  # 일별 수익률 저장

    # 구간별 성과값
    period_metrics = {
        "buy_hold": {
            "Equal": {
                "return": period_metrics["equal"]["return"],
                "std": period_metrics["equal"]["std"],
                "sharpe": period_metrics["equal"]["sharpe"]
            },
            "Value": {
                "return": period_metrics["value"]["return"],
                "std": period_metrics["value"]["std"],
                "sharpe": period_metrics["value"]["sharpe"]
            }
        }
    }

    # 일별 수익률 시계열
    daily_series = {
        "buy_hold": {
            "Equal": daily_returns_dict["equal"],
            "Value": daily_returns_dict["value"]
        }
    }

    return period_metrics, daily_series
