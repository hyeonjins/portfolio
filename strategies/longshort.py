import pandas as pd
import numpy as np
from utils import (
    get_rebalancing_dates, 
    get_is_oos, 
    compute_portfolio_daily_returns, 
    SharpeCalculator,
    apply_weight_strategy
)

class LongShort:
    def __init__(self):
        pass

    def select_portfolios(self, filtered_df, end_date, top_fraction=0.1):
        """
        in‑sample 기간 동안 롱/숏 포트폴리오 후보를 선택
         - Long: return_movement > 0 이며 magnitude_of_change가 'Moderate' 또는 'Large'
         - Short: return_movement == 0 이며 magnitude_of_change가 'Moderate' 또는 'Large'
         (현재 코드는 return_movement 최대값, 최소값 행을 선택해서 롱/숏에 반영함. )
        """
        unique_tickers = filtered_df['ticker'].unique() # uniqe 티커 목록 추출
        top_count = max(1, int(len(unique_tickers) * top_fraction)) # 전체의 10%가 되도록 설정
        
        # 롱 포트폴리오 후보 선택
        long_stocks = filtered_df[
            #(filtered_df['return_movement'] >= 0.5) & 
            (filtered_df['magnitude_of_change'].isin(['Moderate', 'Large'])) # 변화의 크기가 Moderate 또는 Large인 주식 필터링
        ]
        long_top = long_stocks.groupby('ticker')['return_movement'].idxmax().reset_index()\
                    .sort_values('return_movement', ascending=False).head(top_count) 
        long_selected = long_stocks[long_stocks['ticker'].isin(long_top['ticker'])].sort_values('date') # 선택된 롱 주식 정렬
        long_selected['portfolio_date'] = end_date # 포트폴리오 날짜 추가
        
        # 숏 포트폴리오 후보 선택
        short_stocks = filtered_df[
            #(filtered_df['return_movement'] < 0.5) & 
            (filtered_df['magnitude_of_change'].isin(['Moderate', 'Large'])) # 변화의 크기가 Moderate 또는 Large인 주식 필터링
        ]
        short_top = short_stocks.groupby('ticker')['return_movement'].idxmin().reset_index()\
                    .sort_values('return_movement', ascending=True).head(top_count)
        short_selected = short_stocks[short_stocks['ticker'].isin(short_top['ticker'])].sort_values('date') # 선택된 숏 주식 정렬
        short_selected['portfolio_date'] = end_date # 포트폴리오 날짜 추가
        
        # 롱, 숏 포트폴리오 티커 확인
        # print("long:", long_selected['ticker'].nunique(), "long:", long_selected['ticker'].unique())
        # print("short:", short_selected['ticker'].nunique(), "short:", short_selected['ticker'].unique())
        return long_selected, short_selected

def run_backtest_longshort(stock_data, df, rf, frequency="yearly", top_fraction=0.1):
    """
    롱/숏 백테스트 실행
    """
    strategy = LongShort()
    calc = SharpeCalculator(rf)
    
    strategies = ["equal", "value"]
    portfolio_types = ["Long", "Short", "Long-Short"]
    
    period_metrics = {s: {p: {"return": [], "std": [], "sharpe": []} for p in portfolio_types} for s in strategies} # 리밸런싱 구간별 딕셔너리
    daily_series = {s: {p: [] for p in portfolio_types} for s in strategies} # 일별 수익률 딕셔너리
    
    rebalancing_dates = get_rebalancing_dates(stock_data, frequency) # 리밸런싱 날짜 계산
    
    df['date'] = pd.to_datetime(df['date']).dt.normalize() # 날짜 형식 정규화
    
    for i in range(len(rebalancing_dates) - 2):
        insample_data, outsample_data = get_is_oos(stock_data, rebalancing_dates, i) # in-sample, out-sample 데이터 분할
        insample_end = rebalancing_dates[i+1].normalize() # in-sample 마지막 날짜
        current_data = df[df['date'].isin(insample_data.index)].copy() # 현재 in-sample 데이터 추출
        
        long_sel, short_sel = strategy.select_portfolios(current_data, insample_end, top_fraction=top_fraction) #롱, 숏에 속하는 티커 추출
        
        for weight_strategy in strategies:
            long_weighted = apply_weight_strategy(long_sel, weight_strategy) # 롱 포트폴리오에 가중치 적용
            short_weighted = apply_weight_strategy(short_sel, weight_strategy) # 숏 포트폴리오에 가중치 적용
            
            daily_long = compute_portfolio_daily_returns(outsample_data, long_weighted, sign=1) # 롱 포트폴리오의 일일 수익률
            daily_short = compute_portfolio_daily_returns(outsample_data, short_weighted, sign=-1) # 숏 포트폴리오의 일일 수익률
            daily_ls = daily_long + daily_short # 롱-숏 포트폴리오의 일일 수익률
            
            for port_type, series in zip(portfolio_types, [daily_long, daily_short, daily_ls]):
                # 연간 수익률, 표준편차, 샤프 비율 계산
                ann_ret = np.mean(series) * 252
                ann_std = np.std(series) * np.sqrt(252)
                ann_sharpe = (ann_ret - rf) / ann_std
                
                period_metrics[weight_strategy][port_type]["return"].append(ann_ret)
                period_metrics[weight_strategy][port_type]["std"].append(ann_std)
                period_metrics[weight_strategy][port_type]["sharpe"].append(ann_sharpe)
                
                daily_series[weight_strategy][port_type].append(series)
                
    return period_metrics, daily_series