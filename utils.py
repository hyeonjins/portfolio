import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings("ignore")

def get_data(start_date='2012-01-01', end_date='2021-12-31', max_workers=20):
    """
    데이터 로딩, 병합, 주식 데이터 다운로드, 무위험 수익률 계산
    """
    # CSV 파일 읽기
    # df = pd.read_csv('./data/parsed_result_sector_prompt_version_31_0106_cot_10y_all.csv')
    # tbl = pd.read_csv('./data/parsed_result_sector_prompt_version_31_1104_10y_all.csv')
    # df = df.merge(tbl[['permno', 'ticker', 'year', 'month', 'shrout']], on=['permno', 'year', 'month'], how='inner')
    
    df = pd.read_csv('./data/parsed_result_ds_prob_7_snp500_10y_new_2012_2021_250409_all.csv')
    # df_under_12 = df.groupby(['permno', 'year'])['month'].count().reset_index().query('month != 12')['permno'].unique()#.sort_values('month', ascending=False)#['permno'].nunique()#['month'].value_counts()
    # df = df[~df.isin(df_under_12).any(axis=1)]
    
    df['market_cap'] = df['shrout'] * abs(df['prc'])
    df['date'] = pd.to_datetime(df['date'])
    
    # 티커 추출
    unique_tickers = df['ticker'].unique().tolist()
    price_series_list = []

    # 티커별 주가 데이터를 다운로드하는 함수
    def download_ticker_data(ticker_name):
        try:
            ticker = yf.Ticker(ticker_name)
            prices = ticker.history(start=start_date, end=end_date, auto_adjust=True)['Close'].dropna()
            if not prices.empty:
                return ticker_name, prices.rename(ticker_name)
            return None
        except Exception as e:
            return None
    
    price_series_list = []
    
    # 멀티스레딩으로 yfinance에서 데이터 다운로드
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 티커에 대한 future 객체 생성
        future_to_ticker = {executor.submit(download_ticker_data, ticker): ticker for ticker in unique_tickers}
        
        for future in as_completed(future_to_ticker):
            result = future.result()
            if result:
                ticker_name, price_series = result
                price_series_list.append(price_series)

    # 값 결합
    stock_data = pd.concat(price_series_list, axis=1)
    stock_data.index = pd.to_datetime(stock_data.index.tz_localize(None))
    stock_data = stock_data.sort_index()
    
    # 결측치 제거 (모든 티커의 데이터가 존재하는 행만 남김)
    stock_data = stock_data.dropna(axis=0, how='all')
    
    # 무위험 수익률 계산 (연간 평균)
    rf = yf.download('^IRX', start=start_date, end=end_date,
                     auto_adjust=True, progress=False)['Close'].mean().iloc[0] / 100
    
    return df, stock_data, rf

def get_rebalancing_dates(stock_data, frequency='yearly'):
    """
    리밸런싱 날짜 계산 (월별, 분기별, 년도별)
    """
    if frequency == 'monthly':
        rebalancing_dates = stock_data.resample('M').last().index
    elif frequency == 'quarterly':
        rebalancing_dates = stock_data.resample('Q').last().index
    elif frequency == 'yearly':
        monthly_index = stock_data.resample('M').last().index
        rebalancing_dates = monthly_index[monthly_index.month == 6]
    return rebalancing_dates

def get_is_oos(stock_data, rebalancing_dates, idx):
    """
    in‑sample / out‑of‑sample 데이터 분할 (리밸런싱 기간 별)
    """
    insample_start = rebalancing_dates[idx]
    insample_end = rebalancing_dates[idx+1]
    outsample_start = rebalancing_dates[idx+1]
    outsample_end = rebalancing_dates[idx+2]
    
    insample_data = stock_data.loc[insample_start:insample_end].pct_change().iloc[1:]
    outsample_data = stock_data.loc[outsample_start:outsample_end].pct_change().iloc[1:]
    
    # 두 기간에서 공통 데이터 보유 티커 선택
    combined_data = pd.concat([insample_data, outsample_data], axis=0).dropna(axis=1)
    insample_data = combined_data.loc[insample_data.index]
    outsample_data = combined_data.loc[outsample_data.index]
    # print("is:", insample_data.head())
    # print("oos:", outsample_data.head())
    
    return insample_data, outsample_data

def apply_weight_strategy(portfolio_df, weight_strategy='equal'):
    """
    포트폴리오의 가중치 할당 (동일 가중치 또는 시가총액 가중치)
    """
    portfolio_df = portfolio_df.copy()
    if weight_strategy == 'equal':
        portfolio_df['weight'] = 1.0 / len(portfolio_df)
    elif weight_strategy == 'value':
        portfolio_df['weight'] = portfolio_df['market_cap'] / portfolio_df['market_cap'].sum()
    return portfolio_df

def calc_portfolio_metrics(w, mu, cov_matrix, rf):
    """
    포트폴리오의 연간 수익률, 위험 및 Sharpe 비율 계산
    """
    ret = np.dot(w, mu)
    std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = (ret - rf) / std
    return ret, std, sharpe

class SharpeCalculator:
    """
    Sharpe Ratio 계산 3가지 방법
    """
    def __init__(self, rf: float):
        self.rf = rf

    def method1(self, metrics_dict):
        # 리밸런싱 구간별 수익률 평균 및 표준편차 기반 계산
        ret = np.mean(metrics_dict["return"])
        std = np.std(metrics_dict["return"])
        sharpe = (ret - self.rf) / std
        return ret, std, sharpe

    def method2(self, returns_array):
        # 전체 일간 수익률 시계열 기반 계산
        ret = np.mean(returns_array) * 252
        std = np.std(returns_array) * np.sqrt(252)
        sharpe = (ret - self.rf) / std
        return ret, std, sharpe

    def method3(self, metrics_dict):
        # 각 구간별 결과의 평균 후 Sharpe 계산
        ret = np.mean(metrics_dict["return"])
        std = np.mean(metrics_dict["std"])
        sharpe = (ret - self.rf) / std
        avg_sharpe = np.mean(metrics_dict["sharpe"])
        return ret, std, sharpe, avg_sharpe

def compute_portfolio_daily_returns(outsample_data, portfolio_df, sign=1):
    """
    out‑sample 데이터와 포트폴리오 DataFrame을 기반으로
    각 종목의 가중치를 고려한 일별 포트폴리오 수익률 계산
    sign: 1 (롱) 또는 -1 (숏)
    """
    daily_returns = pd.Series(0.0, index=outsample_data.index, dtype=float)
    for _, row in portfolio_df.iterrows():
        ticker = row['ticker']
        if ticker in outsample_data.columns:
            # 해당 티커의 일별 수익률에서 NaN을 0으로 대체
            ticker_returns = outsample_data[ticker].fillna(0)
            daily_returns += sign * ticker_returns * row['weight']
    return daily_returns


def compute_performance_metrics(period_metrics, daily_series, rf):
    """
    3가지 방법으로 포트폴리오 성과 지표 계산
    """
    calc = SharpeCalculator(rf)
    methods = {
        "Method 1": {},
        "Method 2": {},
        "Method 3": {}
    }
    
    # 각 가중치 전략, 포트폴리오 유형 별 성과 계산 (방법 1, 3)
    for scheme, ports in period_metrics.items():
        methods["Method 1"][scheme] = {}
        methods["Method 3"][scheme] = {}
        for port, metrics in ports.items():
            ret1, std1, sharpe1 = calc.method1(metrics)
            methods["Method 1"][scheme][port] = {
                "Return": ret1,
                "Std": std1,
                "Sharpe Ratio": sharpe1
            }
            ret3, std3, sharpe3, avg_sharpe3 = calc.method3(metrics)
            methods["Method 3"][scheme][port] = {
                "Return": ret3,
                "Std": std3,
                "Sharpe Ratio": sharpe3,
                "Avg. Sharpe Ratio": avg_sharpe3
            }
    
    # 방법 2: 모든 일별 수익률 통합 계산
    for scheme, ports in daily_series.items():
        methods["Method 2"][scheme] = {}
        for port, series_list in ports.items():
            combined_daily = pd.concat(series_list) if isinstance(series_list[0], pd.Series) else np.array(series_list)
            ret2, std2, sharpe2 = calc.method2(combined_daily.values if isinstance(combined_daily, pd.Series) else combined_daily)
            methods["Method 2"][scheme][port] = {
                "Return": ret2,
                "Std": std2,
                "Sharpe Ratio": sharpe2
            }
    
    return methods

def print_results(results_dict):
    """
    결과 출력 함수
    """
    for method_name, method_results in results_dict.items():
        print(f"\n{method_name}:")
        for scheme, data in method_results.items():
            df_res = pd.DataFrame(data).T
            print(f"<{scheme}>")
            print(df_res, "\n")