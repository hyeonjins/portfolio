# mvp.py

import numpy as np
from scipy.optimize import minimize
from utils import calc_portfolio_metrics, get_rebalancing_dates, get_is_oos, SharpeCalculator

class MeanVarianceStrategy:
    def __init__(self, rf: float):
        self.rf = rf
            
    def max_sharpe_ratio(self, w, mu, cov_matrix):
        """
        샤프비율 최대화 함수
        """
        ret = np.dot(w, mu) # 포트폴리오 수익률
        std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) # 포트폴리오 표준편차
        return - (ret - self.rf) / std  # 최소화를 위해 음수 변환

    def min_risk(self, w, cov_matrix):
        """
        변동성 최소화 함수
        """
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    def max_return(self, w, mu):
        """
        수익률 최대화 함수
        """
        return -np.dot(w, mu)  # 최소화를 위해 음수 변환

    def get_mvp_weights(self, insample_data):
        """
        MVP 포트폴리오 가중치 계산 함수
        """
        annual_returns_insample = insample_data.mean() * 252 # 연간화 수익률
        cov_matrix_insample = insample_data.cov() * 252 # 연간화 공분산 행렬
        n_assets = len(annual_returns_insample) # 자산 수
        w0 = np.ones(n_assets) / n_assets # 초기 가중치(동일 가중치)
        bounds = [(0, 1) for _ in range(n_assets)] #가중치 범위 (0~1)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} # 가중치 합계가 1이 되도록 제약조건 설정
        
        mu_insample = annual_returns_insample.values # 연간화 수익률 벡터
        cov_insample = cov_matrix_insample.values # 공분산 행렬
        
        # 각 MVP 포트폴리오의 최적 가중치 계산
        weights_sharpe = minimize(self.max_sharpe_ratio, w0, args=(mu_insample, cov_insample),
                                    method='SLSQP', bounds=bounds, constraints=constraints).x
        weights_risk = minimize(self.min_risk, w0, args=(cov_insample,),
                                method='SLSQP', bounds=bounds, constraints=constraints).x
        weights_return = minimize(self.max_return, w0, args=(mu_insample,),
                                  method='SLSQP', bounds=bounds, constraints=constraints).x
        
        return weights_sharpe, weights_risk, weights_return

def run_backtest_meanvar(stock_data, rf, frequency="yearly"):
    """
    평균-분산 전략 백테스트 실행
    """
    strategy = MeanVarianceStrategy(rf)
    calc = SharpeCalculator(rf)
    portfolio_types = ["Maximum Sharpe Ratio", "Minimum Risk", "Maximum Return"]

    # 리밸런싱 구간별 성과 기록용 딕셔너리
    results_outsample = {p: {"return": [], "std": [], "sharpe": []} for p in portfolio_types} # 리밸런싱 구간별 딕셔너리
    daily_returns_dict = {p: [] for p in portfolio_types} # 일별 수익률 딕셔너리
    
    rebalancing_dates = get_rebalancing_dates(stock_data, frequency) # 리밸런싱 날짜 계산

    for i in range(len(rebalancing_dates) - 2):
        insample_data, outsample_data = get_is_oos(stock_data, rebalancing_dates, i)
        weights = {}
        weights["Maximum Sharpe Ratio"], weights["Minimum Risk"], weights["Maximum Return"] = strategy.get_mvp_weights(insample_data)
        
        annual_returns_outsample = outsample_data.mean() * 252 # 연간화 수익률
        cov_matrix_outsample = outsample_data.cov() * 252 # 연간화 공분산 행렬
        mu_outsample = annual_returns_outsample.values # 연간화 수익률 벡터
        cov_outsample = cov_matrix_outsample.values # 공분산 행렬
        
        for port_type in portfolio_types:
            ret, std, sharpe = calc_portfolio_metrics(weights[port_type], mu_outsample, cov_outsample, rf)
            results_outsample[port_type]["return"].append(ret)
            results_outsample[port_type]["std"].append(std)
            results_outsample[port_type]["sharpe"].append(sharpe)
            
            daily_returns = (outsample_data @ weights[port_type]).tolist() # 포트폴리오 일별 수익률
            daily_returns_dict[port_type].extend(daily_returns) # 일별 수익률 저장
    
    period_metrics = {"mvp": results_outsample}  # 리밸런싱 구간별 성과값
    daily_series = {"mvp": daily_returns_dict} # 일별 수익률 시계열
    
    return period_metrics, daily_series
