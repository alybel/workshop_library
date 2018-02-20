import numpy as np


class Backtest(object):
    def __init__(self, trading_signal=None, underlying_series=None, transaction_cost_in_bp=0,
                 time_periods_per_year=250):
        self.trading_signal = trading_signal
        self.underlying_series = underlying_series.loc[self.trading_signal.index]
        self.transaction_cost_in_bp = transaction_cost_in_bp
        self.time_periods_per_year = time_periods_per_year

    @property
    def underlying_returns(self):
        return self.underlying_series.pct_change()

    @property
    def transaction_costs_per_period(self):
        return (self.transaction_cost_in_bp / 10000 * np.abs(self.trading_signal.diff())).shift(1).fillna(0)

    @property
    def returns(self):
        r = self.underlying_returns * self.trading_signal.shift(
            1) - self.transaction_costs_per_period
        r[0] = 0
        return r

    @property
    def performance(self):
        return np.cumprod(self.returns + 1) * 100

    @property
    def backtest_time_span_in_years(self):
        return (self.trading_signal.index.max() - self.trading_signal.index.min()).days / 365.

    @property
    def return_per_annum(self):
        return np.power(self.performance[-2] / 100, (1 / self.backtest_time_span_in_years)) - 1

    @property
    def vola_per_annum(self):
        return self.performance.pct_change().std() * np.sqrt(self.time_periods_per_year)

    @property
    def sharpe_ratio(self):
        return self.return_per_annum / self.vola_per_annum


class SimpleBacktest(Backtest):
    def __init__(self,
                 prediction_series=None,
                 underlying_series=None,
                 transaction_cost_in_bp=0,
                 is_regr_class='class'
                 ):
        underlying_series = underlying_series.loc[prediction_series.index]
        trading_signal = 2 * (prediction_series > 0) - 1 if is_regr_class == 'regr' else 2 * (
                prediction_series > 0.5) - 1
        Backtest.__init__(self, trading_signal=trading_signal, underlying_series=underlying_series,
                          transaction_cost_in_bp=transaction_cost_in_bp)
