import pandas as pd
from collections import defaultdict


def get_X_y(df, target_column, hide_columns, date_column):
    """preapre X and y data from the provided information"""

    if not df.index.name == date_column:
        df.set_index(date_column, inplace=True)

    df.index = pd.to_datetime(df.index)

    if not df.index.is_monotonic_increasing:
        raise AttributeError('provided date column is not monotonically increasing')

    y = df[target_column].copy()
    hide_columns.append(target_column)

    X = df[df.columns.drop(hide_columns, errors='ignore')].copy()
    return X, y


def train_model_and_apply_test(x_train, x_test, y_train, y_test, results, model, round=0):
    """apply training and test"""
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if len(pred.shape) == 2:
        pred = pred[:, 1]
    results['prediction'].extend(pred)
    results['truth'].extend(y_test)
    results['date'].extend(x_test.index)
    results['round'].extend([round] * x_test.shape[0])

    return results


def training(
        df=None,
        backtest_settings=None,
        target=None,
        hide_columns=None,
        model=None,
        date_column=None,
):
    """
    Have several training methods available, all are time-series consistent.
    :param df: DataFrame
    :param backtest_settings: backtest_method [simple_split, walk_forward_rolling, walk_forward_extending], split_ratio, step_size, training_window
    :param hide_columns: columns that should not be part of the training
    :param model: the model to use
    :param date_column: the date column that is in the data
    :return: training_results
    """

    if target is None:
        raise SystemExit('A target column needs to be provided')
    if model is None:
        raise SystemExit('A valid model needs to be provided')
    if date_column is None:
        raise SystemExit('A valid date_column needs to be provided')
    if hide_columns is None:
        hide_columns = []

    # update the default backtest settings (bs)
    bs = {'backtest_method': 'simple_split', 'split_ratio': 0.7, 'step_size': 10, 'training_window': 1000,
          'test_train_diff_days': 1}
    bs.update(backtest_settings)

    X, y = get_X_y(df=df, date_column=date_column, hide_columns=hide_columns, target_column=target)

    results = defaultdict(list)

    if bs['backtest_method'] == 'simple_split':
        cutoff = int(df.shape[0] * bs['split_ratio'])
        x_train, x_test = X[:cutoff], X[cutoff:]
        y_train, y_test = y[:cutoff], y[cutoff:]

        results = train_model_and_apply_test(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                             model=model, results=results, round=0)

    elif bs['backtest_method'] in ['walk_forward_rolling', 'walk_forward_extending']:
        selection_dates = X.index.unique()
        n_dates = len(selection_dates)

        if n_dates < X.shape[0] - 5:
            raise ValueError('provided data has not enough rows to perform a rolling model bulding')

        test_start_idx = bs['training_window'] + bs['test_train_diff_days']
        i = 0
        while test_start_idx < df.shape[0] - bs['step_size']:
            train_start_idx = 0 if bs['backtest_method'] == 'walk_forward_extending' else bs['step_size'] * i
            train_end_idx = bs['training_window'] + i * bs['step_size']
            test_start_idx = train_end_idx + bs['test_train_diff_days']
            test_end_idx = test_start_idx + bs['step_size']
            this_round_train_dates = selection_dates[train_start_idx:train_end_idx]
            this_round_test_dates = selection_dates[test_start_idx:test_end_idx]

            results = train_model_and_apply_test(
                x_train=X.loc[this_round_train_dates],
                x_test=X.loc[this_round_test_dates],
                y_train=y.loc[this_round_train_dates],
                y_test=y.loc[this_round_test_dates],
                model=model,
                results=results,
                round=i
            )
            i += 1

    res_df = pd.DataFrame({'prediction': results['prediction'], 'truth': results['truth'], 'round': results['round']},
                          index=results['date'])
    return res_df


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    from workshop_library import financial_backtest
    import numpy as np

    df = pd.read_csv('../financial_data.csv')
    result = training(
        df=df,
        backtest_settings={'backtest_method': 'walk_forward_rolling', 'training_window': 500, 'step_size': 20,
                           'test_train_diff_days': 1},
        model=LinearRegression(),
        date_column='Date',
        target='ret_1d',
        hide_columns=['ret_10d', 'target']
    )

    bt = financial_backtest.SimpleBacktest(
        prediction_series=result['prediction'],
        underlying_series=df['close'],
        transaction_cost_in_bp=2,
        is_regr_class='regr'
    )

    assert np.isclose(bt.sharpe_ratio, 0.00109117874343)

    from sklearn.linear_model import LogisticRegression
    df['target'] = df['ret_1d'] > 0
    result2 = training(
        df=df,
        backtest_settings={'backtest_method': 'walk_forward_rolling', 'training_window': 500, 'step_size': 20,
                           'test_train_diff_days': 1},
        model=LogisticRegression(),
        date_column='Date',
        target='target',
        hide_columns=['ret_10d', 'target', 'ret_1d']
    )

    bt2 = financial_backtest.SimpleBacktest(
        prediction_series=result2['prediction'],
        underlying_series=df['close'],
        transaction_cost_in_bp=2,
        is_regr_class='class'
    )

    assert np.isclose(bt2.sharpe_ratio, -0.400204689504)