import pandas as pd
from collections import defaultdict


def get_X_y(df, target_column, hide_columns, date_column):
    """preapre X and y data from the provided information"""

    if not df.index.name == date_column:
        df.set_index(date_column, inplace=True)

    y = df[target_column].copy()
    hide_columns.append(target_column)

    X = df[df.columns.drop(hide_columns, errors='ignore')].copy()
    return X, y


def train_model_and_apply_test(x_train, x_test, y_train, y_test, results, model):
    """apply training and test"""
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if len(pred.shape) == 2:
        pred = pred[:, 1]
    results['prediction'].extend(pred)
    results['truth'].extend(y_test)
    results['date'].extend(x_test.index)

    return results


def training(
        df=None,
        backtest_settings=None,
        target='',
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

    # update the default backtest settings (bs)
    bs = {'backtest_method': 'simple_split', 'split_ratio': 0.7, 'step_size': 10, 'training_window': 1000}
    bs.update(backtest_settings)

    X, y = get_X_y(df=df, date_column=date_column, hide_columns=hide_columns, target_column=target)

    results = defaultdict(list)

    if bs['backtest_method'] == 'simple_split':
        cutoff = int(df.shape[0] * bs['split_ratio'])
        x_train, x_test = X[:cutoff], X[cutoff:]
        y_train, y_test = y[:cutoff], y[cutoff:]

        results = train_model_and_apply_test(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                             model=model, results=results)

    res_df = pd.DataFrame({'prediction': results['prediction'], 'truth': results['truth']}, index=results['date'])
    return res_df


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression

    df = pd.read_csv('../../financial_data.csv')
    result = training(
        df=df,
        backtest_settings={'backtest_method': 'simple_split'},
        model=LinearRegression(),
        date_column='Date',
        target='ret_1d',
        hide_columns=['ret_10d', 'target']
    )
    print(result)
