import argparse
import time
import os
import pickle as pkl
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.linear_model import Ridge

pjoin = os.path.join


def mse(y, yhat):
    return np.sqrt(np.mean(np.square(y - yhat)))

def generate_predictors_responses(df, n_lags):
    df_lags = pd.concat([df.shift(i)[["Adj Close"]] if i == 0 else df.shift(i).rename(columns=lambda x: x + f"@-{i}") for i in range(n_lags + 1)], axis=1)
    df_lags = df_lags.dropna()

    return (df_lags.drop(columns=["Adj Close"]), df_lags[["Adj Close"]])

def least_squares_predict(df, train_start, train_end, alpha, n_lags):
    X_df, y_df = generate_predictors_responses(df[["Adj Close"]], n_lags)

    X_train = X_df[(train_start <= X_df.index.get_level_values(1)) & (X_df.index.get_level_values(1) <= train_end)].values
    y_train = y_df[(train_start <= y_df.index.get_level_values(1)) & (y_df.index.get_level_values(1) <= train_end)].values

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    yhat = model.predict(X_df.values) # need all data in order to predict for first n_lags samples of validation set

    yhat_df = pd.DataFrame(
        data=yhat,
        index=y_df.index,
        column="Predictd Log Returns"
    )

    return yhat_df

def least_squares_fit(df, args, alpha, n_lags):
    start_time = time.time()

    # maybe filter out test data here. We don't need it and could speed stuff up a bit
    X_df, y_df = generate_predictors_responses(df[["Adj Close"]], n_lags)

    X_train = X_df[(args.train_start <= X_df.index.get_level_values(1)) & (X_df.index.get_level_values(1) <= args.train_end)].values
    y_train = y_df[(args.train_start <= y_df.index.get_level_values(1)) & (y_df.index.get_level_values(1) <= args.train_end)].values

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    yhat = model.predict(X_df.values) # need all data in order to predict for first n_lags samples of validation set

    yhat_df = pd.DataFrame(
        data=yhat,
        index=y_df.index,
    )
   
    yhat_validation = yhat_df[(args.validation_start <= yhat_df.index.get_level_values(1)) & (yhat_df.index.get_level_values(1) <= args.validation_end)].values
    y_validation = y_df[(args.validation_start <= y_df.index.get_level_values(1)) & (y_df.index.get_level_values(1) <= args.validation_end)].values

    mse_validation = mse(y_validation, yhat_validation)
    end_time = time.time()

    return {"loss": mse_validation, "status": STATUS_OK, "train_validation_time": end_time - start_time}

def main(args):
    max_evals = args.n_trials

    try:
        trials = pkl.load(open(args.hyperopt_trials, "rb"))
        print(f"Loaded saved HyperOpt Trials object from {args.hyperopt_trials}", flush=True)
        n_prev_trials = len(trials.trials)
        max_evals += n_prev_trials
        print(f"Rerunning from {n_prev_trials} trials.", flush=True)
    except:
        trials = Trials()
        print("No saved HyperOpt Trials object found. Starting from scratch", flush=True)

    space = {
        "n_lags": hp.choice("n_lags", [2**i for i in range(3,9)]),
        "alpha": hp.loguniform("alpha", np.log(0.0001), np.log(1000)),
    }

    df = pd.read_parquet(args.file)

    best = fmin(
        fn=lambda hps: least_squares_fit(df, args, **hps),
        space=space,
        algo=tpe.suggest, 
        max_evals=max_evals,
        trials=trials,
    )

    pkl.dump(trials, open(args.hyperopt_trials, "wb"))

    best_acc = min(trial_data["result"]["loss"] for trial_data in trials.trials)
    print(f"Best validation accuracy: {best_acc:.4}", flush=True)
    print(f"Best params: {best}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--file", type=str, default="data/preprocessed/df_log_features.parquet")
    parser.add_argument("--train-start", type=str, default="1998-01-01")
    parser.add_argument("--train-end", type=str, default="2015-12-31")
    parser.add_argument("--validation-start", type=str, default="2016-01-01")
    parser.add_argument("--validation-end", type=str, default="2017-12-31")
    parser.add_argument("--test-start", type=str, default="2018-01-01")
    parser.add_argument("--hyperopt-trials", type=str, required=True)
    args = parser.parse_args()

    main(args)
