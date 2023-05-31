import numpy as np
from model import *
from create_dataset import *
from feature_engineering import reviews_processed, apply_batch, pre_processing, transform
from visualize import error_plot, plot_loss_per_epochs


def tune_SGD(X_train, y_train, params):
    GSDlogred = SGDClassifier(loss='log', learning_rate='optimal', shuffle=True,
                              verbose=0)  # add additional parameters to object /, max_iter=1000
    clf = GridSearchCV(GSDlogred, param_grid=params)
    clf.fit(X_train, y_train)
    return (clf.best_score_, clf.best_params_)


def main():
    # Reading the Data
    s3_bucket = S3BUCKET
    keys_list_indices = 11

    bucket = connect_to_s3bucket(s3_bucket)
    keys_list_filtered, df = create_data(bucket=bucket, keys_list_indices=keys_list_indices)

    # Data preprocessing and feature engineering:
    df_reviews_processed = reviews_processed(df)
    print("Digital_Software has {} rows(data points) and {} columns(features)".format(df_reviews_processed.shape[0],
                                                                                      df_reviews_processed.shape[1]))
    # Using a mini_batch_size of 128, and a batch size of 10,000:
    df_sent = apply_batch(df_reviews_processed, batch_size=128)
    # Make sure the types of the data is compatible with modeling:
    df_processed = pre_processing(df_sent)
    # Add the features of the dataframe that you want to transform and/or combine
    final_df = transform(df_processed)
    print("Final Digital_Software matrix has {} rows(data points) and {} columns(features)".format(final_df.shape[0],
                                                                                                   final_df.shape[1]))
    # Fitting a classification model to a large dataset
    # split to train-test:
    X_train, X_test, y_train, y_test = train_test_split(final_df, np.array(df_processed.iloc[:, -1]), test_size=0.2)

    # standarize using logistic regression both train and test sets
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(X_train, y_train)  # apply scaling on training data
    pipe.fit(X_test, y_test)  # apply scaling on test data

    model = LogisticRegression()
    model.fit(X_train, y_train)
    model_accuracy_train = round(model.score(X_train, y_train) * 100, 2)
    model_accuracy_test = round(model.score(X_test, y_test) * 100, 2)
    print(
        "The accuracy of the Logistic Regression model of Digital_Software Train set is: {}% , and Test set is: {}%".format(
            model_accuracy_train, model_accuracy_test))
    test_error = metrics.mean_squared_error(y_test, model.predict(X_test))
    print("The test error of the Logistic Regression model of Digital_Software is : {}".format(test_error))

    # For different lengths of  𝑛=10×2𝑘, for  𝑘=0,1,..,14.
    # extract only the first 𝑛 values in the train set and use them to fit the logistic regression model.
    log_sample = 15
    lengths = [10 * (2 ** k) for k in range(log_sample)]
    train_error = []
    test_error = []
    for n in lengths:
        if n >= len(X_train):  # If n is larger than the total number of rows, set n to the X_train length
            lengths[lengths.index(n)] = len(X_train)

        X, y = X_train[:n], y_train[:n]
        # print(len(X))
        model.fit(X, y)

        # train_error.append(1-model.score(X, y)) #metrics.mean_squared_error(y, model.predict(X))
        train_error.append(model.error(X, y))
        test_error.append(model.error(X_test, y_test))
        # test_error.append(metrics.mean_squared_error(y_test, model.predict(X_test)))
    # plot the training error vs. the sample size  n  shown on a log-scale
    error_plot(train_error, test_error, lengths)

    # Fitting streaming data using Stochastic Gradient Descent
    bestfit, score = tune_SGD(X_train, y_train, params={
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "penalty": ["l2", "l1", "none"],
        "tol": [0.0001, 0.001, 0.01, 0.1],
        "max_iter": [10, 100, 1000, 10000]
    }
                              )
    print("The SGD tuning Accuracy score is: {}%".format(round(score * 100, 2)))
    print("The SGD tuning best parameters are: {}".format(bestfit))

    Classifier = SGDClassifier(bestfit)
    print("The test error of the final output classifier is: {}".format(
        metrics.mean_squared_error(y_test, Classifier.predict(X_test))))

    epochs = 50
    plot_loss_per_epochs(epochs=epochs, epochLoss=Classifier.logloss(X_train, y_train, epochs=epochs),
                         label='Logistic Loss')


if __name__ == '__main__':
    main()
