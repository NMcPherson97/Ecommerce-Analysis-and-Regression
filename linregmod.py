import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_model1(customer_data):
    customer_data = customer_data.copy()

    customer_data['ltv'] = (
        customer_data['avg_order_value'] * customer_data['total_orders']
    )

    customer_data['loyalty_member'] = customer_data['loyalty_member'].map({
        'Yes': 1,
        'No': 0
    })

    X = customer_data[[
            'account_age_months',
            'loyalty_member',
            'engagement_score',
            'price_sensitivity_index',
            'days_since_last_purchase',
            'browsing_frequency_per_week',
            'satisfaction_score']]

    y = customer_data['ltv']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions)
    }

    return model, X_test, y_test, predictions, metrics


def train_model2(customer_data):
    customer_data = customer_data.copy()

    customer_data['ltv'] = (
        customer_data['avg_order_value'] * customer_data['total_orders']
    )

    customer_data['loyalty_member'] = customer_data['loyalty_member'].map({
        'Yes': 1,
        'No': 0})

    X = customer_data[[
            'account_age_months',
            'loyalty_member',
            'engagement_score',
            'price_sensitivity_index',
            'days_since_last_purchase',
            'browsing_frequency_per_week',
            'satisfaction_score',
            'total_orders',
            'avg_order_value']]

    y = customer_data['ltv']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        'r2': r2_score(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions)}

    return model, X_test, y_test, predictions, metrics