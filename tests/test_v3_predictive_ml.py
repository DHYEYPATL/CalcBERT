import pytest

from ml.forecasting.income_predictor import predict_income
from ml.forecasting.expense_fusion import forecast_expense
from ml.behavioral.personas import get_persona


# =========================
# 🟢 INCOME ENGINE TESTS
# =========================

def test_income_mock_schema():
    result = predict_income(mode="mock")

    assert "predicted_annual_income" in result
    assert "income_range" in result
    assert "confidence" in result

    assert isinstance(result["income_range"], list)
    assert len(result["income_range"]) == 2
    assert 0 <= result["confidence"] <= 1


def test_income_real_data():
    result = predict_income("sample/sample_income.csv")

    assert result["predicted_annual_income"] > 0
    assert result["income_range"][0] <= result["income_range"][1]


def test_income_deterministic():
    r1 = predict_income("sample/sample_income.csv")
    r2 = predict_income("sample/sample_income.csv")

    assert r1 == r2


def test_income_edge_case_empty():
    try:
        result = predict_income([])
        assert result is not None
    except Exception:
        pytest.fail("Income predictor crashed on empty input")


# =========================
# 💸 EXPENSE ENGINE TESTS
# =========================

def test_expense_mock_schema():
    result = forecast_expense(mode="mock")

    assert "monthly_spend_forecast" in result
    assert "subscription_growth_rate" in result
    assert "cash_flow_deficit_months" in result
    assert "confidence" in result

    assert isinstance(result["cash_flow_deficit_months"], list)
    assert 0 <= result["confidence"] <= 1


def test_expense_real_data():
    data = {
        "monthly_spend": [1000, 1200, 1100, 1500, 2000]
    }

    result = forecast_expense(data)

    assert isinstance(result["monthly_spend_forecast"], dict)
    assert result["subscription_growth_rate"] >= 0


def test_expense_variability():
    data = {
        "monthly_spend": [1000, 2000, 1500, 3000]
    }

    result = forecast_expense(data)
    values = list(result["monthly_spend_forecast"].values())

    assert len(set(values)) > 1  # should not be constant


# =========================
# 🧠 BEHAVIOR ENGINE TESTS
# =========================

def test_behavior_mock_schema():
    result = get_persona(mode="mock")

    assert "persona" in result
    assert "confidence" in result
    assert "behavior_flags" in result

    assert isinstance(result["behavior_flags"], list)
    assert 0 <= result["confidence"] <= 1


def test_behavior_real_data():
    data = {
        "transactions": [100, 500, 200, 3000, 50, 700]
    }

    result = get_persona(data)

    assert isinstance(result["persona"], str)


def test_behavior_variation():
    data1 = {"transactions": [100, 100, 100]}
    data2 = {"transactions": [100, 5000, 50, 3000]}

    r1 = get_persona(data1)
    r2 = get_persona(data2)

    assert r1 != r2  # should not always give same persona


# =========================
# 📦 MOCK FILE TESTS
# =========================

import json

def test_mock_income_file():
    with open("mocks/income_forecast.json") as f:
        data = json.load(f)

    assert "predicted_annual_income" in data


def test_mock_expense_file():
    with open("mocks/expense_forecast.json") as f:
        data = json.load(f)

    assert "monthly_spend_forecast" in data


def test_mock_behavior_file():
    with open("mocks/behavior_profile.json") as f:
        data = json.load(f)

    assert "persona" in data