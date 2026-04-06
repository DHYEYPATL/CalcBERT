import json
from ml.forecasting.income_predictor import predict_income
from ml.forecasting.expense_fusion import forecast_expense
from ml.behavioral.personas import get_persona

print("="*50)
print("📈 1. INCOME FORECASTING VARIANCE TESTS")
print("="*50)

# SCENARIO A: Flat, stable income
stable_income_data = {
    "monthly_incomes": [100000, 100000, 100000, 100000],
    "bonus_count": 0,
    "investment_cagr": 0.05
}
print("\n[Scenario A] Stable Employee (Flat 100k/mo, low bonus, standard CAGR):")
res_a = predict_income(stable_income_data)
print(json.dumps(res_a, indent=2))

# SCENARIO B: Spiking, trending upward income (freelancer hitting it big)
volatile_income_data = {
    "monthly_incomes": [50000, 80000, 120000, 200000],
    "bonus_count": 2,
    "investment_cagr": 0.15
}
print("\n[Scenario B] Volatile Freelancer (Trending aggressively 50k->200k, high bonus/CAGR):")
res_b = predict_income(volatile_income_data)
print(json.dumps(res_b, indent=2))


print("\n\n" + "="*50)
print("💸 2. EXPENSE FORECASTING VARIANCE TESTS")
print("="*50)

# SCENARIO C: Decreasing spend (frugal)
frugal_spend_data = {
    "monthly_spend": [80000, 60000, 50000, 45000],
    "projected_monthly_income": 100000
}
print("\n[Scenario C] Frugal Spender (Spend dropping 80k->45k):")
res_c = forecast_expense(frugal_spend_data)
print(json.dumps(res_c, indent=2))

# SCENARIO D: Compounding aggressive spend (lifestyle inflation)
reckless_spend_data = {
    "monthly_spend": [45000, 60000, 80000, 110000],
    "projected_monthly_income": 100000 # Income is 100k, but spend is 110k!
}
print("\n[Scenario D] Reckless Spender (Spend inflating 45k->110k, exceeding income):")
res_d = forecast_expense(reckless_spend_data)
print(json.dumps(res_d, indent=2))


print("\n\n" + "="*50)
print("🧠 3. BEHAVIOR PERSONA VARIANCE TESTS")
print("="*50)

# SCENARIO E: Conservative Saver (low variance, steady)
conservative_data = {
    "transactions": [100, 110, 105, 95, 100, 120, 100]
}
print("\n[Scenario E] Stable low variance transactions (All around $100):")
res_e = get_persona(conservative_data)
print(json.dumps(res_e, indent=2))

# SCENARIO F: Impulsive Spender (huge random spikes)
impulsive_data = {
    "transactions": [50, 45, 60, 5000, 40, 50, 45, 3000]
}
print("\n[Scenario F] Erratic high-spike transactions ($50 avg, but $5000 spikes):")
res_f = get_persona(impulsive_data)
print(json.dumps(res_f, indent=2))
