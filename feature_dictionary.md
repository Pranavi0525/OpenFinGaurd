# Feature Dictionary

## Raw Features (from Kaggle Dataset)

| Feature | Type | Description | Credit Risk Intuition |
|---------|------|-------------|----------------------|
| RevolvingUtilizationOfUnsecuredLines | float | Credit used / total credit limit | Higher = more stressed financially |
| age | int | Borrower age in years | Proxy for credit history length |
| NumberOfTime30-59DaysPastDueNotWorse | int | Late payments (30-59 days) in 2 years | Early warning signal |
| DebtRatio | float | Monthly debt / monthly income | > 0.43 = FHA high-risk threshold |
| MonthlyIncome | float | Monthly gross income ($) | Higher = more repayment capacity |
| NumberOfOpenCreditLinesAndLoans | int | Total active credit accounts | Too many = credit-hungry behavior |
| NumberOfTimes90DaysLate | int | Severely late payments in 2 years | Strongest single default predictor |
| NumberRealEstateLoansOrLines | int | Mortgage/HELOC count | Asset ownership signal |
| NumberOfTime60-89DaysPastDueNotWorse | int | Late payments (60-89 days) in 2 years | Escalation pattern |
| NumberOfDependents | int | Number of financial dependents | Increases financial strain |

## Engineered Features

| Feature | Formula | Why It's Useful |
|---------|---------|-----------------|
| TotalDelinquencies | sum of all 3 delinquency counts | Aggregate payment behavior |
| DelinquencySeverityScore | 30d×1 + 60d×2 + 90d×3 | Weights severity, not just frequency |
| EstimatedMonthlyDebt | DebtRatio × MonthlyIncome | Absolute debt burden |
| DisposableIncome | MonthlyIncome − EstimatedMonthlyDebt | Actual repayment capacity |
| CreditLineDensity | OpenLines / age | Age-normalized credit exposure |

## Target Variable

| Variable | Values | Meaning |
|----------|--------|---------|
| SeriousDlqin2yrs | 0 = No default, 1 = Default | 90+ days delinquent within 2 years |
| Class imbalance | ~6.7% positive | 1 default per ~14 non-defaults |