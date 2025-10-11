# Rebuild the records with markers for major events
from turtle import pd

savings = 1000  # starting savings
father_balance = father_loan
records = []

for date in months:
    year_month = date.strftime("%b-%Y")
    marker = ""

    # Determine income for the month
    if date < pd.Timestamp("2026-01-01"):
        income = income_before
    else:
        income = income_after

    surplus = income - expenses

    # Aug-Dec 2025: saving for visa + car
    if date < pd.Timestamp("2026-01-01"):
        savings += surplus

    # Jan 2026: pay visa + car
    elif date == pd.Timestamp("2026-01-01"):
        savings += surplus
        savings -= (visa + car)
        marker = "Visa + Car Paid"

    # Feb 2026: build emergency + furniture + repay father
    elif date == pd.Timestamp("2026-02-01"):
        savings += surplus
        savings -= 3000  # emergency buffer
        savings -= furniture
        marker = "Emergency Fund + Furniture Bought"
        payment_to_father = min(4750, father_balance)
        father_balance -= payment_to_father
        savings -= payment_to_father
        if father_balance == 0:
            marker += " + Father Loan Cleared"

    # Mar-Dec 2026: repay father 6k/month, save 1k/month
    elif date >= pd.Timestamp("2026-03-01") and date <= pd.Timestamp("2026-12-01"):
        savings += surplus
        payment_to_father = min(6000, father_balance)
        father_balance -= payment_to_father
        savings -= payment_to_father
        savings -= 1000  # moving 1k into "savings pot"
        if father_balance == 0 and marker == "":
            marker = "Father Loan Cleared"

    # Jan 2027: just accumulate
    else:
        savings += surplus

    records.append({
        "Month": year_month,
        "Income": income,
        "Expenses": expenses,
        "Surplus": surplus,
        "Savings Balance": round(savings, 2),
        "Father Loan Remaining": round(father_balance, 2),
        "Event": marker
    })

df_marked = pd.DataFrame(records)
import caas_jupyter_tools

caas_jupyter_tools.display_dataframe_to_user("Financial Plan with Key Events (Aug 2025 - Jan 2027)", df_marked)
