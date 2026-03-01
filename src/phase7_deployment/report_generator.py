import pandas as pd
import numpy as np

print("report_generator module loaded")


# =========================
# CONVERT PROBABILITY → CREDIT SCORE
# =========================

def probability_to_score(prob):

    # safer log scaling
    odds = prob / (1 - prob + 1e-10)

    score = 850 - (np.log(odds + 1e-10) * 50)

    score = np.clip(score, 300, 850)

    return int(score)


# =========================
# ASSIGN RISK LEVEL
# =========================

def assign_risk(prob):

    if prob >= 0.8:
        return "High Risk"

    elif prob >= 0.4:
        return "Medium Risk"

    else:
        return "Low Risk"


# =========================
# ASSIGN STATUS
# =========================

def assign_status(prob):

    if prob >= 0.8:
        return "Blocked"

    elif prob >= 0.4:
        return "Review"

    else:
        return "Safe"


# =========================
# ASSIGN RECOMMENDATION
# =========================

def assign_recommendation(prob):

    if prob >= 0.8:
        return "Block transaction immediately"

    elif prob >= 0.4:
        return "Manual review required"

    else:
        return "No action needed"


# =========================
# MAIN REPORT FUNCTION
# =========================

def generate_report(test_df, probabilities, threshold, output_path):

    print("\nGenerating fraud report...")

    report = test_df.copy().reset_index(drop=True)

    report["Fraud Probability"] = probabilities

    report["Fraud Probability (%)"] = (probabilities * 100).round(2)

    report["Safety Score"] = report["Fraud Probability"].apply(probability_to_score)

    report["Fraud Risk"] = report["Fraud Probability"].apply(assign_risk)

    report["Status"] = report["Fraud Probability"].apply(assign_status)

    report["Recommendation"] = report["Fraud Probability"].apply(assign_recommendation)

    report.to_csv(output_path, index=False)

    print("Report saved to:", output_path)

    return report