import numpy as np
import pandas as pd

# Illustrative assumptions, not real company financials
DEFAULT_COSTS = {"loss_cost": 200.0, "offer_cost": 20.0, "offer_success": 0.30}
MEDIUM_FLOOR = 0.15
# Assumption for the per-customer value view: a customer is worth this many months of their current bill
VALUE_HORIZON_MONTHS = 12


def break_even(costs):
    return costs["offer_cost"] / (costs["offer_success"] * costs["loss_cost"])


def net_benefit(prob, costs):
    return np.asarray(prob) * costs["offer_success"] * costs["loss_cost"] - costs["offer_cost"]


def customer_value(monthly_charges, horizon=VALUE_HORIZON_MONTHS):
    """Revenue at stake if the customer leaves: monthly bill x horizon (illustrative assumption)."""
    return np.nan_to_num(np.asarray(monthly_charges, float)) * horizon


def expected_offer_value(prob, monthly_charges, costs, horizon=VALUE_HORIZON_MONTHS):
    """Expected value of one offer for this customer: P(churn) x success x value at stake - offer cost.

    Unlike net_benefit (which uses one flat loss cost for everyone), this scales with what the customer pays,
    so two customers with the same churn risk are ranked by how much revenue is at stake.
    """
    return np.asarray(prob) * costs["offer_success"] * customer_value(monthly_charges, horizon) - costs["offer_cost"]


def risk_category(prob, threshold):
    """High at/above threshold; Medium needs >=15% and below threshold; otherwise Low."""
    prob = np.asarray(prob)
    medium_floor = min(MEDIUM_FLOOR, threshold)
    return np.where(prob >= threshold, "High", np.where(prob >= medium_floor, "Medium", "Low"))


def recommended_action(prob, threshold, costs):
    offer = np.asarray(prob) >= threshold
    return np.where(offer, "Offer retention", "No offer")


def policy_cost(y_true, offered, costs):
    """Expected total cost: offer cost for each offer, plus churn loss not prevented."""
    y = np.asarray(y_true)
    offered = np.asarray(offered).astype(bool)
    offer_total = offered.sum() * costs["offer_cost"]
    loss = (y * np.where(offered, 1 - costs["offer_success"], 1.0)).sum() * costs["loss_cost"]
    return float(offer_total + loss)


def best_threshold(y_true, prob, costs, grid=None):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    rows = [(t, policy_cost(y_true, np.asarray(prob) >= t, costs)) for t in grid]
    table = pd.DataFrame(rows, columns=["threshold", "expected_cost"])
    return float(table.loc[table.expected_cost.idxmin(), "threshold"]), table


def policy_comparison(y_true, prob, threshold, costs):
    y = np.asarray(y_true)
    prob = np.asarray(prob)
    policies = {
        f"Model @ {threshold:.2f}": prob >= threshold,
        "Model @ 0.50": prob >= 0.5,
        "Contact everyone": np.ones(len(y), bool),
        "No offers": np.zeros(len(y), bool),
    }
    base = policy_cost(y, policies["No offers"], costs)
    rows = []
    for name, offered in policies.items():
        cost = policy_cost(y, offered, costs)
        rows.append({
            "policy": name, "customers_contacted": int(offered.sum()),
            "expected_cost": round(cost, 2), "saving_vs_no_offer": round(base - cost, 2),
        })
    return pd.DataFrame(rows)


def score_customers(X, prob, reasons, threshold, costs):
    prob = np.asarray(prob)
    out = X.reset_index(drop=True).copy()
    value = customer_value(out["Monthly Charges"])
    exp_value = expected_offer_value(prob, out["Monthly Charges"], costs)
    action = recommended_action(prob, threshold, costs)
    out.insert(0, "Churn Probability", np.round(prob, 4))
    out.insert(1, "Risk Category", risk_category(prob, threshold))
    out.insert(2, "Recommended Action", action)
    out.insert(3, "Expected Net Benefit ($)", np.round(net_benefit(prob, costs), 2))
    out.insert(4, "Value at Stake ($)", np.round(value, 2))
    out.insert(5, "Expected Offer Value ($)", np.round(exp_value, 2))
    # Offer priority: rank among customers recommended for an offer, highest expected value first
    rank = pd.Series(exp_value).where(action == "Offer retention").rank(ascending=False, method="first")
    out.insert(6, "Offer Priority", rank.astype("Int64"))
    for i in range(3):
        out.insert(7 + i, f"Reason {i + 1}", [r[i] if len(r) > i else "" for r in reasons])
    return out


def impact_report(y_true, prob, threshold, costs, cltv=None):
    """Policy comparison on held-out data, plus an optional CLTV-based reporting view (not a model feature)."""
    table = policy_comparison(y_true, prob, threshold, costs)
    if cltv is not None:
        y, p, v = np.asarray(y_true), np.asarray(prob), np.asarray(cltv, float)
        offered = p >= threshold
        saved = (y * offered).sum() * costs["offer_success"]
        table.attrs["cltv_note"] = (
            f"Reporting-only CLTV proxy: {saved:.0f} expected retained churners x mean CLTV of churners "
            f"${v[y == 1].mean():,.0f} = ${saved * v[y == 1].mean():,.0f} estimated lifetime value retained, "
            f"before offer costs of ${offered.sum() * costs['offer_cost']:,.0f}."
        )
    return table
