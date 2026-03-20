import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def _build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    token = df.copy()
    token["event"] = token["coin"] + "__" + token["state_updown"].astype(str) + "__" + token["state_vol"].astype(str)
    basket = token.groupby(["date", "event"]).size().unstack(fill_value=0).astype(bool)
    return basket


def mine_association_rules(df: pd.DataFrame, min_support: float = 0.08, min_confidence: float = 0.55, min_lift: float = 1.05, top_k: int = 20) -> pd.DataFrame:
    basket = _build_transactions(df)
    if basket.shape[0] < 10 or basket.shape[1] < 2:
        return pd.DataFrame()
    itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame()
    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return pd.DataFrame()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).head(top_k)
    cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
    return rules[cols]


def compare_rules_by_regime(df: pd.DataFrame, min_support: float, min_confidence: float, min_lift: float, top_k: int = 10) -> pd.DataFrame:
    output = []
    for regime in ["low_vol", "mid_vol", "high_vol"]:
        subset = df[df["state_vol"] == regime]
        rules = mine_association_rules(subset, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, top_k=top_k)
        if not rules.empty:
            rules.insert(0, "regime", regime)
            output.append(rules)
    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True)
