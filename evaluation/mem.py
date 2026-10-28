"""
Run a mixed-effects model predicting ctx_delta from:
- model_size
- modifier type
- their interaction
- split

Random intercepts:
- model_num (seed)
- item
"""

import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
path = DATA_DIR / "eval_results.csv"

df = pd.read_csv(path)
df["type"] = df["type"].astype("category")
df["split"] = df["split"].astype("category")
df["model_num"] = df["model_num"].astype("category")
df["item"] = df["item"].astype("category")


# Make model size numeric if it isn't already
df["model_size"] = pd.to_numeric(df["model_size"])

model = smf.mixedlm(
    formula=(
        "ctx_delta ~ model_size * C(type, Treatment(reference='Dem-Num'))"
        " + C(split, Treatment(reference='train'))"
    ),
    data=df,
    groups=df["model_num"], #random effect
    vc_formula={"item": "0 + C(item)"},
)

result = model.fit(method="lbfgs")


print("\n================ MIXED EFFECTS MODEL ================\n")
print(result.summary())

coef_table = pd.DataFrame({
    "coef": result.params,
    "pvalue": result.pvalues,
    "stderr": result.bse,
})

coef_table.to_csv(DATA_DIR / "mixed_effects_results.csv")
