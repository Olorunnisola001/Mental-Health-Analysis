"""
Mental Health Dataset — Exploratory Data Analysis & Statistical Analysis
========================================================================
Dataset: 2,000 individuals with mental health indicators, lifestyle habits,
         and demographic factors.

Sections:
  1. Data Loading & Overview
  2. Descriptive Statistics
  3. Condition Prevalence
  4. Occupation & Gender Breakdowns
  5. Lifestyle Factor Analysis (Sleep, Screen Time, Physical Activity)
  6. Stress Level Analysis
  7. Correlation Analysis
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. DATA LOADING & OVERVIEW
# ─────────────────────────────────────────────

df = pd.read_csv("mental_health.csv")

print("=" * 60)
print("SECTION 1 — DATA OVERVIEW")
print("=" * 60)
print(f"Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns        : {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")


# ─────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2 — DESCRIPTIVE STATISTICS")
print("=" * 60)
print(df.describe().round(2))

# Categorical breakdowns
print("\nGender distribution:")
print(df["Gender"].value_counts())

print("\nOccupation distribution:")
print(df["Occupation"].value_counts())

print("\nPhysical Activity distribution:")
print(df["Physical_Activity"].value_counts())


# ─────────────────────────────────────────────
# 3. CONDITION PREVALENCE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3 — CONDITION PREVALENCE")
print("=" * 60)

conditions = ["Depression", "Anxiety", "Burnout"]
for cond in conditions:
    rate  = df[cond].mean()
    count = df[cond].sum()
    print(f"  {cond:<12}: {rate:.1%}  ({count} / {len(df)})")

# Co-occurrence: all three conditions at once
all_three = ((df["Depression"] == 1) & (df["Anxiety"] == 1) & (df["Burnout"] == 1)).sum()
print(f"\n  All three    : {all_three / len(df):.1%}  ({all_three} individuals)")

none = ((df["Depression"] == 0) & (df["Anxiety"] == 0) & (df["Burnout"] == 0)).sum()
print(f"  None at all  : {none / len(df):.1%}  ({none} individuals)")


# ─────────────────────────────────────────────
# 4. OCCUPATION & GENDER BREAKDOWNS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4 — BREAKDOWNS BY OCCUPATION & GENDER")
print("=" * 60)

print("\nBy occupation:")
occ_stats = df.groupby("Occupation")[conditions + ["Stress_Level"]].mean().round(3)
print(occ_stats.to_string())

print("\nBy gender:")
gen_stats = df.groupby("Gender")[conditions + ["Stress_Level"]].mean().round(3)
print(gen_stats.to_string())

print("\nBy occupation & gender (Depression rate):")
pivot = df.pivot_table(values="Depression", index="Occupation", columns="Gender", aggfunc="mean")
print(pivot.round(3))


# ─────────────────────────────────────────────
# 5. LIFESTYLE FACTOR ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5 — LIFESTYLE FACTOR ANALYSIS")
print("=" * 60)

# --- Sleep ---
df["Sleep_Cat"] = pd.cut(
    df["Sleep_Hours"],
    bins=[0, 5, 7, 12],
    labels=["<5h (poor)", "5-7h (moderate)", ">7h (good)"]
)
print("\nSleep quality vs mental health:")
print(df.groupby("Sleep_Cat", observed=True)[conditions].mean().round(3))

# --- Screen Time ---
df["Screen_Cat"] = pd.cut(
    df["Daily_Screen_Time"],
    bins=[0, 4, 7, 15],
    labels=["Low (<4h)", "Medium (4-7h)", "High (>7h)"]
)
print("\nDaily screen time vs mental health:")
print(df.groupby("Screen_Cat", observed=True)[conditions].mean().round(3))

# --- Social Media ---
df["Social_Cat"] = pd.cut(
    df["Social_Media_Usage"],
    bins=[0, 3, 6, 15],
    labels=["Low (<3h)", "Medium (3-6h)", "High (>6h)"]
)
print("\nSocial media usage vs mental health:")
print(df.groupby("Social_Cat", observed=True)[conditions].mean().round(3))

# --- Night Usage ---
print("\nNight screen usage vs mental health:")
print(df.groupby("Night_Usage")[conditions + ["Sleep_Hours"]].mean().round(3))

# --- Physical Activity ---
print("\nPhysical activity level vs mental health:")
print(df.groupby("Physical_Activity")[conditions + ["Stress_Level"]].mean().round(3))

# --- Smoking & Alcohol ---
print("\nSmoking vs mental health:")
print(df.groupby("Smoking")[conditions].mean().round(3))

print("\nAlcohol use vs mental health:")
print(df.groupby("Alcohol")[conditions].mean().round(3))


# ─────────────────────────────────────────────
# 6. STRESS LEVEL ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6 — STRESS LEVEL ANALYSIS")
print("=" * 60)

print("\nStress level frequency:")
print(df["Stress_Level"].value_counts().sort_index())

# High vs low stress
df["High_Stress"] = df["Stress_Level"] >= 7
print("\nHigh stress (≥7) vs low stress (<7) — condition rates:")
print(df.groupby("High_Stress")[conditions].mean().round(3))

# Stress × occupation
print("\nAverage stress level by occupation:")
print(df.groupby("Occupation")["Stress_Level"].describe().round(2))

# Stress × sleep
print("\nAverage stress by sleep category:")
print(df.groupby("Sleep_Cat", observed=True)["Stress_Level"].mean().round(2))


# ─────────────────────────────────────────────
# 7. CORRELATION ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 7 — CORRELATION ANALYSIS")
print("=" * 60)

numeric_cols = [
    "Age", "Daily_Screen_Time", "Social_Media_Usage",
    "Sleep_Hours", "Stress_Level", "Work_Study_Hours",
    "Social_Interaction_Score", "Caffeine_Intake",
    "Depression", "Anxiety", "Burnout"
]

corr = df[numeric_cols].corr()

print("\nCorrelation with Depression (sorted):")
print(corr["Depression"].drop("Depression").sort_values(ascending=False).round(3))

print("\nCorrelation with Anxiety (sorted):")
print(corr["Anxiety"].drop("Anxiety").sort_values(ascending=False).round(3))

print("\nCorrelation with Burnout (sorted):")
print(corr["Burnout"].drop("Burnout").sort_values(ascending=False).round(3))

print("\nFull correlation matrix (conditions + stress + sleep):")
focus_cols = ["Stress_Level", "Sleep_Hours", "Daily_Screen_Time",
              "Social_Media_Usage", "Depression", "Anxiety", "Burnout"]
print(df[focus_cols].corr().round(3))


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
