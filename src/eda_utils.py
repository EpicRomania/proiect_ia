import os
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame, Series
from scipy import stats

# Set global styles
plt.rcParams["figure.dpi"] = 110
sns.set_theme(style="whitegrid")


def numeric_summary(df: DataFrame, cols: Optional[List[str]] = None) -> DataFrame:
    """
    Return descriptive statistics (count, mean, std, min, 25%, 50%, 75%, max)
    for the requested numerical columns.
    """
    sub = df[cols] if cols is not None else df
    summary = sub.describe().T
    summary.index.name = "feature"
    return summary


def categorical_summary(df: DataFrame, cols: List[str]) -> DataFrame:
    """
    For each categorical column: non-null count and number of unique values.
    """
    data = {
        "non_null": df[cols].notna().sum(),
        "unique": df[cols].nunique()
    }
    return DataFrame(data)


def boxplots(df: DataFrame,
             cols: List[str],
             n_cols: int = 3,
             save_dir: Optional[str] = None) -> None:
    """
    Draw one vertical box-plot per numerical column; skip if empty.
    """
    if not cols:
        print("boxplots: no numerical columns to plot.")
        return

    n_cols = max(1, min(n_cols, len(cols)))
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharey=False,
        squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, col in zip(axes_flat, cols):
        series = df[col].dropna()
        sns.boxplot(
            y=series,
            ax=ax,
            whis=1.5,
            showcaps=True,
            boxprops={'facecolor': 'white', 'edgecolor': 'steelblue', 'linewidth': 1.2},
            whiskerprops={'color': 'steelblue', 'linewidth': 1},
            capprops={'color': 'steelblue', 'linewidth': 1},
            medianprops={'color': 'firebrick', 'linewidth': 1.5},
            flierprops={'marker': 'o',
                        'markerfacecolor': 'steelblue',
                        'markeredgecolor': 'gray',
                        'markersize': 4,
                        'alpha': 0.6}
        )
        ax.set_title(col)
        ax.set_ylabel("")   

    for ax in axes_flat[len(cols):]:
        fig.delaxes(ax)

    fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "boxplots.png"))
    plt.show()




def countplots(df: DataFrame,
               cols: List[str],
               n_cols: int = 3,
               save_dir: Optional[str] = None) -> None:
    """
    Draw one count-plot per categorical column with pastel colors and annotations,
    avoiding the seaborn palette warning.
    """
    if not cols:
        print("countplots: no categorical columns to plot.")
        return

    n_cols = max(1, min(n_cols, len(cols)))
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, col in zip(axes_flat, cols):
        counts = df[col].value_counts()
        order = counts.index.tolist()
        # draw bars manually
        pastel_colors = sns.color_palette('pastel', len(order))
        bars = ax.bar(
            x=range(len(order)),
            height=counts.values,
            color=pastel_colors,
            edgecolor='black',
            linewidth=1
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45)
        ax.set_title(col)
        ax.set_ylabel('count')
        # annotate bar heights
        for bar in bars:
            h = int(bar.get_height())
            ax.annotate(
                h,
                (bar.get_x() + bar.get_width() / 2, h),
                ha='center', va='bottom',
                fontsize=8
            )

    # drop unused axes
    for ax in axes_flat[len(cols):]:
        fig.delaxes(ax)

    #fig.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "countplots.png"))
    plt.show()





def plot_class_balance(y_train: Series,
                       y_test: Series,
                       save_dir: Optional[str] = None) -> None:
    """
    Bar chart of label frequencies for train vs test.
    """
    df = pd.concat([
        y_train.value_counts().rename("train"),
        y_test.value_counts().rename("test")
    ], axis=1).fillna(0).astype(int)
    df.plot(kind="bar")
    plt.ylabel("frequency")
    plt.title("Class balance (train vs test)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "class_balance.png"))
    plt.show()


def pearson_heatmap(df: DataFrame,
                    cols: List[str],
                    method: Literal["pearson", "kendall", "spearman"] = "pearson",
                    save_dir: Optional[str] = None) -> DataFrame:
    """
    Correlation heatmap for numeric features.
    """
    corr = df[cols].corr(method=method)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(f"{method.capitalize()} correlation")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{method}_corr.png"))
    plt.show()
    return corr


def chi2_categorical(df: DataFrame,
                     cols: List[str],
                     target: Optional[Series] = None,
                     save_path: Optional[str] = None) -> DataFrame:
    """
    Chi-square p-values for each pair of categorical features,
    or between each feature and target if provided.
    """
    results = []
    if target is not None:
        for col in cols:
            table = pd.crosstab(df[col], target)
            chi2, p, *_ = stats.chi2_contingency(table)
            results.append({"feature": col, "chi2": chi2, "p_value": p})
    else:
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                table = pd.crosstab(df[c1], df[c2])
                chi2, p, *_ = stats.chi2_contingency(table)
                results.append({"feature1": c1, "feature2": c2, "chi2": chi2, "p_value": p})
    result_df = DataFrame(results)
    if save_path:
        result_df.to_csv(save_path, index=False)
    return result_df


def eda_memo(numeric_stats: DataFrame,
             cat_stats: DataFrame,
             class_balance: DataFrame,
             corr_matrix: DataFrame,
             chi2_df: DataFrame) -> str:
    """
    Build a plain-text memo summarising main issues for Checkpoint A.
    """
    issues: list[str] = []

    # Missing values
    if (numeric_stats["count"] < len(numeric_stats)).any() or (cat_stats["non_null"] < len(cat_stats)).any():
        issues.append("Missing values present — will need imputation.")

    # Outliers via IQR
    q1, q3 = numeric_stats["25%"], numeric_stats["75%"]
    iqr = q3 - q1
    outlier_feats = numeric_stats[
        (numeric_stats["min"] < q1 - 1.5 * iqr) |
        (numeric_stats["max"] > q3 + 1.5 * iqr)
    ].index.tolist()
    if outlier_feats:
        issues.append(f"Potential outliers in: {', '.join(outlier_feats)}.")

    # High Pearson correlation
    high_corr = (corr_matrix.abs() > 0.9).sum().gt(1)
    redundant = corr_matrix.columns[high_corr].tolist()
    if redundant:
        issues.append(f"High collinearity among: {', '.join(redundant)}.")

    # Class imbalance
    imbalance = class_balance.max(axis=1) / class_balance.min(axis=1)
    if imbalance.max() > 5:
        issues.append("Significant class imbalance detected.")

    # Strong chi-square associations
    if not chi2_df.empty and (chi2_df["p_value"] < 0.01).any():
        issues.append("Some categorical features show strong dependencies.")

    return "\n".join(issues) if issues else "No critical data issues detected."
