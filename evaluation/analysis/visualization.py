from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualization:
    """
    Visualize evaluation metrics (BLEU and METEOR) for translation models
    across different model types and language pairs.
    """

    def __init__(self, data: pd.DataFrame, model_types: Dict[str, str]):
        """
        Initialize with evaluation results and model type annotations.

        Args:
            data (pd.DataFrame): DataFrame with at least ['model_id', 'bleu', 'meteor'].
                                 model_id is expected to follow format: <base_model>_<lang-pair>[_report]
            model_types (Dict[str, str]): Mapping from base_model to model type (e.g. 'llm', 'mt').
        """
        self.df = data.copy()

        # Clean model_id (e.g., remove "_report" suffix)
        self.df["model_id"] = self.df["model_id"].str.replace(
            "_report", "", regex=False
        )

        # Extract base_model and language_pair from model_id
        self.df["base_model"] = self.df["model_id"].apply(lambda x: x.split("_")[0])
        self.df["language_pair"] = self.df["model_id"].apply(lambda x: x.split("_")[1])

        # Map model type from base_model
        self.df["model_type"] = self.df["base_model"].map(model_types)

    def get_bleu_score_table(self) -> pd.DataFrame:
        """
        Return a table of BLEU scores with base models as rows and language pairs as columns.
        """
        return self.df.pivot(index="base_model", columns="language_pair", values="bleu")

    def get_meteor_score_table(self) -> pd.DataFrame:
        """
        Return a table of METEOR scores with base models as rows and language pairs as columns.
        """
        return self.df.pivot(
            index="base_model", columns="language_pair", values="meteor"
        )

    def plot_average_scores_by_type(self) -> None:
        """
        Plot bar chart of average BLEU and METEOR scores grouped by model type (e.g., LLM vs MT).
        """
        avg_by_type = (
            self.df.groupby("model_type")[["bleu", "meteor"]]
            .mean()
            .reset_index()
            .melt(id_vars="model_type", var_name="metric", value_name="score")
        )

        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            data=avg_by_type, x="metric", y="score", hue="model_type", palette="Set2"
        )

        ax.set_title("Average BLEU and METEOR by Model Type")
        ax.set_ylabel("Average Score")
        ax.set_xlabel("Metric")
        plt.ylim(0, 1)
        plt.legend(title="Model Type")
        plt.tight_layout()
        plt.show()

    def plot_grouped_scores(self) -> None:
        """
        Plot side-by-side bar charts of BLEU and METEOR scores grouped by base model.
        Bars are colored by model type.
        """
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # BLEU scores
        sns.barplot(
            data=self.df,
            x="base_model",
            y="bleu",
            hue="model_type",
            ax=axes[0],
            palette="Set2",
            errorbar=None,
        )
        axes[0].set_title("BLEU Scores by Model")
        axes[0].set_ylabel("Score")
        axes[0].set_xlabel("Base Model")
        axes[0].tick_params(axis="x", rotation=45)

        # METEOR scores
        sns.barplot(
            data=self.df,
            x="base_model",
            y="meteor",
            hue="model_type",
            ax=axes[1],
            palette="Set2",
            errorbar=None,
        )
        axes[1].set_title("METEOR Scores by Model")
        axes[1].set_ylabel("Score")
        axes[1].set_xlabel("Base Model")
        axes[1].tick_params(axis="x", rotation=45)

        # Shared legend
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, title="Model Type", loc="upper center", ncol=3)
        for ax in axes:
            legend = ax.get_legend()
            if legend:
                legend.remove()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_scores_by_language_pair(self) -> None:
        """
        Plot bar chart of average BLEU and METEOR scores per language pair
        (aggregated across all models).
        """
        avg_scores = (
            self.df.groupby("language_pair")[["bleu", "meteor"]]
            .mean()
            .reset_index()
            .melt(id_vars="language_pair", var_name="metric", value_name="score")
        )

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=avg_scores, x="language_pair", y="score", hue="metric", palette="Set2"
        )

        ax.set_title("Average BLEU and METEOR Scores per Language Pair")
        ax.set_ylabel("Average Score")
        ax.set_xlabel("Language Pair")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend(title="Metric")
        plt.tight_layout()
        plt.show()

    def plot_scores_by_language_pair_model_type(self) -> None:
        """
        Plot side-by-side bar charts of BLEU and METEOR scores grouped by language pair.
        Bars are colored by model type.
        """
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # BLEU scores
        sns.barplot(
            data=self.df,
            x="language_pair",
            y="bleu",
            hue="model_type",
            ax=axes[0],
            palette="Set2",
            errorbar=None,
        )
        axes[0].set_title("BLEU Scores by Language Pair")
        axes[0].set_ylabel("Score")
        axes[0].set_xlabel("Language Pair")
        axes[0].tick_params(axis="x", rotation=45)

        # METEOR scores
        sns.barplot(
            data=self.df,
            x="language_pair",
            y="meteor",
            hue="model_type",
            ax=axes[1],
            palette="Set2",
            errorbar=None,
        )
        axes[1].set_title("METEOR Scores by Language Pair")
        axes[1].set_ylabel("Score")
        axes[1].set_xlabel("Language Pair")
        axes[1].tick_params(axis="x", rotation=45)

        # Shared legend
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, title="Model Type", loc="upper center", ncol=3)
        for ax in axes:
            legend = ax.get_legend()
            if legend:
                legend.remove()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
