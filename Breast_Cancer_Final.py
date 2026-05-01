import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, filename: str) -> Path:
    file_path = RESULTS_DIR / filename
    df.to_csv(file_path, index=False)
    return file_path


def load_dataset() -> pd.DataFrame:
    local_candidates = [
        BASE_DIR / 'Wisconsin (Diagnostic) Data.csv',
        BASE_DIR / 'breast-cancer-edited.csv',
    ]

    for candidate in local_candidates:
        if candidate.exists():
            return pd.read_csv(candidate)

    raise FileNotFoundError(
        'Dataset file not found. Put Wisconsin (Diagnostic) Data.csv in the project root folder.'
    )


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cleaned = df.copy()

    for col in ['id', 'Unnamed: 32']:
        if col in cleaned.columns:
            cleaned.drop(columns=[col], inplace=True)

    cleaned['diagnosis'] = cleaned['diagnosis'].replace({'M': 1, 'B': 0, 'Malignant': 1, 'Benign': 0})
    cleaned['diagnosis'] = pd.to_numeric(cleaned['diagnosis'], errors='coerce')

    if cleaned['diagnosis'].isna().any():
        raise ValueError('diagnosis column contains unsupported label values after mapping.')

    cleaned['diagnosis'] = cleaned['diagnosis'].astype(int)

    x_data = cleaned.drop(columns=['diagnosis'])
    y_data = cleaned['diagnosis']
    return x_data, y_data


def save_eda_results(df: pd.DataFrame) -> None:
    save_df(df.head(20), '01_dataset_head.csv')
    save_df(df.describe().round(4), '02_dataset_describe.csv')
    save_df(
        df.isnull().sum().reset_index(name='missing_count').rename(columns={'index': 'column'}),
        '03_missing_values.csv',
    )

    grouped = df.groupby('diagnosis').mean(numeric_only=True).reset_index()
    save_df(grouped, '04_groupby_diagnosis_mean.csv')

    fig = px.histogram(
        data_frame=df,
        x='diagnosis',
        color='diagnosis',
        color_discrete_sequence=['#A865C9', '#f6abb6'],
        title='Diagnosis Class Distribution',
    )
    fig.write_html(RESULTS_DIR / '05_diagnosis_histogram.html')


# Keep model list classifier-only to avoid mixed-task metric errors.
def get_classifier_models() -> list[tuple[str, object]]:
    return [
        ('Logistic Regression', LogisticRegression(max_iter=5000, random_state=42, solver='liblinear')),
        ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
        ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42)),
        ('Extra Trees Classifier', ExtraTreesClassifier(random_state=42)),
        ('SVC', SVC(gamma='auto', probability=True, random_state=42)),
        ('GaussianNB', GaussianNB()),
        ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
    ]


def train_test_results(
    models: list[tuple[str, object]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []

    for name, model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rows.append(
            {
                'Model': name,
                'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
                'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)


def cross_validation_results(models: list[tuple[str, object]], x_data: pd.DataFrame, y_data: pd.Series) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_rows = []

    for name, model in models:
        scores = cross_validate(
            model,
            x_data,
            y_data,
            cv=skf,
            scoring=['accuracy', 'f1', 'precision', 'recall'],
            return_train_score=False,
            error_score='raise',
        )
        cv_rows.append(
            {
                'Model': name,
                'CV Accuracy (Mean)': round(scores['test_accuracy'].mean(), 4),
                'CV Accuracy (Std)': round(scores['test_accuracy'].std(), 4),
                'F1 (Mean)': round(scores['test_f1'].mean(), 4),
                'Precision (Mean)': round(scores['test_precision'].mean(), 4),
                'Recall (Mean)': round(scores['test_recall'].mean(), 4),
            }
        )

    return pd.DataFrame(cv_rows).sort_values(by='CV Accuracy (Mean)', ascending=False).reset_index(drop=True)


def ablation_study(x_data: pd.DataFrame, y_data: pd.Series) -> pd.DataFrame:
    mean_features = [c for c in x_data.columns if c.endswith('_mean')]
    se_features = [c for c in x_data.columns if c.endswith('_se')]
    worst_features = [c for c in x_data.columns if c.endswith('_worst')]
    lime_top10 = [
        'radius_worst',
        'area_worst',
        'perimeter_worst',
        'concave points_worst',
        'concave points_mean',
        'texture_worst',
        'perimeter_mean',
        'texture_mean',
        'radius_mean',
        'concavity_worst',
    ]

    feature_subsets = {
        'All Features (Baseline)': x_data.columns.tolist(),
        'Mean Features Only': mean_features,
        'SE Features Only': se_features,
        'Worst Features Only': worst_features,
        'LIME Top-10 Features': [c for c in lime_top10 if c in x_data.columns],
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    rows = []

    for subset_name, features in feature_subsets.items():
        if not features:
            continue

        x_sub = x_data[features]
        model = ExtraTreesClassifier(random_state=42)
        acc_scores = cross_val_score(model, x_sub, y_data, cv=skf, scoring='accuracy')
        f1_scores = cross_val_score(model, x_sub, y_data, cv=skf, scoring='f1')

        rows.append(
            {
                'Feature Subset': subset_name,
                'Num Features': len(features),
                'CV Accuracy (Mean)': round(acc_scores.mean(), 4),
                'CV Accuracy (Std)': round(acc_scores.std(), 4),
                'F1 (Mean)': round(f1_scores.mean(), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(by='CV Accuracy (Mean)', ascending=False).reset_index(drop=True)


def save_confusion_matrix(best_model_name: str, best_model: object, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = best_model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Benign', 'Actual Malignant'],
        columns=['Pred Benign', 'Pred Malignant'],
    )
    save_df(cm_df.reset_index().rename(columns={'index': 'Label'}), '09_confusion_matrix.csv')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f'Confusion Matrix: {best_model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_yticklabels(['Benign', 'Malignant'])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / '10_confusion_matrix.png', dpi=300)
    plt.close(fig)


def save_summary(train_df: pd.DataFrame, cv_df: pd.DataFrame, abl_df: pd.DataFrame) -> None:
    summary = {
        'best_train_test_model': train_df.iloc[0].to_dict(),
        'best_cv_model': cv_df.iloc[0].to_dict(),
        'best_ablation_setting': abl_df.iloc[0].to_dict(),
    }

    with open(RESULTS_DIR / '11_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    with open(RESULTS_DIR / '12_summary.txt', 'w', encoding='utf-8') as file:
        file.write('Breast Cancer Classification Summary\n')
        file.write('===================================\n\n')
        file.write(f"Best Train/Test Model: {train_df.iloc[0]['Model']}\n")
        file.write(f"Accuracy: {train_df.iloc[0]['Accuracy']}\n")
        file.write(f"F1: {train_df.iloc[0]['F1']}\n\n")

        file.write(f"Best Cross-Validation Model: {cv_df.iloc[0]['Model']}\n")
        file.write(f"CV Accuracy Mean: {cv_df.iloc[0]['CV Accuracy (Mean)']}\n")
        file.write(f"CV Accuracy Std: {cv_df.iloc[0]['CV Accuracy (Std)']}\n\n")

        file.write(f"Best Ablation Feature Set: {abl_df.iloc[0]['Feature Subset']}\n")
        file.write(f"Ablation CV Accuracy Mean: {abl_df.iloc[0]['CV Accuracy (Mean)']}\n")


def save_lime_outputs(
    best_model: object,
    scaler: StandardScaler,
    x_train_raw: pd.DataFrame,
    x_test_raw: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    try:
        from lime import lime_tabular
    except Exception as exc:
        status = pd.DataFrame(
            [{'status': f'LIME unavailable: {exc}. Install with pip install lime'}]
        )
        save_df(status, '13_lime_status.csv')
        return

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=x_train_raw.to_numpy(),
        feature_names=feature_names,
        class_names=['Benign', 'Malignant'],
        mode='classification',
    )

    def predict_fn(input_array):
        input_df = pd.DataFrame(input_array, columns=feature_names)
        input_scaled = pd.DataFrame(
            scaler.transform(input_df),
            columns=feature_names,
            index=input_df.index,
        )
        return best_model.predict_proba(input_scaled)

    sample_plot_dir = RESULTS_DIR / 'lime_samples'
    sample_plot_dir.mkdir(parents=True, exist_ok=True)
    local_sample_plot_dir = RESULTS_DIR / 'lime_local_samples'
    local_sample_plot_dir.mkdir(parents=True, exist_ok=True)

    explanation_rows = []
    probability_rows = []
    max_samples = min(10, len(x_test_raw))

    for sample_idx in range(max_samples):
        row_df = x_test_raw.iloc[[sample_idx]]
        row_scaled = pd.DataFrame(
            scaler.transform(row_df),
            columns=feature_names,
            index=row_df.index,
        )
        probs = best_model.predict_proba(row_scaled)[0]
        pred_label = int(probs[1] >= probs[0])
        true_label = int(y_test.iloc[sample_idx])

        probability_rows.append(
            {
                'sample_index': sample_idx,
                'true_label': true_label,
                'predicted_label': pred_label,
                'prob_benign': round(float(probs[0]), 6),
                'prob_malignant': round(float(probs[1]), 6),
            }
        )

        exp = explainer.explain_instance(
            data_row=x_test_raw.iloc[sample_idx].to_numpy(),
            predict_fn=predict_fn,
            num_features=10,
        )

        for rule, weight in exp.as_list():
            matched_feature = 'unparsed_rule'
            for feature in sorted(feature_names, key=len, reverse=True):
                if feature in rule:
                    matched_feature = feature
                    break

            explanation_rows.append(
                {
                    'sample_index': sample_idx,
                    'feature_rule': rule,
                    'parsed_feature': matched_feature,
                    'weight': round(weight, 6),
                    'abs_weight': round(abs(weight), 6),
                }
            )

        if sample_idx < 5:
            exp_rules = exp.as_list()
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(14, 4),
                gridspec_kw={'width_ratios': [1.0, 2.0, 1.6]},
            )

            # Panel 1: prediction probabilities.
            axes[0].barh(['Benign', 'Malignant'], [probs[0], probs[1]], color=['#f39c12', '#2e86c1'])
            axes[0].set_xlim(0.0, 1.0)
            axes[0].set_title('Prediction Probabilities', fontsize=10)
            for bar in axes[0].patches:
                w = bar.get_width()
                axes[0].text(min(w + 0.02, 0.95), bar.get_y() + bar.get_height() / 2, f'{w:.2f}', va='center', fontsize=8)

            # Panel 2: feature contribution bars.
            rules = [item[0] for item in exp_rules]
            weights = [item[1] for item in exp_rules]
            colors = ['#2e86c1' if w < 0 else '#f39c12' for w in weights]
            axes[1].barh(range(len(rules)), weights, color=colors)
            axes[1].set_yticks(range(len(rules)))
            axes[1].set_yticklabels(rules, fontsize=7)
            axes[1].invert_yaxis()
            axes[1].axvline(x=0, color='black', linewidth=0.8)
            axes[1].set_title('Feature Importance (LIME)', fontsize=10)

            # Panel 3: feature values table for explained rules.
            table_features = []
            for rule in rules:
                matched = None
                for feature in sorted(feature_names, key=len, reverse=True):
                    if feature in rule:
                        matched = feature
                        break
                if matched is not None and matched not in table_features:
                    table_features.append(matched)

            table_rows = []
            for feature in table_features[:10]:
                table_rows.append([feature, f"{row_df.iloc[0][feature]:.4f}"])

            axes[2].axis('off')
            table = axes[2].table(
                cellText=table_rows if table_rows else [['N/A', 'N/A']],
                colLabels=['Feature', 'Value'],
                loc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            axes[2].set_title('Feature Values', fontsize=10)

            class_text = f"True: {'Malignant' if true_label == 1 else 'Benign'} | Pred: {'Malignant' if pred_label == 1 else 'Benign'}"
            fig.suptitle(f'LIME Sample {sample_idx} - {class_text}', fontsize=10)
            fig.tight_layout()
            fig.savefig(sample_plot_dir / f'lime_sample_{sample_idx}.png', dpi=300)
            plt.close(fig)

            # Save classic single-panel local explanation for each sample.
            local_rules = [item[0] for item in exp_rules]
            local_weights = [item[1] for item in exp_rules]
            local_colors = ['green' if w >= 0 else 'red' for w in local_weights]

            fig_local, ax_local = plt.subplots(figsize=(8, 3.2))
            ax_local.barh(range(len(local_rules)), local_weights, color=local_colors)
            ax_local.set_yticks(range(len(local_rules)))
            ax_local.set_yticklabels(local_rules, fontsize=7)
            ax_local.invert_yaxis()
            ax_local.axvline(x=0, color='black', linewidth=0.8)
            ax_local.set_title('Local explanation for class Benign', fontsize=10)
            fig_local.tight_layout()
            fig_local.savefig(local_sample_plot_dir / f'lime_local_sample_{sample_idx}.png', dpi=300)
            plt.close(fig_local)

    lime_df = pd.DataFrame(explanation_rows)
    save_df(lime_df, '13_lime_explanations.csv')

    probability_df = pd.DataFrame(probability_rows)
    save_df(probability_df, '18_lime_sample_probabilities.csv')

    global_df = (
        lime_df.groupby('parsed_feature', as_index=False)
        .agg(
            mean_weight=('weight', 'mean'),
            mean_abs_weight=('abs_weight', 'mean'),
            frequency=('parsed_feature', 'count'),
        )
        .sort_values(by='mean_abs_weight', ascending=False)
        .reset_index(drop=True)
    )
    # Keep both legacy and new filenames so no result is lost.
    save_df(global_df, '14_lime_global_summary.csv')
    save_df(global_df, '15_lime_global_summary.csv')

    overview = pd.DataFrame(
        [
            {
                'plots_saved_in': str(sample_plot_dir),
                'num_sample_plots': min(5, max_samples),
                'num_explained_samples': max_samples,
            }
        ]
    )
    save_df(overview, '16_lime_outputs_overview.csv')

    # Create a single paper-ready combined figure with four different samples.
    paper_count = min(4, max_samples)

    # Legacy single-column stacked layout similar to notebook screenshot style.
    if paper_count > 0:
        fig, axes = plt.subplots(paper_count, 1, figsize=(8, 3.4 * paper_count))
        if paper_count == 1:
            axes = [axes]

        for idx in range(paper_count):
            exp = explainer.explain_instance(
                data_row=x_test_raw.iloc[idx].to_numpy(),
                predict_fn=predict_fn,
                num_features=10,
            )
            rules = [item[0] for item in exp.as_list()]
            weights = [item[1] for item in exp.as_list()]
            colors = ['green' if w >= 0 else 'red' for w in weights]

            ax = axes[idx]
            ax.barh(range(len(rules)), weights, color=colors)
            ax.set_yticks(range(len(rules)))
            ax.set_yticklabels(rules, fontsize=7)
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_title('Local explanation for class Benign', fontsize=10)
            ax.text(
                0.5,
                -0.25,
                f"({chr(97 + idx)}) sample-{idx + 1}",
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
            )

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / '15_lime_example_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    if paper_count > 0:
        fig, axes = plt.subplots(paper_count, 3, figsize=(14, 3.8 * paper_count), gridspec_kw={'width_ratios': [1.0, 2.0, 1.6]})
        if paper_count == 1:
            axes = [axes]

        for idx in range(paper_count):
            row_df = x_test_raw.iloc[[idx]]
            row_scaled = pd.DataFrame(
                scaler.transform(row_df),
                columns=feature_names,
                index=row_df.index,
            )
            probs = best_model.predict_proba(row_scaled)[0]
            exp = explainer.explain_instance(
                data_row=x_test_raw.iloc[idx].to_numpy(),
                predict_fn=predict_fn,
                num_features=10,
            )
            rules = [item[0] for item in exp.as_list()]
            weights = [item[1] for item in exp.as_list()]

            ax_prob, ax_imp, ax_tbl = axes[idx]
            ax_prob.barh(['Benign', 'Malignant'], [probs[0], probs[1]], color=['#f39c12', '#2e86c1'])
            ax_prob.set_xlim(0.0, 1.0)
            ax_prob.set_title(f'Sample {idx} Probabilities', fontsize=9)

            colors = ['#2e86c1' if w < 0 else '#f39c12' for w in weights]
            ax_imp.barh(range(len(rules)), weights, color=colors)
            ax_imp.set_yticks(range(len(rules)))
            ax_imp.set_yticklabels(rules, fontsize=6)
            ax_imp.invert_yaxis()
            ax_imp.axvline(x=0, color='black', linewidth=0.7)
            ax_imp.set_title(f'Sample {idx} Importance', fontsize=9)

            table_features = []
            for rule in rules:
                for feature in sorted(feature_names, key=len, reverse=True):
                    if feature in rule and feature not in table_features:
                        table_features.append(feature)
                        break

            rows = [[f, f"{row_df.iloc[0][f]:.3f}"] for f in table_features[:10]]
            ax_tbl.axis('off')
            table = ax_tbl.table(
                cellText=rows if rows else [['N/A', 'N/A']],
                colLabels=['Feature', 'Value'],
                loc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.1)
            ax_tbl.set_title(f'Sample {idx} Values', fontsize=9)

        fig.suptitle('LIME Explanations Across Different Samples', fontsize=12)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / '17_lime_multisample_panel.png', dpi=300)
        plt.close(fig)


def main() -> None:
    dataset = load_dataset()
    save_eda_results(dataset)

    x_data, y_data = prepare_data(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=42,
        stratify=y_data,
    )

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

    models = get_classifier_models()

    train_df = train_test_results(models, x_train_scaled, y_train, x_test_scaled, y_test)
    save_df(train_df, '06_train_test_model_results.csv')

    x_scaled_all = pd.DataFrame(scaler.fit_transform(x_data), columns=x_data.columns)
    cv_df = cross_validation_results(models, x_scaled_all, y_data)
    save_df(cv_df, '07_cross_validation_results.csv')

    abl_df = ablation_study(x_data, y_data)
    save_df(abl_df, '08_ablation_results.csv')

    best_model_name = train_df.iloc[0]['Model']
    best_model = dict(models)[best_model_name]
    best_model.fit(x_train_scaled, y_train)
    save_confusion_matrix(best_model_name, best_model, x_test_scaled, y_test)
    save_lime_outputs(best_model, scaler, x_train, x_test, y_test, x_data.columns.tolist())

    save_summary(train_df, cv_df, abl_df)

    print('All results saved in:', RESULTS_DIR)


if __name__ == '__main__':
    main()
