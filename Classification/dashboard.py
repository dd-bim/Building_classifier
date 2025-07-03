import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import json
import configparser

"""
Dashboard zur Visualisierung der Klassifikations- und Validierungsergebnisse.

Dieses Dash-Interface zeigt Metriken, Feature-Importances, Konfusionsmatrix und weitere Auswertungen
für die Gebäudeklassifikation. Die Erklärungen zu Metriken und Diagrammen sind direkt im Code dokumentiert.
"""

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

# Initialisiere Dash mit Dark Mode
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

# Lade die Ergebnisse
results_path = os.path.join(os.path.dirname(__file__), 'Validierung')
vis_path = os.path.join(os.path.dirname(__file__), 'Validierung')

# Lade die Visualisierungen und Ergebnisse
results_json_path = os.path.join(vis_path, 'validation_results.json')
try:
    with open(results_json_path, 'r') as json_file:
        results_data = json.load(json_file)
    print(f"Loaded results data with {len(results_data.get('levels', []))} levels")
    print(f"Available levels: {[level['level'] for level in results_data.get('levels', [])]}")
except Exception as e:
    print(f"Error loading results data: {e}")
    results_data = {"levels": [], "overall_results": {}}

# Erklärungen für die verschiedenen Metriken
metric_explanations = {
    "F1 Score": "Der harmonische Mittelwert von Präzision und Recall. Diese Metrik ist besonders nützlich, wenn ein Gleichgewicht zwischen Präzision und Recall wichtig ist.",
    "F1 Macro": "Der Durchschnitt der F1-Scores über alle Klassen hinweg, unabhängig von der Klassengröße. Diese Metrik gewichtet alle Klassen gleich.",
    "Precision": "Der Anteil der korrekt vorhergesagten positiven Fälle an allen als positiv vorhergesagten Fällen. Formel: TP / (TP + FP).",
    "Recall": "Der Anteil der tatsächlich positiven Fälle, die korrekt vorhergesagt wurden. Auch bekannt als Sensitivität. Formel: TP / (TP + FN).",
    "Accuracy": "Der Anteil der korrekten Vorhersagen (sowohl positiv als auch negativ) im Verhältnis zur Gesamtanzahl der Vorhersagen. Formel: (TP + TN) / (TP + TN + FP + FN).",
    "Matthews Correlation Coefficient": "Eine Metrik, die die Korrelation zwischen den vorhergesagten und den tatsächlichen Klassen misst. Werte reichen von -1 (perfekte Fehlklassifikation) über 0 (zufällige Vorhersage) bis 1 (perfekte Vorhersage)."
}

# Erklärungen für die Diagramme
diagram_explanations = {
    "confusion-matrix": "Zeigt die tatsächlichen vs. vorhergesagten Klassifikationen. Diagonalwerte sind korrekte Vorhersagen. Hohe Werte außerhalb der Diagonale deuten auf Fehlklassifikationen hin.",
    "precision-recall-curve": "Verhältnis zwischen Precision (Genauigkeit) und Recall (Trefferquote). Eine ideale Kurve liegt nahe bei (1,1).",
    "roc-curve": "Vergleicht Sensitivität (True Positive Rate) mit False Positive Rate. Höhere AUC-Werte zeigen bessere Modellleistung."
}

overall_metric_explanations = {
    "True Positives (TP)": "Die Anzahl der Fälle, bei denen das Modell korrekt vorhergesagt hat, dass sie zur positiven Klasse gehören.",
    "True Negatives (TN)": "Die Anzahl der Fälle, bei denen das Modell korrekt vorhergesagt hat, dass sie zur negativen Klasse gehören.",
    "False Positives (FP)": "Die Anzahl der Fälle, bei denen das Modell fälschlicherweise vorhergesagt hat, dass sie zur positiven Klasse gehören (auch bekannt als Typ-I-Fehler).",
    "False Negatives (FN)": "Die Anzahl der Fälle, bei denen das Modell fälschlicherweise vorhergesagt hat, dass sie zur negativen Klasse gehören (auch bekannt als Typ-II-Fehler).",
    "Accuracy (Gesamt)": "Der Anteil der korrekten Vorhersagen (sowohl positiv als auch negativ) im Verhältnis zur Gesamtanzahl der Vorhersagen.",
    "Sensitivity (Gesamt)": "Auch bekannt als True Positive Rate oder Recall. Zeigt, wie gut das Modell tatsächlich positive Fälle erkennt: TP / (TP + FN).",
    "Specificity (Gesamt)": "Auch bekannt als True Negative Rate. Zeigt, wie gut das Modell tatsächlich negative Fälle erkennt: TN / (TN + FP).",
    "Gewichtete Genauigkeit": "Die Genauigkeit, gewichtet nach der Anzahl der Fälle in jeder Klasse. Dies berücksichtigt Klassen mit ungleicher Verteilung.",
    "Gewichtete Sensitivität": "Die Sensitivität, gewichtet nach der Anzahl der Fälle in jeder Klasse. Dies gleicht Klassen mit ungleicher Verteilung aus.",
    "Gewichtete Spezifität": "Die Spezifität, gewichtet nach der Anzahl der Fälle in jeder Klasse. Dies gleicht Klassen mit ungleicher Verteilung aus.",
    "End-to-End Genauigkeit": "Die Genauigkeit des Modells über den gesamten Prozess hinweg, einschließlich aller Verarbeitungsschritte.",
    "End-to-End F1-Score": "Der harmonische Mittelwert von Präzision und Recall über den gesamten Prozess hinweg. Ein höherer Wert zeigt ein besseres Gleichgewicht zwischen Präzision und Recall.",
    "End-to-End Präzision": "Der Anteil der korrekt vorhergesagten positiven Fälle an allen als positiv vorhergesagten Fällen: TP / (TP + FP).",
    "End-to-End Recall": "Der Anteil der tatsächlich positiven Fälle, die korrekt vorhergesagt wurden: TP / (TP + FN).",
    "Correct by Model": "Anzahl der durch das Modell korrekt klassifizierten Objekte ohne direkte Zuweisung.",
    "Direkte Zuweisungen (Baualter)": "Anzahl der Objekte, die durch regelbasierte Zuweisung klassifiziert wurden (z.B. Baualter).",
    "Korrekt Klassifiziert (Gesamt)": "Summe aller korrekten Vorhersagen durch Modell und Regelwerk (z.B. Baualter)."
}

summary_metrics_tooltips = [
        dbc.Tooltip(overall_metric_explanations["True Positives (TP)"], target="tp-tooltip"),
        dbc.Tooltip(overall_metric_explanations["True Negatives (TN)"], target="tn-tooltip"),
        dbc.Tooltip(overall_metric_explanations["False Positives (FP)"], target="fp-tooltip"),
        dbc.Tooltip(overall_metric_explanations["False Negatives (FN)"], target="fn-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Accuracy (Gesamt)"], target="accuracy-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Sensitivity (Gesamt)"], target="sensitivity-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Specificity (Gesamt)"], target="specificity-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Gewichtete Genauigkeit"], target="weighted-accuracy-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Gewichtete Sensitivität"], target="weighted-sensitivity-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Gewichtete Spezifität"], target="weighted-specificity-tooltip"),
        dbc.Tooltip(overall_metric_explanations["End-to-End Genauigkeit"], target="end-to-end-accuracy-tooltip"),
        dbc.Tooltip(overall_metric_explanations["End-to-End F1-Score"], target="end-to-end-f1-tooltip"),
        dbc.Tooltip(overall_metric_explanations["End-to-End Präzision"], target="end-to-end-precision-tooltip"),
        dbc.Tooltip(overall_metric_explanations["End-to-End Recall"], target="end-to-end-recall-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Correct by Model"], target="correct-by-model-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Direkte Zuweisungen (Baualter)"], target="direct-assignment-tooltip"),
        dbc.Tooltip(overall_metric_explanations["Korrekt Klassifiziert (Gesamt)"], target="correct-total-tooltip")
    ]

# Layout des Dashboards mit Bootstrap
app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H1("📊 Validation Dashboard", className="text-center text-info"), width=12)
    ], className="mb-4"),

    # Dropdown für Level Auswahl
    dbc.Row([
        dbc.Col(html.Label("🔍 Wähle ein Level:", className="fw-bold"), width=3),
        dbc.Col(dcc.Dropdown(
            id='level-dropdown',
            options=[{'label': f'Level {level["level"]}', 'value': level["level"]} for level in results_data.get("levels", [])],
            value=results_data.get("levels", [{}])[0].get("level", "1") if results_data.get("levels") else "1",
            className="mb-2",
            style={"color": "black"}
        ), width=6)
    ], className="mb-4"),

    # Fortschrittsbalken für Modellmetriken (Klickbar für Erklärung)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📈 Modellmetriken"),  
            dbc.CardBody(html.Div(id="metrics-progress", style={"font-size": "18px"}))
        ], className="shadow-sm"), width=12)
    ], className="mb-4"),
    
    # Feature-Importance-Diagramm
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("🔍 Feature Importance"),
            dbc.CardBody(dcc.Graph(id="feature-importance", style={"cursor": "pointer"}))
        ], className="shadow-sm"), width=12)
    ], className="mb-4"),

    # Interaktive Heatmap (Confusion Matrix)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader([
                "🟦 Confusion Matrix",
                html.Br(),
                html.Span(" (Zeigt die tatsächlichen vs. vorhergesagten Klassifikationen. Diagonalwerte sind korrekte Vorhersagen. Hohe Werte außerhalb der Diagonale deuten auf Fehlklassifikationen hin.)", className="text-muted")
            ]),
            dbc.CardBody(dcc.Graph(id="confusion-matrix", style={"cursor": "pointer"}))
        ], className="shadow-sm"), width=12)
    ], className="mb-4"),

    # dbc.Row([
    #     # Precision-Recall Kurve
    #     dbc.Col(dbc.Card([
    #         dbc.CardHeader([
    #             "🔵 Precision-Recall Curve",
    #             html.Br(),
    #             html.Span(" (Verhältnis zwischen Precision (Genauigkeit) und Recall (Trefferquote). Eine ideale Kurve liegt nahe bei (1,1).)", className="text-muted")
    #         ]),
    #         dbc.CardBody(dcc.Graph(id="precision-recall-curve", style={"cursor": "pointer"}))
    #     ], className="shadow-sm"), width=6),

    # # ROC Kurve
    #     dbc.Col(dbc.Card([
    #         dbc.CardHeader([
    #             "🔴 ROC Curve",
    #             html.Br(),
    #             html.Span(" (Vergleicht Sensitivität (True Positive Rate) mit False Positive Rate. Höhere AUC-Werte zeigen bessere Modellleistung.)", className="text-muted")
    #         ]),
    #         dbc.CardBody(dcc.Graph(id="roc-curve", style={"cursor": "pointer"}))
    #     ], className="shadow-sm"), width=6)
    # ], className="mb-4"),

    # Zusammenfassung der True Positive, True Negative, etc.
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📊 Zusammenfassung der Metriken"),
            dbc.CardBody(html.Div(id="summary-metrics", style={"font-size": "18px"}))
        ], className="shadow-sm"), width=12)
    ], className="mb-4"),

    # Gewichtete Metriken und End-to-End-Bewertung
    # dbc.Row([
    #     dbc.Col(dbc.Card([
    #         dbc.CardHeader("📊 Gewichtete Metriken"),
    #         dbc.CardBody([
    #             html.Div([
    #                 html.H5("Gewichtete Genauigkeit", className="text-info"),
    #                 html.H4(f"{results_data['overall_results'].get('weighted_accuracy', 0):.2%}", className="text-center")
    #             ], className="mb-3"),
    #             html.Div([
    #                 html.H5("Gewichtete Sensitivität", className="text-success"),
    #                 html.H4(f"{results_data['overall_results'].get('weighted_sensitivity', 0):.2%}", className="text-center")
    #             ], className="mb-3"),
    #             html.Div([
    #                 html.H5("Gewichtete Spezifität", className="text-warning"),
    #                 html.H4(f"{results_data['overall_results'].get('weighted_specificity', 0):.2%}", className="text-center")
    #             ])
    #         ])
    #     ], className="shadow-sm"), width=6),
    
    dbc.Row([ 
        dbc.Col(dbc.Card([
                dbc.CardHeader("📊 End-to-End-Bewertung"),
                dbc.CardBody([
                    html.Div([
                        html.H5("End-to-End Genauigkeit", className="text-info", id="end-to-end-accuracy-tooltip"),
                        html.H4(f"{results_data['overall_results'].get('end_to_end_accuracy', 0):.2%}", className="text-center")
                    ], className="mb-3"),
                    html.Div([
                        html.H5("Modell-Treffer (korrekt_by_model)", className="text-success", id="correct-by-model-tooltip"),
                        html.H4(f"{results_data['overall_results'].get('correct_by_model', 0)}", className="text-center")
                    ], className="mb-3"),
                    html.Div([
                        html.H5("Direkte Zuweisungen (Baualter)", className="text-warning", id="direct-assignment-tooltip"),
                        html.H4(f"{results_data['overall_results'].get('direct_assignment_count', 0)}", className="text-center")
                    ], className="mb-3"),
                    html.Div([
                        html.H5("Korrekt Klassifiziert (Gesamt)", className="text-info", id="correct-total-tooltip"),
                        html.H4(f"{results_data['overall_results'].get('correct_total', 0)}", className="text-center")
                    ])
                ])
            ], className="shadow-sm"), width=12),
        ], className="mb-4"),
    
    html.Div(summary_metrics_tooltips)
])

# Callback für die Aktualisierung der Visualisierungen
@app.callback(
    [Output('metrics-progress', 'children'),
     Output('feature-importance', 'figure'),
     Output('confusion-matrix', 'figure'),
     #Output('precision-recall-curve', 'figure'),
     #Output('roc-curve', 'figure'),
     Output('summary-metrics', 'children')],
    [Input('level-dropdown', 'value')]
)
def update_dashboard(level):
    try:
        print(f"Updating dashboard for level: {level}")
        level_data = next((item for item in results_data["levels"] if item["level"] == level), None)
        print(f"Found level data: {level_data is not None}")
        
        feature_importance_data = load_feature_importance(level)
        print(f"Feature importance data loaded: {feature_importance_data is not None}")
        
        # Initialisiere overall_results mit Standardwerten
        overall_results = results_data.get("overall_results", {
            'overall_accuracy': 0,
            'overall_sensitivity': 0,
            'overall_specificity': 0,
            'weighted_accuracy': 0,
            'weighted_sensitivity': 0,
            'weighted_specificity': 0,
            'end_to_end_accuracy': 0,
            'end_to_end_f1': 0,
            'end_to_end_precision': 0,
            'end_to_end_recall': 0,
            'total_TP': 0,
            'total_TN': 0,
            'total_FP': 0,
            'total_FN': 0,
            'correct_count': 0,
            'total_count': 0
        })
    except Exception as e:
        print(f"Error in update_dashboard: {e}")
        return (
            [],  # Fortschrittsbalken
            px.bar(title=f"Error loading data: {e}"),  # Feature-Importance-Diagramm
            px.imshow([[0]], text_auto=True, title="Error loading data"),  # Confusion-Matrix-Diagramm
            html.Div(f"Fehler beim Laden der Daten: {e}", className="text-danger")
        )
    
    # if not level_data:
    #     return (
    #         [],  # Fortschrittsbalken
    #         px.bar(title="Keine Feature-Importance-Daten verfügbar"),  # Feature-Importance-Diagramm
    #         px.imshow([[0]], text_auto=True, title="No Data"),  # Confusion-Matrix-Diagramm
    #         px.line(title="Precision-Recall Curve (Placeholder)"),  # Precision-Recall-Kurve
    #         px.line(title="ROC Curve (Placeholder)"),  # ROC-Kurve
    #         html.Div("Für das ausgewählte Level sind keine Daten verfügbar.", className="text-danger")
    #     )

    if not level_data:
        return (
            [],  # Fortschrittsbalken
            px.bar(title="Keine Feature-Importance-Daten verfügbar"),  # Feature-Importance-Diagramm
            px.imshow([[0]], text_auto=True, title="No Data"),  # Confusion-Matrix-Diagramm
            html.Div("Für das ausgewählte Level sind keine Daten verfügbar.", className="text-danger")
        )
        
    metrics = {
        "Accuracy": level_data.get("accuracy", 0),
        "F1 Score": level_data.get("f1_weighted", 0),
        #"F1 Macro": level_data.get("f1_macro", 0),
        "Precision": level_data.get("precision", 0),
        "Recall": level_data.get("recall", 0),
        "Matthews Correlation Coefficient": level_data.get("mcc", 0)
    }
    
    progress_bars = [
        dbc.Progress(
            value=metrics.get(metric, 0) * 100,
            label=f"{metric}: {metrics.get(metric, 0) * 100:.2f}%",
            color="info",
            striped=True,
            animated=True,
            id=f"progress-{metric.replace(' ', '-')}",
            style={"height": "30px", "font-size": "18px"}
        ) for metric in metrics.keys()
    ]

    tooltips = [
        dbc.Tooltip(
            metric_explanations[metric],
            target=f"progress-{metric.replace(' ', '-')}"
        ) for metric in metric_explanations.keys()
    ]
    
    if feature_importance_data:
        fig_feature_importance = create_feature_importance_figure(feature_importance_data)
    else:
        fig_feature_importance = px.bar(title="Keine Feature-Importance-Daten verfügbar")

    class_names = level_data["class_names"]
    conf_matrix = np.array(level_data["conf_matrix"])
    all_class_names = sorted(set(class_names))
    extended_conf_matrix = np.zeros((len(all_class_names), len(all_class_names)), dtype=int)

    for i, class_name in enumerate(class_names):
        for j, class_name in enumerate(class_names):
            if i < conf_matrix.shape[0] and j < conf_matrix.shape[1]:
                extended_conf_matrix[i, j] = conf_matrix[i, j]

    fig_cm = px.imshow(
        extended_conf_matrix,
        text_auto=True,
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        x=all_class_names,
        y=all_class_names,
        color_continuous_scale="Blues"
    )
    
    # Daten für Precision-Recall-Kurve formatieren
    pr_data = pd.DataFrame({
        "Recall": level_data["precision_recall_curve"]["recall"],
        "Precision": level_data["precision_recall_curve"]["precision"]
    })
    
    fig_pr = px.line(
        pr_data,
        x="Recall",
        y="Precision",
        title="Precision-Recall Curve",
        labels={"x": "Recall", "y": "Precision"}
    )
    
    # Daten für ROC-Kurve formatieren
    roc_data = pd.DataFrame({
        "False Positive Rate": level_data["roc_curve"]["fpr"],
        "True Positive Rate": level_data["roc_curve"]["tpr"]
    })
    
    fig_roc = px.line(
        roc_data,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC = {level_data['roc_curve']['roc_auc']:.2f})",
        labels={"x": "False Positive Rate", "y": "True Positive Rate"}
    )

    # Zusammenfassung der True Positive, True Negative, etc.
    overall_results = results_data.get("overall_results", {})

    summary_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("True Positives (TP)", className="text-center text-success", id="tp-tooltip"),
            dbc.CardBody(html.H4(overall_results.get('total_TP', 0), className="text-center"))
        ], className="shadow-sm"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("True Negatives (TN)", className="text-center text-primary", id="tn-tooltip"),
            dbc.CardBody(html.H4(overall_results.get('total_TN', 0), className="text-center"))
        ], className="shadow-sm"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("False Positives (FP)", className="text-center text-danger", id="fp-tooltip"),
            dbc.CardBody(html.H4(overall_results.get('total_FP', 0), className="text-center"))
        ], className="shadow-sm"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardHeader("False Negatives (FN)", className="text-center text-warning", id="fn-tooltip"),
            dbc.CardBody(html.H4(overall_results.get('total_FN', 0), className="text-center"))
        ], className="shadow-sm"), width=3)
    ], className="mb-4")

    # Gesamtmetriken (Accuracy, Sensitivity, Specificity)
    # summary_metrics = dbc.Row([
    #     dbc.Col(dbc.Card([
    #         dbc.CardHeader("Accuracy (Gesamt)", className="text-center text-info", id="accuracy-tooltip"),
    #         dbc.CardBody(html.H4(
    #             f"{(overall_results.get('overall_accuracy') or 0):.2%} "
    #             f"({overall_results.get('correct_count', 0)} / {overall_results.get('total_count', 0)})",
    #             className="text-center"
    #         ))
    #     ], className="shadow-sm"), width=4),

    #     dbc.Col(dbc.Card([
    #         dbc.CardHeader("Sensitivity (Gesamt)", className="text-center text-success", id="sensitivity-tooltip"),
    #         dbc.CardBody(html.H4(f"{(overall_results.get('overall_sensitivity') or 0):.2%}", className="text-center"))
    #     ], className="shadow-sm"), width=4),

    #     dbc.Col(dbc.Card([
    #         dbc.CardHeader("Specificity (Gesamt)", className="text-center text-warning", id="specificity-tooltip"),
    #         dbc.CardBody(html.H4(f"{(overall_results.get('overall_specificity') or 0):.2%}", className="text-center"))
    #     ], className="shadow-sm"), width=4)
    # ], className="mb-4")

    combined_metrics = html.Div([
        summary_cards
        #summary_metrics
    ])

    return (
        progress_bars,  # Fortschrittsbalken
        fig_feature_importance,  # Feature-Importance-Diagramm
        fig_cm,  # Confusion-Matrix-Diagramm
        #fig_pr,  # Precision-Recall-Kurve
        #fig_roc,  # ROC-Kurve
        combined_metrics  # Alle Metriken kombiniert
    )

def load_feature_importance(level):

    feature_importance_dir = os.path.join(os.path.dirname(__file__), config.get('Paths', 'model_dir'))
    importance_file = os.path.join(feature_importance_dir, f'feature_importance_{level}.json')
    try:
        with open(importance_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None
    
def create_feature_importance_figure(feature_importance_data):

    sorted_indices = np.argsort(feature_importance_data['importance'])[::-1]
    sorted_features = [feature_importance_data['features'][i] for i in sorted_indices]
    sorted_importance = [feature_importance_data['importance'][i] for i in sorted_indices]
    
    fig = px.bar(
        x=sorted_importance,
        y=sorted_features,
        orientation='h',
        title="Feature Importance",
        labels={'x': 'Importance', 'y': 'Features'},
        color=sorted_importance,
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
    
# Starte den Server
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)