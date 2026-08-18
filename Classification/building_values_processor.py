import os
import csv
import numpy as np
from scipy import stats
from qgis.core import QgsProject, QgsMessageLog, Qgis
from PyQt5.QtCore import QVariant
from collections import Counter
from .config_loader import get_config

class BuildingValuesProcessor:
    def __init__(self):
        """
        Initialisiert den BuildingValuesProcessor und lädt relevante Dateipfade aus der Konfiguration.
        """
        self.layer_name = 'citydb_filter'

        config = get_config()
        self.output_file = os.path.join(os.path.dirname(__file__), config.get('Paths', 'building_values_file'))
        self.separate_output_file = os.path.join(os.path.dirname(__file__), config.get('Paths', 'values_separate_file'))
        self.csv_output_file = os.path.join(os.path.dirname(__file__), config.get('Paths', 'building_values_csv_file'))
        self.separate_csv_output_file = os.path.join(os.path.dirname(__file__), config.get('Paths', 'values_separate_csv_file'))
        
        self.progress_bar = None

    def process_values(self):
        """
        Berechnet und speichert statistische Kennzahlen für Gebäudemerkmale nach SST_SUB und SST.
        Ergebnisse werden als Text- und CSV-Dateien ausgegeben.
        """
        # Aktiven Layer abrufen
        active_layer = self.get_active_layer()
        
        if not active_layer or active_layer.name() != self.layer_name:
            QgsMessageLog.logMessage(f"Active layer is not {self.layer_name}. Please select the correct layer.", level=Qgis.Critical)
            return

        layer = active_layer
        attributes = ['sst_sub', 'sst']  # Attribute: SST_SUB vor SST
        metrics = ['building_footprint', 'storeys_above_ground', 'length_footprint', 'width_footprint', 'roof_ridge_height', 'eaves_height', 'number_roof_surfaces', 'roof_slope']
        results = {}
        processed_features = set()
        
        total_steps = len(attributes) * len(metrics)
        current_step = 0

        for attribute in attributes:
            # Alle eindeutigen Werte für das Attribut bestimmen
            unique_values = set(f[attribute] for f in layer.getFeatures() if f[attribute] not in [None, 'NULL'])
            for value in unique_values:
                if value in [None, 'NULL']:
                    continue

                # Features mit aktuellem Attributwert filtern
                features = [f for f in layer.getFeatures() if f[attribute] == value and f.id() not in processed_features]
                if not features:
                    continue
                
                processed_features.update(f.id() for f in features)

                if attribute not in results:
                    results[attribute] = {}
                results[attribute][value] = {}

                for metric in metrics:
                    # Werte für das aktuelle Merkmal extrahieren
                    metric_values = [f[metric] for f in features if f[metric] is not None and f[metric] != 'NULL' and isinstance(f[metric], (int, float))]
                    if not metric_values:
                        continue
                    
                    metric_values = np.array([v.value() if isinstance(v, QVariant) else v for v in metric_values])
                    
                    # Berechnung der Statistiken
                    stats = self.calculate_statistics(metric_values)

                    results[attribute][value][metric] = stats
                    
                    current_step += 1
                    if self.progress_bar:
                        progress = int(current_step / total_steps * 50)
                        self.progress_bar.setValue(progress)
                
                # Zähle Dachtypen und Entwicklungstypen
                roof_types = [f['roof_type'] for f in features if f['roof_type'] not in [None, 'NULL']]
                if roof_types:
                    roof_type_counts = Counter(roof_types)
                    total_roof_types = sum(roof_type_counts.values())
                    roof_type_percentages = {k: round(v / total_roof_types * 100, 2) for k, v in roof_type_counts.items()}
                    sorted_roof_types = sorted(roof_type_percentages.items(), key=lambda item: item[1], reverse=True)
                    results[attribute][value]['roof_type'] = sorted_roof_types
                    
                development_types = [f['development_type_code'] for f in features if f['development_type_code'] not in [None, 'NULL']]
                if development_types:
                    development_type_counts = Counter(development_types)
                    total_development_types = sum(development_type_counts.values())
                    development_type_percentages = {k: round(v / total_development_types * 100, 2) for k, v in development_type_counts.items()}
                    sorted_development_types = sorted(development_type_percentages.items(), key=lambda item: item[1], reverse=True)
                    results[attribute][value]['development_type_code'] = sorted_development_types
                    
                results[attribute][value]['count'] = len(features)

        # Schreibe Ergebnisse in Textdatei
        with open(self.output_file, 'w') as f:
            for attribute in sorted(results.keys()):
                categories = results[attribute]
                for category in sorted(categories.keys()):
                    f.write(f"Attribute: {attribute}, Category: {category}\n")
                    f.write(f"Count: {categories[category]['count']}\n")
                    for metric, values in categories[category].items():
                        if metric == 'roof_type':
                            f.write(f"Roof Type: {values}\n")
                        elif metric == 'development_type_code':
                            f.write(f"Development Type Code: {values}\n")
                        elif metric != 'count':
                            f.write(f"{metric.capitalize()} - Min: {values['min']}, Max: {values['max']}, Mean: {values['mean']}, Median: {values['median']}, Std Dev: {values['std_dev']}, CI Mean Low: {values['ci_mean_low']}, CI Mean High: {values['ci_mean_high']}, CI Median Low: {values['ci_median_low']}, CI Median High: {values['ci_median_high']}\n")
                    f.write("\n")

        QgsMessageLog.logMessage(f"Building values saved to {self.output_file}", level=Qgis.Info)
        
        self.write_results_to_csv(results, self.csv_output_file)
        self.process_separate_values(layer, current_step, total_steps)

    def process_separate_values(self, layer, current_step, total_steps):
        """
        Berechnet und speichert Kennzahlen für vordefinierte Gebäudekategorien (z.B. MR, ME, ER, ...).
        Ergebnisse werden als Text- und CSV-Dateien ausgegeben.
        """
        categories = {
            'MR': lambda x: x.startswith('MR'),
            'ME': lambda x: x.startswith('ME'),
            'ER': lambda x: x.startswith('ER'),
            'EE': lambda x: x.startswith('EE'),
            'LW': lambda x: x.startswith('LW'),
            'LWS': lambda x: x.startswith('LWS'),
            'HH': lambda x: x.startswith('HH'),
            '1': lambda x: (len(x) > 2 and x[2] == '1') or (len(x) > 3 and x.startswith('LWS') and x[3] == '1'),
            '2': lambda x: (len(x) > 2 and x[2] == '2') or (len(x) > 3 and x.startswith('LWS') and x[3] == '2'),
            '3': lambda x: (len(x) > 2 and x[2] == '3') or (len(x) > 3 and x.startswith('LWS') and x[3] == '3'),
            '4': lambda x: (len(x) > 2 and x[2] == '4') or (len(x) > 3 and x.startswith('LWS') and x[3] == '4'),
            '5': lambda x: (len(x) > 2 and x[2] == '5') or (len(x) > 3 and x.startswith('LWS') and x[3] == '5'),
            '6': lambda x: (len(x) > 2 and x[2] == '6') or (len(x) > 3 and x.startswith('LWS') and x[3] == '6'),
            '7': lambda x: (len(x) > 2 and x[2] == '7') or (len(x) > 3 and x.startswith('LWS') and x[3] == '7')
        }

        metrics = ['building_footprint', 'storeys_above_ground', 'length_footprint', 'width_footprint', 'roof_ridge_height', 'eaves_height', 'number_roof_surfaces', 'roof_slope']
        results = {}
        
        total_steps += len(categories) * len(metrics)
        
        for category, condition in categories.items():
            # Features nach Kategorie filtern
            features = [f for f in layer.getFeatures() if condition(str(f['sst'].value()) if isinstance(f['sst'], QVariant) else str(f['sst']))]
            if not features:
                continue

            results[category] = {}

            for metric in metrics:
                metric_values = [f[metric] for f in features if f[metric] is not None and f[metric] != 'NULL' and isinstance(f[metric], (int, float))]
                if not metric_values:
                    continue

                metric_values = np.array([v.value() if isinstance(v, QVariant) else v for v in metric_values])

                # Berechnung der Statistiken
                stats = self.calculate_statistics(metric_values)

                results[category][metric] = stats
                
                current_step += 1
                if self.progress_bar:
                    progress = int(current_step / total_steps * 100)
                    self.progress_bar.setValue(progress)

            # Zähle Dachtypen und Entwicklungstypen
            roof_types = [f['roof_type'] for f in features if f['roof_type'] not in [None, 'NULL']]
            if roof_types:
                roof_type_counts = Counter(roof_types)
                total_roof_types = sum(roof_type_counts.values())
                roof_type_percentages = {k: round(v / total_roof_types * 100, 2) for k, v in roof_type_counts.items()}
                sorted_roof_types = sorted(roof_type_percentages.items(), key=lambda item: item[1], reverse=True)
                results[category]['roof_type'] = sorted_roof_types

            development_types = [f['development_type_code'] for f in features if f['development_type_code'] not in [None, 'NULL']]
            if development_types:
                development_type_counts = Counter(development_types)
                total_development_types = sum(development_type_counts.values())
                development_type_percentages = {k: round(v / total_development_types * 100, 2) for k, v in development_type_counts.items()}
                sorted_development_types = sorted(development_type_percentages.items(), key=lambda item: item[1], reverse=True)
                results[category]['development_type_code'] = sorted_development_types
                
            results[category]['count'] = len(features)

        # Schreibe Ergebnisse in Textdatei
        with open(self.separate_output_file, 'w') as f:
            for category in categories.keys():
                if category in results:
                    f.write(f"Category: {category}\n")
                    f.write(f"Count: {results[category]['count']}\n")
                    for metric, values in results[category].items():
                        if metric == 'roof_type':
                            f.write(f"Roof Type: {values}\n")
                        elif metric == 'development_type_code':
                            f.write(f"Development Type Code: {values}\n")
                        elif metric != 'count':
                            f.write(f"{metric.capitalize()} - Min: {values['min']}, Max: {values['max']}, Mean: {values['mean']}, Median: {values['median']}, Std Dev: {values['std_dev']}, CI Mean Low: {values['ci_mean_low']}, CI Mean High: {values['ci_mean_high']}, CI Median Low: {values['ci_median_low']}, CI Median High: {values['ci_median_high']}\n")
                    f.write("\n")

        QgsMessageLog.logMessage(f"Separate building values saved to {self.separate_output_file}", level=Qgis.Info)
        
        self.write_separate_results_to_csv(results, self.separate_csv_output_file)
        
    def calculate_statistics(self, data):   
        """
        Berechnet verschiedene statistische Kennzahlen und Konfidenzintervalle für ein Zahlenarray.
        """
        min_value = round(np.min(data), 2)
        max_value = round(np.max(data), 2)
        mean_value = round(np.mean(data), 2)
        median_value = round(np.median(data), 2)
        std_dev = round(np.std(data, ddof=1), 2)
        
        # Berechnung des Konfidenzintervalls für den Mittelwert
        ci_mean_low, ci_mean_high = self.normal_confidence_interval(data)
        
        # Berechnung des Konfidenzintervalls für den Median mit Bootstrapping
        ci_median_low, ci_median_high = self.bootstrap_confidence_interval(data)
        
        # Sicherstellen, dass ci_low und ci_high innerhalb des Wertebereichs liegen
        ci_mean_low = max(ci_mean_low, min_value)
        ci_mean_high = min(ci_mean_high, max_value)
        ci_median_low = max(ci_median_low, min_value)
        ci_median_high = min(ci_median_high, max_value)
        
        return {
            'min': min_value,
            'max': max_value,
            'mean': mean_value,
            'median': median_value,
            'std_dev': std_dev,
            'ci_mean_low': round(ci_mean_low, 2),
            'ci_mean_high': round(ci_mean_high, 2),
            'ci_median_low': round(ci_median_low, 2),
            'ci_median_high': round(ci_median_high, 2)
        }

    def normal_confidence_interval(self, data, confidence_level=0.95):
        """
        Berechnet das Konfidenzintervall für den Mittelwert einer Stichprobe.
        """
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # Standardabweichung der Stichprobe
        std_err = std_dev / np.sqrt(len(data))  # Standardfehler
        h = std_err * stats.t.ppf((1 + confidence_level) / 2., len(data) - 1)
        return mean - h, mean + h
    
    def bootstrap_confidence_interval(self, data, num_samples=10000, confidence_level=0.95):
        """
        Berechnet das Konfidenzintervall für den Median per Bootstrapping.
        """
        medians = []
        n = len(data)
        for _ in range(num_samples):
            sample = np.random.choice(data, size=n, replace=True)
            medians.append(np.median(sample))
        lower_percentile = (1.0 - confidence_level) / 2.0 * 100
        upper_percentile = (1.0 + confidence_level) / 2.0 * 100
        ci_low = np.percentile(medians, lower_percentile)
        ci_high = np.percentile(medians, upper_percentile)
        return ci_low, ci_high
        
    def write_results_to_csv(self, results, csv_output_file):   
        """
        Schreibt die berechneten Statistiken für alle Attribute/Kategorien in eine CSV-Datei.
        """
        with open(csv_output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'Category', 'Metric', 'Min', 'Max', 'Mean', 
                'Median', 'Std Dev', 'CI Mean Low', 'CI Mean High', 'CI Median Low', 'CI Median High', 'Additional Info'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for attribute, categories in sorted(results.items()):
                for value, metrics in sorted(categories.items()):
                    if isinstance(metrics, dict):
                        for metric, stats in metrics.items():
                            if isinstance(stats, dict):  # Numerische Metriken
                                writer.writerow({
                                    'Category': value,
                                    'Metric': metric,
                                    'Min': round(stats.get('min', ''), 2),
                                    'Max': round(stats.get('max', ''), 2),
                                    'Mean': round(stats.get('mean', ''), 2),
                                    'Median': round(stats.get('median', ''), 2),
                                    'Std Dev': round(stats.get('std_dev', ''), 2),
                                    'CI Mean Low': round(stats.get('ci_mean_low', ''), 2),
                                    'CI Mean High': round(stats.get('ci_mean_high', ''), 2),
                                    'CI Median Low': round(stats.get('ci_median_low', ''), 2),
                                    'CI Median High': round(stats.get('ci_median_high', ''), 2),
                                    'Additional Info': ''
                                })
                            elif isinstance(stats, list):  # Nicht-numerische Metriken (z.B. Dachtyp)
                                for item in stats:
                                    writer.writerow({
                                        'Category': value,
                                        'Metric': metric,
                                        'Min': '',
                                        'Max': '',
                                        'Mean': '',
                                        'Median': '',
                                        'Std Dev': '',
                                        'CI Mean Low': '',
                                        'CI Mean High': '',
                                        'CI Median Low': '',
                                        'CI Median High': '',
                                        'Additional Info': f"{item[0]}: {round(item[1], 2)}%"
                                    })

    def write_separate_results_to_csv(self, results, csv_output_file):
        """
        Schreibt die berechneten Statistiken für vordefinierte Kategorien in eine separate CSV-Datei.
        """
        with open(csv_output_file, 'w', newline='') as csvfile:
            fieldnames = [
                'Category', 'Metric', 'Min', 'Max', 'Mean', 
                'Median', 'Std Dev', 'CI Mean Low', 'CI Mean High', 'CI Median Low', 'CI Median High', 'Additional Info'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for category, metrics in sorted(results.items()):
                if isinstance(metrics, dict):
                    for metric, stats in metrics.items():
                        if isinstance(stats, dict):  # Numerische Metriken
                            writer.writerow({
                                'Category': category,
                                'Metric': metric,
                                'Min': round(stats.get('min', ''), 2),
                                'Max': round(stats.get('max', ''), 2),
                                'Mean': round(stats.get('mean', ''), 2),
                                'Median': round(stats.get('median', ''), 2),
                                'Std Dev': round(stats.get('std_dev', ''), 2),
                                'CI Mean Low': round(stats.get('ci_mean_low', ''), 2),
                                'CI Mean High': round(stats.get('ci_mean_high', ''), 2),
                                'CI Median Low': round(stats.get('ci_median_low', ''), 2),
                                'CI Median High': round(stats.get('ci_median_high', ''), 2),
                                'Additional Info': ''
                            })
                        elif isinstance(stats, list):  # Nicht-numerische Metriken (z.B. Dachtyp)
                            for item in stats:
                                writer.writerow({
                                    'Category': category,
                                    'Metric': metric,
                                    'Min': '',
                                    'Max': '',
                                    'Mean': '',
                                    'Median': '',
                                    'Std Dev': '',
                                    'CI Mean Low': '',
                                    'CI Mean High': '',
                                    'CI Median Low': '',
                                    'CI Median High': '',
                                    'Additional Info': f"{item[0]}: {round(item[1], 2)}%"
                                })
        
    def get_active_layer(self):
        """
        Gibt den aktuell ausgewählten Layer im QGIS-Projekt zurück.
        """
        layer_tree_root = QgsProject.instance().layerTreeRoot()
        selected_layers = layer_tree_root.findLayers()
        if not selected_layers:
            return None
        active_layer = selected_layers[0].layer()
        return active_layer