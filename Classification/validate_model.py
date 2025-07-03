import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib
import json
import configparser
from qgis.core import QgsMessageLog, Qgis, QgsDataSourceUri, QgsVectorLayer, QgsProject
import webbrowser
import subprocess

from .model_trainer import LabelEncoderManager
from .mapping_processor import MappingProcessor

class ValidateModel:
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den ValidateModel mit DB-Verbindung, LabelEncodern und MappingProcessor.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        
        # Lade Pfade aus der config.ini
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        
        self.model_dir = os.path.join(os.path.dirname(__file__), config.get('Paths', 'model_dir'))
        self.vis_path = os.path.join(os.path.dirname(__file__), config.get('Paths', 'vis_path'))
        
        # Initialisiere den LabelEncoderManager        
        self.label_encoder_manager = LabelEncoderManager(self.model_dir)
        self.label_encoders = self.label_encoder_manager.get_label_encoders()
        
        self.mapping_processor = MappingProcessor(conn, cur, connection_params)
        
        self.total_TP = 0
        self.total_FP = 0
        self.total_FN = 0
        self.total_TN = 0

    def load_data_from_db(self, table_name, chunk_size=None):
        """
        Lädt Daten aus einer angegebenen Tabelle der Datenbank als DataFrame.
        Unterstützt Chunk-Loading für große Datensätze.
        """
        if chunk_size:
            # Für große Tabellen: Iterator verwenden
            query = f"SELECT * FROM {table_name}"
            return pd.read_sql(query, self.conn, chunksize=chunk_size)
        else:
            # Standard-Verhalten für kleinere Tabellen
            query = f"SELECT * FROM {table_name}"
            self.cur.execute(query)
            rows = self.cur.fetchall()
            colnames = [desc[0] for desc in self.cur.description]
            df = pd.DataFrame(rows, columns=colnames)
            return df
    
    def save_results_to_db(self, all_data, y_pred, current_data_index):
        """
        Speichert die Vorhersageergebnisse für die angegebenen Indizes in die Datenbank.
        """
        db_filter_ids = all_data.loc[current_data_index, 'db_filter_id']

        # Überprüfen, ob die Länge von y_pred mit den db_filter_ids übereinstimmt
        if len(y_pred) != len(db_filter_ids):
            QgsMessageLog.logMessage("Länge von y_pred und db_filter_ids stimmt nicht überein. Speichern abgebrochen.", level=Qgis.Critical)
            return

        # Erstelle die Liste der Werte, die in die Datenbank geschrieben werden sollen
        update_data = list(zip(y_pred, db_filter_ids))
    
        update_query = '''
            UPDATE "MPSCDresden".validation_data
            SET "results" = %s
            WHERE db_filter_id = %s
        '''
        self.cur.executemany(update_query, update_data)
        self.conn.commit()
        QgsMessageLog.logMessage("Validierungsergebnisse erfolgreich in die Datenbank gespeichert.", level=Qgis.Info)
    
    def calculate_and_log_metrics(self, y_true, y_pred, target_names, level, model, calculate_curves=True, direct_assignments_count=0):
        """
        Berechnet und loggt verschiedene Klassifikationsmetriken für die Vorhersagen.
        """
        unique_classes = np.unique(y_true)
        report = classification_report(y_true, y_pred, target_names=target_names, labels=target_names, output_dict=True)

        # Einzelmetriken aus dem Report
        f1_weighted = report['weighted avg']['f1-score']
        f1_macro = report['macro avg']['f1-score']
        precision_score_value = report['weighted avg']['precision']
        recall_score_value = report['weighted avg']['recall']
        accuracy = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=target_names)

        # Konfusionsmatrix-Bestandteile berechnen
        TP = np.diag(conf_matrix)
        FP = np.array([conf_matrix[j, i] for i in range(conf_matrix.shape[0]) for j in range(i)])
        FN = np.array([conf_matrix[i, j] for i in range(conf_matrix.shape[0]) for j in range(i)])
        TN = np.array([
            np.sum(conf_matrix) - (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - conf_matrix[i, i])
            for i in range(conf_matrix.shape[0])
        ])

        # Initialwerte für beide Pfade
        overall_accuracy = overall_sensitivity = overall_specificity = None
        end_to_end_accuracy = end_to_end_precision = end_to_end_recall = end_to_end_f1 = None

        # Zwischenlevels: Akkumuliere für Gesamtauswertung
        if level != 'end_to_end':
            self.total_TP += int(np.sum(TP))
            self.total_FP += int(np.sum(FP))
            self.total_FN += int(np.sum(FN))
            self.total_TN += int(np.sum(TN))

            dac = direct_assignments_count or 0
            correct_count = self.total_TP + self.total_TN + dac
            total_count = self.total_TP + self.total_TN + self.total_FP + self.total_FN + dac

            overall_accuracy = correct_count / total_count if total_count > 0 else 0
            overall_sensitivity = self.total_TP / (self.total_TP + self.total_FN) if (self.total_TP + self.total_FN) > 0 else 0
            overall_specificity = self.total_TN / (self.total_TN + self.total_FP) if (self.total_TN + self.total_FP) > 0 else 0

        # End-to-End: unabhängig zählen
        else:
            direct_assignment_count = direct_assignments_count or 0
            correct_by_model = (pd.Series(y_true) == pd.Series(y_pred)).sum()
            correct_total = correct_by_model + direct_assignment_count
            total_count = len(y_true)

            end_to_end_accuracy = correct_total / total_count if total_count > 0 else 0
            end_to_end_precision = report['weighted avg']['precision']
            end_to_end_recall = report['weighted avg']['recall']
            end_to_end_f1 = report['weighted avg']['f1-score']

        # Gewichtete Gesamtmetriken (unabhängig von Level)
        weighted_accuracy = report['weighted avg']['recall']
        weighted_sensitivity = sum(
            report[class_name]['recall'] * report[class_name]['support']
            for class_name in unique_classes if class_name in report
        ) / report['weighted avg']['support']
        weighted_specificity = sum(
            (1 - report[class_name]['precision']) * report[class_name]['support']
            for class_name in unique_classes if class_name in report
        ) / report['weighted avg']['support']

        # PR-/ROC-Kurven
        if calculate_curves and model is not None:
            y_true_bin = label_binarize(y_true, classes=model.classes_)
            y_pred_bin = label_binarize(y_pred, classes=model.classes_)

            if y_true_bin.shape[1] > 1 and y_pred_bin.shape[1] > 1:
                precision, recall, fpr, tpr, roc_auc = {}, {}, {}, {}, {}
                for i in range(y_true_bin.shape[1]):
                    precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                all_precision = np.unique(np.concatenate([precision[i] for i in precision]))
                mean_recall = np.zeros_like(all_precision)
                for i in precision:
                    mean_recall += np.interp(all_precision, precision[i][::-1], recall[i][::-1])
                mean_recall /= len(precision)

                all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in fpr:
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= len(fpr)

                precision_recall_curve_data = {'precision': all_precision.tolist(), 'recall': mean_recall.tolist()}
                roc_curve_data = {'fpr': all_fpr.tolist(), 'tpr': mean_tpr.tolist(), 'roc_auc': np.mean(list(roc_auc.values()))}
            else:
                precision_recall_curve_data = {'precision': [], 'recall': []}
                roc_curve_data = {'fpr': [], 'tpr': [], 'roc_auc': 0}
        else:
            precision_recall_curve_data = {'precision': [], 'recall': []}
            roc_curve_data = {'fpr': [], 'tpr': [], 'roc_auc': 0}

        # Ergebnis-Objekt
        results = {
            'level': level,
            'accuracy': accuracy,
            'overall_accuracy': float(overall_accuracy) if overall_accuracy is not None else None,
            'overall_sensitivity': float(overall_sensitivity) if overall_sensitivity is not None else None,
            'overall_specificity': float(overall_specificity) if overall_specificity is not None else None,
            'end_to_end_accuracy': float(end_to_end_accuracy) if end_to_end_accuracy is not None else None,
            'end_to_end_f1': float(end_to_end_f1) if end_to_end_f1 is not None else None,
            'end_to_end_precision': float(end_to_end_precision) if end_to_end_precision is not None else None,
            'end_to_end_recall': float(end_to_end_recall) if end_to_end_recall is not None else None,
            'direct_assignment_count': int(direct_assignments_count or 0),
            'weighted_accuracy': float(weighted_accuracy),
            'weighted_sensitivity': float(weighted_sensitivity),
            'weighted_specificity': float(weighted_specificity),
            'total_TP': self.total_TP,
            'total_TN': self.total_TN,
            'total_FP': self.total_FP,
            'total_FN': self.total_FN,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision': precision_score_value,
            'recall': recall_score_value,
            'mcc': mcc,
            'conf_matrix': conf_matrix.tolist(),
            'TP': TP.tolist(),
            'TN': TN.tolist(),
            'FP': FP.tolist(),
            'FN': FN.tolist(),
            'precision_recall_curve': precision_recall_curve_data,
            'roc_curve': roc_curve_data,
            'class_names': target_names,
            'y_true': y_true.tolist() if not isinstance(y_true, list) else y_true,
            'y_pred': y_pred.tolist() if not isinstance(y_pred, list) else y_pred,
        }

        # Ergänzungen für end_to_end-Level
        if level == "end_to_end":
            results['correct_by_model'] = correct_by_model
            results['correct_total'] = correct_total
            QgsMessageLog.logMessage(
                f"End-to-End: {correct_total} korrekt (Modell: {correct_by_model}, Baualter: {direct_assignment_count}).",
                level=Qgis.Info
            )

        # Speichern
        results_file = os.path.join(self.vis_path, f'validation_results_{level}.txt')
        with open(results_file, 'w') as f:
            for key, value in results.items():
                f.write(f'{key}: {value}\n')

        return results
        
    def prepare_data(self, data, level, is_training=False):      
        """
        Bereitet die Features für die Modellvalidierung vor (inkl. Label-Encoding).
        Lädt Features dynamisch aus dem trainierten Modell.
        
        Returns:
            tuple: (X, model) - Prepared features and loaded model
        """
        X = data.copy()

        # Lade das Modell, um die erwarteten Features zu erhalten
        model_path = os.path.join(self.model_dir, f'model_{level}.pkl')
        if not os.path.exists(model_path):
            QgsMessageLog.logMessage(f"Modell für {level} nicht gefunden: {model_path}", level=Qgis.Critical)
            return pd.DataFrame(), None

        model = joblib.load(model_path)

        # Überprüfe, ob das Modell die erwarteten Features speichert
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
        else:
            QgsMessageLog.logMessage(f"Das Modell für {level} enthält keine Informationen zu den erwarteten Features.", level=Qgis.Critical)
            return pd.DataFrame(), None

        # Entferne die Zielspalte, falls sie in den Daten enthalten ist
        target_column = 'sst'
        if target_column in X.columns:
            X = X.drop(columns=[target_column])

        # Filtere die Daten basierend auf den erwarteten Features
        missing_features = [feature for feature in expected_features if feature not in X.columns]
        if missing_features:
            QgsMessageLog.logMessage(f"Fehlende Features für {level}: {missing_features}", level=Qgis.Critical)
            return pd.DataFrame(), None

        X = X[expected_features]
        
        # Transformiere kategorische Features (verwende immer fit_transform für Robustheit)
        if self.label_encoders:
            for feature in ['roof_type', 'development_type_code', 'neighbor_majority_class', 'building_age']:
                if feature in X.columns and feature in self.label_encoders:
                    if not isinstance(X[feature].iloc[0], str):
                        QgsMessageLog.logMessage(f"Warnung: {feature} ist kein String vor Label-Encoding!", level=Qgis.Warning)
                    X[feature] = X[feature].apply(lambda x: str(x) if pd.notna(x) else None)
                    X[feature] = self.label_encoders[feature].fit_transform(X[feature].astype(str))
        
        return X, model
    
    def map_y_true(self, level, s):
        """
        Mapped die Zielklasse je nach Level auf die gewünschte Gruppierung.
        """
        if pd.isnull(s):
            return None
        s = str(s)
        
        if level == '1':
            if s.startswith('M'): return 'M'
            elif s.startswith('E'): return 'E'
            else: return 'Other'
        
        elif level == '12':
            if s.startswith('HH'): return 'HH'
            elif s.startswith('LW'): return 'LW'
            else: return s

        elif level == '111':
            if s.startswith('MR2'): return 'MR2'
            elif s.startswith('MR3'): return 'MR3'
            elif s.startswith('MR4'): return 'MR4'
            elif s.startswith('MR5'): return 'MR5'
            elif s.startswith('MR6'): return 'MR6'
            elif s.startswith('MR7'): return 'MR7'
            else: return s
            
        else:
            return s
        
    def is_level_relevant(self, results_series, level):
        """
        Gibt eine Maske zurück, ob ein Datensatz für das aktuelle Level relevant ist.
        """
        if level == '121':
            return results_series == 'HH'
        elif level == '122':
            return results_series == 'LW'
        elif level == '112':
            return results_series == 'ME'
        elif level == '113':
            return results_series == 'ER'
        elif level == '114':
            return results_series == 'EE'
        elif level == '111':
            return results_series == 'MR'
        elif level == '1111':
            return results_series.str.startswith('MR', na=False)
        elif level == '12':
            return results_series == 'Other'
        elif level in ['1', '11']:
            return pd.Series([True] * len(results_series), index=results_series.index)
        else:
            return pd.Series([False] * len(results_series), index=results_series.index)

    def process_building_age(self, result, building_age):
        """
        Kombiniert das Ergebnis mit dem bekannten Baualter, falls vorhanden.
        """
        try:
            age_value = str(building_age).strip()
            if "/" in age_value:
                # Mehrere mögliche Altersstufen
                return [f"{result}{part.strip()}" for part in age_value.split("/")], False
            else:
                # Einzelne Altersstufe
                return f"{result}{age_value}", True
        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler bei building_age-Verarbeitung: {building_age}, {e}", level=Qgis.Warning)
            return result, False

    def process_level(self, all_data, level, target_names, results):
        """
        Führt die Validierung für ein bestimmtes Level durch, inklusive direkter Baualterszuweisung und Modellvorhersage.
        """
        QgsMessageLog.logMessage(f"Validierung für {level} gestartet.", level=Qgis.Info)

        # Filterung nach Level
        if level in ['1', '11']:
            current_data = all_data.copy()
        elif level == '12':
            current_data = all_data[all_data['results'] == 'Other'].copy()
        elif level == '121':
            current_data = all_data[all_data['results'] == 'HH'].copy()
        elif level == '122':
            current_data = all_data[all_data['results'] == 'LW'].copy()
        elif level == '112':
            current_data = all_data[all_data['results'] == 'ME'].copy()
        elif level == '113':
            current_data = all_data[all_data['results'] == 'ER'].copy()
        elif level == '114':
            current_data = all_data[all_data['results'] == 'EE'].copy()
        elif level == '111':
            current_data = all_data[(all_data['results'] == 'MR') & (~all_data['results'].isin(['MR5', 'MR6']))].copy()
        elif level == '1111':
            current_data = all_data[all_data['results'].str.startswith('MR', na=False) & (~all_data['results'].isin(['MR5', 'MR6']))].copy()
        else:
            current_data = all_data.copy()
            
        QgsMessageLog.logMessage(
            f"Level {level}: {len(current_data)} Datensätze nach Filterung ({current_data['results'].unique()})",
            level=Qgis.Info
        )

        if current_data.empty:
            QgsMessageLog.logMessage(f"Keine Daten für {level} nach Filterung.", level=Qgis.Warning)
            return all_data

        # Zielspalte
        target_column = 'sst_sub' if level == '1111' else 'sst'

        try:
            y_true = current_data[target_column]
            if level in ['1', '12', '111']:
                y_true = y_true.apply(lambda x: self.map_y_true(level, x))
        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler bei der Datenvorbereitung für {level}: {str(e)}", level=Qgis.Critical)
            return all_data

        # Building-Age-Verarbeitung
        direct_assignments_count = 0
        if level in ['111', '112', '113', '114', '121', '122']:
            if 'possible_results' not in current_data.columns:
                current_data['possible_results'] = None
            if 'direct_assignment' not in current_data.columns:
                current_data['direct_assignment'] = False

            for index, row in current_data.iterrows():
                if pd.notna(row.get('building_age')) and pd.notna(row.get('results')):
                    building_age = row['building_age']
                    updated_result, direct_assignment = self.process_building_age(row['results'], building_age)
                    if isinstance(updated_result, list):
                        current_data.at[index, 'possible_results'] = updated_result
                    else:
                        current_data.at[index, 'results'] = updated_result
                        current_data.at[index, 'direct_assignment'] = direct_assignment
                        if direct_assignment:
                            direct_assignments_count += 1

            all_data.loc[current_data.index, 'results'] = current_data['results']
            all_data.loc[current_data.index, 'direct_assignment'] = current_data['direct_assignment']

        if level != '11':
            # Filter auf target_names, damit nur relevante Datensätze für die Metrik verwendet werden
            QgsMessageLog.logMessage(
                f"Level {level}: {len(current_data)} Datensätze nach Zielklassen-Filterung (target_names={target_names})",
                level=Qgis.Info
            )
            
            if current_data.empty:
                QgsMessageLog.logMessage(f"Keine Daten für {level} nach Zielklassen-Filterung.", level=Qgis.Warning)
                return all_data

            # Für das Modell: alle Daten verwenden
            X, model = self.prepare_data(current_data, level, is_training=False)
            
            # Check if preparation was successful
            if X.empty or model is None:
                QgsMessageLog.logMessage(f"Datenaufbereitung für {level} fehlgeschlagen.", level=Qgis.Critical)
                return all_data

            # Imputer anwenden
            imputer_path = os.path.join(self.model_dir, f'imputer_{level}.pkl')
            if os.path.exists(imputer_path):
                imputer = joblib.load(imputer_path)
                X = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
            else:
                for col in X.select_dtypes(include=[np.number]).columns:
                    X[col] = X[col].fillna(X[col].mean())
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'missing')

            # Vorhersage
            y_prob = model.predict_proba(X)
            y_pred = model.classes_[np.argmax(y_prob, axis=1)]
            
            if level in ['111', '112', '113', '114', '121', '122', '1111']:
                modified_y_pred = []
                
                for idx, pred in zip(current_data.index, y_pred):
                    # Building age aus Originaldaten oder vorhandenem Ergebnis extrahieren
                    building_age = None
                    
                    # Zuerst prüfen, ob building_age direkt verfügbar ist
                    if pd.notna(current_data.loc[idx, 'building_age']):
                        building_age = str(current_data.loc[idx, 'building_age']).strip()
                        # Wenn building_age mehrere Werte enthält (mit / getrennt), ersten verwenden
                        if '/' in building_age:
                            building_age = building_age.split('/')[0].strip()
                    else:
                        # Alternativ versuchen, aus dem vorherigen Ergebnis zu extrahieren
                        current_result = all_data.loc[idx, 'results']
                        if isinstance(current_result, str) and len(current_result) > 2:
                            # Letzte Ziffer(n) extrahieren (z.B. '3' von 'MR3')
                            suffix = ''
                            for char in reversed(current_result):
                                if char.isdigit():
                                    suffix = char + suffix
                                else:
                                    break
                            if suffix:
                                building_age = suffix
                    
                    if building_age:
                        # Klassenpräfix bestimmen (alles außer der Altersklasse)
                        if level == '1111':
                            # Bei Level 1111 nur MRO oder MRG als Präfix verwenden
                            # Originalpräfix aus Vorhersage extrahieren
                            pred_prefix = pred[:3] if len(pred) > 3 else ''
                            if not (pred_prefix == 'MRO' or pred_prefix == 'MRG'):
                                # Fallback zur besseren Option basierend auf Wahrscheinlichkeiten
                                mro_idx = np.where(model.classes_ == f'MRO{building_age}')[0][0] if f'MRO{building_age}' in model.classes_ else -1
                                mrg_idx = np.where(model.classes_ == f'MRG{building_age}')[0][0] if f'MRG{building_age}' in model.classes_ else -1
                                
                                if mro_idx >= 0 and mrg_idx >= 0:
                                    # Beide Optionen verfügbar, wähle die wahrscheinlichere
                                    probs = y_prob[len(modified_y_pred)]
                                    pred_prefix = 'MRO' if probs[mro_idx] > probs[mrg_idx] else 'MRG'
                                elif mro_idx >= 0:
                                    pred_prefix = 'MRO'
                                elif mrg_idx >= 0:
                                    pred_prefix = 'MRG'
                                else:
                                    # Keine gültige Option, behalte Original-Vorhersage
                                    modified_y_pred.append(pred)
                                    QgsMessageLog.logMessage(
                                        f"Warning: Keine gültige MRO/MRG-Klasse mit building_age {building_age} für {current_result}",
                                        level=Qgis.Warning
                                    )
                                    continue
                        else:
                            # Für andere Level: Extrahiere nur den Buchstabenteil
                            pred_prefix = ''.join([c for c in pred if not c.isdigit()])
                        
                        # Kombiniere Präfix mit dem bekannten Building Age
                        modified_pred = f"{pred_prefix}{building_age}"
                        
                        # Prüfen, ob die modifizierte Klasse gültig ist
                        if modified_pred in target_names or modified_pred in model.classes_:
                            modified_y_pred.append(modified_pred)
                        else:
                            # Fallback: Suche nach einer gültigen Klasse mit demselben Building Age
                            valid_classes = [cls for cls in target_names if cls.endswith(building_age)]
                            if valid_classes:
                                # Wähle die ähnlichste Klasse (beginnt mit dem gleichen Buchstaben)
                                matching_classes = [cls for cls in valid_classes if cls.startswith(pred_prefix[0])]
                                if matching_classes:
                                    modified_y_pred.append(matching_classes[0])
                                else:
                                    modified_y_pred.append(valid_classes[0])
                            else:
                                # Falls keine Alternative mit diesem Building Age existiert
                                modified_y_pred.append(pred)
                                QgsMessageLog.logMessage(
                                    f"Warning: Keine gültige Klasse mit building_age {building_age} für Level {level}",
                                    level=Qgis.Warning
                                )
                    else:
                        # Kein Building Age gefunden, verwende die Original-Vorhersage
                        modified_y_pred.append(pred)
                
                # Ersetze Vorhersagen mit modifizierten, die das Building Age respektieren
                if modified_y_pred:
                    y_pred = np.array(modified_y_pred)

            # Ergebnis in DataFrame schreiben
            all_data.loc[current_data.index, 'results'] = y_pred

            # Sicherstellen, dass die Spalte existiert
            direct_assignment_series = current_data['direct_assignment'] if 'direct_assignment' in current_data.columns else pd.Series(False, index=current_data.index)

            # Nur Zielklassen und keine Baualters-Zuweisungen
            metric_mask = y_true.isin(target_names) & (~direct_assignment_series.fillna(False))

            # Robuste Auswahl von y_true und y_pred für Modellmetriken
            y_true_metric = y_true[metric_mask]
            y_pred_full = pd.Series(y_pred, index=current_data.index)
            y_pred_metric = y_pred_full.loc[metric_mask]
            
            if y_true_metric.empty or y_pred_metric.empty:
                QgsMessageLog.logMessage(
                    f"Keine gültigen Modellvorhersagen für Metrikberechnung in Level {level} (nach Ausschluss direkter Zuweisungen).",
                    level=Qgis.Warning
                )
            else:
                metrics = self.calculate_and_log_metrics(
                    y_true=y_true_metric,
                    y_pred=y_pred_metric,
                    target_names=target_names,
                    level=level,
                    model=model,
                    calculate_curves=True,
                    direct_assignments_count=direct_assignments_count if 'direct_assignment' in current_data.columns else 0
                )
                results.append(metrics)

            self.save_results_to_db(all_data, y_pred, current_data.index)

            return all_data

        else:
            valid_results_mask = current_data['results'].isin(['M', 'E'])
            current_data = current_data[valid_results_mask]

            y_pred = current_data.apply(
                lambda row: 'MR' if row['results'] == 'M' and row['proximity'] == 'R'
                else ('ME' if row['results'] == 'M'
                else ('ER' if row['results'] == 'E' and row['proximity'] == 'R'
                else ('EE' if row['results'] == 'E' else row['results']))),
                axis=1
            )

            current_data['results'] = y_pred
            all_data.loc[current_data.index, 'results'] = y_pred
            
            remaining_M_E = all_data[all_data['results'].isin(['M', 'E'])]
            if not remaining_M_E.empty:
                QgsMessageLog.logMessage(
                    f"Nach Level 11 sind noch {len(remaining_M_E)} Datensätze mit 'M' oder 'E' übrig: {remaining_M_E.index.tolist()}",
                    level=Qgis.Warning
                )
        
            return all_data

    def validate(self):
        """
        Führt die Validierung über alle Klassifikationslevels durch und speichert die Ergebnisse.
        """
        self.clear_results_json()

        validation_data = self.load_data_from_db('"MPSCDresden".validation_data')
        if validation_data.empty:
            QgsMessageLog.logMessage("Keine Validierungsdaten vorhanden.", level=Qgis.Critical)
            return

        all_data = validation_data.copy()
        all_data['results'] = None
        all_data['direct_assignment'] = False
        results = []
        overall_results = {}

        levels = self.get_levels()

        max_passes = 10  # Sicherheitslimit
        for i in range(max_passes):
            QgsMessageLog.logMessage(f"Klassifikationsdurchlauf {i+1} gestartet.", level=Qgis.Info)

            previous_results = all_data['results'].copy()

            for level, target_names in levels:
                all_data = self.process_level(all_data, level, target_names, results)

            changed = not all_data['results'].equals(previous_results)
            unfinished = ~all_data['results'].isin(self.final_classes)

            if not unfinished.any():
                QgsMessageLog.logMessage("Alle Datensätze wurden vollständig klassifiziert.", level=Qgis.Info)
                break

            if not changed:
                QgsMessageLog.logMessage("Keine Änderungen mehr – Klassifikation endet, obwohl nicht alles abgeschlossen ist.", level=Qgis.Warning)
                break

        # Auswertung
        self.calculate_end_to_end_metrics(all_data, overall_results)
        self.save_results_to_json(results, overall_results)

        all_data['results'] = all_data['results'].astype(str)
        
        QgsMessageLog.logMessage(
            f"Alle Werte in results (inkl. NaN/None): {all_data['results'].value_counts(dropna=False).to_dict()}",
            level=Qgis.Info
        )
        
        unclassified = all_data[~all_data['results'].isin(self.final_classes)]
        
        if not unclassified.empty:
            QgsMessageLog.logMessage(
                f"{len(unclassified)} Datensätze wurden nicht vollständig klassifiziert. Übersicht: {unclassified['results'].value_counts().to_dict()}",
                level=Qgis.Warning
            )
            
        m_e = unclassified[unclassified['results'].isin(['M', 'E'])]
        if not m_e.empty:
            QgsMessageLog.logMessage(
                f"{len(m_e)} Datensätze haben als Ergebnis 'M' oder 'E' und wurden nicht weiter klassifiziert.",
                level=Qgis.Warning
            )
        QgsMessageLog.logMessage(
            f"Nicht-finale Klassen in results: {set(unclassified['results'].unique())}",
            level=Qgis.Info
        )

        self.save_results_to_db(all_data, all_data['results'], all_data.index)
        self.launch_dashboard()

    def clear_results_json(self):
        """
        Leert die JSON-Datei für die Validierungsergebnisse.
        """
        results_json_path = os.path.join(self.vis_path, 'validation_results.json')
        with open(results_json_path, 'w') as json_file:
            json.dump({"levels": [], "overall_results": {}}, json_file)
        QgsMessageLog.logMessage("JSON-Datei geleert.", level=Qgis.Info)

    def get_levels(self):
        """
        Gibt die Hierarchie der Klassifikationslevels und Zielklassen zurück.
        """
        levels = [
            ('1', ['M', 'E', 'Other']),
            ('11', ['MR', 'ME', 'ER', 'EE']),
            ('12', ['HH', 'LW']),
            ('121', ['HH3', 'HH4', 'HH7']),
            ('122', ['LW1', 'LW2', 'LW3', 'LW7']),
            ('112', ['ME2', 'ME3', 'ME4', 'ME5', 'ME6', 'ME7']),
            ('113', ['ER2', 'ER3', 'ER4', 'ER5', 'ER7']),
            ('114', ['EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE7']),
            ('111', ['MR2', 'MR3', 'MR4', 'MR5', 'MR6', 'MR7']),
            ('1111', ['MRO2', 'MRO3', 'MRO4', 'MRO7', 'MRG2', 'MRG3', 'MRG4', 'MRG7'])
        ]
        
        final_levels = ['121', '122', '112', '113', '114', '1111']
        self.final_classes = [cls for lvl, names in levels if lvl in final_levels for cls in names]
        self.final_classes += ['MR5', 'MR6']
        return levels

    def calculate_end_to_end_metrics(self, all_data, overall_results):
        """
        Berechnet die End-to-End-Metriken für die gesamte Klassifikation.
        """
        end_to_end_y_true = all_data['sst']
        end_to_end_y_pred = all_data['results']

        valid_mask = end_to_end_y_true.notnull() & end_to_end_y_pred.notnull()
        end_to_end_y_true = end_to_end_y_true[valid_mask]
        end_to_end_y_pred = end_to_end_y_pred[valid_mask]

        direct_assignment_count = all_data['direct_assignment'].sum() if 'direct_assignment' in all_data.columns else 0
        correct_by_model = (end_to_end_y_true == end_to_end_y_pred).sum()
        correct_count = correct_by_model + direct_assignment_count
        total_count = len(end_to_end_y_true)

        end_to_end_accuracy = correct_count / total_count if total_count > 0 else 0       
        
        total_count = len(end_to_end_y_true)

        end_to_end_accuracy = correct_count / total_count if total_count > 0 else 0      

        direct_assignments_count = all_data['direct_assignment'].sum()

        if not end_to_end_y_true.empty and not end_to_end_y_pred.empty:
            metrics = self.calculate_and_log_metrics(
                y_true=end_to_end_y_true,
                y_pred=end_to_end_y_pred,
                target_names=None,
                level="end_to_end",
                model=None,
                calculate_curves=False,
                direct_assignments_count=direct_assignments_count
            )
            metrics['correct_by_model'] = correct_by_model
            metrics['direct_assignment_count'] = int(direct_assignment_count)
            metrics['correct_total'] = correct_count

            overall_results.update(metrics)
        else:
            QgsMessageLog.logMessage("Keine gültigen Werte für End-to-End-Metriken verfügbar.", level=Qgis.Warning)

    def save_results_to_json(self, results, overall_results):
        """
        Speichert die Ergebnisse in der JSON-Datei.
        """
        results_json_path = os.path.join(self.vis_path, 'validation_results.json')
        existing_results = {"levels": results, "overall_results": overall_results}

        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return str(obj)  # Fallback für andere Typen

        with open(results_json_path, 'w') as json_file:
            json.dump(existing_results, json_file, indent=4, default=convert_numpy)

        QgsMessageLog.logMessage("Ergebnisse in JSON-Datei gespeichert.", level=Qgis.Info)

    def launch_dashboard(self):
        """
        Startet das Dashboard zur Visualisierung der Ergebnisse.
        """
        subprocess.Popen(['python', 'dashboard.py'])
        webbrowser.open('http://127.0.0.1:8050')
        QgsMessageLog.logMessage("Dashboard gestartet.", level=Qgis.Info)
        
    def load_and_visualize_validation_data(self):
        """
        Lädt die Validierungsdaten als QGIS-Layer und färbt sie nach Kategorie ein.
        """
        uri = QgsDataSourceUri()
        uri.setConnection(self.connection_params['host'], str(self.connection_params['port']), self.connection_params['dbname'], self.connection_params['user'], self.connection_params['password'])
        uri.setDataSource('MPSCDresden', 'validation_data', 'geom', '', 'validation_id')
        
        layer_name = 'Validation Data'
        
        existing_layer = QgsProject.instance().mapLayersByName(layer_name)
        if (existing_layer):
            QgsProject.instance().removeMapLayer(existing_layer[0])

        layer = QgsVectorLayer(uri.uri(False), layer_name, 'postgres')
        if not layer.isValid():
            QgsMessageLog.logMessage("Layer Validation Data is not valid", level=Qgis.Critical)
            return
        
        QgsProject.instance().addMapLayer(layer)
        
        self.mapping_processor.categorize_and_colorize(layer)