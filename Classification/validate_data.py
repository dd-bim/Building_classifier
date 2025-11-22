import os
import json
import numpy as np
import pandas as pd
import joblib
import configparser
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef
)
from qgis.core import QgsMessageLog, Qgis, QgsDataSourceUri, QgsVectorLayer, QgsProject
import sys
import subprocess
import webbrowser

from .model_trainer import LabelEncoderManager

class ValidateData:
    """
    Neue, klar strukturierte Validierungspipeline:
    - sequentielle Level-Validierung (Filterung anhand vorheriger Ergebnisse)
    - building_age als Feature, Constraint und direkte Zuweisung in Endlevels
    - Level 11 und 1111 sind deterministisch und gelten als korrekt
    - pro-Level Metriken + End-to-End Metriken
    - Dashboard-kompatible JSON-Ausgabe
    """

    # Features identisch zur Trainings-Pipeline
    CORE_FEATURES = [
        'roof_type', 'storeys_above_ground', 'building_footprint',
        'roof_ridge_height', 'eaves_height', 'storey_height',
        'number_roof_surfaces', 'roof_slope', 'development_type_code',
        'building_age'
    ]
    SIMPLE_GEOM_FEATURES = ['length_footprint', 'width_footprint', 'building_volume']
    ADV_GEOM_FEATURES = ['compactness', 'convexity', 'rectangularity']
    NEIGH_FEATURES = ['neighbour_density', 'neighbour_avg_size', 'neighbour_min_distance', 'neighbour_majority_class']
    RATIO_FEATURES = ['ground_area_per_storey', 'height_to_area_ratio', 'footprint_ratio', 'roof_height_ratio']

    CATEGORICAL = ['roof_type', 'development_type_code', 'neighbour_majority_class', 'building_age']

    # Leveldefinitionen (Trainings-/Modell-Levels, Zielklassen)
    LEVELS: List[Tuple[str, List[str]]] = [
        ('1',   ['M', 'E', 'Other']),
        ('11',  ['MR', 'ME', 'ER', 'EE']),               # rule-based
        ('12',  ['HH', 'LW']),
        ('121', ['HH3', 'HH4']),                         # Endlevel, direkt wenn building_age vorhanden
        ('122', ['LW1', 'LW2', 'LW3', 'LW7']),           # Endlevel, direkt wenn building_age vorhanden
        ('112', ['ME2', 'ME3', 'ME4', 'ME5', 'ME6', 'ME7']),  # Endlevel, direkt wenn building_age vorhanden
        ('113', ['ER2', 'ER3', 'ER4', 'ER5', 'ER7']),         # Endlevel, direkt wenn building_age vorhanden
        ('114', ['EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE7']),  # Endlevel, direkt wenn building_age vorhanden
        ('111', ['MR2', 'MR3', 'MR4', 'MR5', 'MR6', 'MR7']),  # Endlevel, direkt wenn building_age vorhanden
        ('1111',['MRO2', 'MRO3', 'MRO4', 'MRO7', 'MRG2', 'MRG3', 'MRG4', 'MRG7'])  # rule-based
    ]

    # Endlevels mit direkter Baualterszuweisung (kein Modell bei vorhandenem building_age)
    # Hinweis: Direkte Zuweisung nur für atomare Ages (z.B. "3"); zusammengesetzte Ages ("1/2","5/6") dienen nur als Constraints.
    ENDLEVELS_DIRECT_AGE = {'111', '112', '113', '114', '121', '122'}
    # NEU: Levels ohne Modelle, für die keine Metriken berechnet oder exportiert werden
    SKIP_METRIC_LEVELS = {'11', '1111'}

    def __init__(self, conn, cur, connection_params):
        self.conn = conn
        self.cur = cur
        self.connection_params = connection_params

        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self.model_dir = os.path.join(os.path.dirname(__file__), config.get('Paths', 'model_dir'))
        self.vis_path = os.path.join(os.path.dirname(__file__), config.get('Paths', 'vis_path'))

        self.label_encoders = LabelEncoderManager(self.model_dir).get_label_encoders()

        # Zähler für End-to-End / Summen
        self.reset_counters()

        # final_classes für Vollständigkeitsprüfung
        self.final_classes = set(['MR5', 'MR6']) | set(sum([names for lvl, names in self.LEVELS if lvl in {'121','122','112','113','114','1111'}], []))

        # Neu: baue dynamische Suffix-Mappings je Präfix (HH/LW/MR/ME/ER/EE)
        self._build_level_maps()

    def _build_level_maps(self):
        """
        Baut Hilfsstrukturen:
        - self.level_targets: Map von Level -> Klassenliste
        - self.suffix_by_prefix: Map von Präfix -> erlaubte Endstufen-Suffixe (Ziffern) aus verfügbaren Klassen
        """
        self.level_targets: Dict[str, List[str]] = {lvl: names for (lvl, names) in self.LEVELS}
        mapping = {'HH': '121', 'LW': '122', 'MR': '111', 'ME': '112', 'ER': '113', 'EE': '114'}
        self.suffix_by_prefix: Dict[str, set] = {}
        for prefix, lvl in mapping.items():
            names = self.level_targets.get(lvl, [])
            suffixes = set()
            for n in names:
                if n.startswith(prefix):
                    suffixes.add(n[len(prefix):])  # z.B. 'MR5' -> '5'
            self.suffix_by_prefix[prefix] = suffixes
        QgsMessageLog.logMessage(f"Suffix mapping: { {k: sorted(list(v)) for k,v in self.suffix_by_prefix.items()} }", level=Qgis.Info)

    def reset_counters(self):
        self.total_TP = 0
        self.total_FP = 0
        self.total_TN = 0
        self.total_FN = 0
        self.total_direct_assignments = 0  # Nur Baualters-Zuweisungen

    # -----------------------------
    # Datenzugriff + Utilities
    # -----------------------------
    def load_validation_data(self) -> pd.DataFrame:
        query = 'SELECT * FROM "MPSCDresden".validation_data'
        self.cur.execute(query)
        rows = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        return pd.DataFrame(rows, columns=colnames)

    def save_results_to_db(self, all_data: pd.DataFrame):
        # Persistiere finale results in validation_data.results
        update_data = list(zip(all_data['results'].astype(str), all_data['db_filter_id']))
        update_query = '''
            UPDATE "MPSCDresden".validation_data
            SET "results" = %s
            WHERE db_filter_id = %s
        '''
        self.cur.executemany(update_query, update_data)
        self.conn.commit()
        QgsMessageLog.logMessage("Validation results saved to DB.", level=Qgis.Info)

    def safe_label_transform(self, col: pd.Series, feature: str) -> np.ndarray:
        enc = self.label_encoders.get(feature)
        if enc is None:
            return col.astype('category').cat.codes.values

        # normalize values to str and handle NAs
        vals = col.apply(lambda x: str(x) if pd.notna(x) else ("unknown" if feature == 'building_age' else "None")).astype(str)

        classes = set(getattr(enc, 'classes_', []))
        # fallback mapping
        fallback = 'unknown' if 'unknown' in classes else (next(iter(classes)) if classes else None)

        def map_val(v: str) -> str:
            return v if v in classes else fallback

        mapped = vals.map(map_val)
        # If encoder has not been fit, fit on current values
        if not hasattr(enc, 'classes_') or len(enc.classes_) == 0:
            enc.fit(mapped.values)
        # transform
        return enc.transform(mapped.values)

    def prepare_features_for_level(self, df: pd.DataFrame, level: str) -> Tuple[pd.DataFrame, Optional[object], List[str]]:
        # expected features
        expected = self.CORE_FEATURES + self.SIMPLE_GEOM_FEATURES + self.ADV_GEOM_FEATURES + self.NEIGH_FEATURES + self.RATIO_FEATURES

        # load model
        model_path = os.path.join(self.model_dir, f'model_{level}.pkl')
        if not os.path.exists(model_path):
            QgsMessageLog.logMessage(f"No model found for level {level}: {model_path}", level=Qgis.Warning)
            return pd.DataFrame(), None, []

        model = joblib.load(model_path)

        # feature filtering
        missing = [f for f in expected if f not in df.columns]
        if missing:
            QgsMessageLog.logMessage(f"Missing features for level {level}: {missing}", level=Qgis.Critical)
            return pd.DataFrame(), None, []

        X = df[expected].copy()

        # encode categoricals
        for c in self.CATEGORICAL:
            X[c] = self.safe_label_transform(X[c], c)

        # simple imputation
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        # no string cols should remain after encoding; guard anyway
        for col in X.select_dtypes(include=['object', 'category']).columns:
            mode = X[col].mode().iloc[0] if not X[col].mode().empty else 0
            X[col] = X[col].fillna(mode)

        return X, model, expected

    # -----------------------------
    # building_age Constraints (generalized across levels)
    # -----------------------------
    def parse_building_age(self, value) -> List[str]:
        if pd.isna(value):
            return []
        s = str(value).strip()
        if not s:
            return []
        return [p.strip() for p in s.split('/')] if '/' in s else [s]

    def allowed_groups_level12(self, ages: List[str]) -> List[str]:
        """
        Erlaubte Gruppen (HH/LW) in Level 12 dynamisch bestimmen:
        - HH erlaubt, wenn es in Level 121 eine Klasse HH{age} gibt
        - LW erlaubt, wenn es in Level 122 eine Klasse LW{age} gibt
        """
        if not ages:
            return ['HH', 'LW']
        allowed = []
        hh_suffixes = self.suffix_by_prefix.get('HH', set())
        lw_suffixes = self.suffix_by_prefix.get('LW', set())

        if any(a in hh_suffixes for a in ages):
            allowed.append('HH')
        if any(a in lw_suffixes for a in ages):
            allowed.append('LW')

        return allowed

    def allowed_groups_level11(self, ages: List[str]) -> List[str]:
        """
        Erlaubte Gruppen in Level 11 (MR/ME/ER/EE) auf Basis verfügbarer Endlevel-Suffixe.
        Zulässig, wenn es in 111/112/113/114 eine Klasse mit passender Age gibt.
        """
        if not ages:
            return ['MR', 'ME', 'ER', 'EE']

        allowed = []
        for prefix in ['MR', 'ME', 'ER', 'EE']:
            suffixes = self.suffix_by_prefix.get(prefix, set())
            if any(a in suffixes for a in ages):
                allowed.append(prefix)

        return allowed

    def allowed_top_level_for_ages(self, ages: List[str]) -> List[str]:
        """
        Erlaubte Top-Level Optionen in Level 1 (M/E/Other) auf Basis der Downstream-Möglichkeiten:
        - M erlaubt, wenn MR oder ME später mit passender Age möglich
        - E erlaubt, wenn ER oder EE später mit passender Age möglich
        - Other erlaubt, wenn HH oder LW später mit passender Age möglich
        """
        if not ages:
            return ['M', 'E', 'Other']

        allowed_lvl11 = set(self.allowed_groups_level11(ages))
        allowed_lvl12 = set(self.allowed_groups_level12(ages))

        allowed = []
        if {'MR', 'ME'} & allowed_lvl11:
            allowed.append('M')
        if {'ER', 'EE'} & allowed_lvl11:
            allowed.append('E')
        if allowed_lvl12:
            allowed.append('Other')
        return allowed

    def direct_assign_endlevel(self, level: str, ages: List[str], target_names: List[str]) -> Optional[str]:
        """
        Direkte Zuweisung nur bei atomarem Alter (ein Wert). Zusammengesetzte Ages (z.B. 1/2, 5/6)
        dienen ausschließlich als Constraints.
        """
        if not ages or len(ages) != 1:
            return None
        age = ages[0]
        prefix = {'111': 'MR', '112': 'ME', '113': 'ER', '114': 'EE', '121': 'HH', '122': 'LW'}.get(level)
        if not prefix:
            return None
        candidate = f"{prefix}{age}"
        return candidate if candidate in target_names else None

    def restrict_with_constraints(self, level: str, row: pd.Series, target_names: List[str], model_classes: np.ndarray, probs: np.ndarray, pred_label: str) -> Optional[str]:
        """
        Erzwingt building_age Constraints über alle Levels:
        - Level 1: nur M/E/Other erlauben, die downstream Endklassen für Age besitzen
        - Level 12: nur HH/LW erlauben, die Endklassen für Age besitzen
        - Endlevel (111/112/113/114/121/122): nur Klassen mit passender Altersziffer
        Gibt None zurück, wenn keine gültige Klasse möglich ist (Pfad wird blockiert).
        """
        ages = self.parse_building_age(row.get('building_age'))

        # Level 1: M/E/Other nur erlauben, wenn der Pfad downstream möglich ist
        if level == '1':
            allowed_top = self.allowed_top_level_for_ages(ages)
            if not allowed_top:
                return None
            if pred_label in allowed_top:
                return pred_label
            # Wähle beste erlaubte Top-Level-Option nach Wahrscheinlichkeit
            allowed_idx = [np.where(model_classes == cls)[0][0] for cls in allowed_top if cls in model_classes]
            return model_classes[allowed_idx[np.argmax(probs[allowed_idx])]] if allowed_idx else None

        # Level 12: HH/LW nur erlauben, wenn spätere Subklasse existiert
        if level == '12':
            allowed_groups = self.allowed_groups_level12(ages)
            if not allowed_groups:
                return None
            if pred_label in allowed_groups:
                return pred_label
            if len(allowed_groups) == 1:
                return allowed_groups[0]
            allowed_idx = [np.where(model_classes == cls)[0][0] for cls in allowed_groups if cls in model_classes]
            return model_classes[allowed_idx[np.argmax(probs[allowed_idx])]] if allowed_idx else None

        # Endlevel-Constraints: nur Klassen mit passender Altersziffer zulassen
        endlevel_prefix_by_level = {
            '111': 'MR', '112': 'ME', '113': 'ER', '114': 'EE', '121': 'HH', '122': 'LW'
        }
        if level in endlevel_prefix_by_level and ages:
            prefix = endlevel_prefix_by_level[level]
            allowed_classes = [c for c in target_names if any(c == f"{prefix}{a}" for a in ages)]
            if not allowed_classes:
                return None
            if pred_label in allowed_classes:
                return pred_label
            allowed_idx = [np.where(model_classes == cls)[0][0] for cls in allowed_classes if cls in model_classes]
            return model_classes[allowed_idx[np.argmax(probs[allowed_idx])]] if allowed_idx else None

        # Default
        return pred_label

    # -----------------------------
    # Ground truth mapping per level
    # -----------------------------
    def map_truth_for_level(self, level: str, sst: Optional[str], sst_sub: Optional[str]) -> Optional[str]:
        if pd.isna(sst) and pd.isna(sst_sub):
            return None
        s = str(sst) if pd.notna(sst) else ''
        sub = str(sst_sub) if pd.notna(sst_sub) else ''

        if level == '1':
            if s.startswith('M'): return 'M'
            if s.startswith('E'): return 'E'
            return 'Other'
        if level == '11':
            if s.startswith('MR'): return 'MR'
            if s.startswith('ME'): return 'ME'
            if s.startswith('ER'): return 'ER'
            if s.startswith('EE'): return 'EE'
            return None
        if level == '12':
            if s.startswith('HH'): return 'HH'
            if s.startswith('LW'): return 'LW'
            return None
        if level == '121':
            if s.startswith('HH3'): return 'HH3'
            if s.startswith('HH4'): return 'HH4'
            return None
        if level == '122':
            for k in ['LW1','LW2','LW3','LW7']:
                if s.startswith(k): return k
            return None
        if level == '112':
            for k in ['ME2','ME3','ME4','ME5','ME6','ME7']:
                if s.startswith(k): return k
            return None
        if level == '113':
            for k in ['ER2','ER3','ER4','ER5','ER7']:
                if s.startswith(k): return k
            return None
        if level == '114':
            for k in ['EE1','EE2','EE3','EE4','EE5','EE7']:
                if s.startswith(k): return k
            return None
        if level == '111':
            for k in ['MR2','MR3','MR4','MR5','MR6','MR7']:
                if s.startswith(k): return k
            return None
        if level == '1111':
            # truth for sub-classes from sst_sub if available (MRO/MRG per digit)
            if sub.startswith('MRO') or sub.startswith('MRG'):
                return sub
            # construct from MR? + neighbourhood? Fallback to None if unknown
            return None
        return None

    # -----------------------------
    # Deterministic levels (rules)
    # -----------------------------
    def predict_level_11(self, df: pd.DataFrame) -> pd.Series:
        """
        Rule-based Zuweisung MR/ME/ER/EE mit Age-Constraints:
        - Bevorzugt MR/ER bei proximity 'R'
        - Fällt auf ME/EE zurück, wenn MR/ER für die gegebene Age nicht möglich
        - Wenn keine Option möglich, bleibt vorheriges Ergebnis (M/E) bestehen (Pfad blockiert)
        """
        def rule(row):
            res = row.get('results')
            prox = row.get('proximity')
            ages = self.parse_building_age(row.get('building_age'))
            allowed = set(self.allowed_groups_level11(ages))

            if res == 'M':
                # bevorzugt MR bei R, sonst ME – aber nur wenn erlaubt
                if prox == 'R' and 'MR' in allowed:
                    return 'MR'
                if 'ME' in allowed:
                    return 'ME'
                if 'MR' in allowed:
                    return 'MR'
                return res  # keine zulässige MR/ME-Option -> blockiere
            if res == 'E':
                if prox == 'R' and 'ER' in allowed:
                    return 'ER'
                if 'EE' in allowed:
                    return 'EE'
                if 'ER' in allowed:
                    return 'ER'
                return res  # keine zulässige ER/EE-Option -> blockiere
            return res
        return df.apply(rule, axis=1)

    def predict_level_1111(self, df: pd.DataFrame) -> pd.Series:
        # Split MR? into MRO?/MRG? by neighbouring_buildings; only for MR2/3/4/7
        def rule(row):
            res = row.get('results')
            if not isinstance(res, str) or not res.startswith('MR'):
                return res
            if res in ['MR5','MR6']:  # no split
                return res
            age_digit = res[2:]
            nb = row.get('neighbouring_buildings')
            if pd.isna(nb):
                return res
            return f"{'MRO' if nb < 2 else 'MRG'}{age_digit}"
        return df.apply(rule, axis=1)

    # -----------------------------
    # Metrics
    # -----------------------------
    def compute_level_metrics(self, y_true: List[str], y_pred: List[str], target_names: List[str], level: str, direct_assignment_count: int = 0) -> Dict:
        if len(y_true) == 0:
            # empty metrics scaffold
            return {
                'level': level,
                'accuracy': 0.0, 'f1_weighted': 0.0, 'f1_macro': 0.0,
                'precision': 0.0, 'recall': 0.0, 'mcc': 0.0,
                'conf_matrix': [[0 for _ in target_names] for _ in target_names],
                'precision_recall_curve': {'precision': [], 'recall': []},
                'roc_curve': {'fpr': [], 'tpr': [], 'roc_auc': 0.0},
                'class_names': target_names,
                'y_true': [], 'y_pred': [],
                'direct_assignment_count': int(direct_assignment_count)
            }

        report = classification_report(y_true, y_pred, labels=target_names, target_names=target_names, output_dict=True, zero_division=0)
        conf = confusion_matrix(y_true, y_pred, labels=target_names)
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 and len(set(y_pred)) > 1 else 0.0

        return {
            'level': level,
            'accuracy': float(acc),
            'f1_weighted': float(report['weighted avg']['f1-score']),
            'f1_macro': float(report['macro avg']['f1-score']),
            'precision': float(report['weighted avg']['precision']),
            'recall': float(report['weighted avg']['recall']),
            'mcc': float(mcc),
            'conf_matrix': conf.tolist(),
            'precision_recall_curve': {'precision': [], 'recall': []},  # kept empty (dashboard handles)
            'roc_curve': {'fpr': [], 'tpr': [], 'roc_auc': 0.0},
            'class_names': target_names,
            'y_true': list(y_true),
            'y_pred': list(y_pred),
            'direct_assignment_count': int(direct_assignment_count)
        }

    # -----------------------------
    # Hauptablauf
    # -----------------------------
    def validate(self) -> Dict[str, Dict]:
        # load data
        all_data = self.load_validation_data()
        if all_data.empty:
            QgsMessageLog.logMessage("No validation data found.", level=Qgis.Critical)
            return {}

        # reset state
        self.reset_counters()
        all_data['results'] = None
        all_data['source'] = None  # 'model' | 'age_direct' | 'rule'

        # NEW: stage snapshots for transparency
        for lvl in ['1','11','12','121','122','112','113','114','111','1111']:
            all_data[f'pred_{lvl}'] = None

        per_level_results: List[Dict] = []

        for level, target_names in self.LEVELS:
            # Filter rows for current level based on previous results
            if level == '1':
                current_idx = all_data.index
            elif level == '11':
                current_idx = all_data.index  # based on level 1 result inside rule
            elif level == '12':
                current_idx = all_data.index[all_data['results'] == 'Other']
            elif level == '121':
                current_idx = all_data.index[all_data['results'] == 'HH']
            elif level == '122':
                current_idx = all_data.index[all_data['results'] == 'LW']
            elif level == '112':
                current_idx = all_data.index[all_data['results'] == 'ME']
            elif level == '113':
                current_idx = all_data.index[all_data['results'] == 'ER']
            elif level == '114':
                current_idx = all_data.index[all_data['results'] == 'EE']
            elif level == '111':
                current_idx = all_data.index[all_data['results'] == 'MR']
            elif level == '1111':
                mask = all_data['results'].astype(str).str.startswith('MR', na=False) & (~all_data['results'].isin(['MR5','MR6']))
                current_idx = all_data.index[mask]
            else:
                current_idx = all_data.index

            if len(current_idx) == 0:
                # Für 11 und 1111 keine leeren Metriken erzeugen
                if level not in self.SKIP_METRIC_LEVELS:
                    per_level_results.append(self.compute_level_metrics([], [], target_names, level))
                continue

            current = all_data.loc[current_idx].copy()

            # y_true mapping für Metriken
            y_true = [
                self.map_truth_for_level(level, row.get('sst'), row.get('sst_sub'))
                for _, row in current.iterrows()
            ]
            truth_mask = [yt in target_names for yt in y_true]
            truth_idx = current_idx[truth_mask]
            y_true_use = [yt for (yt, keep) in zip(y_true, truth_mask) if keep]

            # deterministisches Level 11 (keine Metriken/kein Export in Dashboard)
            if level == '11':
                y_pred_series = self.predict_level_11(current)
                all_data.loc[current_idx, 'results'] = y_pred_series.values
                all_data.loc[current_idx, 'source'] = 'rule'
                # NEW: snapshot
                all_data.loc[current_idx, 'pred_11'] = y_pred_series.values
                # Keine per_level_results.append(...) für Level 11
                continue

            # deterministisches Level 1111 (keine Metriken/kein Export in Dashboard)
            if level == '1111':
                y_pred_series = self.predict_level_1111(current)
                all_data.loc[current_idx, 'results'] = y_pred_series.values
                all_data.loc[current_idx, 'source'] = 'rule'
                # NEW: snapshot
                all_data.loc[current_idx, 'pred_1111'] = y_pred_series.values
                # Keine per_level_results.append(...) für Level 1111
                continue

            # handle endlevels direct assignment if building_age present
            level_direct_assignments = 0
            direct_assign_mask = pd.Series(False, index=current_idx)
            if level in self.ENDLEVELS_DIRECT_AGE:
                assigned = []
                for idx, row in current.iterrows():
                    ages = self.parse_building_age(row.get('building_age'))
                    cls = self.direct_assign_endlevel(level, ages, target_names)
                    assigned.append(cls)
                assigned_series = pd.Series(assigned, index=current_idx)

                assignable = assigned_series.notna()
                if assignable.any():
                    all_data.loc[current_idx[assignable], 'results'] = assigned_series[assignable].values
                    all_data.loc[current_idx[assignable], 'source'] = 'age_direct'
                    # NEW: snapshot
                    all_data.loc[current_idx[assignable], f'pred_{level}'] = assigned_series[assignable].values
                    direct_assign_mask.loc[current_idx[assignable]] = True
                    level_direct_assignments = int(assignable.sum())
                    self.total_direct_assignments += level_direct_assignments

                model_idx = current_idx[~direct_assign_mask.loc[current_idx]]
            else:
                model_idx = current_idx

            # run model if any rows remain and model exists
            y_pred_model = pd.Series(index=model_idx, dtype=object)
            if len(model_idx) > 0:
                X, model, _ = self.prepare_features_for_level(all_data.loc[model_idx], level)
                if model is not None and not X.empty:
                    prob = model.predict_proba(X)
                    base_pred = model.classes_[np.argmax(prob, axis=1)]
                    constrained = []
                    # Wende Constraints an; wenn None zurückkommt, belasse vorheriges Ergebnis (Pfad blockiert)
                    for i, idx in enumerate(model_idx):
                        new_label = self.restrict_with_constraints(
                            level,
                            all_data.loc[idx],
                            target_names,
                            model.classes_,
                            prob[i],
                            base_pred[i]
                        )
                        if new_label is None:
                            # blockiert -> Vorzustand behalten
                            new_label = all_data.loc[idx, 'results']
                        constrained.append(new_label)
                    y_pred_model = pd.Series(constrained, index=model_idx)
                    all_data.loc[model_idx, 'results'] = y_pred_model.values
                    all_data.loc[model_idx, 'source'] = 'model'
                    # NEW: snapshot
                    all_data.loc[model_idx, f'pred_{level}'] = y_pred_model.values
                else:
                    # no model -> keep previous results (should not happen for model levels)
                    pass

            # NEW: also snapshot non-end deterministic/path levels:
            # Level 1 and 12 are model-based; snapshots already set above for model_idx.
            # If any rows for level without model (unlikely), fill snapshot from current results to avoid None.
            if len(current_idx) > 0 and all_data.loc[current_idx, f'pred_{level}'].isna().any():
                all_data.loc[current_idx, f'pred_{level}'] = all_data.loc[current_idx, 'results']

            # collect predictions for metrics (include both model and direct age to reflect correctness)
            y_pred_full = all_data.loc[truth_idx, 'results'].astype(str).tolist()
            per_level_results.append(self.compute_level_metrics(y_true_use, y_pred_full, target_names, level, direct_assignment_count=level_direct_assignments))

        # finalize DB + overall metrics
        self.save_results_to_db(all_data)

        overall_results = self.compute_overall_metrics(all_data)
        self.save_results_to_json(per_level_results, overall_results)

        # Optional: visualize validation layer in QGIS
        self.load_and_visualize_validation_data()

        # Neu: Dashboard automatisch starten
        self.launch_dashboard()

        return {lvl['level']: lvl for lvl in per_level_results}

    def compute_overall_metrics(self, all_data: pd.DataFrame) -> Dict:
        # End-to-end correctness: final results vs sst
        valid_truth = all_data['sst'].notna()
        y_true = all_data.loc[valid_truth, 'sst'].astype(str)
        y_pred = all_data.loc[valid_truth, 'results'].astype(str)

        correct_mask = (y_true == y_pred)
        correct_total = int(correct_mask.sum())

        # count "correct_by_model" and "direct_assignment_count" on final results
        direct_assignment_count = int((all_data['source'] == 'age_direct').sum())
        correct_by_model = int(((all_data['source'] == 'model') & correct_mask.reindex(all_data.index, fill_value=False)).sum())

        total_count = int(valid_truth.sum())
        end_to_end_accuracy = (correct_total / total_count) if total_count > 0 else 0.0

        # NEW: derive truth per stage to build funnel and attribute errors
        def truth_map(level: str, row) -> Optional[str]:
            return self.map_truth_for_level(level, row.get('sst'), row.get('sst_sub'))

        stage_truth = all_data[valid_truth].copy()
        stage_truth['truth_1'] = stage_truth.apply(lambda r: truth_map('1', r), axis=1)
        stage_truth['truth_11'] = stage_truth.apply(lambda r: truth_map('11', r), axis=1)
        stage_truth['truth_12'] = stage_truth.apply(lambda r: truth_map('12', r), axis=1)

        # predictions per stage (snapshots)
        pred_1 = all_data.loc[valid_truth, 'pred_1'].astype(object)
        pred_11 = all_data.loc[valid_truth, 'pred_11'].astype(object)
        pred_12 = all_data.loc[valid_truth, 'pred_12'].astype(object)

        # masks
        m_total = stage_truth.index
        m_after_1 = (pred_1.values == stage_truth['truth_1'].values)
        # split path by truth
        is_other = (stage_truth['truth_1'] == 'Other').values
        branch_correct_other = (pred_12.values == stage_truth['truth_12'].values)
        branch_correct_me = (pred_11.values == stage_truth['truth_11'].values)

        m_after_branch = np.zeros_like(m_after_1, dtype=bool)
        m_after_branch[is_other] = m_after_1[is_other] & branch_correct_other[is_other]
        m_after_branch[~is_other] = m_after_1[~is_other] & branch_correct_me[~is_other]

        final_correct = (all_data.loc[valid_truth, 'results'].astype(str).values == stage_truth['sst'].astype(str).values)
        m_final_correct = m_after_branch & final_correct

        # NEW: integrity check – funnel final vs overall correct
        funnel_final_calc = int(m_final_correct.sum())
        if funnel_final_calc != correct_total:
            QgsMessageLog.logMessage(
                f"Funnel/overall mismatch: funnel_final={funnel_final_calc} vs correct_total={correct_total}",
                level=Qgis.Warning
            )

        # stage error attribution
        l1_errors = int((~m_after_1).sum())
        branch_errors = int((m_after_1 & ~m_after_branch).sum())
        endlevel_errors = int((m_after_branch & ~final_correct).sum())

        # source breakdown accuracy on final results
        source_accuracy = {}
        for src in ['model', 'age_direct', 'rule']:
            smask = (all_data['source'] == src) & valid_truth
            n = int(smask.sum())
            c = int(((all_data['results'] == all_data['sst']) & smask).sum())
            source_accuracy[src] = {
                'count': n,
                'correct': c,
                'accuracy': (c / n) if n > 0 else 0.0
            }

        funnel = {
            'total': total_count,
            'after_level_1': int(m_after_1.sum()),
            'after_branch_11_12': int(m_after_branch.sum()),
            # CHANGED: use global correct_total for clarity
            'final_correct': int(correct_total),
            'rates': {
                'after_level_1': (int(m_after_1.sum()) / total_count) if total_count > 0 else 0.0,
                'after_branch_11_12': (int(m_after_branch.sum()) / total_count) if total_count > 0 else 0.0,
                'final': end_to_end_accuracy
            }
        }

        stage_error_breakdown = {
            'level_1_errors': l1_errors,
            'branch_errors_11_12': branch_errors,
            'endlevel_errors': endlevel_errors
        }

        # NEW: "First wrong stage" – zeigt, wo ein Sample zuerst falsch abbiegt
        n = total_count
        wrong_stage = np.full(n, 'correct', dtype=object)
        wrong_stage[~m_after_1] = 'level_1'
        mask_branch_error = m_after_1 & ~m_after_branch
        wrong_stage[mask_branch_error] = 'branch'
        mask_endlevel_error = m_after_branch & ~final_correct
        wrong_stage[mask_endlevel_error] = 'endlevel'

        expected_at_first = np.empty(n, dtype=object); expected_at_first[:] = None
        predicted_at_first = np.empty(n, dtype=object); predicted_at_first[:] = None

        # Level-1-Fehler: erwartetes truth_1 vs. pred_1
        mask_l1 = (wrong_stage == 'level_1')
        if np.any(mask_l1):
            expected_at_first[mask_l1] = stage_truth['truth_1'].values[mask_l1]
            predicted_at_first[mask_l1] = pred_1.values[mask_l1]

        # Branch-Fehler: erwartetes truth_11/12 vs. pred_11/12, abhängig von truth_1
        idx_branch = np.where(wrong_stage == 'branch')[0]
        if idx_branch.size > 0:
            exp_branch = np.where(is_other[idx_branch],
                                  stage_truth['truth_12'].values[idx_branch],
                                  stage_truth['truth_11'].values[idx_branch])
            pred_branch = np.where(is_other[idx_branch],
                                   pred_12.values[idx_branch],
                                   pred_11.values[idx_branch])
            expected_at_first[idx_branch] = exp_branch
            predicted_at_first[idx_branch] = pred_branch

        # Endlevel-Fehler: erwartetes sst vs. finale results
        idx_end = np.where(wrong_stage == 'endlevel')[0]
        if idx_end.size > 0:
            expected_at_first[idx_end] = stage_truth['sst'].astype(str).values[idx_end]
            predicted_at_first[idx_end] = all_data.loc[valid_truth, 'results'].astype(str).values[idx_end]

        # Aggregationen
        first_wrong_stage_counts = {
            'level_1': int(np.sum(wrong_stage == 'level_1')),
            'branch': int(np.sum(wrong_stage == 'branch')),
            'endlevel': int(np.sum(wrong_stage == 'endlevel')),
            'correct': int(np.sum(wrong_stage == 'correct'))
        }

        # Breakdown je Pfad (Ground-Truth Top-Level: M/E/Other)
        truth1_arr = stage_truth['truth_1'].values
        first_wrong_stage_by_truth1 = {}
        for grp in ['M', 'E', 'Other']:
            gmask = (truth1_arr == grp)
            first_wrong_stage_by_truth1[grp] = {
                'total': int(np.sum(gmask)),
                'level_1': int(np.sum((wrong_stage == 'level_1') & gmask)),
                'branch': int(np.sum((wrong_stage == 'branch') & gmask)),
                'endlevel': int(np.sum((wrong_stage == 'endlevel') & gmask)),
                'correct': int(np.sum((wrong_stage == 'correct') & gmask))
            }

        # Top Fehlabbieger (expected -> predicted an der ersten falschen Stufe)
        df_wrong = pd.DataFrame({
            'stage': wrong_stage,
            'expected': expected_at_first,
            'predicted': predicted_at_first
        })
        df_wrong = df_wrong[df_wrong['stage'] != 'correct']
        top_wrong = []
        if not df_wrong.empty:
            agg = df_wrong.groupby(['stage', 'expected', 'predicted']).size().reset_index(name='count')
            agg = agg.sort_values('count', ascending=False).head(10)
            top_wrong = agg.to_dict(orient='records')

        return {
            'correct_by_model': correct_by_model,
            'direct_assignment_count': direct_assignment_count,
            'correct_total': correct_total,
            'end_to_end_accuracy': end_to_end_accuracy,
            'total_count': total_count,
            'total_TP': correct_total,
            'total_TN': 0,
            'total_FP': int((~correct_mask).sum()),
            'total_FN': 0,
            # existing breakdowns
            'funnel': funnel,
            'stage_error_breakdown': stage_error_breakdown,
            'source_accuracy': source_accuracy,
            # NEW: Fehlabbieger-Diagnostik
            'first_wrong_stage_counts': first_wrong_stage_counts,
            'first_wrong_stage_by_path': first_wrong_stage_by_truth1,
            'top_wrong_turns': top_wrong
        }

    def build_dashboard_payload(self, levels: List[Dict], overall_results: Dict) -> Dict:
        """
        Reduzierte Kennzahlen für das Dashboard:
        - funnel: Durchlaufzählungen und Raten
        - stage_errors: Attribution der Fehler zu Stufen
        - source_accuracy: Performance je Quelle (model / age_direct / rule)
        - end_to_end_accuracy: finale Accuracy (zur Einordnung)
        - per_level_accuracy: einfache Liste Level -> Accuracy
        """
        per_level_accuracy = {
            lvl['level']: lvl.get('accuracy', 0.0)
            for lvl in levels
        }
        return {
            'funnel': overall_results.get('funnel', {}),
            'stage_errors': overall_results.get('stage_error_breakdown', {}),
            'source_accuracy': overall_results.get('source_accuracy', {}),
            'end_to_end_accuracy': overall_results.get('end_to_end_accuracy', 0.0),
            'per_level_accuracy': per_level_accuracy,
            # NEW: Klar sichtbar "wo falsch abgebogen"
            'wrong_stage_counts': overall_results.get('first_wrong_stage_counts', {}),
            'wrong_stage_by_path': overall_results.get('first_wrong_stage_by_path', {}),
            'wrong_turns_top': overall_results.get('top_wrong_turns', [])
        }

    def save_results_to_json(self, levels: List[Dict], overall_results: Dict):
        os.makedirs(self.vis_path, exist_ok=True)
        out_path = os.path.join(self.vis_path, 'validation_results.json')
        with open(out_path, 'w') as f:
            json.dump({'levels': levels, 'overall_results': overall_results}, f, indent=4)
        QgsMessageLog.logMessage(f"Validation JSON written: {out_path}", level=Qgis.Info)

        dashboard_payload = self.build_dashboard_payload(levels, overall_results)
        summary_path = os.path.join(self.vis_path, 'dashboard_summary.json')
        with open(summary_path, 'w') as f2:
            json.dump(dashboard_payload, f2, indent=4)
        QgsMessageLog.logMessage(f"Dashboard summary written: {summary_path}", level=Qgis.Info)

    def load_and_visualize_validation_data(self):
        try:
            uri = QgsDataSourceUri()
            uri.setConnection(
                self.connection_params['host'],
                str(self.connection_params['port']),
                self.connection_params['dbname'],
                self.connection_params['user'],
                self.connection_params['password']
            )
            uri.setDataSource('MPSCDresden', 'validation_data', 'geom', '', 'validation_id')
            layer_name = 'Validation Data'

            existing_layer = QgsProject.instance().mapLayersByName(layer_name)
            if existing_layer:
                QgsProject.instance().removeMapLayer(existing_layer[0])

            layer = QgsVectorLayer(uri.uri(False), layer_name, 'postgres')
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                QgsMessageLog.logMessage("Validation Data layer loaded.", level=Qgis.Info)
            else:
                QgsMessageLog.logMessage("Validation Data layer invalid.", level=Qgis.Warning)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error loading validation layer: {e}", level=Qgis.Warning)

    # -----------------------------
    # Dashboard launcher
    # -----------------------------
    def launch_dashboard(self):
        """
        Startet das Dash-Dashboard in einem separaten Prozess und öffnet den Browser.
        Verwendet denselben Ansatz wie validate_model.py, um kein neues QGIS-Fenster zu öffnen.
        """
        # Starte Dash analog zu validate_model.py
        subprocess.Popen(['python', 'dashboard.py'])
        webbrowser.open('http://127.0.0.1:8050')
        QgsMessageLog.logMessage("Dashboard launched at http://127.0.0.1:8050", level=Qgis.Info)
