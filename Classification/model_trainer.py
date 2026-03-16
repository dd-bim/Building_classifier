import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import configparser
from qgis.core import QgsMessageLog, Qgis, QgsProject, QgsDataSourceUri, QgsVectorLayer

from .mapping_processor import MappingProcessor


class LabelEncoderManager:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.label_encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        self.label_encoders = self.load_label_encoders()

    def load_label_encoders(self):
        if os.path.exists(self.label_encoders_path):
            with open(self.label_encoders_path, 'rb') as f:
                label_encoders = joblib.load(f)
        else:
            label_encoders = {}

        for feature in ['roof_type', 'development_type_code', 'neighbour_majority_class', 'building_age']:
            if feature not in label_encoders:
                label_encoders[feature] = LabelEncoder()
        
        return label_encoders

    def save_label_encoders(self):
        with open(self.label_encoders_path, 'wb') as f:
            joblib.dump(self.label_encoders, f)

    def get_label_encoders(self):
        return self.label_encoders
    
class ModelTrainer:  
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den ModelTrainer mit DB-Verbindung, LabelEncodern und MappingProcessor.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        
        self.model_dir = os.path.join(os.path.dirname(__file__), config.get('Paths', 'model_dir'))
        
        self.label_encoder_manager = LabelEncoderManager(self.model_dir)
        self.label_encoders = self.label_encoder_manager.get_label_encoders()
        
        self.mapping_processor = MappingProcessor(conn, cur, connection_params)
            
    def get_label_encoders(self):
        """
        Gibt die geladenen LabelEncoder zurück.
        """
        return self.label_encoders
    
    def save_label_encoders(self):
        """
        Speichert die LabelEncoder auf der Festplatte.
        """
        self.label_encoder_manager.save_label_encoders()
            
    def load_data_from_db(self, table_name):
        """
        Lädt Daten aus einer angegebenen Tabelle der Datenbank als DataFrame.
        """
        query = f"SELECT * FROM {table_name}"
        self.cur.execute(query)
        rows = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        return df
    
    def prepare_data(self, data, target_column, warm_start: bool = False):
        """
        Bereitet die Features und Zielvariable für das Training vor (inkl. Label-Encoding).
        - warm_start=False (Default): Originales Verhalten, LabelEncoder werden gefittet (fit_transform).
        - warm_start=True: Vorhandene LabelEncoder nur transformieren, Fehler bei unbekannten Kategorien.
        """

        core_features = [
            'roof_type', 'storeys_above_ground', 'building_footprint', 
            'roof_ridge_height', 'eaves_height', 'storey_height', 
            'number_roof_surfaces', 'roof_slope', 'development_type_code',
            'building_age'
        ]
        
        # Einfache geometrische Features aus citydb_processor.py (absolute Dimensionen)
        simple_geometric_features = [
            'length_footprint',     # Gebäudelänge
            'width_footprint',      # Gebäudebreite
            'building_volume'       # Gebäudevolumen (absolut)
        ]
        
        # Advanced geometric features (geometrische Charakteristika)
        geometric_features = [
            'compactness',         # Form-Kompaktheit
            'convexity',          # Konvexität der Form  
            'rectangularity'      # Rechteckigkeit
        ]
        
        # Neighborhood features (Kontext)
        neighbourhood_features = [
            'neighbour_density', 'neighbour_avg_size', 'neighbour_min_distance', 'neighbour_majority_class'
        ]
    
        ratio_features = [
            'ground_area_per_storey',  # Grundfläche pro Stockwerk
            'height_to_area_ratio',    # Höhen-zu-Flächen-Verhältnis
            'footprint_ratio',          # Längen-zu-Breiten-Verhältnis
            'roof_height_ratio'
        ]
        
        features = core_features + simple_geometric_features + geometric_features + neighbourhood_features + ratio_features
        X = data[features].copy()
        y = data[target_column]

        # building_age wird als kategorisches Feature encoded
        for feature in ['roof_type', 'development_type_code', 'neighbour_majority_class', 'building_age']:
            if feature in X.columns and feature in self.label_encoders:
                if not isinstance(X[feature].iloc[0], str):
                    QgsMessageLog.logMessage(f"Warnung: {feature} ist kein String vor Label-Encoding!", level=Qgis.Warning)
                # Missing-Handling konsistent zum bisherigen Verhalten
                if feature == 'building_age':
                    X[feature] = X[feature].apply(lambda x: str(x) if pd.notna(x) else "unknown")
                else:
                    X[feature] = X[feature].apply(lambda x: str(x) if pd.notna(x) else None)

                le = self.label_encoders[feature]
                if not warm_start:
                    X[feature] = le.fit_transform(X[feature].astype(str))
                else:
                    # Warmstart: nur transformieren, bei unbekannten Klassen abbrechen
                    if not hasattr(le, 'classes_'):
                        raise ValueError(f"LabelEncoder für {feature} ist ungefitttet; Warmstart nicht möglich.")
                    unseen = set(X[feature].astype(str).unique()) - set(le.classes_)
                    if unseen:
                        raise ValueError(f"Unbekannte Kategorien in {feature}: {unseen}. Warmstart nicht möglich.")
                    X[feature] = le.transform(X[feature].astype(str))

        return X, y

    def calculate_adaptive_class_weights(self, y_train, target_names, level_name):
        """
        Berechnet intelligente Klassengewichte, die über sklearn's 'balanced' hinausgehen.
        
        STRATEGIE: 
        - Seltene Klassen bekommen höheres Gewicht als sklearn's balanced
        - Verwechslungspaargruppen (MR5/MR6, LW1/LW2) bekommen ausgewogene Gewichte
        - Domain Knowledge über wichtige vs. unwichtige Klassen fließt ein
        """
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        n_classes = len(target_names)
        
        # Basis-Gewichte (sklearn balanced Formel)
        base_weights = {}
        for class_name in target_names:
            count = class_counts.get(class_name, 1)  # Mindestens 1 um Division durch 0 zu vermeiden
            base_weights[class_name] = total_samples / (n_classes * count)
        
        # ADAPTIVE ANPASSUNGEN:
        adaptive_weights = base_weights.copy()
        
        # 1. KONFUSIONS-PAARE: MR5/MR6 und LW1/LW2 ausbalancieren
        confusion_pairs = [('MR5', 'MR6'), ('LW1', 'LW2')]
        for class_a, class_b in confusion_pairs:
            if class_a in adaptive_weights and class_b in adaptive_weights:
                # Gleiche Gewichte für verwechselte Klassen
                avg_weight = (adaptive_weights[class_a] + adaptive_weights[class_b]) / 2
                adaptive_weights[class_a] = avg_weight
                adaptive_weights[class_b] = avg_weight
                QgsMessageLog.logMessage(f"Confusion pair {class_a}/{class_b} balanced: {avg_weight:.3f}", level=Qgis.Info)
        
        # 2. SELTENE KLASSEN: Extra Boost für sehr seltene Klassen
        min_count = min(class_counts.values()) if class_counts else 1
        for class_name in target_names:
            count = class_counts.get(class_name, 1)
            if count <= max(2, min_count * 1.5):  # Sehr seltene Klassen
                boost_factor = 1.5  # 50% zusätzliches Gewicht
                adaptive_weights[class_name] *= boost_factor
                QgsMessageLog.logMessage(f"Rare class {class_name} boosted: x{boost_factor}", level=Qgis.Info)
        
        # 3. LEVEL-SPEZIFISCHE ANPASSUNGEN
        if level_name in ['111', '122']:  # Levels mit bekannten Konfusionsproblemen
            # Stärkere Gewichtung für unterscheidende Features
            for class_name in target_names:
                if class_name in ['MR5', 'MR6', 'LW1', 'LW2']:
                    adaptive_weights[class_name] *= 1.3  # 30% zusätzliches Gewicht
        
        # Normalisierung: Gewichte zwischen 0.1 und 10.0 begrenzen
        max_weight = max(adaptive_weights.values())
        min_weight = min(adaptive_weights.values())
        
        if max_weight / min_weight > 50:  # Zu extreme Gewichte vermeiden
            scale_factor = 50 / (max_weight / min_weight)
            for class_name in adaptive_weights:
                adaptive_weights[class_name] = min_weight + (adaptive_weights[class_name] - min_weight) * scale_factor
        
        # Log final weights
        weights_info = [f"{cls}: {weight:.3f} (n={class_counts.get(cls, 0)})" 
                       for cls, weight in sorted(adaptive_weights.items())]
        QgsMessageLog.logMessage(f"Adaptive weights for {level_name}: {', '.join(weights_info)}", level=Qgis.Info)
        
        return adaptive_weights

    def get_target_names_for_level(self, level):
        """
        Gibt die verfügbaren Zielnamen für ein bestimmtes Level zurück.
        Wird von validate_model.py und classify_data.py verwendet.
        """
        levels = self.level_definition()
        for level_name, column, logic, target_names in levels:
            if level_name == level:
                return target_names
        return []

    def split_and_save_data(self):
        """
        Teilt die Daten in Trainings-, Validierungs- und Klassifikationsdaten auf und speichert sie in der Datenbank.
        """
        try:
            data = self.load_data_from_db('"MPSCDresden".citydb_filter')

            if data.empty:
                QgsMessageLog.logMessage("Die Tabelle 'citydb_filter' enthält keine Daten.", level=Qgis.Critical)
                return

            target_column = 'sst'

            classification_data = data[data[target_column].isna()]
            classification_data.loc[:, 'training'] = 'c'

            data_with_target = data.dropna(subset=[target_column])
            if data_with_target.empty:
                QgsMessageLog.logMessage("Keine Daten mit Zielvariablen vorhanden.", level=Qgis.Critical)
                return
            
            data_with_target = self.filter_valid_classes(data_with_target, target_column)

            if data_with_target.empty:
                QgsMessageLog.logMessage("Keine gültigen Daten nach dem Entfernen von Klassen mit weniger als 2 Vertretern.", level=Qgis.Critical)
                return

            stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_idx, val_idx in stratified_split.split(data_with_target, data_with_target[target_column]):
                train_data = data_with_target.iloc[train_idx]
                validation_data = data_with_target.iloc[val_idx]

            train_data.loc[:, 'training'] = 't'
            validation_data.loc[:, 'training'] = 'v'
            classification_data.loc[:, 'training'] = 'c'

            self.cur.execute('''
                ALTER TABLE "MPSCDresden".citydb_filter ADD COLUMN IF NOT EXISTS "training" VARCHAR(1);
            ''')
            self.conn.commit()

            combined_data = pd.concat([train_data, validation_data, classification_data])
            for index, row in combined_data.iterrows():
                self.cur.execute(f'''
                    UPDATE "MPSCDresden".citydb_filter
                    SET "training" = %s
                    WHERE db_filter_id = %s
                ''', (row['training'], row['db_filter_id']))
            self.conn.commit()

            # Tabellen für Training, Validierung und Klassifikation erstellen
            self.cur.execute('''
                DROP TABLE IF EXISTS "MPSCDresden".train_data;
                CREATE TABLE "MPSCDresden".train_data AS
                SELECT *, NULL::VARCHAR AS results FROM "MPSCDresden".citydb_filter WHERE "training" = 't';
                ALTER TABLE "MPSCDresden".train_data
                ADD COLUMN train_id SERIAL PRIMARY KEY,
                ADD CONSTRAINT fk_gml_id FOREIGN KEY (gml_id) REFERENCES "MPSCDresden".citydb_filter (gml_id),
                ADD CONSTRAINT train_data_db_filter_id_unique UNIQUE (db_filter_id);

                DROP TABLE IF EXISTS "MPSCDresden".validation_data;
                CREATE TABLE "MPSCDresden".validation_data AS
                SELECT *, NULL::VARCHAR AS results FROM "MPSCDresden".citydb_filter WHERE "training" = 'v';
                ALTER TABLE "MPSCDresden".validation_data
                ADD COLUMN validation_id SERIAL PRIMARY KEY,
                ADD CONSTRAINT fk_gml_id FOREIGN KEY (gml_id) REFERENCES "MPSCDresden".citydb_filter (gml_id),
                ADD CONSTRAINT validation_data_db_filter_id_unique UNIQUE (db_filter_id);

                DROP TABLE IF EXISTS "MPSCDresden".classification_data CASCADE;
                CREATE TABLE "MPSCDresden".classification_data AS
                SELECT *, NULL::VARCHAR AS results FROM "MPSCDresden".citydb_filter WHERE "training" = 'c';
                ALTER TABLE "MPSCDresden".classification_data
                ADD COLUMN classification_id SERIAL PRIMARY KEY,
                ADD CONSTRAINT fk_gml_id FOREIGN KEY (gml_id) REFERENCES "MPSCDresden".citydb_filter (gml_id),
                ADD CONSTRAINT classification_data_db_filter_id_unique UNIQUE (db_filter_id);
            ''')
            self.conn.commit()

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Aufteilen der Daten: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            raise e
        
    def level_definition(self):
        """
        Definiert die Hierarchie und Logik der Klassifikations-Levels.
        """
        levels = [
            (
                '1', 'sst', 
                lambda row: 'M' if str(row['sst'])[0] == 'M' else ('E' if str(row['sst'])[0] == 'E' else 'Other'), 
                ['M', 'E', 'Other']
            ),
            (
                '11', 'results', 
                lambda row: 'MR' if row['results'] == 'M' and row['proximity'] == 'R' else (
                    'ME' if row['results'] == 'M' else (
                    'ER' if row['results'] == 'E' and row['proximity'] == 'R' else (
                    'EE' if row['results'] == 'E' else row['results']))), 
                ['MR', 'ME', 'ER', 'EE']
            ),
            (
                '12', 'sst', 
                lambda row: 'HH' if row['results'] == 'Other' and str(row['sst'])[0:2] == 'HH' else (
                    'LW' if row['results'] == 'Other' and str(row['sst'])[0:2] == 'LW' else None), 
                ['HH', 'LW']
            ),
            (
                '121', 'sst', 
                lambda row: 'HH3' if row['results'] == 'HH' and str(row['sst'])[0:3] == 'HH3' else (
                    'HH4' if row['results'] == 'HH' and str(row['sst'])[0:3] == 'HH4' else None), 
                ['HH3', 'HH4']
            ),
            (
                '122', 'sst', 
                lambda row: 'LW1' if row['results'] == 'LW' and str(row['sst'])[0:3] == 'LW1' else (
                    'LW2' if row['results'] == 'LW' and str(row['sst'])[0:3] == 'LW2' else (
                    'LW3' if row['results'] == 'LW' and str(row['sst'])[0:3] == 'LW3' else (
                    'LW7' if row['results'] == 'LW' and str(row['sst'])[0:3] == 'LW7' else None))), 
                ['LW1', 'LW2', 'LW3', 'LW7']
            ),
            (
                '112', 'sst', 
                lambda row: 'ME2' if row['results'] == 'ME' and str(row['sst'])[0:3] == 'ME2' else (
                    'ME3' if row['results'] == 'ME' and str(row['sst'])[0:3] == 'ME3' else (
                    'ME4' if row['results'] == 'ME' and str(row['sst'])[0:3] == 'ME4' else (
                    'ME5' if row['results'] == 'ME' and str(row['sst'])[0:3] == 'ME5' else (
                    'ME6' if row['results'] == 'ME' and str(row['sst'])[0:3] == 'ME6' else (
                    'ME7' if row['results'] == 'ME' and str(row['sst'])[0:3] == 'ME7' else None))))), 
                ['ME2', 'ME3', 'ME4', 'ME5', 'ME6', 'ME7']
            ),
            (
                '113', 'sst', 
                lambda row: 'ER2' if row['results'] == 'ER' and str(row['sst'])[0:3] == 'ER2' else (
                    'ER3' if row['results'] == 'ER' and str(row['sst'])[0:3] == 'ER3' else (
                    'ER4' if row['results'] == 'ER' and str(row['sst'])[0:3] == 'ER4' else (
                    'ER5' if row['results'] == 'ER' and str(row['sst'])[0:3] == 'ER5' else (
                    'ER7' if row['results'] == 'ER' and str(row['sst'])[0:3] == 'ER7' else None)))), 
                ['ER2', 'ER3', 'ER4', 'ER5', 'ER7']
            ),
            (
                '114', 'sst', 
                lambda row: 'EE1' if row['results'] == 'EE' and str(row['sst'])[0:3] == 'EE1' else (
                    'EE2' if row['results'] == 'EE' and str(row['sst'])[0:3] == 'EE2' else (
                    'EE3' if row['results'] == 'EE' and str(row['sst'])[0:3] == 'EE3' else (
                    'EE4' if row['results'] == 'EE' and str(row['sst'])[0:3] == 'EE4' else (
                    'EE5' if row['results'] == 'EE' and str(row['sst'])[0:3] == 'EE5' else (
                    'EE7' if row['results'] == 'EE' and str(row['sst'])[0:3] == 'EE7' else None))))), 
                ['EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE7']
            ),
            (
                '111', 'sst', 
                lambda row: 'MR2' if row['results'] == 'MR' and str(row['sst'])[0:3] == 'MR2' else (
                    'MR3' if row['results'] == 'MR' and str(row['sst'])[0:3] == 'MR3' else (
                    'MR4' if row['results'] == 'MR' and str(row['sst'])[0:3] == 'MR4' else (
                    'MR5' if row['results'] == 'MR' and str(row['sst'])[0:3] == 'MR5' else (
                    'MR6' if row['results'] == 'MR' and str(row['sst'])[0:3] == 'MR6' else (
                    'MR7' if row['results'] == 'MR' and str(row['sst'])[0:3] == 'MR7' else None))))), 
                ['MR2', 'MR3', 'MR4', 'MR5', 'MR6', 'MR7']
            )
        ]
        
        return levels

    def train_level(self, X_train, y_train, target_names, level_name, warm_start=False):
        try:
            # Datenvalidierung
            if X_train.empty or y_train.empty:
                QgsMessageLog.logMessage(f"Keine Daten für das Training von {level_name} vorhanden.", level=Qgis.Warning)
                return None, None

            # Nur gültige Zielklassen verwenden
            valid_indices = y_train.isin(target_names)
            X_train = X_train[valid_indices]
            y_train = y_train[valid_indices]
            if y_train.empty:
                QgsMessageLog.logMessage(f"Keine gültigen Zielwerte für {level_name}.", level=Qgis.Warning)
                return None, None

            # Klassen mit weniger als 2 Einträgen filtern
            class_counts = y_train.value_counts()
            insufficient_classes = class_counts[class_counts < 2].index.tolist()
            if insufficient_classes:
                QgsMessageLog.logMessage(f"Unzureichende Klassen für {level_name}: {insufficient_classes}", level=Qgis.Warning)
                valid_indices = ~y_train.isin(insufficient_classes)
                X_train = X_train[valid_indices]
                y_train = y_train[valid_indices]
            if X_train.empty or y_train.empty:
                QgsMessageLog.logMessage(f"Keine validen Daten nach Filterung für {level_name}.", level=Qgis.Warning)
                return None, None

            # NaN-Behandlung
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                median_value = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_value)
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                mode_value = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else 'unknown'
                X_train[col] = X_train[col].fillna(mode_value)

            # WARMSTART: vorhandenes Modell laden und erweitern
            if warm_start:
                model_path = os.path.join(self.model_dir, f'model_{level_name}.pkl')
                if os.path.exists(model_path):
                    try:
                        prev_model = joblib.load(model_path)
                        if isinstance(prev_model, RandomForestClassifier) and hasattr(prev_model, 'classes_'):
                            prev_classes = set(prev_model.classes_.tolist())
                            y_classes = set(y_train.unique().tolist())
                            # Prüfe Konsistenz aller Trees
                            mixed_trees = any(
                                getattr(est, "n_classes_", len(y_classes)) != len(y_classes)
                                for est in getattr(prev_model, "estimators_", [])
                            )
                            if prev_classes != y_classes or prev_model.n_classes_ != len(y_classes) or mixed_trees:
                                QgsMessageLog.logMessage(
                                    f"Klassenänderung oder inkonsistente Bäume für {level_name}: "
                                    f"alt={sorted(prev_classes)}, neu={sorted(y_classes)}, "
                                    f"mixed_trees={mixed_trees}. Fallback auf Full-Retrain.",
                                    level=Qgis.Warning
                                )
                            else:
                                old_n = prev_model.n_estimators
                                add_trees = max(50, int(old_n * 0.2))
                                new_n = old_n + add_trees
                                prev_model.set_params(warm_start=True, n_estimators=new_n, n_jobs=-1, random_state=42)
                                QgsMessageLog.logMessage(
                                    f"Warmstart {level_name}: Bäume {old_n}->{new_n}", level=Qgis.Info
                                )
                                prev_model.fit(X_train, y_train)
                                importance_df = pd.DataFrame({
                                    'Feature': X_train.columns.tolist(),
                                    'Importance': prev_model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                joblib.dump(prev_model, model_path)
                                return prev_model, importance_df
                    except Exception as e:
                        QgsMessageLog.logMessage(f"Warmstart-Laden fehlgeschlagen ({level_name}): {e}", level=Qgis.Warning)
                # Falls kein Modell/Fehler/Neue Klassen: Full-Retrain (weiter unten)

            # ADAPTIVE CLASS WEIGHTS & GridSearch (Full-Retrain)
            adaptive_weights = self.calculate_adaptive_class_weights(y_train, target_names, level_name)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', adaptive_weights]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

            n_samples = len(X_train)
            cv_folds = min(5, max(3, n_samples // 50))
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                n_jobs=1,
                scoring='balanced_accuracy',
                verbose=1
            )
            QgsMessageLog.logMessage(f"Starte RandomForest GridSearch für {level_name} mit {n_samples} Samples", level=Qgis.Info)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            QgsMessageLog.logMessage(
                f"RandomForest für {level_name}: Score={best_score:.4f}, Params={grid_search.best_params_}",
                level=Qgis.Info
            )
            importance_df = pd.DataFrame({
                'Feature': X_train.columns.tolist(),
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            joblib.dump(best_model, os.path.join(self.model_dir, f'model_{level_name}.pkl'))
            # Ergebnisse loggen
            results_path = os.path.join(self.model_dir, f'results_{level_name}.txt')
            with open(results_path, 'w') as f:
                f.write(f"{'='*60}\nRandomForest Model Training Summary for {level_name}\n{'='*60}\n\n")
                f.write(f"🔹 Best Score: {best_score:.4f}\n")
                f.write(f"🔹 Best Parameters:\n{grid_search.best_params_}\n\n")
                f.write(f"🔹 Training Samples: {n_samples}\n")
                f.write(f"🔹 CV Folds: {cv_folds}\n")
                f.write(f"🔹 Features Used: {len(X_train.columns.tolist())}\n\n")
                f.write("🔹 Top 10 Feature Importances:\n")
                f.write(importance_df.head(10).to_string(index=False) + "\n\n")
                f.write("🔹 Training Configuration:\n")
                f.write(" - RandomForest with adaptive class weights\n")
                f.write(" - Grid search for optimal hyperparameters\n")
                f.write(" - Balanced accuracy scoring\n")
                f.write(" - NaN values filled with median/mode\n")

            QgsMessageLog.logMessage(f"RandomForest training results for {level_name} saved", level=Qgis.Info)

            return best_model, importance_df

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Training von {level_name}: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            return None, None
        
    def train(self, warm_start=False):
        """
        Trainiert die Klassifikationsmodelle für alle Level.
        warm_start=True: bestehende Modelle werden erweitert (kein Re-Splitting, nur train_data verwenden).
        """
        # KEIN Re-Splitting beim Warmstart – train_data wird als Basis genutzt
        if warm_start:
            QgsMessageLog.logMessage("Warm Start: Bestehende Modelle mit neuen Trainingsdaten weitertrainieren.", level=Qgis.Info)
            return self.train_warm_start()

        # Lade die Trainingsdaten
        QgsMessageLog.logMessage("Training aller Level gestartet...", level=Qgis.Info)
        train_data = self.load_data_from_db('"MPSCDresden".train_data')
        train_data = self.filter_valid_classes(train_data, 'sst')
        if train_data.empty:
            QgsMessageLog.logMessage("Keine gültigen Trainingsdaten nach Filterung der Klassen mit weniger als 2 Einträgen.", level=Qgis.Critical)
            return

        train_data['results'] = None
        levels = self.level_definition()
        for level_name, column, logic, target_names in levels:
            if level_name == '11':
                train_data['results'] = train_data.apply(logic, axis=1)
            else:
                train_data[level_name] = train_data.apply(logic, axis=1)
                level_data = train_data[train_data[level_name].isin(target_names)]
                if level_data.empty:
                    QgsMessageLog.logMessage(f"Keine gültigen Daten für Level {level_name} nach Filterung.", level=Qgis.Warning)
                    continue
                # Encoder fit (vollständiges Training)
                X_train, y_train = self.prepare_data(level_data, level_name, warm_start=False)
                trained_model, importance_df = self.train_level(
                    X_train, y_train, target_names, level_name, warm_start=False
                )
                if trained_model and importance_df is not None:
                    predictions = trained_model.predict(X_train)
                    predictions_series = pd.Series(predictions, index=level_data.index)
                    train_data.loc[level_data.index, 'results'] = predictions_series
                    self.save_model(trained_model, level_name)
                    self.save_feature_importance(level_name, importance_df)
        try:
            self.save_label_encoders()
        except Exception as e:
            QgsMessageLog.logMessage(f"Warnung: LabelEncoder konnten nicht gespeichert werden: {e}", level=Qgis.Warning)
        QgsMessageLog.logMessage("Models saved for all levels", level=Qgis.Info)
        return train_data
    
    def train_warm_start(self):
        """
        Inkrementelles Training ohne Re-Splitting:
        - nutzt bestehende train_data
        - verwendet prepare_data(..., warm_start=True), Fallback auf prepare_data(..., False) bei neuen Kategorien
        """
        QgsMessageLog.logMessage("Warm Start: Bestehende Modelle mit neuen Trainingsdaten weitertrainieren.", level=Qgis.Info)

        # Rekonstruiere train/validation Tabellen aus citydb_filter.training Flags,
        # damit neu hinzugefügte Nachkartierungen sicher in den Tabellen landen.
        try:
            # IDs aus citydb_filter holen
            self.cur.execute('SELECT db_filter_id FROM "MPSCDresden".citydb_filter WHERE training = %s', ('t',))
            train_ids = [int(r[0]) for r in self.cur.fetchall()]
            self.cur.execute('SELECT db_filter_id FROM "MPSCDresden".citydb_filter WHERE training = %s', ('v',))
            val_ids = [int(r[0]) for r in self.cur.fetchall()]

            # Tabellen neu befüllen (update_train_and_validation_tables erwartet DataFrames mit db_filter_id)
            train_df = pd.DataFrame({'db_filter_id': train_ids})
            val_df = pd.DataFrame({'db_filter_id': val_ids})
            self.update_train_and_validation_tables(train_df, val_df)
            QgsMessageLog.logMessage(f"Train/Validation Tabellen vor Warm-Start rekonstruiert: {len(train_ids)} train, {len(val_ids)} val", level=Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(f"Warnung: Rekonstruktion der Train/Validation Tabellen vor Warm-Start fehlgeschlagen: {e}", level=Qgis.Warning)
            # Fortfahren: versuche trotzdem die bestehenden train_data zu laden

        train_data = self.load_data_from_db('"MPSCDresden".train_data')
        train_data = self.filter_valid_classes(train_data, 'sst')
        if train_data.empty:
            QgsMessageLog.logMessage("Keine gültigen Trainingsdaten nach Filterung der Klassen mit weniger als 2 Einträgen.", level=Qgis.Critical)
            return

        train_data['results'] = None
        levels = self.level_definition()
        for level_name, column, logic, target_names in levels:
            if level_name == '11':
                train_data['results'] = train_data.apply(logic, axis=1)
            else:
                train_data[level_name] = train_data.apply(logic, axis=1)
                level_data = train_data[train_data[level_name].isin(target_names)]
                if level_data.empty:
                    QgsMessageLog.logMessage(f"Keine gültigen Daten für Level {level_name} nach Filterung.", level=Qgis.Warning)
                    continue

                # Warmstart-Encoder verwenden, bei Problemen Full-Retrain
                try:
                    X_train, y_train = self.prepare_data(level_data, level_name, warm_start=True)
                except Exception as enc_err:
                    QgsMessageLog.logMessage(f"Encoder-Problem ({level_name}): {enc_err}. Fallback auf Full-Retrain.", level=Qgis.Warning)
                    X_train, y_train = self.prepare_data(level_data, level_name, warm_start=False)

                trained_model, importance_df = self.train_level(
                    X_train, y_train, target_names, level_name, warm_start=True
                )
                if trained_model and importance_df is not None:
                    predictions = trained_model.predict(X_train)
                    predictions_series = pd.Series(predictions, index=level_data.index)
                    train_data.loc[level_data.index, 'results'] = predictions_series
                    self.save_model(trained_model, level_name)
                    self.save_feature_importance(level_name, importance_df)

        try:
            self.save_label_encoders()
        except Exception as e:
            QgsMessageLog.logMessage(f"Warnung: LabelEncoder konnten nicht gespeichert werden: {e}", level=Qgis.Warning)

        QgsMessageLog.logMessage("Warmstart-Training abgeschlossen.", level=Qgis.Info)
        return train_data

    def filter_valid_classes(self, data, target_column='sst', min_samples=2):
        """
        Filtert Daten nach Klassen mit mindestens min_samples Einträgen.
        
        :param data: DataFrame mit den zu filternden Daten
        :param target_column: Name der Zielvariable-Spalte
        :param min_samples: Mindestanzahl Samples pro Klasse
        :return: Gefiltertes DataFrame
        """
        if data.empty or target_column not in data.columns:
            return data
            
        class_counts = data[target_column].value_counts()
        valid_classes = class_counts[class_counts >= min_samples].index
        filtered_data = data[data[target_column].isin(valid_classes)]
        
        if len(valid_classes) < len(class_counts):
            removed_classes = class_counts[class_counts < min_samples].index.tolist()
            QgsMessageLog.logMessage(
                f"Entfernte Klassen mit < {min_samples} Samples: {removed_classes}", 
                level=Qgis.Info
            )
        
        return filtered_data
    
    def update_train_and_validation_tables(self, train_data, validation_data):
        """
        Aktualisiert die train_data und validation_data Tabellen für Retraining.
        Set-basierte Inserts direkt aus citydb_filter mit korrekter Typzuordnung.
        """
        try:
            # IDs sammeln
            train_ids = train_data['db_filter_id'].dropna().astype(int).tolist()
            val_ids = validation_data['db_filter_id'].dropna().astype(int).tolist()

            # Wenn keine IDs vorhanden, Tabellen leeren und zurück
            self.cur.execute('TRUNCATE "MPSCDresden".train_data RESTART IDENTITY CASCADE')
            self.cur.execute('TRUNCATE "MPSCDresden".validation_data RESTART IDENTITY CASCADE')
            self.conn.commit()
            if not train_ids and not val_ids:
                QgsMessageLog.logMessage("Keine IDs für Train/Validation vorhanden. Tabellen geleert.", level=Qgis.Warning)
                return

            # Hilfsfunktion zum Laden geordneter Spaltennamen
            def get_columns(table, exclude_cols):
                self.cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'MPSCDresden' AND table_name = %s
                    ORDER BY ordinal_position
                """, (table,))
                cols = [r[0] for r in self.cur.fetchall()]
                return [c for c in cols if c not in exclude_cols]

            # Spaltenlisten besorgen (ohne PK-Spalten)
            train_cols = get_columns('train_data', exclude_cols=['train_id'])
            val_cols = get_columns('validation_data', exclude_cols=['validation_id'])

            # Select-Listen bauen: für alle Spalten aus citydb_filter, für 'results' NULL::varchar
            def build_select_list(cols):
                sel = []
                for c in cols:
                    if c == 'results':
                        sel.append("NULL::varchar AS results")
                    else:
                        sel.append(f'cf."{c}"')
                return ', '.join(sel)

            train_cols_sql = ', '.join([f'"{c}"' for c in train_cols])
            val_cols_sql = ', '.join([f'"{c}"' for c in val_cols])

            train_select_sql = build_select_list(train_cols)
            val_select_sql = build_select_list(val_cols)

            # Train-Daten einfügen
            if train_ids:
                self.cur.execute(
                    f'''
                    INSERT INTO "MPSCDresden".train_data ({train_cols_sql})
                    SELECT {train_select_sql}
                    FROM "MPSCDresden".citydb_filter cf
                    WHERE cf.db_filter_id = ANY(%s)
                    ''',
                    (train_ids,)
                )

            # Validation-Daten einfügen
            if val_ids:
                self.cur.execute(
                    f'''
                    INSERT INTO "MPSCDresden".validation_data ({val_cols_sql})
                    SELECT {val_select_sql}
                    FROM "MPSCDresden".citydb_filter cf
                    WHERE cf.db_filter_id = ANY(%s)
                    ''',
                    (val_ids,)
                )

            self.conn.commit()
            QgsMessageLog.logMessage("Train/Validation Tabellen für Retraining aktualisiert.", level=Qgis.Info)
            
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Aktualisieren der Train/Validation Tabellen: {str(e)}", level=Qgis.Critical)
            raise e

    def load_and_visualize_training_data(self):
        """
        Lädt und visualisiert die Trainingsdaten in QGIS.
        Wird vom building_classificator_dialog.py nach dem Training aufgerufen.
        """
        try:
            uri = QgsDataSourceUri()
            uri.setConnection(
                self.connection_params['host'],
                str(self.connection_params['port']),
                self.connection_params['dbname'],
                self.connection_params['user'],
                self.connection_params['password']
            )
            uri.setDataSource(
                'MPSCDresden',
                'train_data',
                'geom',
                '',
                'train_id'
            )

            layer_name = 'Training Data'
            # Entferne ggf. alten Layer
            existing_layer = QgsProject.instance().mapLayersByName(layer_name)
            if existing_layer:
                QgsProject.instance().removeMapLayer(existing_layer[0])

            layer = QgsVectorLayer(uri.uri(False), layer_name, 'postgres')
            if not layer.isValid():
                QgsMessageLog.logMessage("Training Data Layer ist ungültig", level=Qgis.Critical)
                return

            QgsProject.instance().addMapLayer(layer)
            QgsMessageLog.logMessage("Training Data Layer erfolgreich geladen.", level=Qgis.Info)
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Laden der Trainingsdaten: {str(e)}", level=Qgis.Critical)
    
    def save_model(self, model, level_name):
        
        if model:
            model_path = os.path.join(self.model_dir, f'model_{level_name}.pkl')
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(model, model_path)
            QgsMessageLog.logMessage(f"Model for {level_name} saved at {model_path}", level=Qgis.Info)
            
    def save_feature_importance(self, level_name, importance_df):
        try:
            # Feature Importance-Daten erstellen
            importance_data = {
                "features": importance_df["Feature"].tolist(),
                "importance": importance_df["Importance"].tolist()
            }

            # Datei speichern
            importance_file = os.path.join(self.model_dir, f'feature_importance_{level_name}.json')
            os.makedirs(self.model_dir, exist_ok=True)
            with open(importance_file, 'w') as f:
                json.dump(importance_data, f, indent=4)

            QgsMessageLog.logMessage(
                f"Feature Importance für Level {level_name} gespeichert: {importance_file}",
                level=Qgis.Info
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Fehler beim Speichern der Feature Importance für Level {level_name}: {str(e)}",
                level=Qgis.Critical
            )
            
    def split_train_validation(self, data, target_column):
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, val_idx in stratified_split.split(data, data[target_column]):
            train_data = data.iloc[train_idx]
            validation_data = data.iloc[val_idx]
        return train_data, validation_data