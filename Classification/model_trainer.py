import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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

        # Sicherstellen, dass alle erwarteten LabelEncoder vorhanden sind
        for feature in ['roof_type', 'development_type_code', 'neighbor_majority_class', 'building_age']:
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
    
    def prepare_data(self, data, target_column):     
        """
        Bereitet die Features und Zielvariable für das Training vor (inkl. Label-Encoding).
        """
        # features = [
        #     'roof_type', 'storeys_above_ground', 'building_footprint', 'length_footprint', 
        #     'width_footprint', 'roof_ridge_height', 'eaves_height', 'storey_height', 'number_roof_surfaces', 
        #     'roof_slope', 'neighbor_density', 'neighbor_avg_size', 'neighbor_min_distance', 'neighbor_majority_class', 
        #     'development_type_code', 'ground_area_per_storey', 'footprint_ratio', 
        #     'height_to_area_ratio', 'roof_height_ratio', 'storey_height_ratio', 'roof_slope_to_height_ratio', 
        #     'building_volume'
        # ]
        
        # Core geometric features (wichtigste)
        core_features = [
            'roof_type', 'storeys_above_ground', 'building_footprint', 
            'roof_ridge_height', 'eaves_height', 'storey_height', 
            'number_roof_surfaces', 'roof_slope', 'development_type_code',
            'building_age'
        ]
        
        # Advanced geometric features (geometrische Charakteristika)
        geometric_features = [
            'compactness', 
            'convexity',  
            'rectangularity',     
            'vertex_count'
        ]
        
        # Neighborhood features (Kontext)
        neighborhood_features = [
            'neighbor_density', 'neighbor_avg_size', 'neighbor_min_distance', 'neighbor_majority_class'
        ]
        
        # Derived ratios (abgeleitete Verhältnisse)
        ratio_features = [
            'ground_area_per_storey', 
            'height_to_area_ratio',
            'footprint_ratio',
            'roof_height_ratio',
            'building_volume'
        ]
        
        features = core_features + geometric_features + neighborhood_features + ratio_features
        
        X = data[features].copy()
        y = data[target_column]
    
        for feature in ['roof_type', 'development_type_code', 'neighbor_majority_class', 'building_age']:
            if feature in X.columns and feature in self.label_encoders:
                if not isinstance(X[feature].iloc[0], str):
                    QgsMessageLog.logMessage(f"Warnung: {feature} ist kein String vor Label-Encoding!", level=Qgis.Warning)
                X[feature] = X[feature].apply(lambda x: str(x) if pd.notna(x) else None)
                X[feature] = self.label_encoders[feature].fit_transform(X[feature].astype(str))

        return X, y

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

                DROP TABLE IF EXISTS "MPSCDresden".classification_data;
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
            ),
            (
                '1111', 'sst_sub', 
                lambda row: 'MRO2' if row['results'] == 'MR2' and str(row['sst_sub'])[0:4] == 'MRO2' else (
                    'MRO3' if row['results'] == 'MR3' and str(row['sst_sub'])[0:4] == 'MRO3' else (
                    'MRO4' if row['results'] == 'MR4' and str(row['sst_sub'])[0:4] == 'MRO4' else (
                    'MRO7' if row['results'] == 'MR7' and str(row['sst_sub'])[0:4] == 'MRO7' else (
                    'MRG2' if row['results'] == 'MR2' and str(row['sst_sub'])[0:4] == 'MRG2' else (
                    'MRG3' if row['results'] == 'MR3' and str(row['sst_sub'])[0:4] == 'MRG3' else (
                    'MRG4' if row['results'] == 'MR4' and str(row['sst_sub'])[0:4] == 'MRG4' else (
                    'MRG7' if row['results'] == 'MR7' and str(row['sst_sub'])[0:4] == 'MRG7' else None))))))), 
                ['MRO2', 'MRO3', 'MRO4', 'MRO7', 'MRG2', 'MRG3', 'MRG4', 'MRG7']
            )
        ]
        
        return levels

    def train_level(self, X_train, y_train, target_names, level_name, warm_start=False, existing_model=None):
        try:
            # Datenvalidierung
            if X_train.empty or y_train.empty:
                QgsMessageLog.logMessage(f"Keine Daten für das Training von {level_name} vorhanden.", level=Qgis.Warning)
                return None

            # Nur gültige Zielklassen verwenden
            valid_indices = y_train.isin(target_names)
            X_train = X_train[valid_indices]
            y_train = y_train[valid_indices]

            if y_train.empty:
                QgsMessageLog.logMessage(f"Keine gültigen Zielwerte für {level_name}.", level=Qgis.Warning)
                return None

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
                return None

            # Hoch korrelierte Features entfernen
            corr_matrix = X_train.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.8)]
            X_train = X_train.drop(columns=high_corr_features)
            QgsMessageLog.logMessage(f"Entfernte hoch korrelierte Features: {high_corr_features}", level=Qgis.Info)

            # Feature Importance zur Auswahl
            base_model = RandomForestClassifier(class_weight="balanced", random_state=42, warm_start=warm_start)
            base_model.fit(X_train, y_train)
            importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': base_model.feature_importances_})
            important_features = importance_df[importance_df["Importance"] >= 0.01]["Feature"].tolist()
            
            # Falls keine wichtigen Features gefunden wurden, behalte mindestens die wichtigsten Features
            if not important_features:
                QgsMessageLog.logMessage(f"Keine Features mit Importance >= 0.01 für {level_name}, verwende Top-Features", level=Qgis.Warning)
                important_features = importance_df.nlargest(min(3, len(importance_df)), 'Importance')['Feature'].tolist()
            
            X_train_selected = X_train[important_features]

            QgsMessageLog.logMessage(f"Behaltene Features für {level_name}: {important_features}", level=Qgis.Info)
            
            # Prüfe, ob noch Features vorhanden sind
            if X_train_selected.empty or len(important_features) == 0:
                QgsMessageLog.logMessage(f"Keine Features für Training von {level_name} nach Filterung verfügbar.", level=Qgis.Warning)
                return None

            # Imputer anwenden
            num_features = X_train_selected.select_dtypes(include=[np.number]).columns.tolist()
            cat_features = X_train_selected.select_dtypes(include=['object', 'category']).columns.tolist()

            imputer = ColumnTransformer([
                ('num', SimpleImputer(strategy='mean'), num_features),
                ('cat', SimpleImputer(strategy='most_frequent'), cat_features)
            ])

            X_train_imputed = pd.DataFrame(
                imputer.fit_transform(X_train_selected),
                columns=num_features + cat_features,
                index=X_train_selected.index
            )

            joblib.dump(imputer, os.path.join(self.model_dir, f'imputer_{level_name}.pkl'))

            # Modelltraining mit GridSearchCV
            param_grid = {
                'n_estimators': [150, 250, 350],
                'max_depth': [15, 25, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'criterion': ['gini', 'entropy']
            }

            final_model = existing_model or RandomForestClassifier(class_weight="balanced", random_state=42, warm_start=warm_start)
            scorer = make_scorer(f1_score, average='weighted')

            grid_search = GridSearchCV(estimator=final_model, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring=scorer)
            grid_search.fit(X_train_imputed, y_train)
            best_model = grid_search.best_estimator_

            # Modell & Feature Info speichern
            joblib.dump(best_model, os.path.join(self.model_dir, f'model_{level_name}.pkl'))

            removed_features = set(X_train.columns) - set(important_features)
            QgsMessageLog.logMessage(f"Entfernte Features für {level_name}: {removed_features}", level=Qgis.Info)

            # Ergebnisse loggen
            y_pred_train = best_model.predict(X_train_imputed)
            pred_counts_train = pd.Series(y_pred_train).value_counts()

            results_path = os.path.join(self.model_dir, f'results_{level_name}.txt')
            with open(results_path, 'w') as f:
                f.write(f"{'='*60}\nModel Training Summary for {level_name}\n{'='*60}\n\n")
                f.write(f"🔹 Best Hyperparameters:\n{grid_search.best_params_}\n\n")
                f.write(f"🔹 Important Features:\n" + "\n".join(f" - {feat}" for feat in important_features) + "\n\n")
                f.write(f"🔹 Removed Features:\n" + ("\n".join(f" - {feat}" for feat in removed_features) if removed_features else "None") + "\n\n")
                f.write("🔹 Feature Importances:\n")
                f.write(importance_df.to_string(index=False) + "\n\n")
                f.write("🔹 Predictions in Training Data:\n")
                f.write(str(pred_counts_train) + "\n")

            QgsMessageLog.logMessage(f"Training results for {level_name} saved at {results_path}", level=Qgis.Info)

            return best_model, importance_df

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Training von {level_name}: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            raise e
        
    def train(self, warm_start=False, retrain_level=None):
        # Beim Retraining: neue Split und Tabellen-Update
        # Beim normalen Training: verwende bereits existierende train_data Tabelle
        if retrain_level is not None:
            QgsMessageLog.logMessage("Retraining: Erstelle neue Train/Validation Split...", level=Qgis.Info)
            # Lade alle Daten aus citydb_filter für neuen Split
            all_data = self.load_data_from_db('"MPSCDresden".citydb_filter')
            # Filtere nur Daten mit SST-Werten für Training/Validation
            data_with_target = all_data.dropna(subset=['sst'])
            
            # Filtere Klassen mit weniger als 2 Einträgen heraus
            data_with_target = self.filter_valid_classes(data_with_target, 'sst')
            
            if data_with_target.empty:
                QgsMessageLog.logMessage("Keine gültigen Daten für Retraining nach Filterung.", level=Qgis.Critical)
                return
                
            # Neuer Split für Retraining
            train_data, validation_data = self.split_train_validation(data_with_target, target_column='sst')
            self.update_train_and_validation_tables(train_data, validation_data)
        else:
            QgsMessageLog.logMessage("Normales Training: Verwende existierende Trainingsdaten...", level=Qgis.Info)
            # Lade die bereits gesplitteten Trainingsdaten
            train_data = self.load_data_from_db('"MPSCDresden".train_data')
            
            # Filtere Klassen mit weniger als 2 Einträgen heraus
            train_data = self.filter_valid_classes(train_data, 'sst')

        if train_data.empty:
            QgsMessageLog.logMessage("Keine gültigen Trainingsdaten nach Filterung der Klassen mit weniger als 2 Einträgen.", level=Qgis.Critical)
            return

        train_data['results'] = None

        levels = self.level_definition()

        for level_name, column, logic, target_names in levels:
            if retrain_level and level_name != retrain_level:
                continue

            if level_name == '11':
                train_data['results'] = train_data.apply(logic, axis=1)
            else:
                train_data[level_name] = train_data.apply(logic, axis=1)
                level_data = train_data[train_data[level_name].isin(target_names)]

                X_train, y_train = self.prepare_data(level_data, level_name)

                model_path = os.path.join(self.model_dir, f'model_{level_name}.pkl')
                if warm_start and os.path.exists(model_path):
                    model = joblib.load(model_path)
                    model.warm_start = True
                    QgsMessageLog.logMessage(f"Loaded existing model for {level_name} for retraining.", level=Qgis.Info)
                else:
                    model = None

                trained_model, importance_df = self.train_level(
                    X_train, y_train, target_names, level_name, warm_start=warm_start, existing_model=model
                )

                if trained_model:
                    X_train_selected = X_train[trained_model.feature_names_in_]
                    predictions = trained_model.predict(X_train_selected)
                    predictions = pd.Series(predictions, index=level_data.index).reindex(level_data.index, fill_value='fail')
                    train_data.loc[level_data.index, 'results'] = predictions
                    self.save_model(trained_model, level_name)
                    self.save_feature_importance(level_name, importance_df)

        QgsMessageLog.logMessage("Models saved for all levels", level=Qgis.Info)
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
        DEPRECATED für normales Training! 
        Diese Methode sollte nur beim Retraining verwendet werden, da sie bestehende 
        Tabellen überschreibt. Für initiales Setup verwende split_and_save_data().
        
        Erstellt neue train_data und validation_data Tabellen mit Primärschlüsseln.
        """
        # TRAIN DATA - Erstelle Tabelle mit korrektem Primärschlüssel von Anfang an
        self.cur.execute('DROP TABLE IF EXISTS "MPSCDresden".train_data')
        self.conn.commit()
        
        # Tabelle mit Primärschlüssel und passenden Spalten anlegen
        columns = ', '.join([f'"{col}" TEXT' for col in train_data.columns])
        self.cur.execute(f'''
            CREATE TABLE "MPSCDresden".train_data (
                train_id SERIAL PRIMARY KEY,
                {columns}
            )
        ''')
        self.conn.commit()
        
        # Daten einfügen
        if not train_data.empty:
            for _, row in train_data.iterrows():
                insert_cols = ', '.join([f'"{col}"' for col in train_data.columns])
                placeholders = ', '.join(['%s'] * len(train_data.columns))
                self.cur.execute(
                    f'INSERT INTO "MPSCDresden".train_data ({insert_cols}) VALUES ({placeholders})',
                    tuple(row)
                )
            self.conn.commit()

        # VALIDATION DATA - Erstelle Tabelle mit korrektem Primärschlüssel von Anfang an
        self.cur.execute('DROP TABLE IF EXISTS "MPSCDresden".validation_data')
        self.conn.commit()
        
        columns = ', '.join([f'"{col}" TEXT' for col in validation_data.columns])
        self.cur.execute(f'''
            CREATE TABLE "MPSCDresden".validation_data (
                validation_id SERIAL PRIMARY KEY,
                {columns}
            )
        ''')
        self.conn.commit()
        
        if not validation_data.empty:
            for _, row in validation_data.iterrows():
                insert_cols = ', '.join([f'"{col}"' for col in validation_data.columns])
                placeholders = ', '.join(['%s'] * len(validation_data.columns))
                self.cur.execute(
                    f'INSERT INTO "MPSCDresden".validation_data ({insert_cols}) VALUES ({placeholders})',
                    tuple(row)
                )
            self.conn.commit()
    
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
            
    def load_and_visualize_training_data(self):
        uri = QgsDataSourceUri()
        uri.setConnection(self.connection_params['host'], str(self.connection_params['port']), self.connection_params['dbname'], self.connection_params['user'], self.connection_params['password'])
        uri.setDataSource('MPSCDresden', 'train_data', 'geom', '', 'train_id')
        
        layer_name = 'Training Data'

        existing_layer = QgsProject.instance().mapLayersByName(layer_name)
        if existing_layer:
            QgsProject.instance().removeMapLayer(existing_layer[0])

        layer = QgsVectorLayer(uri.uri(False), layer_name, 'postgres')
        if not layer.isValid():
            QgsMessageLog.logMessage("Layer Training Data is not valid", level=Qgis.Critical)
            return
        
        QgsProject.instance().addMapLayer(layer)

        self.mapping_processor.categorize_and_colorize(layer)