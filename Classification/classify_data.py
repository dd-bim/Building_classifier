import os
import joblib
import pandas as pd
import numpy as np
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from qgis.core import QgsMessageLog, Qgis, QgsVectorLayer, QgsDataSourceUri, QgsProject
from .citydb_processor import CityDBProcessor
from .model_trainer import ModelTrainer, LabelEncoderManager
from .validate_model import ValidateModel
from .mapping_processor import MappingProcessor

class ClassifyData:
    """
    Führt die Klassifikation von Gebäudedaten durch und verwaltet die zugehörigen Datenbankoperationen.

    Diese Klasse enthält Methoden zum Laden, Klassifizieren, Speichern und Visualisieren von Gebäudedaten
    sowie zur Generierung von Berichten über die Klassifikationsergebnisse.
    """

    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert die ClassifyData-Instanz.

        :param conn: Datenbankverbindung (z.B. psycopg2 connection)
        :param cur: Datenbank-Cursor
        :param connection_params: Dictionary mit Verbindungsparametern
        """
        self.conn = conn
        self.cur = cur
        self.connection_params = connection_params
        
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        
        self.model_dir = os.path.join(os.path.dirname(__file__), config.get('Paths', 'model_dir'))

        self.label_encoder_manager = LabelEncoderManager(self.model_dir)
        self.label_encoders = self.label_encoder_manager.get_label_encoders()
        
        self.citydb_processor = CityDBProcessor(conn, cur, connection_params)
        self.model_trainer = ModelTrainer(conn, cur, connection_params)
        self.model_validator = ValidateModel(conn, cur, connection_params)
        self.mapping_processor = MappingProcessor(conn, cur, connection_params)
        
    def set_label_encoders(self, label_encoders):
        """
        Setzt die LabelEncoder für die Klassifikation.
        """
        self.label_encoders = label_encoders

    def ensure_level_columns(self):
        """
        Stellt sicher, dass alle benötigten Level- und Confidence-Spalten sowie Indizes in der Tabelle classification_data existieren.
        """
        level_columns = ['1', '11', '12', '121', '122', '112', '113', '114', '111', '1111']

        for column in level_columns:
            self.cur.execute(f'''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name = 'classification_data' 
                        AND column_name = '{column}'
                    ) THEN
                        ALTER TABLE "MPSCDresden".classification_data 
                        ADD COLUMN "{column}" VARCHAR;
                    END IF;
                END $$;
            ''')
        
            # Sicherstellen, dass die Confidence-Spalte existiert
            confidence_column = f"{column}_confidence"
            self.cur.execute(f'''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name = 'classification_data' 
                        AND column_name = '{confidence_column}'
                    ) THEN
                        ALTER TABLE "MPSCDresden".classification_data 
                        ADD COLUMN "{confidence_column}" FLOAT;
                    END IF;
                END $$;
            ''')
        self.conn.commit()

        index_columns = ['db_filter_id', 'sst'] + level_columns
        for column in index_columns:
            self.cur.execute(f'''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_indexes
                        WHERE tablename = 'classification_data'
                        AND indexname = 'idx_classification_data_{column}'
                    ) THEN
                        CREATE INDEX idx_classification_data_{column}
                        ON "MPSCDresden".classification_data ("{column}");
                    END IF;
                END $$;
            ''')
        self.conn.commit()

        # Gesamtconfidence-Spalte anlegen, falls nicht vorhanden
        self.cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'classification_data'
                    AND column_name = 'overall_confidence'
                ) THEN
                    ALTER TABLE "MPSCDresden".classification_data
                    ADD COLUMN overall_confidence FLOAT;
                END IF;
            END $$;
        """)
        self.conn.commit()

        QgsMessageLog.logMessage("Level-Spalten und Indizes in der Datenbank sichergestellt.", level=Qgis.Info)

    def load_classification_data(self):
        """
        Lädt die Klassifikationsdaten aus der Datenbank als DataFrame.
        """
        return self.model_trainer.load_data_from_db('"MPSCDresden".classification_data')
    
    def create_result_relation(self):
        """
        Erstellt die Ergebnistabelle für finale Klassifikationsergebnisse, falls sie noch nicht existiert.
        """
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS "MPSCDresden".classification_results (
                result_id SERIAL PRIMARY KEY,
                db_filter_id INTEGER REFERENCES "MPSCDresden".classification_data(db_filter_id),
                level VARCHAR(10),
                predicted_class VARCHAR(255),
                confidence FLOAT,
                classification_source VARCHAR(50),
                created_at TIMESTAMP DEFAULT NOW()
                );
        ''')
        self.conn.commit()

    def classify(self):
        """
        Führt die vollständige Klassifikation der Gebäudedaten durch.

        - Setzt alle Klassifikationsspalten zurück
        - Führt die Klassifikation für alle Level durch
        - Speichert die Ergebnisse in der Datenbank
        - Generiert einen Confidence-Report
        - Visualisiert die Ergebnisse in QGIS
        """
        self.ensure_level_columns()
        level_columns = ['1', '11', '12', '121', '122', '112', '113', '114', '111', '1111']

        # Setze alle Level- und Confidence-Spalten auf NULL
        set_null_sql = ', '.join([f'"{col}" = NULL, "{col}_confidence" = NULL' for col in level_columns])
        self.cur.execute(f'''
            UPDATE "MPSCDresden".classification_data
            SET {set_null_sql}
        ''')
        self.cur.execute('UPDATE "MPSCDresden".classification_data SET "sst" = NULL')
        self.conn.commit()

        level_groups = [
            [('1', ['M', 'E', 'Other'])],
            [('11', ['MR', 'ME', 'ER', 'EE']), ('12', ['HH', 'LW'])],
            [('121', ['HH3', 'HH4']), ('122', ['LW1', 'LW2', 'LW3', 'LW7'])],
            [('112', ['ME2', 'ME3', 'ME4', 'ME5', 'ME6', 'ME7']), ('113', ['ER2', 'ER3', 'ER4', 'ER5', 'ER7']), ('114', ['EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE7']), ('111', ['MR2', 'MR3', 'MR4', 'MR5', 'MR6', 'MR7'])],
            [('1111', ['MRO2', 'MRO3', 'MRO4', 'MRO7', 'MRG2', 'MRG3', 'MRG4', 'MRG7'])]
        ]

        max_iterations = 5
        iteration = 0
        num_classified_prev = -1

        while iteration < max_iterations:
            QgsMessageLog.logMessage(f"Starte Klassifikationsiteration {iteration+1}", level=Qgis.Info)

            classification_data = self.load_classification_data()
            if classification_data.empty:
                QgsMessageLog.logMessage("Keine Klassifizierungsdaten vorhanden.", level=Qgis.Critical)
                return

            current_data = classification_data.copy()
            current_data['sst'] = None

            #level_columns = ['1', '11', '12', '121', '122', '112', '113', '114', '111', '1111']

            for group in level_groups:
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    futures = {executor.submit(self.process_level, level, current_data, level_columns): level for level, target_names in group}
                        
                    for future in as_completed(futures):
                        level = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            QgsMessageLog.logMessage(f"Fehler bei der Klassifikation für {level}: {str(e)}", level=Qgis.Critical)

            for index, row in current_data.iterrows():
                for level in reversed(level_columns):
                    if pd.notnull(row[level]):
                        current_data.at[index, 'sst'] = row[level]
                        self.cur.execute(f'''
                            UPDATE "MPSCDresden".classification_data
                            SET "sst" = %s
                            WHERE db_filter_id = %s
                        ''', (row[level], row['db_filter_id']))
                        break
            self.conn.commit()
            # --- Ende bestehender Klassifikationscode ---

            # Prüfe, ob sich noch etwas ändert
            classification_data_after = self.load_classification_data()
            num_classified_now = classification_data_after['sst'].notnull().sum()
            QgsMessageLog.logMessage(f"Iteration {iteration+1}: {num_classified_now} Gebäude klassifiziert.", level=Qgis.Info)
            if num_classified_now == num_classified_prev:
                QgsMessageLog.logMessage("Keine neuen Klassifikationen mehr, beende Iteration.", level=Qgis.Info)
                break
            num_classified_prev = num_classified_now

            # Nachbarschaftsattribute neu berechnen
            self.citydb_processor.calculate_neighbors()

            iteration += 1

        QgsMessageLog.logMessage("Iterative Klassifizierung abgeschlossen.", level=Qgis.Info)
        self.load_and_visualize_classification_data()
        self.generate_confidence_report(current_data, level_columns)
        self.save_classification_results(current_data, level_columns)

        # Gesamtconfidence als Produkt der Confidences auf dem Pfad zur sst-Klasse
        update_queries = []
        for idx, row in current_data.iterrows():
            sst = row.get("sst")
            if pd.notnull(sst):
                confidences = []
                found = False
                for level in level_columns:
                    val = row.get(level)
                    conf = row.get(f"{level}_confidence")
                    if pd.notnull(val) and pd.notnull(conf):
                        confidences.append(conf)
                        if val == sst:
                            found = True
                            break
                if found and confidences:
                    overall_conf = float(np.prod(confidences) ** (1/len(confidences)))
                else:
                    overall_conf = None

                # Quelle bestimmen
                if pd.notnull(row.get("building_age")):
                    source = "Baualter"
                    source_id = 2
                else:
                    source = "Modell"
                    source_id = 3

                update_queries.append((overall_conf, source_id, source, row['db_filter_id']))

        if update_queries:
            self.cur.executemany("""
                UPDATE "MPSCDresden".classification_data
                SET overall_confidence = %s,
                    classification_source_id = %s,
                    classification_source = %s
                WHERE db_filter_id = %s
            """, update_queries)
            self.conn.commit()
            QgsMessageLog.logMessage(f"Gesamtconfidence und Quelle für {len(update_queries)} Gebäude gespeichert.", level=Qgis.Info)
        else:
            QgsMessageLog.logMessage("Keine Gesamtconfidence/Quelle zu speichern.", level=Qgis.Warning)

    def process_level(self, level, current_data, level_columns):
        """
        Führt die Klassifikation für ein bestimmtes Level durch und aktualisiert die entsprechenden Spalten im DataFrame und in der Datenbank.
        """
        QgsMessageLog.logMessage(f"Klassifizierung für {level} gestartet.", level=Qgis.Info)

        try:
            current_column = level
            confidence_column = f"{level}_confidence"

            if level == '1':
                current_data[current_column], current_data[confidence_column] = self.run_classification(level, current_data)

            elif level == '11':
                QgsMessageLog.logMessage(f"Starte regelbasierte Zuordnung für {level}. Anzahl der Datensätze: {current_data.shape[0]}", level=Qgis.Info)
                current_data[current_column] = current_data.apply(
                    lambda row: 'MR' if row['1'] == 'M' and row['proximity'] == 'R' else (
                        'ME' if row['1'] == 'M' else (
                        'ER' if row['1'] == 'E' and row['proximity'] == 'R' else (
                        'EE' if row['1'] == 'E' else None))),
                    axis=1
                )
                # Konfidenz für die regelbasierte Zuordnung
                current_data[confidence_column] = 1.0
                QgsMessageLog.logMessage(f"Regelbasierte Zuordnung abgeschlossen für {level}. Werteverteilung: {current_data[current_column].value_counts().to_dict()}", level=Qgis.Info)

            elif level == '12':
                filtered_data = current_data[current_data['1'] == 'Other']
                QgsMessageLog.logMessage(f"Anzahl der Datensätze für {level} nach Filterung: {filtered_data.shape[0]}", level=Qgis.Info)
                if not filtered_data.empty:
                    current_data.loc[filtered_data.index, current_column], current_data.loc[filtered_data.index, confidence_column] = self.run_classification(level, filtered_data)

            elif level in ['121', '122', '111', '112', '113', '114']:
                # Bestimme den korrekten Filter basierend auf der Hierarchie
                if level == '121':
                    filtered_data = current_data[current_data['12'] == 'HH']
                    prefix = 'HH'
                elif level == '122':
                    filtered_data = current_data[current_data['12'] == 'LW']
                    prefix = 'LW'
                elif level == '111':
                    filtered_data = current_data[current_data['11'] == 'MR']
                    prefix = 'MR'
                elif level == '112':
                    filtered_data = current_data[current_data['11'] == 'ME']
                    prefix = 'ME'
                elif level == '113':
                    filtered_data = current_data[current_data['11'] == 'ER']
                    prefix = 'ER'
                elif level == '114':
                    filtered_data = current_data[current_data['11'] == 'EE']
                    prefix = 'EE'
                    
                QgsMessageLog.logMessage(f"Anzahl der Datensätze für {level} nach Filterung: {filtered_data.shape[0]}", level=Qgis.Info)
                
                if not filtered_data.empty:
                    # Teile die Daten basierend auf dem Vorhandensein von building_age
                    age_mask = filtered_data['building_age'].notna()
                    direct_data = filtered_data[age_mask]
                    model_data = filtered_data[~age_mask]

                    # Verarbeite Daten mit building_age direkt
                    if not direct_data.empty:
                        QgsMessageLog.logMessage(f"Verarbeite {len(direct_data)} Datensätze mit building_age für {level}.", level=Qgis.Info)
                        
                        for idx, row in direct_data.iterrows():
                            result = self.handle_direct_building_age_assignment(row, prefix=prefix)
                            
                            # Fall 1: Einzelnes Ergebnis (z.B. "MR3")
                            if not isinstance(result, list):
                                current_data.at[idx, current_column] = result
                                current_data.at[idx, confidence_column] = 1.0
                            
                            # Fall 2: Mehrere mögliche Ergebnisse (z.B. ["MR1", "MR2"])
                            else:
                                # Wir müssen hier das Modell für die engere Auswahl verwenden
                                # Erstellen wir eine temporäre Kopie der Daten für diesen einen Datensatz
                                temp_data = pd.DataFrame([row.to_dict()])
                                
                                if not temp_data.empty:
                                    # Führe die Klassifikation durch
                                    pred, conf = self.run_classification(level, temp_data)
                                    
                                    if pred is not None and len(pred) > 0:
                                        # Überprüfe, ob die Vorhersage in den möglichen Ergebnissen ist
                                        if pred[0] in result:
                                            current_data.at[idx, current_column] = pred[0]
                                            current_data.at[idx, confidence_column] = conf[0] if conf is not None else 0.5
                                        else:
                                            # Falls nicht, wähle das erste mögliche Ergebnis
                                            current_data.at[idx, current_column] = result[0]
                                            current_data.at[idx, confidence_column] = 0.5  # Mittlere Konfidenz
                                    else:
                                        # Fallback: Erstes Element der Liste verwenden
                                        current_data.at[idx, current_column] = result[0]
                                        current_data.at[idx, confidence_column] = 0.5
                        
                        QgsMessageLog.logMessage(f"{len(direct_data)} direkte Zuordnungen für {level} verarbeitet.", level=Qgis.Info)

                    # Verarbeite Daten ohne building_age mit dem Modell
                    if not model_data.empty:
                        QgsMessageLog.logMessage(f"Klassifiziere {len(model_data)} Datensätze mit Modell für {level}.", level=Qgis.Info)
                        current_data.loc[model_data.index, current_column], current_data.loc[model_data.index, confidence_column] = self.run_classification(level, model_data)

            elif level == '1111':
                # Filtere Daten, die für die weitere Klassifikation in Frage kommen
                filtered_data = current_data[current_data['111'].isin(['MR2', 'MR3', 'MR4', 'MR7'])]
                QgsMessageLog.logMessage(f"Anzahl der Datensätze für {level} nach Filterung: {filtered_data.shape[0]}", level=Qgis.Info)
                
                if not filtered_data.empty:
                    # Für jeden Datensatz die Baualtersstufe beibehalten und nur zwischen O und G entscheiden
                    for idx, row in filtered_data.iterrows():
                        mr_level = row['111']  # z.B. 'MR3'
                        if pd.isna(mr_level):
                            continue
                        
                        # Extrahiere die Nummer aus MR-Level (z.B. '3' aus 'MR3')
                        age_number = mr_level[2:]
                        
                        # Erstelle temporäre Daten für die Modellvorhersage
                        temp_data = pd.DataFrame([row.to_dict()])
                        
                        # Führe Modellklassifikation durch, aber nur als Entscheidungshilfe
                        pred, conf = self.run_classification(level, temp_data)
                        
                        if pred is not None and len(pred) > 0:
                            predicted_class = pred[0]
                            # Prüfe ob die Vorhersage eine O- oder G-Variante ist
                            is_o_variant = 'MRO' in predicted_class
                            is_g_variant = 'MRG' in predicted_class
                            
                            # Erstelle die korrekten Klassen mit der korrekten Baualtersstufe
                            correct_o_class = f"MRO{age_number}"
                            correct_g_class = f"MRG{age_number}"
                            
                            # Treffe Entscheidung basierend auf der Modellvorhersage, aber behalte die Baualtersstufe bei
                            if is_o_variant:
                                current_data.at[idx, current_column] = correct_o_class
                            elif is_g_variant:
                                current_data.at[idx, current_column] = correct_g_class
                            else:
                                # Fallback: Wähle O-Variante, wenn die Vorhersage weder O noch G ist
                                current_data.at[idx, current_column] = correct_o_class
                                QgsMessageLog.logMessage(f"Für {mr_level} wurde eine unerwartete Klasse vorhergesagt: {predicted_class}. Verwende Fallback {correct_o_class}.", level=Qgis.Warning)
                            

                            # Übernehme die Konfidenz oder setze sie auf einen mittleren Wert
                            current_data.at[idx, confidence_column] = conf[0] if conf is not None else 0.5
                        else:
                            # Bei Fehlern default zu O-Variante
                            correct_o_class = f"MRO{age_number}"
                            current_data.at[idx, current_column] = correct_o_class
                            current_data.at[idx, confidence_column] = 0.5
                            QgsMessageLog.logMessage(f"Fehler bei der Klassifikation für {mr_level} in Level {level}. Verwende Fallback {correct_o_class}.", level=Qgis.Warning)
            
            # Aktualisiere die Datenbank mit den neuen Klassifikationen
            self.batch_update(current_data, current_column, confidence_column)

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler bei der Klassifikation für {level}: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(f"Stacktrace: {traceback.format_exc()}", level=Qgis.Critical)
            
    def handle_direct_building_age_assignment(self, row, prefix):
        """
        Erzeugt die Zielklasse(n) für direkte Baualterszuweisungen.
        """
        age_value = str(row.get("building_age")).strip()
        if "/" in age_value:
            return [f"{prefix}{part.strip()}" for part in age_value.split("/")]
        else:
            return f"{prefix}{age_value}"
    
    def batch_update(self, data, column, confidence_column):
        """
        Führt ein Batch-Update der Klassifikationsergebnisse für ein Level in der Datenbank durch.
        """
        update_queries = [
            (row[column], row[confidence_column], row['db_filter_id']) for _, row in data.iterrows() if pd.notnull(row[column])
        ]
        
        if update_queries:
            self.cur.executemany(f'''
                UPDATE "MPSCDresden".classification_data
                SET "{column}" = %s, "{confidence_column}" = %s
                WHERE db_filter_id = %s
            ''', update_queries)
            self.conn.commit()
            
            QgsMessageLog.logMessage(f"Batch-Updates für {column} und {confidence_column} abgeschlossen. Anzahl: {len(update_queries)}", level=Qgis.Info)
        else:
            QgsMessageLog.logMessage(f"Keine gültigen Daten zum Aktualisieren für {column} und {confidence_column}.", level=Qgis.Warning)
    
    def prepare_data(self, data, level):
        """
        Bereitet die Eingabedaten für die Modellklassifikation eines Levels vor (Feature-Auswahl und Label-Encoding).
        """
        X = data.copy()

        # Lade das Modell, um die erwarteten Features zu erhalten
        model_path = os.path.join(self.model_dir, f'model_{level}.pkl')
        if not os.path.exists(model_path):
            QgsMessageLog.logMessage(f"Modell für {level} nicht gefunden: {model_path}", level=Qgis.Critical)
            return pd.DataFrame()

        model = joblib.load(model_path)

        # Überprüfe, ob das Modell die erwarteten Features speichert
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
        else:
            QgsMessageLog.logMessage(f"Das Modell für {level} enthält keine Informationen zu den erwarteten Features.", level=Qgis.Critical)
            return pd.DataFrame()

        # Entferne die Zielspalte, falls sie in den Daten enthalten ist
        target_column = 'sst'
        if target_column in X.columns:
            X = X.drop(columns=[target_column])

        # Filtere die Daten basierend auf den erwarteten Features
        missing_features = [feature for feature in expected_features if feature not in X.columns]
        if missing_features:
            QgsMessageLog.logMessage(f"Fehlende Features für {level}: {missing_features}", level=Qgis.Critical)
            return pd.DataFrame()

        X = X[expected_features]

        # Transformiere kategorische Features
        if self.label_encoders:
            for feature in ['roof_type', 'development_type_code', 'neighbor_majority_class', 'building_age']:
                if feature in X.columns and feature in self.label_encoders:
                    X[feature] = X[feature].apply(
                        lambda x: self.label_encoders[feature].transform([x])[0] if pd.notna(x) and x in self.label_encoders[feature].classes_ else -1
                    )

        return X
    
    def run_classification(self, level, data):
        """
        Führt die Modellklassifikation für ein bestimmtes Level durch und gibt Vorhersagen und Konfidenzen zurück.
        """
        X = self.prepare_data(data, level)
        
        if X.empty:
            QgsMessageLog.logMessage(f"Keine Eingabedaten für {level}.", level=Qgis.Warning)
            return None, None

        model_path = os.path.join(self.model_dir, f'model_{level}.pkl')
        if not os.path.exists(model_path):
            QgsMessageLog.logMessage(f"Modell für {level} nicht gefunden: {model_path}", level=Qgis.Critical)
            return None, None

        model = joblib.load(model_path)

        # Vorhersagen durchführen
        y_pred = model.predict(X)

        try:
            y_pred_proba = model.predict_proba(X)
            y_confidence = y_pred_proba.max(axis=1)
        except AttributeError:
            y_confidence = None
            QgsMessageLog.logMessage(f"Das Modell für {level} unterstützt keine Wahrscheinlichkeitsvorhersagen.", level=Qgis.Warning)
                    
        return y_pred, y_confidence
    
    def generate_confidence_report(self, current_data, level_columns):
        """
        Erstellt einen Bericht über die Verteilung der Konfidenzwerte für jede Klassifikationsstufe.

        :param current_data: DataFrame mit den aktuellen Klassifikationsergebnissen
        :param level_columns: Liste der Level-Spalten
        """
        try:
            # Lese den Pfad aus der Konfiguration
            config = configparser.ConfigParser()
            config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
            output_path = os.path.join(os.path.dirname(__file__), config.get('Paths', 'confidence_report'))

            QgsMessageLog.logMessage("Generiere Confidence-Report...", level=Qgis.Info)
            
            confidence_ranges = [
                (0.9, 1.0, "> 0.9"),
                (0.8, 0.9, "0.8 - 0.9"),
                (0.7, 0.8, "0.7 - 0.8"),
                (0.6, 0.7, "0.6 - 0.7"),
                (0.5, 0.6, "0.5 - 0.6"),
                (0.0, 0.5, "< 0.5")
            ]

            report_lines = []
            report_lines.append("Confidence Report\n")
            report_lines.append("=" * 50 + "\n")

            for level in level_columns:
                confidence_column = f"{level}_confidence"
                if confidence_column not in current_data.columns or level not in current_data.columns:
                    continue

                report_lines.append(f"Level: {level}\n")
                report_lines.append("-" * 50 + "\n")

                for target_value in current_data[level].dropna().unique():
                    report_lines.append(f"  Target value: {target_value}\n")
                    filtered_data = current_data[current_data[level] == target_value]

                    for lower, upper, label in confidence_ranges:
                        count = filtered_data[(filtered_data[confidence_column] > lower) & (filtered_data[confidence_column] <= upper)].shape[0]
                        report_lines.append(f"    {label}: {count}\n")

                report_lines.append("\n")

            # Schreibe den Report in die Datei
            with open(output_path, "w", encoding="utf-8") as file:
                file.writelines(report_lines)

            QgsMessageLog.logMessage(f"Confidence-Report erfolgreich erstellt: {output_path}", level=Qgis.Info)

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Generieren des Confidence-Reports: {str(e)}", level=Qgis.Critical)
            
    def save_classification_results(self, current_data, level_columns):
        """
        Speichert die finalen Klassifikationsergebnisse (sst) in die Ergebnistabelle classification_results.
        """
        try:
            # Lösche alle bestehenden Ergebnisse, um Duplikate zu vermeiden
            self.cur.execute('DELETE FROM "MPSCDresden".classification_results')
            self.conn.commit()
            
            QgsMessageLog.logMessage(f"Speichere {len(current_data)} finale sst-Datensätze in classification_results", level=Qgis.Info)
            
            insert_values = []
            for idx, row in current_data.iterrows():
                sst = row.get("sst")
                if pd.notnull(sst):
                    overall_conf = row.get("overall_confidence")
                    # Quelle bestimmen
                    if pd.notnull(row.get("building_age")):
                        source = "Baualter"
                    else:
                        source = "Modell"
                    insert_values.append((
                        row['db_filter_id'],
                        "sst",
                        sst,
                        overall_conf,
                        source
                    ))
    
            if insert_values:
                self.cur.executemany("""
                    INSERT INTO "MPSCDresden".classification_results
                    (db_filter_id, level, predicted_class, confidence, classification_source)
                    VALUES (%s, %s, %s, %s, %s)
                """, insert_values)
                self.conn.commit()
                QgsMessageLog.logMessage(f"{len(insert_values)} finale Klassifikationsergebnisse (sst) erfolgreich gespeichert.", level=Qgis.Info)
            else:
                QgsMessageLog.logMessage("Keine finale Klassifikationsergebnisse (sst) zum Speichern vorhanden.", level=Qgis.Warning)
    
        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Speichern der Klassifikationsergebnisse: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(f"Stacktrace: {traceback.format_exc()}", level=Qgis.Critical)
            self.conn.rollback()

    def load_and_visualize_classification_data(self):
        """
        Lädt die finale Klassifikationstabelle als QGIS-Layer und färbt sie nach Kategorie ein.
        """
        uri = QgsDataSourceUri()
        uri.setConnection(self.connection_params['host'], str(self.connection_params['port']), self.connection_params['dbname'], self.connection_params['user'], self.connection_params['password'])
        uri.setDataSource('MPSCDresden', 'classification_data', 'geom', '', 'db_filter_id')
        
        layer_name = 'Classification Data'
        
        existing_layer = QgsProject.instance().mapLayersByName(layer_name)
        if existing_layer:
            QgsProject.instance().removeMapLayer(existing_layer[0])
        
        layer = QgsVectorLayer(uri.uri(False), layer_name, 'postgres')
        if not layer.isValid():
            QgsMessageLog.logMessage("Layer Classification Data is not valid", level=Qgis.Critical)
            return
        
        QgsProject.instance().addMapLayer(layer)
        
        self.mapping_processor.categorize_and_colorize(layer)