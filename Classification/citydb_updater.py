import pandas as pd
import os
from qgis.core import QgsMessageLog, Qgis
from .config_loader import get_config

class CityDBUpdater:
    """
    Aktualisiert die 3DCityDB Version 5 mit generischen Attributen aus den Klassifikationsergebnissen.

    Diese Klasse bietet Methoden zum:
    - Laden der Klassifikationsergebnisse aus der classification_data-Tabelle
    - Einfügen von generischen sst-Attributen in die property-Tabelle der 3DCityDB
    - Aktualisierung bestehender sst-Attribute in der 3DCityDB
    - Optionale Filterung nach individueller Confidence-Schwellwert
    """

    def __init__(self, conn, cur, connection_params, confidence_threshold=None):
        """
        Initialisiert den CityDBUpdater mit DB-Verbindung und Konfigurationsparametern.

        :param conn: Datenbankverbindung (z.B. psycopg2 connection)
        :param cur: Datenbank-Cursor
        :param connection_params: Dictionary mit Verbindungsparametern
        :param confidence_threshold: Optionaler Schwellwert für die individuelle Confidence (z.B. 0.8 für 80%)
                                    Nur Klassifikationen >= diesem Wert werden übertragen
        """
        self.conn = conn
        self.cur = cur
        self.connection_params = connection_params
        self.confidence_threshold = confidence_threshold

        self.config = get_config()
        self.schema = self.config.get('Database', 'schema')

    def load_classification_results(self):
        """
        Lädt die Klassifikationsergebnisse (gml_id, sst, confidence) aus der citydb_filter-Tabelle.
        Wendet optional einen Confidence-Schwellwert an.
        :return: DataFrame mit den Spalten gml_id, sst, confidence
        """
        # Basis-Query aus citydb_filter
        query = f'''
            SELECT gml_id, sst, confidence
            FROM "{self.schema}".citydb_filter
            WHERE sst IS NOT NULL
        '''
        
        # Optionale Confidence-Filterung
        if self.confidence_threshold is not None:
            query += f' AND confidence >= {self.confidence_threshold}'
            QgsMessageLog.logMessage(f"Confidence threshold activated: >= {self.confidence_threshold}", level=Qgis.Info)
        
        self.cur.execute(query)
        rows = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        
        threshold_info = f" (Confidence >= {self.confidence_threshold})" if self.confidence_threshold is not None else ""
        QgsMessageLog.logMessage(f"Classification results loaded{threshold_info}: {len(df)} records", level=Qgis.Info)

        return df

    def get_building_feature_ids(self):
        """
        Lädt die Zuordnung von gml_id zu feature_id aus der 3DCityDB.
        :return: DataFrame mit gml_id und feature_id
        """
        query = '''
            SELECT p.feature_id, p.val_uri as gml_id
            FROM citydb.property p
            WHERE p.name = 'externalReference'
            AND p.val_uri IS NOT NULL
        '''
        self.cur.execute(query)
        rows = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        QgsMessageLog.logMessage(f"Feature IDs found in 3DCityDB: {len(df)}", level=Qgis.Info)
        return df

    def update_citydb_properties(self):
        """
        Fügt sst-Attribute als generische Properties in die 3DCityDB property-Tabelle ein.
        Aktualisiert bestehende sst-Attribute oder erstellt neue.
        Berücksichtigt optional einen Confidence-Schwellwert.
        Optimiert mit Batch-Operationen für bessere Performance.
        """
        try:
            QgsMessageLog.logMessage("Starting CityDB property update...", level=Qgis.Info)
            
            # Lade Klassifikationsergebnisse (mit optionalem Confidence-Filter)
            classification_results = self.load_classification_results()
            if classification_results.empty:
                if self.confidence_threshold is not None:
                    QgsMessageLog.logMessage(f"No classification results found with confidence >= {self.confidence_threshold}.", level=Qgis.Warning)
                else:
                    QgsMessageLog.logMessage("No classification results found.", level=Qgis.Warning)
                return

            # Lade Feature-IDs aus der 3DCityDB
            building_features = self.get_building_feature_ids()
            if building_features.empty:
                QgsMessageLog.logMessage("No building feature IDs found in the 3DCityDB.", level=Qgis.Warning)
                return

            # Merge Klassifikationsergebnisse mit Feature-IDs
            merged_data = pd.merge(
                classification_results, 
                building_features, 
                on='gml_id', 
                how='inner'
            )

            if merged_data.empty:
                QgsMessageLog.logMessage("No matches found between classification results and 3DCityDB features.", level=Qgis.Warning)
                return

            unmatched = len(classification_results) - len(merged_data)
            unmatched_info = f" ({unmatched} skipped due to no matching feature ID)" if unmatched > 0 else ""
            QgsMessageLog.logMessage(f"{len(merged_data)} buildings found for processing{unmatched_info}", level=Qgis.Info)

            # BATCH-OPTIMIERUNG: Lade alle existierenden sst-Properties auf einmal
            feature_ids = tuple(merged_data['feature_id'].tolist())
            
            existing_properties_query = """
            SELECT feature_id, id, val_string 
            FROM citydb.property 
            WHERE feature_id = ANY(%s) AND name = 'sst'
            """
            self.cur.execute(existing_properties_query, (list(feature_ids),))
            existing_rows = self.cur.fetchall()
            
            # Erstelle Lookup für existierende Properties
            existing_properties = {}
            for feature_id, prop_id, val_string in existing_rows:
                existing_properties[feature_id] = (prop_id, val_string)
            
            QgsMessageLog.logMessage(f"Existing sst properties: {len(existing_properties)}", level=Qgis.Info)

            # Bereite Batch-Updates und Inserts vor
            updates_batch = []
            inserts_batch = []
            skipped_low_confidence = 0
            skipped_unchanged = 0

            for index, row in merged_data.iterrows():
                feature_id = row['feature_id']
                sst = row['sst']
                confidence = row['confidence']

                # Zusätzliche Confidence-Prüfung
                if self.confidence_threshold is not None and confidence < self.confidence_threshold:
                    skipped_low_confidence += 1
                    continue

                if feature_id in existing_properties:
                    # Aktualisierung nur, wenn sich der Wert tatsächlich ändert
                    property_id, current_value = existing_properties[feature_id]
                    if current_value != sst:
                        updates_batch.append((sst, property_id))
                    else:
                        skipped_unchanged += 1
                else:
                    # Neuer Eintrag erforderlich
                    inserts_batch.append((feature_id, 'sst', sst))

            # BATCH-UPDATE: Alle Updates auf einmal
            if updates_batch:
                update_query = """
                UPDATE citydb.property
                SET val_string = %s
                WHERE id = %s
                """
                self.cur.executemany(update_query, updates_batch)

            # BATCH-INSERT: Alle neuen Einträge auf einmal
            if inserts_batch:
                # Ermittle Startwert für neue IDs
                self.cur.execute("SELECT COALESCE(MAX(id), 0) FROM citydb.property")
                start_id = self.cur.fetchone()[0] + 1

                # Füge IDs zu den Insert-Daten hinzu
                inserts_with_ids = [
                    (start_id + i, feature_id, name, val_string)
                    for i, (feature_id, name, val_string) in enumerate(inserts_batch)
                ]

                insert_query = """
                INSERT INTO citydb.property (id, feature_id, name, val_string)
                VALUES (%s, %s, %s, %s)
                """
                self.cur.executemany(insert_query, inserts_with_ids)

            self.conn.commit()

            summary_message = (
                f"3DCityDB property table successfully updated:\n"
                f"- Buildings processed: {len(merged_data)}\n"
                f"- New sst attributes: {len(inserts_batch)}\n"
                f"- Updated sst attributes: {len(updates_batch)}\n"
                f"- Already up to date (unchanged): {skipped_unchanged}"
            )

            if self.confidence_threshold is not None:
                summary_message += f"\n- Skipped (confidence below {self.confidence_threshold}): {skipped_low_confidence}"

            QgsMessageLog.logMessage(summary_message, level=Qgis.Success)

        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error updating 3DCityDB properties: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            raise e