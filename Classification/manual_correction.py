from .config_loader import get_schema, get_layer_name
import webbrowser
import random
from qgis.gui import QgsMapToolIdentifyFeature
from qgis.core import QgsProject, QgsMessageLog, Qgis, QgsCoordinateReferenceSystem, QgsCoordinateTransform

from .model_trainer import ModelTrainer

class ManualCorrection:
    """
    Unterstützt die manuelle Korrektur von Gebäudeklassifikationen in QGIS.

    Diese Klasse bietet Methoden zum:
    - Auswählen eines Gebäudes im QGIS-Layer
    - Öffnen der Position in Google Street View
    - Korrigieren der Klassifikation (sst) eines Gebäudes
    - Übertragen des Gebäudes in die Trainingsdaten
    - Retraining der Modelle nach manueller Korrektur
    """

    def __init__(self, iface, conn, cur, connection_params):
        """
        Initialisiert die ManualCorrection-Instanz.

        :param iface: QGIS-Interface
        :param conn: Datenbankverbindung (z.B. psycopg2 connection)
        :param cur: Datenbank-Cursor
        :param connection_params: Dictionary mit Verbindungsparametern
        """
        self.iface = iface
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        self.schema = get_schema()
        self.selected_db_filter_id = None

    def select_building(self):
        """
        Aktiviert das Auswahlwerkzeug in QGIS, um ein Gebäude im Layer 'Classification Data' auszuwählen.
        Öffnet nach Auswahl die Position in Google Street View.
        """
        layers = QgsProject.instance().mapLayersByName(get_layer_name('classification_data'))
        if not layers:
            QgsMessageLog.logMessage(f"Layer '{get_layer_name('classification_data')}' not found.", level=Qgis.Critical)
            return
        self.layer = layers[0]
        canvas = self.iface.mapCanvas()
        self.map_tool = QgsMapToolIdentifyFeature(canvas)
        self.map_tool.setLayer(layers[0])
        self.map_tool.featureIdentified.connect(self.feature_selected)
        canvas.setMapTool(self.map_tool)
        QgsMessageLog.logMessage("Select a building by clicking on it.", level=Qgis.Info)

    def feature_selected(self, feature):
        """
        Callback, wenn ein Gebäude im Layer ausgewählt wurde.
        Öffnet die Position in Google Street View.
        """
        self.selected_db_filter_id = feature['db_filter_id']
        self.selected_sst = feature['sst']
        geometry = feature.geometry()

        # Immer Schwerpunkt verwenden (vermeidet asPoint() Fehler bei Polygonen)
        centroid_geom = geometry.centroid()
        if centroid_geom.isEmpty():
            QgsMessageLog.logMessage("Selected geometry has no centroid.", level=Qgis.Warning)
            return

        # Transformiere die Koordinaten in WGS84
        source_crs = self.layer.crs()
        dest_crs = QgsCoordinateReferenceSystem('EPSG:4326')
        transform = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())
        try:
            point = transform.transform(centroid_geom.asPoint())
        except Exception as e:
            QgsMessageLog.logMessage(f"Coordinate transformation failed: {e}", level=Qgis.Critical)
            return
        
        lat = point.y()
        lon = point.x()
        street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        webbrowser.open(street_view_url)
        QgsMessageLog.logMessage(
            f"Selected building with db_filter_id {self.selected_db_filter_id} as {self.selected_sst}. Opened Google Street View at {lat}, {lon}.",
            level=Qgis.Info
        )

    def correct_building(self, new_sst):
        """
        Korrigiert die Klassifikation (sst) des aktuell ausgewählten Gebäudes und überträgt es in die Trainingsdaten.
        """
        if self.selected_db_filter_id and new_sst:
            try:
                self.cur.execute(f'''
                    UPDATE "{self.schema}".classification_data
                    SET sst = %s
                    WHERE db_filter_id = %s
                ''', (new_sst, self.selected_db_filter_id))
                self.conn.commit()
                QgsMessageLog.logMessage(f"Building with db_filter_id {self.selected_db_filter_id} corrected to sst {new_sst}.", level=Qgis.Info)
                self.update_training_data()
            except Exception as e:
                QgsMessageLog.logMessage(f"Error correcting building with db_filter_id {self.selected_db_filter_id}: {str(e)}", level=Qgis.Critical)
                raise e

    def update_training_data(self):
        """
        - Übernimmt sst aus classification_data nach citydb_filter
        - Markiert Quelle als 'Nachkartierung' (classification_source_id = 4)
        - Überträgt sst analog nach citydb_mirror (Quelle der Wahrheit, wie bei der CSV-Nachkartierung)
        - Entfernt Einträge aus classification_results und classification_data
        - Keine sofortige Zuweisung zu Train/Validation (Batch-Zuweisung separat)
        """
        try:
            self.cur.execute(f'''
                UPDATE "{self.schema}".citydb_filter cf
                SET sst = cd.sst,
                    classification_source_id = 4,
                    classification_source = 'Nachkartierung'
                FROM "{self.schema}".classification_data cd
                WHERE cf.db_filter_id = cd.db_filter_id
                  AND cf.db_filter_id = %s
            ''', (self.selected_db_filter_id,))
            self.conn.commit()
            QgsMessageLog.logMessage(f"citydb_filter.sst + source 'Nachkartierung' updated for db_filter_id {self.selected_db_filter_id}.", level=Qgis.Info)

            # citydb_mirror analog zu citydb_filter aktualisieren (gleiches Muster wie
            # citydb_extender.update_sst_from_csv). sst_sub wird zurückgesetzt, da die interaktive
            # Korrektur keine Unterklasse liefert.
            self.cur.execute(f'''
                UPDATE "{self.schema}".citydb_mirror cm
                SET sst = cf.sst,
                    sst_sub = NULL
                FROM "{self.schema}".citydb_filter cf
                WHERE cm.gml_id = cf.gml_id
                  AND cf.db_filter_id = %s
            ''', (self.selected_db_filter_id,))
            self.conn.commit()
            QgsMessageLog.logMessage(f"citydb_mirror.sst updated analogously for db_filter_id {self.selected_db_filter_id}.", level=Qgis.Info)

            self.cur.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = 'classification_results')",
                (self.schema,)
            )
            if self.cur.fetchone()[0]:
                self.cur.execute(f'DELETE FROM "{self.schema}".classification_results WHERE db_filter_id = %s', (self.selected_db_filter_id,))
                self.conn.commit()
            self.cur.execute(f'DELETE FROM "{self.schema}".classification_data WHERE db_filter_id = %s', (self.selected_db_filter_id,))
            self.conn.commit()
            QgsMessageLog.logMessage(f"Record {self.selected_db_filter_id} removed from classification_data.", level=Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error updating training data for db_filter_id {self.selected_db_filter_id}: {str(e)}", level=Qgis.Critical)
            raise e

    def assign_manual_corrections_to_train_validation(self):
        """
        Weist alle bislang korrigierten Datensätze (Quelle='Nachkartierung', ID=4) 80:20 train/validation zu.
        Es werden vollständige Kopien aus citydb_filter eingefügt.
        """
        try:
            # Kandidaten: verifiziert (Nachkartierung), sst vorhanden, noch nicht in Train/Validation
            self.cur.execute(f'''
                SELECT cf.db_filter_id
                FROM "{self.schema}".citydb_filter cf
                LEFT JOIN "{self.schema}".train_data t ON t.db_filter_id = cf.db_filter_id
                LEFT JOIN "{self.schema}".validation_data v ON v.db_filter_id = cf.db_filter_id
                WHERE cf.classification_source_id = 4
                  AND cf.sst IS NOT NULL
                  AND t.db_filter_id IS NULL
                  AND v.db_filter_id IS NULL
            ''')
            candidate_ids = [int(r[0]) for r in self.cur.fetchall()]
            if not candidate_ids:
                QgsMessageLog.logMessage("No new re-surveys found for assignment.", level=Qgis.Info)
                return

            # zufällige 80:20-Aufteilung
            random.Random(42).shuffle(candidate_ids)
            val_size = max(1, int(len(candidate_ids) * 0.2)) if len(candidate_ids) >= 5 else 0
            val_ids = candidate_ids[:val_size]
            train_ids = candidate_ids[val_size:]

            # Helfer: Spaltenliste aus Zieltabellen und SELECT-Liste aus citydb_filter
            def get_columns(table, exclude_cols):
                self.cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'MPSCDresden' AND table_name = %s
                    ORDER BY ordinal_position
                """, (table,))
                cols = [r[0] for r in self.cur.fetchall()]
                return [c for c in cols if c not in exclude_cols]

            def build_select_list(cols):
                parts = []
                for c in cols:
                    if c == 'results':
                        parts.append("NULL::varchar AS results")
                    else:
                        parts.append(f'cf."{c}"')
                return ', '.join(parts)

            if train_ids:
                cols_t = get_columns('train_data', exclude_cols=['train_id'])
                self.cur.execute(
                    f'''
                    INSERT INTO "{self.schema}".train_data ({', '.join(f'"{c}"' for c in cols_t)})
                    SELECT {build_select_list(cols_t)}
                    FROM "{self.schema}".citydb_filter cf
                    WHERE cf.db_filter_id = ANY(%s)
                    ''',
                    (train_ids,)
                )
            if val_ids:
                cols_v = get_columns('validation_data', exclude_cols=['validation_id'])
                self.cur.execute(
                    f'''
                    INSERT INTO "{self.schema}".validation_data ({', '.join(f'"{c}"' for c in cols_v)})
                    SELECT {build_select_list(cols_v)}
                    FROM "{self.schema}".citydb_filter cf
                    WHERE cf.db_filter_id = ANY(%s)
                    ''',
                    (val_ids,)
                )
            self.conn.commit()

            # training-Flags setzen
            if train_ids:
                self.cur.execute(
                    f'UPDATE "{self.schema}".citydb_filter SET training = %s WHERE db_filter_id = ANY(%s)',
                    ('t', train_ids)
                )
            if val_ids:
                self.cur.execute(
                    f'UPDATE "{self.schema}".citydb_filter SET training = %s WHERE db_filter_id = ANY(%s)',
                    ('v', val_ids)
                )
            self.conn.commit()

            QgsMessageLog.logMessage(f"Re-surveys assigned: {len(train_ids)} train, {len(val_ids)} validation.", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error assigning re-surveys: {str(e)}", level=Qgis.Critical)
            raise e

    def retrain_all_levels(self):
        """
        Retraining nach Nachkartierungen:
        - Batch-Zuweisung 80:20 in train/validation
        - Warmstart-Training (Modelle erweitern, kein Re-Splitting)
        """
        self.assign_manual_corrections_to_train_validation()
        model_trainer = ModelTrainer(self.conn, self.cur, self.connection_params)
        model_trainer.train_warm_start()
        model_trainer.save_label_encoders()
        QgsMessageLog.logMessage("Retraining (warm start) completed: models extended, encoders saved.", level=Qgis.Info)