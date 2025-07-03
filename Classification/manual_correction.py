import webbrowser
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
        self.selected_db_filter_id = None

    def select_building(self):
        """
        Aktiviert das Auswahlwerkzeug in QGIS, um ein Gebäude im Layer 'Classification Data' auszuwählen.
        Öffnet nach Auswahl die Position in Google Street View.
        """
        layers = QgsProject.instance().mapLayersByName('Classification Data')
        if not layers:
            QgsMessageLog.logMessage("Layer 'Classification Data' not found.", level=Qgis.Critical)
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
        if geometry.isMultipart():
            geometry = geometry.centroid()
            
        # Transformiere die Koordinaten in WGS84
        source_crs = self.layer.crs()
        dest_crs = QgsCoordinateReferenceSystem('EPSG:4326')
        transform = QgsCoordinateTransform(source_crs, dest_crs, QgsProject.instance())
        point = transform.transform(geometry.asPoint())
        
        lat = point.y()
        lon = point.x()
        street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        webbrowser.open(street_view_url)
        QgsMessageLog.logMessage(f"Selected building with db_filter_id {self.selected_db_filter_id} as {self.selected_sst}. Opened Google Street View at {lat}, {lon}.", level=Qgis.Info)

    def correct_building(self, new_sst):
        """
        Korrigiert die Klassifikation (sst) des aktuell ausgewählten Gebäudes und überträgt es in die Trainingsdaten.
        """
        if self.selected_db_filter_id and new_sst:
            try:
                self.cur.execute('''
                    UPDATE "MPSCDresden".classification_data
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
        Überträgt das aktuell ausgewählte Gebäude aus classification_data nach train_data,
        löscht es aus classification_data und aktualisiert citydb_filter mit neuem sst-Wert und training='t'.
        """
        try:
            # Abrufen der Spaltennamen aus train_data
            self.cur.execute('''
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'train_data'
            ''')
            train_data_columns = {row[0] for row in self.cur.fetchall()}

            # Abrufen der Spaltennamen aus classification_data
            self.cur.execute('''
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'classification_data'
            ''')
            classification_data_columns = {row[0] for row in self.cur.fetchall()}

            # Gemeinsame Spalten ermitteln, aber Level-Spalten & ID-Spalten ausschließen
            common_columns = {col for col in train_data_columns.intersection(classification_data_columns) 
                            if not col.startswith('level_') and col not in ['train_id', 'classification_id']}
            
            # Sicherstellen, dass 'training' nicht doppelt vorkommt
            if "training" in classification_data_columns and "training" not in common_columns:
                common_columns.add("training")
            
            columns_str = ', '.join(common_columns)
            columns_select_str = ', '.join(common_columns)

            if not columns_str:
                QgsMessageLog.logMessage("No valid columns to insert into train_data!", level=Qgis.Critical)
                return

            # INSERT-Abfrage ohne explizite Angabe von train_id (wird automatisch vergeben)
            insert_query = f'''
                INSERT INTO "MPSCDresden".train_data ({columns_str})
                SELECT {columns_select_str}
                FROM "MPSCDresden".classification_data
                WHERE db_filter_id = %s
            '''
            
            self.cur.execute(insert_query, (self.selected_db_filter_id,))
            self.conn.commit()

            QgsMessageLog.logMessage(f"Building with db_filter_id {self.selected_db_filter_id} moved to training data.", level=Qgis.Info)

            # Vor dem Löschen aus classification_data: Lösche zugehörigen Eintrag aus classification_results
            delete_results_query = '''
                DELETE FROM "MPSCDresden".classification_results
                WHERE db_filter_id = %s
            '''
            self.cur.execute(delete_results_query, (self.selected_db_filter_id,))
            self.conn.commit()
            
            # Löschen des Datensatzes aus classification_data
            delete_query = '''
                DELETE FROM "MPSCDresden".classification_data
                WHERE db_filter_id = %s
            '''
            self.cur.execute(delete_query, (self.selected_db_filter_id,))
            self.conn.commit()

            QgsMessageLog.logMessage(f"Deleted building with db_filter_id {self.selected_db_filter_id} from classification_data.", level=Qgis.Info)
            
            # Aktualisierung von citydb_filter mit neuem sst-Wert und Training = 't'
            update_query = '''
                UPDATE "MPSCDresden".citydb_filter
                SET sst = (SELECT sst FROM "MPSCDresden".classification_data WHERE db_filter_id = %s),
                    training = 't'
                WHERE db_filter_id = %s
            '''
            self.cur.execute(update_query, (self.selected_db_filter_id, self.selected_db_filter_id))
            self.conn.commit()

            QgsMessageLog.logMessage(f"Updated citydb_filter for db_filter_id {self.selected_db_filter_id}.", level=Qgis.Info)
        
        except Exception as e:
            QgsMessageLog.logMessage(f"Error updating training data for db_filter_id {self.selected_db_filter_id}: {str(e)}", level=Qgis.Critical)
            raise e

    def retrain_all_levels(self):
        """
        Startet das Retraining aller Klassifikationsmodelle nach manueller Korrektur.
        """
        model_trainer = ModelTrainer(self.conn, self.cur, self.connection_params)
        
        # Hole die korrekten Level-Namen aus der level_definition
        levels_definition = model_trainer.level_definition()
        level_names = [level[0] for level in levels_definition]  # Extrahiere level_name aus (level_name, column, logic, target_names)
        
        for level_name in level_names:
            model_trainer.train(warm_start=True, retrain_level=level_name)
            QgsMessageLog.logMessage(f"Model for level {level_name} retrained.", level=Qgis.Info)