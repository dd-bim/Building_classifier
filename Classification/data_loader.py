from qgis.core import QgsVectorLayer, QgsField, QgsFeature, QgsGeometry, QgsProject, QgsMessageLog, Qgis, QgsDataSourceUri, QgsVectorLayerExporter, QgsCoordinateReferenceSystem
from qgis.PyQt.QtCore import QVariant
import pandas as pd
import os
import configparser

class DataLoader:
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den DataLoader mit DB-Verbindung und lädt Pfade aus der Konfiguration.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        
        self.paths = {
            'building_development': os.path.join(os.path.dirname(__file__), config.get('Paths', 'building_development')),
            'parcels': os.path.join(os.path.dirname(__file__), config.get('Paths', 'parcels'))
        }
        
    @staticmethod
    def load_csv(filename):
        """
        Lädt eine CSV-Datei als DataFrame, falls vorhanden, sonst leeres DataFrame.
        """
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep=';', encoding='utf-8')
            return df
        else:
            QgsMessageLog.logMessage(f"File {filename} does not exist.", level=Qgis.Warning)
            return pd.DataFrame()

    @staticmethod
    def filter_data(data_type, df):
        """
        Filtert und transformiert die Daten je nach Typ (building_development/parcels).
        """
        if data_type == 'building_development':
            filter_criteria = ['A1', 'B1', 'C1', 'D1']
            
            # Filtere nach desk2-Präfix
            df = df[df['desk2'].str.startswith(tuple(filter_criteria))]
            
            # Wähle relevante Spalten
            df = df[['blocknr', 'sst_liste', 'sst_lv_2_liste', 'sst_lv_3_liste', 'desk3', 'shape']]
            
            # Extrahiere Geometrie und SRID
            df[['srid_building_development', 'geometry_building_development']] = df['shape'].apply(DataLoader.extract_geometry).apply(pd.Series)
            df = df.drop(columns=['shape'])
            QgsMessageLog.logMessage(f"Filtered building_development DataFrame columns: {df.columns}", level=Qgis.Info)
            QgsMessageLog.logMessage(f"Filtered building_development DataFrame head: {df.head()}", level=Qgis.Info)

        elif data_type == 'parcels':
            # Wähle relevante Spalten
            df = df[['id', 'shape']]
            
            # Extrahiere Geometrie und SRID
            df[['srid_parcels', 'geometry_parcels']] = df['shape'].apply(DataLoader.extract_geometry).apply(pd.Series)
            df = df.drop(columns=['shape'])
            QgsMessageLog.logMessage(f"Filtered parcels DataFrame columns: {df.columns}", level=Qgis.Info)
            QgsMessageLog.logMessage(f"Filtered parcels DataFrame head: {df.head()}", level=Qgis.Info)

        return df
    
    @staticmethod
    def extract_geometry(shape):
        """
        Extrahiert SRID und WKT-Geometrie aus einem shape-String.
        """
        parts = shape.split(';')
        srid = parts[0].replace('SRID=', '') if len(parts) > 1 else None
        wkt = parts[1] if len(parts) > 1 else parts[0]
        return srid, wkt
    
    @staticmethod
    def set_crs_for_layer(layer, epsg_code):
        """
        Setzt das Koordinatenreferenzsystem (CRS) für einen Layer.
        """
        try:
            crs = QgsCoordinateReferenceSystem(f"EPSG:{epsg_code}")
            layer.setCrs(crs)
            QgsMessageLog.logMessage(f"CRS set to EPSG:{epsg_code} for layer {layer.name()}", level=Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error setting CRS for layer {layer.name()}: {str(e)}", level=Qgis.Critical)

    @staticmethod
    def create_vector_layer(df, layer_name, geometry_column, srid_column):
        """
        Erstellt einen temporären Vektorlayer aus einem DataFrame und fügt ihn zu QGIS hinzu.
        """
        # Entferne ggf. bestehenden Layer mit gleichem Namen
        existing_layer = QgsProject.instance().mapLayersByName(layer_name)
        if existing_layer:
            QgsProject.instance().removeMapLayer(existing_layer[0])
        
        # Definiere Felder (Attribute)
        fields = [QgsField(col, QVariant.String) for col in df.columns if col not in [geometry_column, srid_column]]
        fields.append(QgsField(geometry_column, QVariant.String))
        fields.append(QgsField(srid_column, QVariant.String))
        
        srid = df[srid_column].iloc[0]
        crs = f"EPSG:{srid}" if srid else 'EPSG:4326'

        # Erstelle Memory-Layer
        layer = QgsVectorLayer(f'Polygon?crs={crs}', layer_name, 'memory')
        provider = layer.dataProvider()
        provider.addAttributes(fields)
        layer.updateFields()

        # Füge Features hinzu
        for i, (_, row) in enumerate(df.iterrows()):
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromWkt(row[geometry_column]))
            attributes = [row[col] for col in df.columns if col not in [geometry_column, srid_column]]
            attributes.append(row[geometry_column])
            attributes.append(row[srid_column])
            feature.setAttributes(attributes)
            provider.addFeature(feature)

        QgsProject.instance().addMapLayer(layer)
        return layer
    
    def export_layer_to_citydb(self, layer, table_name, drop_existing=False):
        """
        Exportiert einen QGIS-Layer in eine CityDB-Tabelle (PostGIS).
        """
        QgsMessageLog.logMessage(f"Starting export of layer to table {table_name}", level=Qgis.Info)
        
        try:
            # Prüfe, ob Tabelle existiert
            self.cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'MPSCDresden' AND table_name = '{table_name}');")
            table_exists = self.cur.fetchone()[0]
            
            if table_exists:
                # Lösche bestehende Tabelle
                self.cur.execute(f'DROP TABLE IF EXISTS "MPSCDresden".{table_name};')
                self.conn.commit()
                QgsMessageLog.logMessage(f"Table {table_name} dropped", level=Qgis.Info)
            
            # Setze Datenbankverbindung
            uri = QgsDataSourceUri()
            uri.setConnection(self.connection_params['host'], self.connection_params['port'], self.connection_params['dbname'], self.connection_params['user'], self.connection_params['password'])
            uri.setDataSource('MPSCDresden', table_name, 'geom')

            # Exportiere Layer
            error = QgsVectorLayerExporter.exportLayer(layer, uri.uri(), 'postgres', layer.crs(), False)
            if error[0] != QgsVectorLayerExporter.NoError:
                QgsMessageLog.logMessage(f"Error exporting layer to CityDB: {error[1]}", level=Qgis.Warning)
            else:
                QgsMessageLog.logMessage(f"Layer {layer.name()} exported to CityDB table {table_name}", level=Qgis.Info)

        except Exception as e:
            QgsMessageLog.logMessage(f"Error in export_layer_to_citydb: {str(e)}", level=Qgis.Critical)
    
    @staticmethod
    def load_layer_from_db(connection_params, schema, table, geom_column='geom'):
        """
        Lädt einen Layer direkt aus der Datenbank in QGIS.
        """
        try:
            uri = QgsDataSourceUri()
            uri.setConnection(connection_params['host'], connection_params['port'], connection_params['dbname'], connection_params['user'], connection_params['password'])
            uri.setDataSource(schema, table, geom_column)
            layer = QgsVectorLayer(uri.uri(), table, 'postgres')
            
            if not layer.isValid():
                QgsMessageLog.logMessage(f"Layer {table} is not valid", level=Qgis.Critical)
                return None
            
            QgsProject.instance().addMapLayer(layer)
            QgsMessageLog.logMessage(f"Layer {table} loaded from database", level=Qgis.Info)
            return layer
        except Exception as e:
            QgsMessageLog.logMessage(f"Error loading layer {table} from database: {str(e)}", level=Qgis.Critical)
            return None