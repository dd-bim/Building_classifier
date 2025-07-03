from qgis.core import QgsMessageLog, Qgis, QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsRandomColorRamp, QgsSymbol

from .data_loader import DataLoader
from .geometry_processor import GeometryProcessor

class MappingProcessor:
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den MappingProcessor mit DB-Verbindung und Verbindungsparametern.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        
        self.data_loader = DataLoader(conn, cur, connection_params)
        self.geometry_processor = GeometryProcessor(conn, cur, connection_params)
        
    def add_additional_columns(self):
        """
        Fügt zusätzliche Spalten zu 'kartierung_dd_gesamt' hinzu, falls sie nicht existieren.
        """
        try:
            add_columns_query = """
            ALTER TABLE "MPSCDresden".kartierung_dd_gesamt
            ADD COLUMN IF NOT EXISTS topo_id INTEGER,
            ADD COLUMN IF NOT EXISTS guid_alkis VARCHAR,
            ADD COLUMN IF NOT EXISTS blocknr VARCHAR,
            ADD COLUMN IF NOT EXISTS development_type VARCHAR,
            ADD COLUMN IF NOT EXISTS development_type_lv2 VARCHAR,
            ADD COLUMN IF NOT EXISTS development_type_lv3 VARCHAR,
            ADD COLUMN IF NOT EXISTS development_type_code VARCHAR;
            """
            self.cur.execute(add_columns_query)
            self.conn.commit()
            QgsMessageLog.logMessage("Additional columns added to kartierung_dd_gesamt", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error adding additional columns to kartierung_dd_gesamt: {str(e)}", level=Qgis.Critical)
    
    def transfer_attributes_to_kartierung_dd_gesamt(self):
        """
        Überträgt Attribute von built_up_parcel nach kartierung_dd_gesamt, falls Geometrien sich überschneiden.
        """
        try:
            # Transfer attributes from built_up_parcel to kartierung_dd_gesamt
            transfer_query = """
            UPDATE "MPSCDresden".kartierung_dd_gesamt k
            SET 
                topo_id = COALESCE(k.topo_id, b.topo_id),
                guid_alkis = COALESCE(k.guid_alkis, b.guid_alkis),
                blocknr = COALESCE(k.blocknr, b.blocknr),
                development_type = COALESCE(k.development_type, b.development_type),
                development_type_lv2 = COALESCE(k.development_type_lv2, b.development_type_lv2),
                development_type_lv3 = COALESCE(k.development_type_lv3, b.development_type_lv3),
                development_type_code = COALESCE(k.development_type_code, b.development_type_code)
            FROM "MPSCDresden".built_up_parcel b
            WHERE ST_Intersects(k.geom, b.geom);
            """
            self.cur.execute(transfer_query)
            updated_rows = self.cur.rowcount
            self.conn.commit()
            QgsMessageLog.logMessage(f"Attributes transferred to kartierung_dd_gesamt, {updated_rows} rows updated", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error transferring attributes to kartierung_dd_gesamt: {str(e)}", level=Qgis.Critical)
            
    def compare_and_categorize_layers(self):
        """
        Vergleicht Attribute und kategorisiert die Einträge in kartierung_dd_gesamt anhand von festen Regeln.
        Die Regeln ergeben sich aus der Wohngebäudematrix und der Bezeichnung des Gebäudetyps der Blockkarte.
        Gibt eine Auswertung der Kategorisierung im Log aus.
        """
        try:
            # Add the new attribute check_type if it doesn't exist
            self.cur.execute("""
            ALTER TABLE "MPSCDresden".kartierung_dd_gesamt
            ADD COLUMN IF NOT EXISTS check_type BOOLEAN;
            """)
            self.conn.commit()

            # Compare attributes and categorize the new QGIS layer
            compare_query = """
            UPDATE "MPSCDresden".kartierung_dd_gesamt
            SET check_type = CASE
                WHEN ("development_type_code" = 'A11' AND "sstg" IN ('LW', 'LWS', 'EE', 'ER')) OR
                     ("development_type_code" = 'A12' AND "sstg" IN ('EE', 'ER')) OR
                     ("development_type_code" = 'A13' AND "sst" IN ('ME1', 'ME2', 'ME3', 'ME4', 'ME5', 'ME7')) OR
                     ("development_type_code" = 'B11' AND "sstg" = 'ER') OR
                     ("development_type_code" = 'B12' AND ("sstg" IN ('MRO', 'MRG', 'ME') OR "sst" = 'MR5')) OR
                     ("development_type_code" = 'B13' AND "sst" = 'MR6') OR
                     ("development_type_code" = 'B14' AND "sst" = 'ME6') OR
                     ("development_type_code" = 'C11' AND "sstg" IN ('MRG', 'MRO')) OR
                     ("development_type_code" = 'C12' AND "sst" = 'MR6')
                THEN TRUE
                WHEN ("development_type_code" IS NOT NULL AND "sstg" IS NOT NULL) OR
                     ("development_type_code" IS NOT NULL AND "sst" IS NOT NULL)
                THEN FALSE
                ELSE NULL
            END;
            """
            self.cur.execute(compare_query)
            self.conn.commit()
            
            # Berechnung der Ergebnisse
            self.cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE check_type = TRUE) AS correct_count,
                COUNT(*) FILTER (WHERE check_type = FALSE) AS incorrect_count,
                COUNT(*) FILTER (WHERE check_type IS NULL) AS not_categorized_count,
                COUNT(*) AS total_count
            FROM "MPSCDresden".kartierung_dd_gesamt;
            """)
            result = self.cur.fetchone()
            correct_count = result[0]
            incorrect_count = result[1]
            not_categorized_count = result[2]
            total_count = result[3]

            correct_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
            incorrect_percentage = (incorrect_count / total_count) * 100 if total_count > 0 else 0
            not_categorized_percentage = (not_categorized_count / total_count) * 100 if total_count > 0 else 0
            
            # Ausgabe der Ergebnisse auf der Konsole
            QgsMessageLog.logMessage(f"Correct categorizations: {correct_count} ({correct_percentage:.2f}%)", level=Qgis.Info)
            QgsMessageLog.logMessage(f"Incorrect categorizations: {incorrect_count} ({incorrect_percentage:.2f}%)", level=Qgis.Info)
            QgsMessageLog.logMessage(f"Not categorized: {not_categorized_count} ({not_categorized_percentage:.2f}%)", level=Qgis.Info)
            
            QgsMessageLog.logMessage("Attributes compared and categorized in kartierung_dd_gesamt", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error comparing and categorizing layers: {str(e)}", level=Qgis.Critical)
            
    def categorize_and_colorize(self, layer):
        """
        Kategorisiert und färbt einen QGIS-Layer nach dem Attribut 'SST' ein.
        """
        try:
            if not layer.isValid():
                QgsMessageLog.logMessage("Layer is not valid", level=Qgis.Critical)
                return
            
            categories = []
            
            # Definiere die Kategorien basierend auf dem Attribut "SST"
            unique_values = layer.uniqueValues(layer.fields().indexFromName('SST'))
            for value in unique_values:
                symbol = QgsSymbol.defaultSymbol(layer.geometryType())
                category = QgsRendererCategory(value, symbol, str(value))
                categories.append(category)
            
            renderer = QgsCategorizedSymbolRenderer('SST', categories)
            layer.setRenderer(renderer)
            
            # Zufällige Farben zuweisen
            color_ramp = QgsRandomColorRamp()
            renderer.updateColorRamp(color_ramp)
            
            QgsMessageLog.logMessage("Categorized and colorized layer created", level=Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error in categorize_and_colorize: {str(e)}", level=Qgis.Critical)