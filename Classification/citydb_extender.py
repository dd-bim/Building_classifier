import configparser
import os
import pandas as pd
from qgis.core import QgsMessageLog, Qgis

class CityDBExtender:
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den CityDBExtender mit DB-Verbindung und Verbindungsparametern.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        
        # Lade Pfade aus der config.ini
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self.paths = {
            'additional_sst': os.path.join(os.path.dirname(__file__), config.get('Paths', 'additional_sst')),
            'building_age_monuments': os.path.join(os.path.dirname(__file__), config.get('Paths', 'building_age_monuments'))
        }
        
    def add_additional_columns(self):
        """
        Fügt zusätzliche Spalten für Baualter, Genehmigungsjahr und Quelle zu citydb_filter hinzu.
        """
        try:
            add_columns_query = """
            ALTER TABLE "MPSCDresden".citydb_filter
            ADD COLUMN IF NOT EXISTS building_age VARCHAR,
            ADD COLUMN IF NOT EXISTS baugenehmigung_year INTEGER,
            ADD COLUMN IF NOT EXISTS baujahr DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS building_age_source VARCHAR
            """
            self.cur.execute(add_columns_query)
            self.conn.commit()
            QgsMessageLog.logMessage("Additional columns added to citydb_filter", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error adding additional columns to citydb_filter: {str(e)}", level=Qgis.Critical)
            
    def reset_columns(self):
        """
        Setzt die Spalten building_age, baugenehmigung_year und baujahr in citydb_filter auf NULL.
        """
        try:
            reset_columns_query = """
            UPDATE "MPSCDresden".citydb_filter
            SET building_age = NULL,
                baugenehmigung_year = NULL,
                baujahr = NULL
            """
            self.cur.execute(reset_columns_query)
            self.conn.commit()
            QgsMessageLog.logMessage("Columns building_age, baugenehmigung_year, and baujahr reset to NULL", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error resetting columns: {str(e)}", level=Qgis.Critical)
        
    def add_new_buildings(self):
        """
        Aktualisiert building_age und baugenehmigung_year für Gebäude, die mit Neubauten-Geometrien überlappen.
        Setzt zusätzlich building_age_source auf 'neubauten', um die Herkunft zu kennzeichnen.
        """
        try:
            add_new_buildings_query = """
            UPDATE "MPSCDresden".citydb_filter cf
            SET building_age = '7',
                baugenehmigung_year = EXTRACT(YEAR FROM nb.baugenehmi::DATE),
                building_age_source = 'neubauten'
            FROM "MPSCDresden".neubauten nb
            WHERE ST_Intersects(cf.geom, nb.geom)
                AND nb.baugenehmi IS NOT NULL
            """
            self.cur.execute(add_new_buildings_query)
            self.conn.commit()
            QgsMessageLog.logMessage("Building age and source updated for intersecting buildings (neubauten)", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error updating building age/source: {str(e)}", level=Qgis.Critical)

    def update_building_age(self):
        """
        Aktualisiert building_age und baujahr anhand der Tabelle baualter und Zuordnungstabellen.
        Setzt zusätzlich building_age_source auf 'baualter', wenn der Wert aus dieser Quelle stammt.
        """
        try:
            update_building_age_query = """
            UPDATE "MPSCDresden".citydb_filter cf
            SET building_age = CASE
                    WHEN cf.building_age IS NULL AND ba.baujahr > 0 THEN
                        CASE
                            WHEN ba.baujahr < 1870 THEN '1/2'
                            WHEN ba.baujahr >= 1870 AND ba.baujahr <= 1918 THEN '3'
                            WHEN ba.baujahr > 1918 AND ba.baujahr <= 1945 THEN '4'
                            WHEN ba.baujahr > 1945 AND ba.baujahr < 1970 THEN '5'
                            WHEN ba.baujahr >= 1970 AND ba.baujahr <= 1990 THEN '5/6'
                            WHEN ba.baujahr > 1990 THEN '7'
                            ELSE NULL
                        END
                    ELSE cf.building_age
                END,
                baujahr = CASE
                    WHEN ba.baujahr > 0 THEN ba.baujahr
                    ELSE cf.baujahr
                END,
                building_age_source = CASE
                    WHEN cf.building_age IS NULL AND ba.baujahr > 0 THEN 'baualter'
                    ELSE cf.building_age_source
                END
            FROM "MPSCDresden".baualter ba
            WHERE ST_Intersects(cf.geom, ba.geom)
            """
            self.cur.execute(update_building_age_query)
            self.conn.commit()
            QgsMessageLog.logMessage("Building age and source updated based on Baujahr mapping", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error updating building age/source with Baujahr mapping: {str(e)}", level=Qgis.Critical)
            
    def update_building_age_from_monuments(self):
        """
        Aktualisiert das Attribut building_age in citydb_filter anhand der Denkmaldaten (building_age_monuments).
        Die Zuordnung erfolgt über die ID (entspricht gml_id). Zusätzlich wird building_age_source auf 'monument' gesetzt.
        Gibt gml_id aus, die nicht zugeordnet werden konnten.
        """
        try:
            csv_path = self.paths['building_age_monuments']
            if not os.path.exists(csv_path):
                QgsMessageLog.logMessage(f"CSV-Datei für building_age_monuments nicht gefunden: {csv_path}", level=Qgis.Critical)
                return

            df = pd.read_csv(csv_path, dtype=str, sep=';')

            if 'ID' not in df.columns or 'Baualtersstufe' not in df.columns:
                QgsMessageLog.logMessage("CSV muss die Spalten 'ID' und 'Baualtersstufe' enthalten.", level=Qgis.Critical)
                return

            update_count = 0
            skipped_gml_ids = []
            for idx, row in df.iterrows():
                gml_id = row.get('ID')
                building_age = row.get('Baualtersstufe')
                if not gml_id or not building_age or pd.isna(gml_id) or pd.isna(building_age):
                    skipped_gml_ids.append(gml_id)
                    continue
                update_query = """
                    UPDATE "MPSCDresden".citydb_filter
                    SET building_age = %s,
                        building_age_source = 'monument'
                    WHERE gml_id = %s
                """
                self.cur.execute(update_query, (building_age, gml_id))
                update_count += self.cur.rowcount

            self.conn.commit()
            QgsMessageLog.logMessage(f"{update_count} Gebäude mit building_age aus building_age_monuments.csv aktualisiert.", level=Qgis.Info)
            if skipped_gml_ids:
                QgsMessageLog.logMessage(
                    f"{len(skipped_gml_ids)} Zeilen in building_age_monuments.csv hatten leere Werte und wurden übersprungen. gml_id: {skipped_gml_ids}",
                    level=Qgis.Warning
                )
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Aktualisieren von building_age aus Denkmaldaten: {str(e)}", level=Qgis.Critical)
            
    def update_sst_from_csv(self):
        """
        Aktualisiert die Spalten sst und sst_sub in citydb_filter anhand einer CSV-Datei.
        """
        try:
            csv_path = self.paths['additional_sst']
            if not os.path.exists(csv_path):
                QgsMessageLog.logMessage(f"CSV-Datei für additional_sst nicht gefunden: {csv_path}", level=Qgis.Critical)
                return

            # CSV einlesen
            df = pd.read_csv(csv_path, dtype={'gml_id': str}, sep=';')
            if 'gml_id' not in df.columns or 'sst' not in df.columns or 'sst_sub' not in df.columns:
                QgsMessageLog.logMessage("CSV muss die Spalten 'gml_id', 'sst' und 'sst_sub' enthalten.", level=Qgis.Critical)
                return

            csv_ids = set(df['gml_id'].astype(str))
            self.cur.execute('SELECT gml_id FROM "MPSCDresden".citydb_filter')
            db_ids = set(row[0] for row in self.cur.fetchall())
            fehlende = csv_ids - db_ids
            QgsMessageLog.logMessage(f"Nicht in DB gefundene gml_id aus der Nachkartierung: {fehlende}", level=Qgis.Warning)

            update_count = 0
            for _, row in df.iterrows():
                gml_id = row['gml_id']
                sst = row['sst']
                sst_sub = row['sst_sub']
                if pd.isna(sst_sub) or str(sst_sub).strip() == "":
                    update_query = """
                        UPDATE "MPSCDresden".citydb_filter
                        SET sst = %s
                        WHERE gml_id = %s
                    """
                    self.cur.execute(update_query, (sst, gml_id))
                else:
                    update_query = """
                        UPDATE "MPSCDresden".citydb_filter
                        SET sst = %s, sst_sub = %s
                        WHERE gml_id = %s
                    """
                    self.cur.execute(update_query, (sst, sst_sub, gml_id))
                update_count += self.cur.rowcount

            self.conn.commit()
            QgsMessageLog.logMessage(f"{update_count} Gebäude mit sst/sst_sub aus additional_sst.csv aktualisiert.", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Aktualisieren von sst/sst_sub aus CSV: {str(e)}", level=Qgis.Critical)

    def set_classification_source_kartierung(self):
        """
        Setzt classification_source und classification_source_id auf 'Kartierung' (1) für alle Datensätze mit sst oder sst_sub,
        sofern noch keine Quelle gesetzt ist.
        """
        try:
            self.cur.execute("""
                UPDATE "MPSCDresden".citydb_filter
                SET classification_source_id = 1,
                    classification_source = 'Kartierung'
                WHERE (sst IS NOT NULL OR sst_sub IS NOT NULL)
                  AND (classification_source IS NULL OR classification_source = '');
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("classification_source für Kartierung gesetzt.", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Setzen von classification_source Kartierung: {str(e)}", level=Qgis.Critical)