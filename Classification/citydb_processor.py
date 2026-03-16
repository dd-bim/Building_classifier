import configparser
import os
from qgis.core import QgsMessageLog, Qgis
import pandas as pd

class CityDBProcessor:
    """
    Bereitet die citydb_filter-Tabelle für die Gebäudeklassifikation vor.

    Diese Klasse bietet Methoden zum:
    - Erstellen und Befüllen der citydb_filter-Tabelle aus der CityDB
    - Übertragen und Bereinigen von Attributen und Geometrien
    - Berechnen von abgeleiteten Merkmalen (Feature Engineering)
    - Durchführen von Nachbarschaftsanalysen und Clustering
    - Fortschrittsanzeige für alle Verarbeitungsschritte
    """
    
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den CityDBProcessor mit DB-Verbindung und Verbindungsparametern.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        self.progress_bar = None
        self.total_steps = 0
        self.current_step = 0
        
        # Lade Pfade aus der config.ini
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        
        self.csv_path = os.path.join(os.path.dirname(__file__), config.get('Paths', 'building_function_csv'))
        
    def set_progress_bar(self, progress_bar):
        """
        Setzt eine externe Fortschrittsanzeige für die Verarbeitung.
        """
        self.progress_bar = progress_bar

    def update_progress(self):
        """
        Aktualisiert die Fortschrittsanzeige, falls vorhanden.
        """
        if self.progress_bar:
            progress = int(self.current_step / self.total_steps * 100)
            self.progress_bar.setValue(progress)

    def create_tables(self):
        """
        Erstellt die Tabelle citydb_filter und legt alle relevanten Indizes an.
        """
        try:
            # Erstelle die Tabelle db_filter
            self.cur.execute("""
            CREATE TABLE IF NOT EXISTS "MPSCDresden".citydb_filter (
                db_filter_id SERIAL PRIMARY KEY,
                cityobject_id INTEGER,
                gml_id VARCHAR(255) UNIQUE,
                cluster_id INTEGER,
                address VARCHAR(255),
                function VARCHAR(255),
                roof_type VARCHAR(255),
                storeys_above_ground INTEGER,
                building_footprint DOUBLE PRECISION,
                length_footprint DOUBLE PRECISION,
                width_footprint DOUBLE PRECISION,
                roof_ridge_height DOUBLE PRECISION,
                eaves_height DOUBLE PRECISION,
                storey_height DOUBLE PRECISION,
                number_roof_surfaces INTEGER,
                roof_slope DOUBLE PRECISION,
                proximity CHAR(1),
                neighbouring_buildings INTEGER,
                neighbour_density INTEGER,
                neighbour_avg_size DOUBLE PRECISION,
                neighbour_min_distance DOUBLE PRECISION,
                neighbour_majority_class VARCHAR(255),
                mapping_id DOUBLE PRECISION,
                SST VARCHAR(255),
                SST_SUB VARCHAR(255),
                classification_source_id INTEGER,
                classification_source VARCHAR(255),
                confidence DOUBLE PRECISION,
                ID_ALKIS VARCHAR(255),
                development_type_code VARCHAR(255),
                ground_area_per_storey DOUBLE PRECISION,
                footprint_ratio DOUBLE PRECISION,
                height_to_area_ratio DOUBLE PRECISION,
                roof_height_ratio DOUBLE PRECISION,
                building_volume DOUBLE PRECISION,
                compactness DOUBLE PRECISION,
                convexity DOUBLE PRECISION,
                rectangularity DOUBLE PRECISION,
                geom GEOMETRY(MULTIPOLYGON, 25833)
            );
            """)
            # Sinnvolle Indizes für schnelle Abfragen
            self.cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_cityobject_id ON "MPSCDresden".citydb_filter (cityobject_id);
            CREATE INDEX IF NOT EXISTS idx_gml_id ON "MPSCDresden".citydb_filter (gml_id);
            CREATE INDEX IF NOT EXISTS idx_function ON "MPSCDresden".citydb_filter (function);
            CREATE INDEX IF NOT EXISTS idx_SST ON "MPSCDresden".citydb_filter (SST);
            CREATE INDEX IF NOT EXISTS idx_SST_SUB ON "MPSCDresden".citydb_filter (SST_SUB);
            CREATE INDEX IF NOT EXISTS idx_cluster_id ON "MPSCDresden".citydb_filter (cluster_id);
            CREATE INDEX IF NOT EXISTS idx_building_footprint ON "MPSCDresden".citydb_filter (building_footprint);
            CREATE INDEX IF NOT EXISTS idx_storeys_above_ground ON "MPSCDresden".citydb_filter (storeys_above_ground);
            CREATE INDEX IF NOT EXISTS citydb_filter_geom_idx ON "MPSCDresden".citydb_filter USING GIST (geom);
            CREATE INDEX IF NOT EXISTS idx_function_sst ON "MPSCDresden".citydb_filter (function, SST);
            CREATE INDEX IF NOT EXISTS idx_cluster_sst ON "MPSCDresden".citydb_filter (cluster_id, SST);
            """)
            
            self.conn.commit()
            QgsMessageLog.logMessage("Table db_filter created and indexed successfully", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to create and index tables: {str(e)}", level=Qgis.Critical)
            
    def fill_table(self):
        """
        Füllt die Tabelle citydb_filter mit Grunddaten (ohne Geometrie) aus der CityDB.
        Zuerst werden alle Gebäude eingefügt, dann wird separat gefiltert.
        """
        try:
            # Insertiere erstmal alle Gebäude (objectclass_id = 901)
            self.cur.execute("""            
            INSERT INTO "MPSCDresden".citydb_filter(
                cityobject_id
            )
            SELECT DISTINCT f.id AS cityobject_id
            FROM citydb.feature f
            WHERE f.objectclass_id = 901;
            """)
            
            # Hinzufügen der gml_id
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET gml_id = p.val_uri
            FROM citydb.property p
            WHERE cf.cityobject_id = p.feature_id
            AND p.name = 'externalReference'
            AND p.val_uri IS NOT NULL;
            """)
            
            # Hinzufügen der function
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET 
                function = p.val_string
            FROM citydb.property p
            WHERE cf.cityobject_id = p.feature_id
            AND p.name = 'function';
            """)
            
            # Robuste Adresszuordnung mit Subquery-Ansatz
            # Dieser Ansatz ist zuverlässiger als JOIN-basierte UPDATEs
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET address = (
                SELECT CONCAT(a.street, ' ', COALESCE(a.house_number, ''))
                FROM citydb.property p
                JOIN citydb.address a ON p.val_address_id = a.id
                WHERE p.feature_id = cf.cityobject_id
                AND p.name = 'address'
                AND p.val_address_id IS NOT NULL
                AND a.street IS NOT NULL
                AND a.street != ''
                AND a.street != '0'
                LIMIT 1
            )
            WHERE EXISTS (
                SELECT 1
                FROM citydb.property p
                JOIN citydb.address a ON p.val_address_id = a.id
                WHERE p.feature_id = cf.cityobject_id
                AND p.name = 'address'
                AND p.val_address_id IS NOT NULL
                AND a.street IS NOT NULL
                AND a.street != ''
                AND a.street != '0'
            );
            """)
            
            # Erweiterte Logging für besseres Debugging
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".citydb_filter WHERE address IS NOT NULL;
            """)
            buildings_with_address = self.cur.fetchone()[0]
            
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".citydb_filter;
            """)
            total_buildings = self.cur.fetchone()[0]
            
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".citydb_filter WHERE function IN ('1000', '1100');
            """)
            function_buildings = self.cur.fetchone()[0]
            
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".citydb_filter WHERE function IN ('1000', '1100') AND address IS NOT NULL;
            """)
            function_buildings_with_address = self.cur.fetchone()[0]
            
            QgsMessageLog.logMessage(f"Address UPDATE Results:", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Total buildings in filter: {total_buildings}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Buildings with addresses: {buildings_with_address}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Function 1000/1100 buildings: {function_buildings}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Function 1000/1100 with addresses: {function_buildings_with_address}", level=Qgis.Warning)
            
            # Konservative Filterung - nur offensichtlich leere Adressen entfernen
            self.cur.execute("""
            DELETE FROM "MPSCDresden".citydb_filter
            WHERE address IS NULL;
            """)
            
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".citydb_filter;
            """)
            remaining_buildings = self.cur.fetchone()[0]
            
            QgsMessageLog.logMessage(f"- Buildings remaining after address filtering: {remaining_buildings}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"  (Note: Function filtering (1000/1100 only) will be applied next)", level=Qgis.Warning)
            
            self.conn.commit()
            QgsMessageLog.logMessage("Table citydb_filter filled with basic attributes (cityobject_id, gml_id, function, address) and conservatively filtered", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to fill table with basic attributes: {str(e)}", level=Qgis.Critical)
    
    def update_function_from_csv(self):
        """
        Aktualisiert das Attribut 'function' in citydb_filter anhand einer CSV-Datei.
        """
        try:
            df = pd.read_csv(self.csv_path, delimiter=';')

            # Iteriere über die Zeilen des DataFrames und aktualisiere die Funktion in der Datenbank
            for index, row in df.iterrows():
                gml_id = row['ID']
                function = row['GFK']
                self.cur.execute('''
                    UPDATE "MPSCDresden".citydb_filter
                    SET function = %s
                    WHERE gml_id = %s
                ''', (function, gml_id))
            self.conn.commit()
            QgsMessageLog.logMessage("Function updated successfully from CSV", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to update function from CSV: {str(e)}", level=Qgis.Critical)
    
    def intersect_and_update_citydb_filter(self):
        """
        Überträgt Attribute aus kartierung_dd_gesamt nach citydb_filter und setzt Quelle Kartierung (1) für übernommene sst.
        """
        try:
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET 
                mapping_id = k."id",
                SST = k."sst",
                SST_SUB = k."sst_sub",
                development_type_code = k."development_type_code"
            FROM "MPSCDresden".kartierung_dd_gesamt k
            WHERE (
                  cf.gml_id = k."id_alkis"
               OR (
                    k."id_alkis" IS NULL
                    AND trim(lower(k.str || ' ' || k.hnr)) = trim(lower(cf.address))
                  )
            );
            """)
            self.conn.commit()

            # Quelle Kartierung setzen (nur wenn sst vorhanden und noch keine Quelle)
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET classification_source_id = 1,
                classification_source = 'Kartierung'
            WHERE sst IS NOT NULL
              AND (classification_source_id IS NULL OR classification_source_id <> 1);
            """)
            self.conn.commit()

            QgsMessageLog.logMessage("citydb_filter updated with Kartierung attributes + source.", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error intersecting and updating citydb_filter: {str(e)}", level=Qgis.Critical)
            
    def clean_sst_data(self):
        """
        Cleans the sst and sst_sub columns in citydb_filter.
        Replaces sst with sst_sub when sst_sub is a valid target value.
        """
        try:
            QgsMessageLog.logMessage("=== SST DATA CLEANING STARTED ===", level=Qgis.Info)
            
            # STEP 1: Define valid target values (Hard-Coding)
            valid_targets = ['ER2', 'ER3', 'ER4', 'ER5', 'ER7',
                            'EE1', 'EE2', 'EE3', 'EE4', 'EE5', 'EE7',
                            'HH3', 'HH4',
                            'MR5', 'MR6',
                            'MRG2', 'MRO2', 'MRG3', 'MRO3', 'MRG4', 'MRO4', 'MRG7', 'MRO7',
                            'ME2', 'ME3', 'ME4', 'ME5', 'ME6', 'ME7',
                            'LW1', 'LW2', 'LW3', 'LW7']
            
            valid_str = ','.join([f"'{val}'" for val in valid_targets])
            
            # STEP 2: Replace sst with sst_sub (when valid)
            self.cur.execute(f"""
            UPDATE "MPSCDresden".citydb_filter
            SET sst = sst_sub
            WHERE sst_sub IS NOT NULL 
            AND sst_sub IN ({valid_str});
            """)
            
            # STEP 3: LWS → LW conversion
            lws_mapping = {'LWS1': 'LW1', 'LWS2': 'LW2', 'LWS3': 'LW3'}
            
            for old, new in lws_mapping.items():
                self.cur.execute(f"""
                UPDATE "MPSCDresden".citydb_filter
                SET sst = '{new}'
                WHERE sst = '{old}';
                """)
        
            # STEP 4: Clear sst_sub column
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET sst_sub = NULL;
            """)
            
            self.conn.commit()
            QgsMessageLog.logMessage("SST data cleaning completed successfully", level=Qgis.Info)
            
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to clean SST data: {str(e)}", level=Qgis.Critical)
            
    def filter_table(self):
        """
        Filtert die Tabelle citydb_filter nach zulässigen Funktionswerten (1000, 1100).
        """
        try:
            # Zähle vor der Filterung
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".citydb_filter;
            """)
            before_filtering = self.cur.fetchone()[0]

            # Zusatzdiagnose: prüfe, ob die Kartierungsliste NULL-Werte enthält
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".kartierung_dd_gesamt
            WHERE id_alkis IS NULL;
            """)
            null_ids = self.cur.fetchone()[0]
            if null_ids > 0:
                QgsMessageLog.logMessage(f"Warning: kartierung_dd_gesamt enthält {null_ids} NULL id_alkis."
                                         " NOT IN wird dadurch neutralisiert.", level=Qgis.Warning)

            # Löschen mit NOT EXISTS, inklusive Adressabgleich für NULL-id_alkis
            self.cur.execute("""
            DELETE FROM "MPSCDresden".citydb_filter cf
            WHERE function NOT IN ('1000', '1100')
              AND NOT EXISTS (
                  SELECT 1
                  FROM "MPSCDresden".kartierung_dd_gesamt k
                  WHERE (
                      k.id_alkis = cf.gml_id
                      OR (
                          k.id_alkis IS NULL
                          AND trim(lower(k.str || ' ' || k.hnr)) = trim(lower(cf.address))
                      )
                  )
              );
            """)
            
            # Zähle nach der Filterung
            self.cur.execute("""
            SELECT COUNT(*) FROM "MPSCDresden".citydb_filter;
            """)
            after_filtering = self.cur.fetchone()[0]
            
            removed_buildings = before_filtering - after_filtering
            
            QgsMessageLog.logMessage(f"Function filtering completed:", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Buildings before function filtering: {before_filtering}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Buildings removed (non-1000/1100): {removed_buildings}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Final buildings (1000/1100 with addresses): {after_filtering}", level=Qgis.Warning)
            
            self.conn.commit()
            QgsMessageLog.logMessage("Table citydb_filter filtered successfully", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to filter table citydb_filter: {str(e)}", level=Qgis.Critical)
            
    def fill_remaining_attributes_and_geometry(self):
        """
        Füllt die verbleibenden Attribute und Geometrie für die gefilterten Gebäude.
        Wird nach der Filterung aufgerufen, um nur noch relevante Gebäude zu verarbeiten.
        """
        try:
            # EINFACHE UND DIREKTE GEOMETRIE-VERARBEITUNG
            # Sammelt alle Ground Surfaces pro Building und vereint sie mit robuster Methode
            self.cur.execute("""
            WITH building_ground_surfaces AS (
                -- Sammle alle Ground Surface Geometrien pro Building
                SELECT 
                    p.feature_id AS building_id,
                    array_agg(g.geometry) AS all_geometries,
                    COUNT(*) AS surface_count
                FROM citydb.property p
                JOIN citydb.feature ground_f ON p.val_feature_id = ground_f.id
                JOIN citydb.geometry_data g ON ground_f.id = g.feature_id
                WHERE p.name = 'boundary'
                AND ground_f.objectclass_id = 710  -- Ground Surface
                AND g.geometry IS NOT NULL
                GROUP BY p.feature_id
                HAVING COUNT(*) > 0
            ),
            unified_building_geometry AS (
                SELECT 
                    bgs.building_id,
                    bgs.surface_count,
                    -- ROBUSTE GEOMETRIE-VEREINIGUNG mit 3 Fallback-Strategien
                    COALESCE(
                        -- Strategie 1: Direkte Union mit 3D→2D Konvertierung (löst GEOS-Probleme)
                        (SELECT ST_Union(ST_Force2D(ST_MakeValid(geom)))
                         FROM unnest(bgs.all_geometries) AS geom
                         WHERE geom IS NOT NULL),
                        
                        -- Strategie 2: Buffer-Reparatur bei Topology-Problemen  
                        (SELECT ST_Union(ST_Buffer(ST_Force2D(ST_MakeValid(geom)), 0.001))
                         FROM unnest(bgs.all_geometries) AS geom
                         WHERE geom IS NOT NULL),
                        
                        -- Strategie 3: Collect als garantierter Fallback
                        (SELECT ST_Multi(ST_Collect(ST_Force2D(ST_MakeValid(geom))))
                         FROM unnest(bgs.all_geometries) AS geom
                         WHERE geom IS NOT NULL)
                    ) AS final_geometry
                FROM building_ground_surfaces bgs
            )
            UPDATE "MPSCDresden".citydb_filter cf
            SET geom = ubg.final_geometry
            FROM unified_building_geometry ubg
            WHERE cf.cityobject_id = ubg.building_id
            AND ubg.final_geometry IS NOT NULL;
            """)
            
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET 
                storeys_above_ground = p.val_int
            FROM citydb.property p
            WHERE cf.cityobject_id = p.feature_id
            AND p.name = 'storeysAboveGround';
            """)
            
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET 
                roof_type = p.val_string
            FROM citydb.property p
            WHERE cf.cityobject_id = p.feature_id
            AND p.name = 'roofType';
            """)
            
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET roof_ridge_height = p.val_double
            FROM citydb.property p
            WHERE cf.cityobject_id = p.feature_id
            AND p.name = 'value'
            AND p.parent_id IN (
                SELECT p2.id FROM citydb.property p2 
                WHERE p2.feature_id = p.feature_id 
                AND p2.name = 'height'
            );
            """)
            
            self.conn.commit()
            
            # Überprüfe Erfolgsrate der robusten Geometrie-Verarbeitung
            self.cur.execute("""
            SELECT 
                COUNT(*) AS total_buildings,
                COUNT(*) FILTER (WHERE geom IS NOT NULL) AS successful_geometries,
                ROUND(100.0 * COUNT(*) FILTER (WHERE geom IS NOT NULL) / COUNT(*), 2) AS success_rate
            FROM "MPSCDresden".citydb_filter;
            """)
            
            result = self.cur.fetchone()
            total_buildings = result[0]
            successful_geometries = result[1] 
            success_rate = result[2]
            
            QgsMessageLog.logMessage(
                f"ROBUSTE GEOMETRIE-VERARBEITUNG ABGESCHLOSSEN:\n"
                f"- Gebäude insgesamt: {total_buildings}\n"
                f"- Erfolgreiche Geometrien: {successful_geometries}\n"
                f"- Erfolgsrate: {success_rate}% (Ziel: 100%)", 
                level=Qgis.Info
            )
            
            if success_rate >= 99.0:
                QgsMessageLog.logMessage("✓ HERVORRAGEND: Nahezu 100% Erfolgsrate erreicht!", level=Qgis.Info)
            elif success_rate >= 90.0:
                QgsMessageLog.logMessage("✓ GUT: Hohe Erfolgsrate erreicht", level=Qgis.Info)
            else:
                QgsMessageLog.logMessage("⚠ WARNUNG: Niedrige Erfolgsrate - weitere Optimierung nötig", level=Qgis.Warning)
                
            QgsMessageLog.logMessage("Remaining attributes and geometry added successfully to filtered buildings", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to fill remaining attributes and geometry: {str(e)}", level=Qgis.Critical)
                
    def calculate_footprint(self):
        """
        Berechnet Grundfläche, Länge und Breite des Gebäude-Footprints in einem optimierten Query.
        Entfernt Gebäude mit einer Grundfläche kleiner als 30 m².
        """
        try:
            # Erst die Grundfläche berechnen
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET building_footprint = ST_Area(geom)
            WHERE geom IS NOT NULL;
            """)
            
            # Gebäude mit zu kleiner Grundfläche entfernen
            self.cur.execute("""
            DELETE FROM "MPSCDresden".citydb_filter
            WHERE building_footprint < 30;
            """)
            
            # Dann Länge und Breite für die verbleibenden Gebäude berechnen
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET 
                length_footprint = ST_XMax(ST_Envelope(geom)) - ST_XMin(ST_Envelope(geom)),
                width_footprint = ST_YMax(ST_Envelope(geom)) - ST_YMin(ST_Envelope(geom))
            WHERE geom IS NOT NULL;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Table citydb_filter filled successfully with the footprint attributes and filtered by minimum area", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to fill table with the footprint attributes: {str(e)}", level=Qgis.Critical)
            
    def calculate_eaves_height(self):
        """
        Berechnet und aktualisiert die Traufhöhe (eaves_height) in citydb_filter - nur für gefilterte Gebäude.
        """
        try:            
            self.cur.execute("""
            WITH roof AS (
                SELECT f.id AS roof_id
                FROM citydb.feature f
                WHERE objectclass_id = 712
            ),
            roof_height AS (
                SELECT
                    p.feature_id AS roof_id,
                    CAST(p.val_string AS DOUBLE PRECISION) AS Z_Min
                FROM citydb.property p
                JOIN roof r ON p.feature_id = r.roof_id
                WHERE p.name = 'Z_Min'
            ),
            boundary_link AS (
                SELECT
                    p.feature_id AS building_id,
                    rh.Z_Min AS eaves_height
                FROM citydb.property p
                JOIN roof_height rh ON p.val_feature_id = rh.roof_id
                JOIN "MPSCDresden".citydb_filter cf ON p.feature_id = cf.cityobject_id
                WHERE p.name = 'boundary'
            )
            
            UPDATE "MPSCDresden".citydb_filter cf
            SET eaves_height = bl.eaves_height
            FROM boundary_link bl
            WHERE cf.cityobject_id = bl.building_id;
            """)
            
            self.conn.commit()
            QgsMessageLog.logMessage("Eaves height calculated and updated successfully in citydb_filter", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to calculate and update eaves height in citydb_filter: {str(e)}", level=Qgis.Critical)
            
    def calculate_storey_height(self):
        """
        Berechnet die Geschosshöhe und aktualisiert die Tabelle citydb_filter.
        """
        try:
            self.cur.execute("""
            WITH valid_buildings AS (
                SELECT 
                    cf.cityobject_id
                FROM 
                    "MPSCDresden".citydb_filter cf
                JOIN 
                    citydb.property p ON cf.cityobject_id = p.feature_id
                WHERE 
                    p.name = 'DatenquelleGeschoss'
                    AND p.val_string IN ('1', '2', '3', '4')
            )
            UPDATE "MPSCDresden".citydb_filter cf
            SET storey_height = cf.eaves_height / NULLIF(cf.storeys_above_ground, 0)
            FROM valid_buildings vb
            WHERE cf.cityobject_id = vb.cityobject_id;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Storey height calculated and updated successfully in citydb_filter", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to calculate and update storey height in citydb_filter: {str(e)}", level=Qgis.Critical)
    
    def count_roof_surfaces(self):
        """
        Zählt die Anzahl der Dachflächen und aktualisiert citydb_filter - nur für gefilterte Gebäude.
        """
        try:
            self.cur.execute("""
            WITH roof AS (
                SELECT f.id AS roof_id
                FROM citydb.feature f
                WHERE objectclass_id = 712
            ),
            boundary_mapping AS (
                SELECT 
                    p.feature_id AS building_id,
                    p.val_feature_id AS roof_id
                FROM citydb.property p
                JOIN "MPSCDresden".citydb_filter cf ON p.feature_id = cf.cityobject_id
                WHERE p.name = 'boundary'
            ),
            roof_surface_counts AS (
                SELECT 
                    bm.building_id,
                    COUNT(r.roof_id) AS number_roof_surfaces
                FROM boundary_mapping bm
                JOIN roof r ON bm.roof_id = r.roof_id
                GROUP BY bm.building_id
            )

            UPDATE "MPSCDresden".citydb_filter cf
            SET number_roof_surfaces = rsc.number_roof_surfaces
            FROM roof_surface_counts rsc
            WHERE cf.cityobject_id = rsc.building_id;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Number of roof surfaces calculated and updated successfully in citydb_filter", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to calculate and update number of roof surfaces in citydb_filter: {str(e)}", level=Qgis.Critical)
            
    def calculate_roof_slope(self):
        """
        Berechnet die Dachneigung und aktualisiert citydb_filter.
        """
        try:
            self.cur.execute("""
            WITH roof AS (
                SELECT f.id AS roof_id
                FROM citydb.feature f
                WHERE objectclass_id = 712
            ),
            boundary_mapping AS (
                SELECT 
                    p.feature_id AS building_id,
                    p.val_feature_id AS roof_id
                FROM citydb.property p
                WHERE p.name = 'boundary'
            ),
            largest_roof_surface AS (
                SELECT 
                    bm.building_id,
                    f.id AS surface_geometry_id,
                    MAX(ST_Area(f.envelope)) AS max_area
                FROM boundary_mapping bm
                JOIN roof r ON bm.roof_id = r.roof_id
                JOIN citydb.feature f ON r.roof_id = f.id
                GROUP BY bm.building_id, f.id
            ),
            roof_slope_values AS (
                SELECT 
                    lrs.building_id,
                    MAX(CAST(p.val_string AS DOUBLE PRECISION)) AS roof_slope
                FROM largest_roof_surface lrs
                JOIN citydb.property p ON lrs.surface_geometry_id = p.feature_id
                WHERE p.name = 'NORMAL_H'
                GROUP BY lrs.building_id
            )

            UPDATE "MPSCDresden".citydb_filter cf
            SET roof_slope = rsv.roof_slope
            FROM roof_slope_values rsv
            WHERE cf.cityobject_id = rsv.building_id;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Roof slope calculated and updated successfully in citydb_filter", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to calculate and update roof slope in citydb_filter: {str(e)}", level=Qgis.Critical)

    def calculate_clusters(self, buffer_distance=100):
        """
        Berechnet Cluster-IDs für Gebäude basierend auf räumlicher Nähe.
        """
        try:
            self.cur.execute("""
            WITH clusters AS (
                SELECT 
                    gml_id, 
                    ST_ClusterDBSCAN(geom, eps := %s, minpoints := 1) OVER () AS cluster_id
                FROM "MPSCDresden".citydb_filter
            )
            UPDATE "MPSCDresden".citydb_filter cf
            SET cluster_id = c.cluster_id
            FROM clusters c
            WHERE cf.gml_id = c.gml_id;
            """, (buffer_distance,))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler bei der Cluster-Berechnung: {str(e)}", level=Qgis.Critical)

    def set_default_values(self):
        """
        Setzt Standardwerte für Nachbarschaftsattribute in citydb_filter.
        """
        try:
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET proximity = 'E',
                neighbouring_buildings = 0,
                neighbour_density = 0,
                neighbour_avg_size = NULL,
                neighbour_min_distance = NULL,
                neighbour_majority_class = NULL;
            """)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Setzen der Standardwerte: {str(e)}", level=Qgis.Critical)

    def calculate_neighbours(self, buffer_distance=100):
        """
        Berechnet Nachbarschaftsmerkmale für jedes Gebäude (Dichte, Größe, Abstand, Mehrheitsklasse).
        Optimiert mit räumlichen Indizes und 2D-Geometrien.
        """
        try:
            # Optimierte Nachbarschaftsberechnung ohne temporäre 2D-Geometrie
            # Schritt 1: Basis-Nachbarschaftsdaten berechnen
            self.cur.execute("""
            WITH neighbour_data AS (
                SELECT 
                    a.gml_id AS target_gml_id,
                    a.cluster_id,
                    COUNT(b.gml_id) AS neighbour_density,
                    AVG(b.building_footprint) AS neighbour_avg_size,
                    MIN(ST_Distance(a.geom, b.geom)) AS neighbour_min_distance
                FROM "MPSCDresden".citydb_filter a
                JOIN "MPSCDresden".citydb_filter b 
                ON a.cluster_id = b.cluster_id
                AND ST_DWithin(a.geom, b.geom, %s) 
                WHERE a.gml_id != b.gml_id
                AND a.geom IS NOT NULL
                AND b.geom IS NOT NULL
                GROUP BY a.gml_id, a.cluster_id
            )
            UPDATE "MPSCDresden".citydb_filter cf
            SET 
                proximity = CASE 
                    WHEN nd.neighbour_min_distance <= 1.5 THEN 'R'
                    ELSE 'E'
                END,
                neighbour_density = nd.neighbour_density,
                neighbour_avg_size = COALESCE(nd.neighbour_avg_size, NULL),
                neighbour_min_distance = COALESCE(nd.neighbour_min_distance, NULL)
            FROM neighbour_data nd
            WHERE cf.gml_id = nd.target_gml_id;
            """, (buffer_distance,))
            
            self.conn.commit()
            
            # Schritt 2: Nahe Nachbarn zählen
            self.cur.execute("""
            WITH close_neighbours AS (
                SELECT 
                    a.gml_id AS target_gml_id,
                    COUNT(b.gml_id) AS neighbouring_buildings
                FROM "MPSCDresden".citydb_filter a
                JOIN "MPSCDresden".citydb_filter b 
                ON ST_DWithin(a.geom, b.geom, 1.5)
                WHERE a.gml_id != b.gml_id
                AND a.geom IS NOT NULL
                AND b.geom IS NOT NULL
                GROUP BY a.gml_id
            )
            UPDATE "MPSCDresden".citydb_filter cf
            SET neighbouring_buildings = COALESCE(cn.neighbouring_buildings, 0)
            FROM close_neighbours cn
            WHERE cf.gml_id = cn.target_gml_id;
            """)
            
            self.conn.commit()
            
            # Schritt 3: Mehrheitsklasse berechnen (vereinfacht)
            self.cur.execute("""
            WITH cluster_majority AS (
                SELECT 
                    cluster_id,
                    sst,
                    COUNT(*) as class_count,
                    ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY COUNT(*) DESC) as rank
                FROM "MPSCDresden".citydb_filter 
                WHERE sst IS NOT NULL 
                GROUP BY cluster_id, sst
            ),
            majority_per_cluster AS (
                SELECT cluster_id, sst as majority_sst
                FROM cluster_majority 
                WHERE rank = 1
            )
            UPDATE "MPSCDresden".citydb_filter cf
            SET neighbour_majority_class = mpc.majority_sst
            FROM majority_per_cluster mpc
            WHERE cf.cluster_id = mpc.cluster_id;
            """)
            
            self.conn.commit()
            QgsMessageLog.logMessage("Nachbarschaftsmerkmale erfolgreich berechnet & gespeichert.", level=Qgis.Info)
        
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler bei der Berechnung der Nachbarschaftsmerkmale: {str(e)}", level=Qgis.Critical)
            
    def add_feature_engineering_attributes(self):
        """
        Fügt abgeleitete Merkmale (Feature Engineering) zu citydb_filter hinzu.
        """
        try:
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET 
                ground_area_per_storey = building_footprint / NULLIF(storeys_above_ground, 0),
                footprint_ratio = length_footprint / NULLIF(width_footprint, 0),
                height_to_area_ratio = roof_ridge_height / NULLIF(building_footprint, 0),
                roof_height_ratio = (roof_ridge_height - eaves_height) / NULLIF(roof_ridge_height, 0),
                building_volume = building_footprint * roof_ridge_height
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Feature engineering attributes added successfully to citydb_filter", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to add feature engineering attributes to citydb_filter: {str(e)}", level=Qgis.Critical)
    
    def calculate_geometric_features(self):
        """
        Berechnet geometrische Features aus der Gebäudegrundrissgeometrie für das Machine Learning.
        Optimiert mit robusten Berechnungen und besserer Fehlerbehandlung.
        """
        try:
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET                 
                compactness = CASE 
                    WHEN ST_Perimeter(geom) > 0 
                    THEN (4 * PI() * ST_Area(geom)) / (ST_Perimeter(geom) * ST_Perimeter(geom))
                    ELSE NULL 
                END,
                convexity = CASE 
                    WHEN ST_Area(ST_ConvexHull(geom)) > 0
                    THEN ST_Area(geom) / ST_Area(ST_ConvexHull(geom))
                    ELSE NULL 
                END,
                rectangularity = CASE 
                    WHEN ST_Area(ST_OrientedEnvelope(geom)) > 0
                    THEN ST_Area(geom) / ST_Area(ST_OrientedEnvelope(geom))
                    ELSE NULL 
                END
            WHERE geom IS NOT NULL 
            AND ST_IsValid(geom) 
            AND ST_Area(geom) > 0;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Geometric features calculated successfully", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to calculate geometric features: {str(e)}", level=Qgis.Critical)