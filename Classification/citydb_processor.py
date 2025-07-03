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
                neighbor_density INTEGER,
                neighbor_avg_size DOUBLE PRECISION,
                neighbor_min_distance DOUBLE PRECISION,
                neighbor_majority_class VARCHAR(255),
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
                vertex_count INTEGER,
                rectangularity DOUBLE PRECISION,
                geom GEOMETRY(MULTIPOLYGONZ, 25833)
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
        """
        try:
            # Insertiere Grundobjekte
            self.cur.execute("""            
            INSERT INTO "MPSCDresden".citydb_filter(
                cityobject_id
            )
            SELECT f.id AS cityobject_id
            FROM citydb.feature f
            WHERE objectclass_id = 901;
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
            
            self.conn.commit()
            QgsMessageLog.logMessage("Table citydb_filter filled with basic attributes (cityobject_id, gml_id, function)", level=Qgis.Info)
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
            
    def filter_table(self):
        """
        Filtert die Tabelle citydb_filter nach zulässigen Funktionswerten (1000, 1100).
        """
        try:
            self.cur.execute("""
            DELETE FROM "MPSCDresden".citydb_filter
            WHERE function NOT IN ('1000', '1100');
            """)
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
            # Alle Attribute und Geometrie in einem einzigen optimierten Query
            self.cur.execute("""
            WITH ground AS (
                SELECT f.id AS ground_id
                FROM citydb.feature f
                WHERE objectclass_id = 710
            ),
            ground_geometry AS (
                SELECT
                    g.feature_id AS ground_id,
                    -- Stelle sicher, dass die Geometrie valide ist und verwende ST_Union für alle Fälle
                    ST_MakeValid(ST_Union(
                        CASE 
                            WHEN ST_IsValid(g.geometry) THEN g.geometry
                            ELSE ST_MakeValid(g.geometry)
                        END
                    )) AS unified_geom
                FROM citydb.geometry_data g
                JOIN ground gr ON g.feature_id = gr.ground_id
                WHERE g.geometry IS NOT NULL
                GROUP BY g.feature_id
            ),
            boundary_link AS (
                SELECT
                    p.feature_id AS building_id,
                    gg.unified_geom AS geom
                FROM citydb.property p
                JOIN ground_geometry gg ON p.val_feature_id = gg.ground_id
                WHERE p.name = 'boundary'
                AND gg.unified_geom IS NOT NULL
            )
            
            UPDATE "MPSCDresden".citydb_filter cf
            SET geom = bl.geom
            FROM boundary_link bl
            WHERE cf.cityobject_id = bl.building_id;
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
            QgsMessageLog.logMessage("Remaining attributes and geometry added successfully to filtered buildings", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to fill remaining attributes and geometry: {str(e)}", level=Qgis.Critical)
                
    def calculate_footprint(self):
        """
        Berechnet Grundfläche, Länge und Breite des Gebäude-Footprints in einem optimierten Query.
        """
        try:
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET 
                building_footprint = ST_Area(ST_Force2D(geom)),
                length_footprint = ST_XMax(ST_Envelope(ST_Force2D(geom))) - ST_XMin(ST_Envelope(ST_Force2D(geom))),
                width_footprint = ST_YMax(ST_Envelope(ST_Force2D(geom))) - ST_YMin(ST_Envelope(ST_Force2D(geom)))
            WHERE geom IS NOT NULL;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Table citydb_filter filled successfully with the footprint attributes", level=Qgis.Info)
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
            
    def intersect_and_update_citydb_filter(self):
        """
        Überträgt Attribute aus kartierung_dd_gesamt nach citydb_filter per Geometrie-Verschnitt.
        """
        try:
            # Geometrischer Verschnitt der Tabellen und Aktualisierung von citydb_filter
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter cf
            SET 
                mapping_id = k."id",
                SST = k."sst",
                SST_SUB = k."sst_sub",
                ID_ALKIS = k."guid_alkis",
                development_type_code = k."development_type_code"
            FROM "MPSCDresden".kartierung_dd_gesamt k
            WHERE ST_Intersects(cf.geom, k.geom);
            """)
            self.conn.commit()
            
            QgsMessageLog.logMessage("citydb_filter updated with attributes from kartierung_dd_gesamt", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error intersecting and updating citydb_filter: {str(e)}", level=Qgis.Critical)
            
    def clean_sst_data(self):
        """
        Bereinigt die Spalten sst und sst_sub in citydb_filter nach festen Regeln.
        """
        try:
            # Bereinigungsregeln für sst_sub
            sst_sub_rules = [
                ('MR2', ['MRG2', 'MRO2'], 'MRO2'),
                ('MR3', ['MRG3', 'MRO3'], 'MRO3'),
                ('MR4', ['MRG4', 'MRO4'], 'MRO4'),
                ('MR7', ['MRG7', 'MRO7'], 'MRO7')
            ]

            for sst, valid_subs, default_sub in sst_sub_rules:
                self.cur.execute(f"""
                UPDATE "MPSCDresden".citydb_filter
                SET sst_sub = '{default_sub}'
                WHERE sst = '{sst}' AND sst_sub NOT IN ({','.join([f"'{sub}'" for sub in valid_subs])}) AND sst_sub IS NOT NULL;
                """)

            # Bereinigung der Daten für MR5 und MR6
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter
            SET sst_sub = NULL
            WHERE sst IN ('MR5', 'MR6');
            """)

            # Umwandlung von LWS1, LWS2 und LWS3 zu LW1, LW2 und LW3
            lws_to_lw_mapping = {
                'LWS1': 'LW1',
                'LWS2': 'LW2',
                'LWS3': 'LW3'
            }

            for old_value, new_value in lws_to_lw_mapping.items():
                self.cur.execute(f"""
                UPDATE "MPSCDresden".citydb_filter
                SET sst = '{new_value}'
                WHERE sst = '{old_value}';
                """)

            self.conn.commit()
            QgsMessageLog.logMessage("sst and sst_sub columns cleaned successfully in citydb_filter", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to clean sst and sst_sub columns in citydb_filter: {str(e)}", level=Qgis.Critical)

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
                neighbor_density = 0,
                neighbor_avg_size = NULL,
                neighbor_min_distance = NULL,
                neighbor_majority_class = NULL;
            """)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Setzen der Standardwerte: {str(e)}", level=Qgis.Critical)

    def calculate_neighbors(self, buffer_distance=100):
        """
        Berechnet Nachbarschaftsmerkmale für jedes Gebäude (Dichte, Größe, Abstand, Mehrheitsklasse).
        Optimiert mit räumlichen Indizes und vorberechneten 2D-Geometrien.
        """
        try:
            # Erstelle temporäre 2D-Geometrie-Spalte für bessere Performance
            self.cur.execute("""
            ALTER TABLE "MPSCDresden".citydb_filter 
            ADD COLUMN IF NOT EXISTS geom_2d GEOMETRY(MULTIPOLYGON, 25833);
            """)
            
            self.cur.execute("""
            UPDATE "MPSCDresden".citydb_filter 
            SET geom_2d = ST_Force2D(geom)
            WHERE geom_2d IS NULL AND geom IS NOT NULL;
            """)
            
            # Erstelle räumlichen Index für 2D-Geometrien
            self.cur.execute("""
            CREATE INDEX IF NOT EXISTS citydb_filter_geom_2d_idx 
            ON "MPSCDresden".citydb_filter USING GIST (geom_2d);
            """)
            
            # Optimierte Nachbarschaftsberechnung mit vorberechneten Distanzen
            self.cur.execute("""
            WITH neighbor_data AS (
                SELECT 
                    a.gml_id AS target_gml_id,
                    a.cluster_id,
                    COUNT(b.gml_id) AS neighbor_density,
                    AVG(b.building_footprint) AS neighbor_avg_size,
                    MIN(ST_Distance(a.geom_2d, b.geom_2d)) AS neighbor_min_distance
                FROM "MPSCDresden".citydb_filter a
                JOIN "MPSCDresden".citydb_filter b 
                ON a.cluster_id = b.cluster_id
                AND ST_DWithin(a.geom_2d, b.geom_2d, %s) 
                WHERE a.gml_id != b.gml_id
                AND a.geom_2d IS NOT NULL
                AND b.geom_2d IS NOT NULL
                GROUP BY a.gml_id, a.cluster_id
            ),
            majority_class AS (
                SELECT 
                    nd.target_gml_id,
                    (
                        SELECT b.sst 
                        FROM "MPSCDresden".citydb_filter b
                        WHERE b.cluster_id = nd.cluster_id 
                        AND b.sst IS NOT NULL
                        GROUP BY b.sst 
                        ORDER BY COUNT(*) DESC 
                        LIMIT 1
                    ) AS neighbor_majority_class
                FROM neighbor_data nd
            )
            UPDATE "MPSCDresden".citydb_filter cf
            SET 
                proximity = CASE 
                    WHEN nd.neighbor_min_distance <= 1.5 THEN 'R'
                    ELSE 'E'
                END,
                neighbor_density = nd.neighbor_density,
                neighbor_avg_size = COALESCE(nd.neighbor_avg_size, NULL),
                neighbor_min_distance = COALESCE(nd.neighbor_min_distance, NULL),
                neighbor_majority_class = COALESCE(mc.neighbor_majority_class, NULL)
            FROM neighbor_data nd
            LEFT JOIN majority_class mc ON nd.target_gml_id = mc.target_gml_id
            WHERE cf.gml_id = nd.target_gml_id;
            """, (buffer_distance,))
            
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
                    WHEN ST_Perimeter(ST_Force2D(geom)) > 0 
                    THEN (4 * PI() * ST_Area(ST_Force2D(geom))) / (ST_Perimeter(ST_Force2D(geom)) * ST_Perimeter(ST_Force2D(geom)))
                    ELSE NULL 
                END,
                convexity = CASE 
                    WHEN ST_Area(ST_ConvexHull(ST_Force2D(geom))) > 0
                    THEN ST_Area(ST_Force2D(geom)) / ST_Area(ST_ConvexHull(ST_Force2D(geom)))
                    ELSE NULL 
                END,
                vertex_count = CASE 
                    WHEN ST_GeometryType(ST_Force2D(geom)) LIKE '%POLYGON%'
                    THEN ST_NPoints(ST_ExteriorRing(ST_GeometryN(ST_Force2D(geom), 1)))
                    ELSE ST_NPoints(ST_Force2D(geom))
                END,
                rectangularity = CASE 
                    WHEN ST_Area(ST_OrientedEnvelope(ST_Force2D(geom))) > 0
                    THEN ST_Area(ST_Force2D(geom)) / ST_Area(ST_OrientedEnvelope(ST_Force2D(geom)))
                    ELSE NULL 
                END
            WHERE geom IS NOT NULL 
            AND ST_IsValid(geom) 
            AND ST_Area(ST_Force2D(geom)) > 0;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Geometric features calculated successfully", level=Qgis.Info)
            self.current_step += 1
            self.update_progress()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to calculate geometric features: {str(e)}", level=Qgis.Critical)