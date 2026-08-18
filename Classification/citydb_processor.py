import os
from qgis.core import QgsMessageLog, Qgis, QgsProject
from .config_loader import get_config, get_layer_name, get_option
from .building_tracer import trace_building

class CityDBProcessor:
    """
    Bereitet die citydb_filter-Tabelle für die Gebäudeklassifikation vor.

    Diese Klasse bietet Methoden zum:
    - Erstellen und Befüllen der citydb_filter-Tabelle aus der CityDB
    - Übertragen und Bereinigen von Attributen und Geometrien
    - Berechnen von abgeleiteten Merkmalen (Feature Engineering)
    - Durchführen von Nachbarschaftsanalysen und Clustering
    """
    
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den CityDBProcessor mit DB-Verbindung und Verbindungsparametern.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur

        config = get_config()
        self.schema = config.get('Database', 'schema')
        
    def create_tables(self):
        """
        Erstellt die Tabellen citydb_mirror (alle Gebäude aus 3DCityDB, ungefiltert) und
        citydb_filter (gefilterte, angereicherte Daten für die Klassifikation).
        Bei recreate_tables=true werden vorhandene Tabellen zuerst gedroppt.
        """
        try:
            recreate = get_option('recreate_tables', 'false').strip().lower() == 'true'
            if recreate:
                self.cur.execute(f'DROP TABLE IF EXISTS "{self.schema}".citydb_mirror CASCADE;')
                self.cur.execute(f'DROP TABLE IF EXISTS "{self.schema}".citydb_filter CASCADE;')
                QgsMessageLog.logMessage("Tables citydb_mirror and citydb_filter were cleared and recreated (recreate_tables=true).", level=Qgis.Warning)

            # Spiegel-Tabelle: alle Gebäude aus 3DCityDB, ungefiltert
            self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{self.schema}".citydb_mirror (
                db_mirror_id SERIAL PRIMARY KEY,
                cityobject_id INTEGER,
                gml_id VARCHAR(255) UNIQUE,
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
                sst VARCHAR(255),
                sst_sub VARCHAR(255),
                geom GEOMETRY(MULTIPOLYGON, 25833)
            );
            """)
            self.cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_mirror_cityobject_id ON "{self.schema}".citydb_mirror (cityobject_id);
            CREATE INDEX IF NOT EXISTS idx_mirror_gml_id ON "{self.schema}".citydb_mirror (gml_id);
            CREATE INDEX IF NOT EXISTS idx_mirror_function ON "{self.schema}".citydb_mirror (function);
            CREATE INDEX IF NOT EXISTS idx_mirror_address ON "{self.schema}".citydb_mirror (address);
            CREATE INDEX IF NOT EXISTS idx_mirror_building_footprint ON "{self.schema}".citydb_mirror (building_footprint);
            CREATE INDEX IF NOT EXISTS citydb_mirror_geom_idx ON "{self.schema}".citydb_mirror USING GIST (geom);
            """)

            # Gefilterte Tabelle: nur relevante Gebäude, angereichert für die Klassifikation
            self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS "{self.schema}".citydb_filter (
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
            self.cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_cityobject_id ON "{self.schema}".citydb_filter (cityobject_id);
            CREATE INDEX IF NOT EXISTS idx_gml_id ON "{self.schema}".citydb_filter (gml_id);
            CREATE INDEX IF NOT EXISTS idx_function ON "{self.schema}".citydb_filter (function);
            CREATE INDEX IF NOT EXISTS idx_SST ON "{self.schema}".citydb_filter (SST);
            CREATE INDEX IF NOT EXISTS idx_SST_SUB ON "{self.schema}".citydb_filter (SST_SUB);
            CREATE INDEX IF NOT EXISTS idx_cluster_id ON "{self.schema}".citydb_filter (cluster_id);
            CREATE INDEX IF NOT EXISTS idx_building_footprint ON "{self.schema}".citydb_filter (building_footprint);
            CREATE INDEX IF NOT EXISTS idx_storeys_above_ground ON "{self.schema}".citydb_filter (storeys_above_ground);
            CREATE INDEX IF NOT EXISTS citydb_filter_geom_idx ON "{self.schema}".citydb_filter USING GIST (geom);
            CREATE INDEX IF NOT EXISTS idx_function_sst ON "{self.schema}".citydb_filter (function, SST);
            CREATE INDEX IF NOT EXISTS idx_cluster_sst ON "{self.schema}".citydb_filter (cluster_id, SST);
            """)

            self.conn.commit()
            QgsMessageLog.logMessage("Tables citydb_mirror and citydb_filter created and indexed successfully", level=Qgis.Info)

            # View sofort (re-)erstellen, damit bereits geladene QGIS-Layer keine Fehler werfen
            self.cur.execute(f"""
            CREATE OR REPLACE VIEW "{self.schema}".citydb_filter_view AS
            SELECT db_filter_id, gml_id, sst, classification_source_id, classification_source, confidence, geom
            FROM "{self.schema}".citydb_filter;
            """)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to create and index tables: {str(e)}", level=Qgis.Critical)
            
    def fill_mirror_table(self):
        """
        Füllt die Tabelle citydb_mirror mit allen Gebäuden aus der 3DCityDB und deren CityDB-Attributen.
        Diese Tabelle ist der ungefilterte Spiegel des 3DCityDB-Gebäudebestands. Die Filterung nach Adresse,
        Funktion und Mindestgröße erfolgt anschließend in populate_filter_from_mirror().
        """
        try:
            # Neue Gebäude identifizieren (inkrementeller Insert – bestehende Zeilen bleiben erhalten)
            self.cur.execute("DROP TABLE IF EXISTS temp_new_buildings;")
            self.cur.execute(f"""
            CREATE TEMP TABLE temp_new_buildings AS
            SELECT DISTINCT f.id AS cityobject_id
            FROM citydb.feature f
            WHERE f.objectclass_id = 901
            AND NOT EXISTS (
                SELECT 1 FROM "{self.schema}".citydb_mirror cr
                WHERE cr.cityobject_id = f.id
            );
            """)
            self.cur.execute(f"""
            INSERT INTO "{self.schema}".citydb_mirror(cityobject_id)
            SELECT cityobject_id FROM temp_new_buildings;
            """)

            # gml_id aus OBJEKT_ID (val_string)
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET gml_id = p.val_string
            FROM citydb.property p
            WHERE cr.cityobject_id = p.feature_id
            AND p.name = 'OBJEKT_ID'
            AND p.val_string IS NOT NULL
            AND cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)

            # function
            # 1. Priorität: bevorzugte Funktion (1000/1100) aus Building ODER irgendeinem BuildingPart
            # 2. Fallback: erster verfügbarer Wert aus Building ODER irgendeinem BuildingPart
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET function = COALESCE(
                (SELECT p.val_string
                 FROM citydb.property p
                 WHERE p.feature_id = cr.cityobject_id
                 AND p.name = 'function'
                 AND p.val_string IN ('1000', '1100', '31001_1000', '31001_1100')
                 UNION ALL
                 SELECT p.val_string
                 FROM citydb.property bp
                 JOIN citydb.property p ON p.feature_id = bp.val_feature_id
                 WHERE bp.feature_id = cr.cityobject_id
                 AND bp.name = 'buildingPart'
                 AND p.name = 'function'
                 AND p.val_string IN ('1000', '1100', '31001_1000', '31001_1100')
                 LIMIT 1),
                (SELECT p.val_string
                 FROM citydb.property p
                 WHERE p.feature_id = cr.cityobject_id
                 AND p.name = 'function'
                 AND p.val_string IS NOT NULL
                 UNION ALL
                 SELECT p.val_string
                 FROM citydb.property bp
                 JOIN citydb.property p ON p.feature_id = bp.val_feature_id
                 WHERE bp.feature_id = cr.cityobject_id
                 AND bp.name = 'buildingPart'
                 AND p.name = 'function'
                 AND p.val_string IS NOT NULL
                 LIMIT 1)
            )
            WHERE cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)

            # Adresse (primär aus CityDB address-Tabelle)
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET address = (
                SELECT CONCAT(a.street, ' ', COALESCE(a.house_number, ''))
                FROM citydb.property p
                JOIN citydb.address a ON p.val_address_id = a.id
                WHERE p.feature_id = cr.cityobject_id
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
                WHERE p.feature_id = cr.cityobject_id
                AND p.name = 'address'
                AND p.val_address_id IS NOT NULL
                AND a.street IS NOT NULL
                AND a.street != ''
                AND a.street != '0'
            )
            AND cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)

            # Adresse Fallback: aus Building Parts
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET address = (
                SELECT CONCAT(a.street, ' ', COALESCE(a.house_number, ''))
                FROM citydb.property bp_prop
                JOIN citydb.property p ON p.feature_id = bp_prop.val_feature_id
                JOIN citydb.address a ON p.val_address_id = a.id
                WHERE bp_prop.feature_id = cr.cityobject_id
                AND bp_prop.name = 'buildingPart'
                AND p.name = 'address'
                AND p.val_address_id IS NOT NULL
                AND a.street IS NOT NULL
                AND a.street != ''
                AND a.street != '0'
                LIMIT 1
            )
            WHERE cr.address IS NULL;
            """)

            self.cur.execute(f"SELECT COUNT(*) FROM \"{self.schema}\".citydb_mirror WHERE address IS NOT NULL;")
            buildings_with_address = self.cur.fetchone()[0]
            self.cur.execute(f"SELECT COUNT(*) FROM \"{self.schema}\".citydb_mirror;")
            total_buildings = self.cur.fetchone()[0]
            self.cur.execute(f"SELECT COUNT(*) FROM \"{self.schema}\".citydb_mirror WHERE function IN ('1000', '1100', '31001_1000', '31001_1100');")
            function_buildings = self.cur.fetchone()[0]
            self.cur.execute(f"SELECT COUNT(*) FROM \"{self.schema}\".citydb_mirror WHERE function IN ('1000', '1100', '31001_1000', '31001_1100') AND address IS NOT NULL;")
            function_buildings_with_address = self.cur.fetchone()[0]

            QgsMessageLog.logMessage(f"citydb_mirror base attributes loaded:", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Buildings total: {total_buildings}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Of which with address: {buildings_with_address}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Function 1000/1100: {function_buildings}", level=Qgis.Warning)
            QgsMessageLog.logMessage(f"- Function 1000/1100 with address: {function_buildings_with_address}", level=Qgis.Warning)

            self.conn.commit()

            # --- Geometrie (Ground Surfaces, inkl. BuildingParts) ---
            self.cur.execute(f"""
            WITH raw_ground_surfaces AS (
                SELECT
                    p.feature_id AS building_id,
                    g.geometry
                FROM citydb.property p
                JOIN citydb.feature bldg
                    ON bldg.id = p.feature_id
                    AND bldg.objectclass_id = 901
                JOIN citydb.feature ground_f ON p.val_feature_id = ground_f.id
                JOIN citydb.geometry_data g ON ground_f.id = g.feature_id
                WHERE p.name = 'boundary'
                AND ground_f.objectclass_id = 710
                AND g.geometry IS NOT NULL

                UNION ALL

                SELECT
                    bp_link.feature_id AS building_id,
                    g.geometry
                FROM citydb.property part_p
                JOIN citydb.feature ground_f ON part_p.val_feature_id = ground_f.id
                JOIN citydb.geometry_data g ON ground_f.id = g.feature_id
                JOIN citydb.property bp_link
                    ON bp_link.val_feature_id = part_p.feature_id
                    AND bp_link.name = 'buildingPart'
                WHERE part_p.name = 'boundary'
                AND ground_f.objectclass_id = 710
                AND g.geometry IS NOT NULL
            ),
            building_ground_surfaces AS (
                SELECT
                    building_id,
                    array_agg(geometry) AS all_geometries,
                    COUNT(*) AS surface_count
                FROM raw_ground_surfaces
                GROUP BY building_id
                HAVING COUNT(*) > 0
            ),
            unified_building_geometry AS (
                SELECT
                    bgs.building_id,
                    bgs.surface_count,
                    COALESCE(
                        (SELECT ST_Union(ST_Force2D(ST_MakeValid(geom)))
                         FROM unnest(bgs.all_geometries) AS geom
                         WHERE geom IS NOT NULL),
                        (SELECT ST_Union(ST_Buffer(ST_Force2D(ST_MakeValid(geom)), 0.001))
                         FROM unnest(bgs.all_geometries) AS geom
                         WHERE geom IS NOT NULL),
                        (SELECT ST_Multi(ST_Collect(ST_Force2D(ST_MakeValid(geom))))
                         FROM unnest(bgs.all_geometries) AS geom
                         WHERE geom IS NOT NULL)
                    ) AS final_geometry
                FROM building_ground_surfaces bgs
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET geom = ubg.final_geometry
            FROM unified_building_geometry ubg
            WHERE cr.cityobject_id = ubg.building_id
            AND ubg.final_geometry IS NOT NULL
            AND cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)

            # --- Grundfläche, Länge, Breite (kein DELETE – Filterung erfolgt in populate_filter_from_raw) ---
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror
            SET building_footprint = ST_Area(geom)
            WHERE geom IS NOT NULL
            AND cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror
            SET
                length_footprint = ST_XMax(ST_Envelope(geom)) - ST_XMin(ST_Envelope(geom)),
                width_footprint  = ST_YMax(ST_Envelope(geom)) - ST_YMin(ST_Envelope(geom))
            WHERE geom IS NOT NULL
            AND cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)

            # --- storeysAboveGround (+ Fallback aus Building Parts) ---
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET storeys_above_ground = (
                SELECT MAX(p.val_int)
                FROM citydb.property p
                WHERE p.feature_id = cr.cityobject_id
                AND p.name = 'storeysAboveGround'
                AND p.val_int IS NOT NULL
            )
            WHERE cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)
            self.cur.execute(f"""
            WITH part_areas AS (
                SELECT
                    bp.feature_id AS building_id,
                    bp.val_feature_id AS part_id,
                    COALESCE(SUM(ST_Area(ST_Force2D(g.geometry))), 0) AS part_area
                FROM citydb.property bp
                JOIN citydb.property pp ON pp.feature_id = bp.val_feature_id AND pp.name = 'boundary'
                JOIN citydb.feature gf ON pp.val_feature_id = gf.id AND gf.objectclass_id = 710
                JOIN citydb.geometry_data g ON gf.id = g.feature_id
                WHERE bp.name = 'buildingPart'
                GROUP BY bp.feature_id, bp.val_feature_id
            ),
            largest_part AS (
                SELECT DISTINCT ON (building_id)
                    building_id,
                    part_id
                FROM part_areas
                ORDER BY building_id, part_area DESC
            ),
            part_value AS (
                SELECT lp.building_id, p.val_int AS storeys
                FROM largest_part lp
                JOIN citydb.property p ON p.feature_id = lp.part_id
                WHERE p.name = 'storeysAboveGround'
                AND p.val_int IS NOT NULL
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET storeys_above_ground = pv.storeys
            FROM part_value pv
            WHERE cr.cityobject_id = pv.building_id
            AND cr.storeys_above_ground IS NULL;
            """)

            # --- roofType (+ Fallback aus Building Parts) ---
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET roof_type = (
                SELECT p.val_string
                FROM citydb.property p
                WHERE p.feature_id = cr.cityobject_id
                AND p.name = 'roofType'
                AND p.val_string IS NOT NULL
                LIMIT 1
            )
            WHERE cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)
            self.cur.execute(f"""
            WITH part_areas AS (
                SELECT
                    bp.feature_id AS building_id,
                    bp.val_feature_id AS part_id,
                    COALESCE(SUM(ST_Area(ST_Force2D(g.geometry))), 0) AS part_area
                FROM citydb.property bp
                JOIN citydb.property pp ON pp.feature_id = bp.val_feature_id AND pp.name = 'boundary'
                JOIN citydb.feature gf ON pp.val_feature_id = gf.id AND gf.objectclass_id = 710
                JOIN citydb.geometry_data g ON gf.id = g.feature_id
                WHERE bp.name = 'buildingPart'
                GROUP BY bp.feature_id, bp.val_feature_id
            ),
            largest_part AS (
                SELECT DISTINCT ON (building_id)
                    building_id,
                    part_id
                FROM part_areas
                ORDER BY building_id, part_area DESC
            ),
            part_value AS (
                SELECT lp.building_id, p.val_string AS roof_type
                FROM largest_part lp
                JOIN citydb.property p ON p.feature_id = lp.part_id
                WHERE p.name = 'roofType'
                AND p.val_string IS NOT NULL
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET roof_type = pv.roof_type
            FROM part_value pv
            WHERE cr.cityobject_id = pv.building_id
            AND cr.roof_type IS NULL;
            """)

            # --- roof_ridge_height (3 Fallbacks) ---
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET roof_ridge_height = (
                SELECT MAX(p.val_double)
                FROM citydb.property p
                WHERE p.feature_id = cr.cityobject_id
                AND p.name = 'value'
                AND p.parent_id IN (
                    SELECT p2.id FROM citydb.property p2
                    WHERE p2.feature_id = cr.cityobject_id
                    AND p2.name = 'height'
                )
                AND p.val_double IS NOT NULL
            )
            WHERE cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)
            self.cur.execute(f"""
            WITH part_areas AS (
                SELECT
                    bp.feature_id AS building_id,
                    bp.val_feature_id AS part_id,
                    COALESCE(SUM(ST_Area(ST_Force2D(g.geometry))), 0) AS part_area
                FROM citydb.property bp
                JOIN citydb.property pp ON pp.feature_id = bp.val_feature_id AND pp.name = 'boundary'
                JOIN citydb.feature gf ON pp.val_feature_id = gf.id AND gf.objectclass_id = 710
                JOIN citydb.geometry_data g ON gf.id = g.feature_id
                WHERE bp.name = 'buildingPart'
                GROUP BY bp.feature_id, bp.val_feature_id
            ),
            largest_part AS (
                SELECT DISTINCT ON (building_id)
                    building_id,
                    part_id
                FROM part_areas
                ORDER BY building_id, part_area DESC
            ),
            part_value AS (
                SELECT lp.building_id, MAX(p.val_double) AS ridge_height
                FROM largest_part lp
                JOIN citydb.property height_p ON height_p.feature_id = lp.part_id AND height_p.name = 'height'
                JOIN citydb.property p ON p.parent_id = height_p.id AND p.name = 'value'
                WHERE p.val_double IS NOT NULL
                GROUP BY lp.building_id
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET roof_ridge_height = pv.ridge_height
            FROM part_value pv
            WHERE cr.cityobject_id = pv.building_id
            AND cr.roof_ridge_height IS NULL;
            """)
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET roof_ridge_height = (
                SELECT MAX(CAST(p.val_string AS DOUBLE PRECISION))
                FROM citydb.property p
                WHERE p.feature_id = cr.cityobject_id
                AND p.name = 'MaxHBuilding'
                AND p.val_string IS NOT NULL
            )
            WHERE cr.roof_ridge_height IS NULL;
            """)
            self.cur.execute(f"""
            WITH part_areas AS (
                SELECT
                    bp.feature_id AS building_id,
                    bp.val_feature_id AS part_id,
                    COALESCE(SUM(ST_Area(ST_Force2D(g.geometry))), 0) AS part_area
                FROM citydb.property bp
                JOIN citydb.property pp ON pp.feature_id = bp.val_feature_id AND pp.name = 'boundary'
                JOIN citydb.feature gf ON pp.val_feature_id = gf.id AND gf.objectclass_id = 710
                JOIN citydb.geometry_data g ON gf.id = g.feature_id
                WHERE bp.name = 'buildingPart'
                GROUP BY bp.feature_id, bp.val_feature_id
            ),
            largest_part AS (
                SELECT DISTINCT ON (building_id)
                    building_id,
                    part_id
                FROM part_areas
                ORDER BY building_id, part_area DESC
            ),
            part_value AS (
                SELECT lp.building_id, MAX(CAST(p.val_string AS DOUBLE PRECISION)) AS ridge_height
                FROM largest_part lp
                JOIN citydb.property p ON p.feature_id = lp.part_id
                WHERE p.name = 'MaxHBuilding'
                AND p.val_string IS NOT NULL
                GROUP BY lp.building_id
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET roof_ridge_height = pv.ridge_height
            FROM part_value pv
            WHERE cr.cityobject_id = pv.building_id
            AND cr.roof_ridge_height IS NULL;
            """)

            # --- eaves_height ---
            # Primär: MAX(Z_Min) aus direkten Dachflächen des Gebäudes
            self.cur.execute(f"""
            WITH direct_eaves AS (
                SELECT
                    p.feature_id AS building_id,
                    MAX(CAST(prop.val_string AS DOUBLE PRECISION)) AS eaves_height
                FROM citydb.property p
                JOIN citydb.feature rf ON p.val_feature_id = rf.id AND rf.objectclass_id = 712
                JOIN citydb.property prop ON prop.feature_id = rf.id AND prop.name = 'Z_Min'
                WHERE p.name = 'boundary'
                AND prop.val_string IS NOT NULL
                GROUP BY p.feature_id
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET eaves_height = de.eaves_height
            FROM direct_eaves de
            WHERE cr.cityobject_id = de.building_id
            AND (cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings)
                 OR cr.eaves_height IS NULL);
            """)
            # Fallback: Z_Min vom größten BuildingPart
            self.cur.execute(f"""
            WITH part_areas AS (
                SELECT
                    bp.feature_id AS building_id,
                    bp.val_feature_id AS part_id,
                    COALESCE(SUM(ST_Area(ST_Force2D(g.geometry))), 0) AS part_area
                FROM citydb.property bp
                JOIN citydb.property pp ON pp.feature_id = bp.val_feature_id AND pp.name = 'boundary'
                JOIN citydb.feature gf ON pp.val_feature_id = gf.id AND gf.objectclass_id = 710
                JOIN citydb.geometry_data g ON gf.id = g.feature_id
                WHERE bp.name = 'buildingPart'
                GROUP BY bp.feature_id, bp.val_feature_id
            ),
            largest_part AS (
                SELECT DISTINCT ON (building_id)
                    building_id,
                    part_id
                FROM part_areas
                ORDER BY building_id, part_area DESC
            ),
            part_eaves AS (
                SELECT
                    lp.building_id,
                    MAX(CAST(prop.val_string AS DOUBLE PRECISION)) AS eaves_height
                FROM largest_part lp
                JOIN citydb.property pp ON pp.feature_id = lp.part_id AND pp.name = 'boundary'
                JOIN citydb.feature rf ON pp.val_feature_id = rf.id AND rf.objectclass_id = 712
                JOIN citydb.property prop ON prop.feature_id = rf.id AND prop.name = 'Z_Min'
                WHERE prop.val_string IS NOT NULL
                GROUP BY lp.building_id
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET eaves_height = pe.eaves_height
            FROM part_eaves pe
            WHERE cr.cityobject_id = pe.building_id
            AND cr.eaves_height IS NULL;
            """)

            # --- storey_height ---
            # Immer neu berechnen für alle Gebäude mit gültigen Eingabewerten.
            # Kein storey_height IS NULL-Guard, da sich eaves_height oder storeys_above_ground
            # zwischen Läufen ändern können. Kostet nichts, da nur eine Division.
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror cr
            SET storey_height = cr.eaves_height / cr.storeys_above_ground
            WHERE cr.eaves_height IS NOT NULL
            AND cr.storeys_above_ground IS NOT NULL
            AND cr.storeys_above_ground > 0;
            """)

            # --- number_roof_surfaces ---
            # Zählt Dachflächen aus direkten Boundary-Links und BuildingParts (dedupliziert)
            self.cur.execute(f"""
            WITH direct_roofs AS (
                SELECT
                    p.feature_id AS building_id,
                    f.id AS roof_id
                FROM citydb.property p
                JOIN citydb.feature f ON p.val_feature_id = f.id AND f.objectclass_id = 712
                WHERE p.name = 'boundary'
            ),
            part_roofs AS (
                SELECT
                    bp.feature_id AS building_id,
                    f.id AS roof_id
                FROM citydb.property bp
                JOIN citydb.property pp ON pp.feature_id = bp.val_feature_id AND pp.name = 'boundary'
                JOIN citydb.feature f ON pp.val_feature_id = f.id AND f.objectclass_id = 712
                WHERE bp.name = 'buildingPart'
            ),
            all_roofs AS (
                SELECT building_id, roof_id FROM direct_roofs
                UNION
                SELECT building_id, roof_id FROM part_roofs
            ),
            roof_surface_counts AS (
                SELECT building_id, COUNT(roof_id) AS number_roof_surfaces
                FROM all_roofs
                GROUP BY building_id
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET number_roof_surfaces = rsc.number_roof_surfaces
            FROM roof_surface_counts rsc
            WHERE cr.cityobject_id = rsc.building_id
            AND cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)

            # --- roof_slope ---
            # Ermittelt NORMAL_H der flächenmäßig größten Dachfläche je Gebäude
            # inkl. Dachflächen aus BuildingParts
            self.cur.execute(f"""
            WITH direct_boundary AS (
                SELECT
                    p.feature_id AS building_id,
                    f.id AS roof_id,
                    ST_Area(f.envelope) AS surface_area
                FROM citydb.property p
                JOIN citydb.feature f ON p.val_feature_id = f.id AND f.objectclass_id = 712
                WHERE p.name = 'boundary'
            ),
            part_boundary AS (
                SELECT
                    bp.feature_id AS building_id,
                    f.id AS roof_id,
                    ST_Area(f.envelope) AS surface_area
                FROM citydb.property bp
                JOIN citydb.property pp ON pp.feature_id = bp.val_feature_id AND pp.name = 'boundary'
                JOIN citydb.feature f ON pp.val_feature_id = f.id AND f.objectclass_id = 712
                WHERE bp.name = 'buildingPart'
            ),
            all_surfaces AS (
                SELECT building_id, roof_id, surface_area FROM direct_boundary
                UNION ALL
                SELECT building_id, roof_id, surface_area FROM part_boundary
            ),
            largest_surface AS (
                SELECT DISTINCT ON (building_id)
                    building_id,
                    roof_id
                FROM all_surfaces
                WHERE surface_area IS NOT NULL
                ORDER BY building_id, surface_area DESC
            ),
            roof_slope_values AS (
                SELECT
                    ls.building_id,
                    CAST(p.val_string AS DOUBLE PRECISION) AS roof_slope
                FROM largest_surface ls
                JOIN citydb.property p ON ls.roof_id = p.feature_id
                WHERE p.name = 'NORMAL_H'
                AND p.val_string IS NOT NULL
            )
            UPDATE "{self.schema}".citydb_mirror cr
            SET roof_slope = rsv.roof_slope
            FROM roof_slope_values rsv
            WHERE cr.cityobject_id = rsv.building_id
            AND cr.cityobject_id IN (SELECT cityobject_id FROM temp_new_buildings);
            """)

            self.cur.execute("SELECT COUNT(*) FROM temp_new_buildings;")
            new_count = self.cur.fetchone()[0]
            self.cur.execute("DROP TABLE IF EXISTS temp_new_buildings;")
            self.conn.commit()

            self.cur.execute(f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE geom IS NOT NULL) AS with_geom,
                ROUND(100.0 * COUNT(*) FILTER (WHERE geom IS NOT NULL) / NULLIF(COUNT(*), 0), 2) AS rate
            FROM "{self.schema}".citydb_mirror;
            """)
            result = self.cur.fetchone()
            QgsMessageLog.logMessage(
                f"citydb_mirror updated ({new_count} new buildings added):\n"
                f"- Buildings total: {result[0]}\n"
                f"- With geometry: {result[1]}\n"
                f"- Success rate: {result[2]}%",
                level=Qgis.Info
            )
        except Exception as e:
            self.cur.execute("DROP TABLE IF EXISTS temp_new_buildings;")
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to fill citydb_mirror: {str(e)}", level=Qgis.Critical)

    def _process_address_layer(self, layer_key):
        """
        Hilfsmethode: Verarbeitet einen Adress-Ergänzungs-Layer.
        - Gebäude bereits in citydb_mirror (gml_id = Objekt_ID): Adresse wird gesetzt, sofern noch NULL.
        - Gebäude noch nicht in citydb_mirror: werden mit gml_id und Adresse neu eingefügt.
          Weitere Attribute (SST, Funktion etc.) folgen in intersect_and_update_citydb_filter.
        :param layer_key: Schlüssel des Layers in config.ini (z.B. 'ergaenzung_adressen')
        :return: (inserted, updated, skipped) Zähler
        """
        layer_name = get_layer_name(layer_key)
        layers = QgsProject.instance().mapLayersByName(layer_name)
        if not layers:
            QgsMessageLog.logMessage(f"Layer '{layer_name}' not found in project.", level=Qgis.Warning)
            return 0, 0, 0

        layer = layers[0]
        inserted = 0
        updated = 0
        skipped = 0

        for feature in layer.getFeatures():
            objekt_id = feature['Objekt_ID']
            strasse = feature['Strasse'] or ''
            hausnummer = feature['Hausnummer'] or ''
            address = f"{strasse} {hausnummer}".strip()

            if not objekt_id or not strasse:
                skipped += 1
                continue

            # Prüfen ob Gebäude bereits in citydb_mirror vorhanden
            self.cur.execute(
                f'SELECT 1 FROM "{self.schema}".citydb_mirror WHERE gml_id = %s',
                (objekt_id,)
            )
            if self.cur.fetchone():
                # Adresse aktualisieren, falls noch nicht gesetzt
                self.cur.execute(f"""
                    UPDATE "{self.schema}".citydb_mirror
                    SET address = %s
                    WHERE gml_id = %s
                    AND address IS NULL;
                """, (address, objekt_id))
                updated += self.cur.rowcount
            else:
                # Neues Gebäude einfügen – function='1000' (Wohngebäude), da beide Ergänzungs-Layer
                # per Definition Wohngebäude enthalten. Weitere Attribute (SST etc.) folgen in
                # intersect_and_update_citydb_filter.
                self.cur.execute(f"""
                    INSERT INTO "{self.schema}".citydb_mirror (gml_id, address, function)
                    VALUES (%s, %s, '31001_1000');
                """, (objekt_id, address))
                inserted += self.cur.rowcount

        return inserted, updated, skipped

    def update_address_from_shp(self):
        """
        Ergänzt fehlende Adressen in citydb_mirror anhand der QGIS-Layer
        'Ergaenzung_HH_Addressen' und 'Ergaenzung_Addressen_WG_gr30qm'.
        Gebäude, die noch nicht in citydb_mirror enthalten sind, werden neu eingefügt.
        Weitere Attribute (SST, Funktion etc.) werden anschließend durch
        intersect_and_update_citydb_filter befüllt.
        """
        try:
            inserted, updated, skipped = self._process_address_layer('ergaenzung_adressen')
            self.conn.commit()
            QgsMessageLog.logMessage(
                f"update_address_from_shp (HH): {inserted} newly inserted, {updated} addresses added, {skipped} skipped.",
                level=Qgis.Info
            )

            inserted_wg, updated_wg, skipped_wg = self._process_address_layer('ergaenzung_adressen_30')
            self.conn.commit()
            QgsMessageLog.logMessage(
                f"update_address_from_shp (WG): {inserted_wg} newly inserted, {updated_wg} addresses added, {skipped_wg} skipped.",
                level=Qgis.Info
            )
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error in update_address_from_shp: {str(e)}", level=Qgis.Critical)

    def apply_kartierung_to_mirror(self):
        """
        Überträgt sst/sst_sub aus kartierung_dd_gesamt in citydb_mirror
        via id_alkis-Abgleich oder Adressabgleich.
        Muss vor populate_filter_from_mirror aufgerufen werden, damit kartierte
        Gebäude beim Filtern erkannt und unabhängig von Funktion/Grundfläche
        übernommen werden.
        """
        try:
            # Spalten in citydb_mirror sicherstellen
            self.cur.execute(f"""
                ALTER TABLE "{self.schema}".citydb_mirror
                ADD COLUMN IF NOT EXISTS sst VARCHAR(255),
                ADD COLUMN IF NOT EXISTS sst_sub VARCHAR(255);
            """)
            self.conn.commit()

            # sst/sst_sub aus kartierung_dd_gesamt in citydb_mirror schreiben
            self.cur.execute(f"""
                UPDATE "{self.schema}".citydb_mirror m
                SET sst     = k.sst,
                    sst_sub = k.sst_sub
                FROM "{self.schema}".kartierung_dd_gesamt k
                WHERE k.id_alkis = m.gml_id
                   OR (
                       k.id_alkis IS NULL
                       AND m.address IS NOT NULL
                       AND trim(lower(k.str || ' ' || k.hnr)) = trim(lower(m.address))
                   );
            """)
            updated = self.cur.rowcount
            self.conn.commit()
            QgsMessageLog.logMessage(
                f"apply_kartierung_to_mirror: {updated} buildings in citydb_mirror updated with sst/sst_sub from kartierung_dd_gesamt.",
                level=Qgis.Info
            )
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error in apply_kartierung_to_mirror: {str(e)}", level=Qgis.Critical)
            
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
            UPDATE "{self.schema}".citydb_mirror
            SET sst = sst_sub
            WHERE sst_sub IS NOT NULL 
            AND sst_sub IN ({valid_str});
            """)
            
            # STEP 3: LWS → LW conversion
            lws_mapping = {'LWS1': 'LW1', 'LWS2': 'LW2', 'LWS3': 'LW3'}
            
            for old, new in lws_mapping.items():
                self.cur.execute(f"""
                UPDATE "{self.schema}".citydb_mirror
                SET sst = '{new}'
                WHERE sst = '{old}';
                """)
        
            # STEP 4: Clear sst_sub column
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_mirror
            SET sst_sub = NULL;
            """)
            
            self.conn.commit()
            QgsMessageLog.logMessage("SST data cleaning completed successfully", level=Qgis.Info)
            
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to clean SST data: {str(e)}", level=Qgis.Critical)
            
    def populate_filter_from_mirror(self):
        """
        Überführt Gebäude aus citydb_mirror in citydb_filter.
        Kartierte Gebäude (sst IS NOT NULL in citydb_mirror, gesetzt durch apply_kartierung_to_mirror)
        werden immer übernommen – unabhängig von Adresse, Funktion oder Grundfläche.
        Nicht-kartierte Gebäude durchlaufen die normale Filterlogik:
        - Adresse vorhanden
        - Funktion 1000/1100
        - Grundfläche >= 30 m² (oder NULL)
        Die Tabelle citydb_filter wird vor dem INSERT geleert.
        """
        try:
            self.cur.execute(f'TRUNCATE "{self.schema}".citydb_filter RESTART IDENTITY CASCADE;')

            # --- Diagnose: Filteranalyse ---
            self.cur.execute(f"""
            SELECT
                COUNT(*)                                                                             AS total,
                COUNT(*) FILTER (WHERE sst IS NOT NULL)                                              AS kartiert,
                COUNT(*) FILTER (WHERE sst IS NULL AND address IS NULL)                              AS no_address,
                COUNT(*) FILTER (WHERE sst IS NULL AND address IS NOT NULL
                    AND function IN ('1000', '1100', '31001_1000', '31001_1100')
                    AND building_footprint IS NOT NULL AND building_footprint < 30)                  AS func_too_small,
                COUNT(*) FILTER (WHERE sst IS NULL AND address IS NOT NULL
                    AND function IN ('1000', '1100', '31001_1000', '31001_1100')
                    AND (building_footprint IS NULL OR building_footprint >= 30))                    AS func_match
            FROM "{self.schema}".citydb_mirror;
            """)
            d = self.cur.fetchone()
            total_mirror, kartiert, no_addr, func_too_small, func_match = d
            QgsMessageLog.logMessage(
                f"populate_filter_from_mirror – filter analysis ({total_mirror} buildings in citydb_mirror):\n"
                f"  Surveyed (sst set, always adopted):              {kartiert}\n"
                f"  Not surveyed, no address (filtered out):         {no_addr}\n"
                f"  Function 1000/1100, footprint under 30 m2:       {func_too_small}  (filtered out)\n"
                f"  Function 1000/1100, footprint >= 30 or NULL:     {func_match}  (adopted)",
                level=Qgis.Warning
            )

            self.cur.execute(f"""
            INSERT INTO "{self.schema}".citydb_filter (
                cityobject_id, gml_id, function, address, roof_type,
                storeys_above_ground, building_footprint, length_footprint, width_footprint,
                roof_ridge_height, eaves_height, storey_height, number_roof_surfaces,
                roof_slope, sst, sst_sub, geom
            )
            SELECT
                r.cityobject_id, r.gml_id, r.function, r.address, r.roof_type,
                r.storeys_above_ground, r.building_footprint, r.length_footprint, r.width_footprint,
                r.roof_ridge_height, r.eaves_height,
                CASE
                    WHEN r.eaves_height IS NOT NULL
                     AND r.storeys_above_ground IS NOT NULL
                     AND r.storeys_above_ground > 0
                    THEN r.eaves_height / r.storeys_above_ground
                    ELSE r.storey_height
                END AS storey_height,
                r.number_roof_surfaces,
                r.roof_slope, r.sst, r.sst_sub, r.geom
            FROM "{self.schema}".citydb_mirror r
            WHERE
                -- Kartierte Gebäude: immer übernehmen, unabhängig von Adresse/Funktion/Grundfläche
                r.sst IS NOT NULL
                OR (
                    -- Normale Filterlogik für nicht-kartierte Gebäude
                    r.address IS NOT NULL
                    AND r.function IN ('1000', '1100', '31001_1000', '31001_1100')
                    AND (r.building_footprint IS NULL OR r.building_footprint >= 30)
                );
            """)

            self.cur.execute(f'SELECT COUNT(*) FROM "{self.schema}".citydb_filter;')
            count = self.cur.fetchone()[0]
            self.conn.commit()
            filtered_out = total_mirror - count
            QgsMessageLog.logMessage(
                f"populate_filter_from_mirror – result:\n"
                f"  TOTAL in citydb_filter:    {count} (of {total_mirror} = {round(100.0 * count / total_mirror, 1) if total_mirror else 0} %)\n"
                f"  Filtered out total:        {filtered_out}",
                level=Qgis.Warning
            )
            self.trace_building()  # DIAGNOSE
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to populate citydb_filter from citydb_mirror: {str(e)}", level=Qgis.Critical)

    def intersect_and_update_citydb_filter(self):
        """
        Überträgt development_type_code aus kartierung_dd_gesamt nach citydb_filter und setzt
        Quelle Kartierung (1) für übernommene sst.
        sst/sst_sub werden hier NICHT mehr aus kartierung_dd_gesamt übernommen: Sie kommen bereits
        bereinigt aus citydb_mirror (apply_kartierung_to_mirror + clean_sst_data) über
        populate_filter_from_mirror in citydb_filter an. Ein erneutes Überschreiben aus
        kartierung_dd_gesamt würde die dortigen unbereinigten Rohwerte (z.B. LWS statt LW)
        zurück nach citydb_filter holen.
        """
        try:
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_filter cf
            SET
                mapping_id = k."id",
                development_type_code = k."development_type_code"
            FROM "{self.schema}".kartierung_dd_gesamt k
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
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_filter
            SET classification_source_id = 1,
                classification_source = 'Kartierung'
            WHERE sst IS NOT NULL
              AND (classification_source_id IS NULL OR classification_source_id <> 1);
            """)
            self.conn.commit()

            QgsMessageLog.logMessage("citydb_filter updated with Kartierung development_type_code + source.", level=Qgis.Info)
            self.trace_building()  # DIAGNOSE
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error intersecting and updating citydb_filter: {str(e)}", level=Qgis.Critical)

    def calculate_clusters(self, buffer_distance=100):
        """
        Berechnet Cluster-IDs für Gebäude basierend auf räumlicher Nähe.
        """
        try:
            self.cur.execute(f"""
            WITH clusters AS (
                SELECT 
                    gml_id, 
                    ST_ClusterDBSCAN(geom, eps := %s, minpoints := 1) OVER () AS cluster_id
                FROM "{self.schema}".citydb_filter
            )
            UPDATE "{self.schema}".citydb_filter cf
            SET cluster_id = c.cluster_id
            FROM clusters c
            WHERE cf.gml_id = c.gml_id;
            """, (buffer_distance,))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error in cluster calculation: {str(e)}", level=Qgis.Critical)

    def set_default_values(self):
        """
        Setzt Standardwerte für Nachbarschaftsattribute in citydb_filter.
        """
        try:
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_filter
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
            QgsMessageLog.logMessage(f"Error setting default values: {str(e)}", level=Qgis.Critical)

    def calculate_neighbours(self, buffer_distance=100):
        """
        Berechnet Nachbarschaftsmerkmale für jedes Gebäude (Dichte, Größe, Abstand, Mehrheitsklasse).
        Optimiert mit räumlichen Indizes und 2D-Geometrien.
        """
        try:
            # Optimierte Nachbarschaftsberechnung ohne temporäre 2D-Geometrie
            # Schritt 1: Basis-Nachbarschaftsdaten berechnen
            self.cur.execute(f"""
            WITH neighbour_data AS (
                SELECT 
                    a.gml_id AS target_gml_id,
                    a.cluster_id,
                    COUNT(b.gml_id) AS neighbour_density,
                    AVG(b.building_footprint) AS neighbour_avg_size,
                    MIN(ST_Distance(a.geom, b.geom)) AS neighbour_min_distance
                FROM "{self.schema}".citydb_filter a
                JOIN "{self.schema}".citydb_filter b 
                ON a.cluster_id = b.cluster_id
                AND ST_DWithin(a.geom, b.geom, %s) 
                WHERE a.gml_id != b.gml_id
                AND a.geom IS NOT NULL
                AND b.geom IS NOT NULL
                GROUP BY a.gml_id, a.cluster_id
            )
            UPDATE "{self.schema}".citydb_filter cf
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
            self.cur.execute(f"""
            WITH close_neighbours AS (
                SELECT 
                    a.gml_id AS target_gml_id,
                    COUNT(b.gml_id) AS neighbouring_buildings
                FROM "{self.schema}".citydb_filter a
                JOIN "{self.schema}".citydb_filter b 
                ON ST_DWithin(a.geom, b.geom, 1.5)
                WHERE a.gml_id != b.gml_id
                AND a.geom IS NOT NULL
                AND b.geom IS NOT NULL
                GROUP BY a.gml_id
            )
            UPDATE "{self.schema}".citydb_filter cf
            SET neighbouring_buildings = COALESCE(cn.neighbouring_buildings, 0)
            FROM close_neighbours cn
            WHERE cf.gml_id = cn.target_gml_id;
            """)
            
            self.conn.commit()
            
            # Schritt 3: Mehrheitsklasse berechnen (vereinfacht)
            self.cur.execute(f"""
            WITH cluster_majority AS (
                SELECT 
                    cluster_id,
                    sst,
                    COUNT(*) as class_count,
                    ROW_NUMBER() OVER (PARTITION BY cluster_id ORDER BY COUNT(*) DESC) as rank
                FROM "{self.schema}".citydb_filter 
                WHERE sst IS NOT NULL 
                GROUP BY cluster_id, sst
            ),
            majority_per_cluster AS (
                SELECT cluster_id, sst as majority_sst
                FROM cluster_majority 
                WHERE rank = 1
            )
            UPDATE "{self.schema}".citydb_filter cf
            SET neighbour_majority_class = mpc.majority_sst
            FROM majority_per_cluster mpc
            WHERE cf.cluster_id = mpc.cluster_id;
            """)
            
            self.conn.commit()
            QgsMessageLog.logMessage("Neighbourhood features successfully calculated & saved.", level=Qgis.Info)

        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error calculating neighbourhood features: {str(e)}", level=Qgis.Critical)

    def correct_invalid_mr_sst(self):
        """
        Korrigiert invalide MR-Werte (MR2, MR3, MR4, MR7) in citydb_filter.sst, für die
        beim Mapping/clean_sst_data keine sst_sub-Unterklasse (MRO/MRG) vorlag.
        Die Zuordnung erfolgt anhand der Nachbarschaftsbeziehung (neighbouring_buildings),
        analog zur Vergabe in validate_data.split_mr_leaf_truth / classify_data.predict_level_1111:
        - neighbouring_buildings < 2  -> MRO{Baualtersstufe} (offen/freistehend)
        - neighbouring_buildings >= 2 -> MRG{Baualtersstufe} (geschlossen/Reihenhaus)
        Gebäude ohne bekannte neighbouring_buildings bleiben unverändert (invalide),
        da keine Nachbarschaftsinformation zur Entscheidung vorliegt.
        Muss nach calculate_neighbours() aufgerufen werden, da neighbouring_buildings
        erst dort berechnet wird.
        """
        try:
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_filter
            SET sst = CASE
                    WHEN neighbouring_buildings < 2 THEN 'MRO' || SUBSTRING(sst FROM 3)
                    ELSE 'MRG' || SUBSTRING(sst FROM 3)
                END
            WHERE sst IN ('MR2', 'MR3', 'MR4', 'MR7')
            AND neighbouring_buildings IS NOT NULL;
            """)
            corrected = self.cur.rowcount
            self.conn.commit()
            QgsMessageLog.logMessage(
                f"correct_invalid_mr_sst: {corrected} invalid MR values corrected to MRO/MRG based on neighbourhood.",
                level=Qgis.Info
            )
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error in correct_invalid_mr_sst: {str(e)}", level=Qgis.Critical)

    def add_feature_engineering_attributes(self):
        """
        Fügt abgeleitete Merkmale (Feature Engineering) zu citydb_filter hinzu.
        """
        try:
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_filter
            SET 
                ground_area_per_storey = building_footprint / NULLIF(storeys_above_ground, 0),
                footprint_ratio = length_footprint / NULLIF(width_footprint, 0),
                height_to_area_ratio = roof_ridge_height / NULLIF(building_footprint, 0),
                roof_height_ratio = (roof_ridge_height - eaves_height) / NULLIF(roof_ridge_height, 0),
                building_volume = building_footprint * roof_ridge_height
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("Feature engineering attributes added successfully to citydb_filter", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to add feature engineering attributes to citydb_filter: {str(e)}", level=Qgis.Critical)
    
    # ==========================================================================
    # DIAGNOSE – gesamten Block auskommentieren zum Deaktivieren
    # ==========================================================================
    def trace_building(self):
        """Delegiert an building_tracer.py. OBJEKT_ID hier eintragen und Methode aufrufen."""
        # ===== HIER OBJEKT_ID EINTRAGEN =====
        OBJEKT_ID = "DESNALK0q5001lVe"   # <-- anpassen
        # ====================================
        trace_building(self.cur, self.schema, OBJEKT_ID)
    # ==========================================================================

    def calculate_geometric_features(self):
        """
        Berechnet geometrische Features aus der Gebäudegrundrissgeometrie für das Machine Learning.
        Optimiert mit robusten Berechnungen und besserer Fehlerbehandlung.
        """
        try:
            self.cur.execute(f"""
            UPDATE "{self.schema}".citydb_filter
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
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Failed to calculate geometric features: {str(e)}", level=Qgis.Critical)