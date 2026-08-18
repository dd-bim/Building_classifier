from .config_loader import get_schema
from qgis.core import QgsMessageLog, Qgis

class GeometryProcessor:
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den GeometryProcessor mit DB-Verbindung und Verbindungsparametern.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        self.schema = get_schema()
        self.kartierung_dd_gesamt_layer = None

    def create_built_up_parcel_table(self):
        """
        Legt die Tabelle 'built_up_parcel' an. Eine vorhandene Tabelle wird immer gedroppt,
        da der SRID von den aktuellen Quelldaten abhängt und eine alte Tabelle zu
        Koordinatensystem-Konflikten führen würde.
        """
        try:
            # Prüfen, ob `parcels` Daten hat
            self.cur.execute(f'SELECT COUNT(*) FROM "{self.schema}".parcels;')
            parcel_count = self.cur.fetchone()[0]

            if parcel_count == 0:
                QgsMessageLog.logMessage("Parcels table is empty, skipping built_up_parcel creation", level=Qgis.Warning)
                return

            # SRID aus aktuellen Quelldaten abrufen
            self.cur.execute(f'SELECT ST_SRID(geom) FROM "{self.schema}".parcels WHERE geom IS NOT NULL LIMIT 1;')
            srid = self.cur.fetchone()
            srid = srid[0] if srid else 4326

            # Immer droppen und neu anlegen (SRID muss zu Quelldaten passen)
            self.cur.execute(f'DROP TABLE IF EXISTS "{self.schema}".built_up_parcel CASCADE;')
            self.cur.execute(f"""
                CREATE TABLE "{self.schema}".built_up_parcel (
                    TOPO_ID SERIAL PRIMARY KEY,
                    GUID_ALKIS VARCHAR UNIQUE NOT NULL,
                    blocknr VARCHAR,
                    development_type VARCHAR,
                    development_type_Lv2 VARCHAR,
                    development_type_Lv3 VARCHAR,
                    development_type_Code VARCHAR,
                    geom geometry(Polygon, {srid})
                );
            """)
            self.conn.commit()
            QgsMessageLog.logMessage(f"built_up_parcel table (re)created with SRID {srid}", level=Qgis.Info)

        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error creating built_up_parcel table in CityDB: {e}", level=Qgis.Critical)
            
    def create_indexes(self):
        """
        Stellt sicher, dass GIST-Indizes auf den Geometriespalten existieren.
        """
        try:
            # Indizes für schnellere Geometrieabfragen
            self.cur.execute(f'CREATE INDEX IF NOT EXISTS idx_parcels_geom ON "{self.schema}".parcels USING GIST (geom);')
            self.cur.execute(f'CREATE INDEX IF NOT EXISTS idx_building_development_geom ON "{self.schema}".building_development USING GIST (geom);')
            self.cur.execute(f'CREATE INDEX IF NOT EXISTS idx_built_up_parcel_geom ON "{self.schema}".built_up_parcel USING GIST (geom);')
            self.conn.commit()
            QgsMessageLog.logMessage("Indexes on geometry columns ensured", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error creating indexes in CityDB: {str(e)}", level=Qgis.Critical)
        
    def process_overlapping_geometries(self):
        """
        Verarbeitet überlappende Geometrien zwischen parcels und building_development und schreibt sie in built_up_parcel.
        """
        try:
            # Prüfe, ob benötigte Tabellen existieren
            self.cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = '{self.schema}' AND table_name = 'parcels');")
            parcels_exists = self.cur.fetchone()[0]

            self.cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = '{self.schema}' AND table_name = 'building_development');")
            building_development_exists = self.cur.fetchone()[0]
            
            if not parcels_exists or not building_development_exists:
                QgsMessageLog.logMessage("Required tables are not present in the database. Overlapping geometries process skipped.", level=Qgis.Warning)
                return
        
            self.cur.execute("BEGIN;")

            # SRID-Werte abrufen
            self.cur.execute(f"SELECT Find_SRID('{self.schema}', 'parcels', 'geom');")
            parcels_srid = self.cur.fetchone()[0]

            self.cur.execute(f"SELECT Find_SRID('{self.schema}', 'building_development', 'geom');")
            building_development_srid = self.cur.fetchone()[0]
            
            # Target SRID anpassen: Verwende parcels_srid als Standard, falls beide verfügbar sind
            if parcels_srid and building_development_srid:
                target_srid = parcels_srid  # Oder building_development_srid, je nach Präferenz
            elif parcels_srid:
                target_srid = parcels_srid
            elif building_development_srid:
                target_srid = building_development_srid
            else:
                target_srid = 4326  # Fallback auf WGS84
                
            # built_up_parcel SRID abrufen, falls Tabelle existiert
            try:
                self.cur.execute(f"SELECT Find_SRID('{self.schema}', 'built_up_parcel', 'geom');")
                existing_target_srid = self.cur.fetchone()[0]
                if existing_target_srid and existing_target_srid != target_srid:
                    QgsMessageLog.logMessage(f"Warning: built_up_parcel has SRID {existing_target_srid}, but using {target_srid} from source tables", level=Qgis.Warning)
            except:
                # Tabelle existiert noch nicht, das ist normal
                pass

            QgsMessageLog.logMessage(f"SRIDs - Target: {target_srid}, Parcels: {parcels_srid}, Buildings: {building_development_srid}", level=Qgis.Info)

            batch_size = 1000 # Verarbeitung als Batch, um Zeit zu sparen

            # SQL-Query mit SRID-Transformation, falls nötig
            parcels_geom = f"ST_Transform(a.geom, {target_srid})" if parcels_srid != target_srid else "a.geom"
            building_geom = f"ST_Transform(b.geom, {target_srid})" if building_development_srid != target_srid else "b.geom"
            
            # Cursor für große Datenmengen verwenden
            overlap_query = f"""
                DECLARE overlap_cursor CURSOR FOR 
                SELECT 
                    a.id AS GUID_ALKIS,
                    b.blocknr,
                    b.sst_liste AS development_type,
                    b.sst_lv_2_liste AS development_type_Lv2,
                    b.sst_lv_3_liste AS development_type_Lv3,
                    b.desk3 AS development_type_Code,
                    {parcels_geom} AS geom
                FROM "{self.schema}".parcels a
                JOIN "{self.schema}".building_development b 
                    ON ST_Intersects({parcels_geom}, {building_geom}) 
                    AND NOT ST_Touches({parcels_geom}, {building_geom});
            """
            
            self.cur.execute(overlap_query)

            while True:
                self.cur.execute(f"FETCH {batch_size} FROM overlap_cursor;")
                rows = self.cur.fetchall()

                if not rows:
                    break

                # Batch-Insert für Performance
                self.cur.executemany(f"""
                    INSERT INTO "{self.schema}".built_up_parcel 
                    (GUID_ALKIS, blocknr, development_type, development_type_Lv2, development_type_Lv3, development_type_Code, geom)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (GUID_ALKIS) DO NOTHING;
                """, rows)

                QgsMessageLog.logMessage(f"Processed {len(rows)} rows", level=Qgis.Info)

            self.cur.execute("CLOSE overlap_cursor;")  # Cursor schließen

            # Transaktion erst nach allen Inserts committen
            self.cur.execute("COMMIT;")

            # Finales Logging
            self.cur.execute(f'SELECT COUNT(*) FROM "{self.schema}".built_up_parcel;')
            row_count = self.cur.fetchone()[0]
            QgsMessageLog.logMessage(f"Inserted {row_count} rows into built_up_parcel", level=Qgis.Info)
            QgsMessageLog.logMessage("Overlapping geometries process completed successfully", level=Qgis.Info)

        except Exception as e:
            self.conn.rollback()  # Falls ein Fehler auftritt, wird alles zurückgesetzt
            QgsMessageLog.logMessage(f"Error in overlapping geometries process: {str(e)}", level=Qgis.Critical)
                            
    def transform_built_up_parcel_table(self, target_epsg):
        """
        Transformiert die Geometriespalte der Tabelle 'built_up_parcel' auf das gewünschte EPSG-Koordinatensystem.
        """
        try:
            transform_query = f"""
            ALTER TABLE "{self.schema}".built_up_parcel
            ALTER COLUMN geom TYPE geometry(Polygon, {target_epsg})
            USING ST_Transform(geom, {target_epsg});
            """
            self.cur.execute(transform_query)
            self.conn.commit()
            QgsMessageLog.logMessage(f"Transformed 'built_up_parcel' table to EPSG:{target_epsg}", level=Qgis.Info)
        
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error transforming 'built_up_parcel' table: {str(e)}", level=Qgis.Critical)