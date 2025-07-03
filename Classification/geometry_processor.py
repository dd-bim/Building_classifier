from qgis.core import QgsMessageLog, Qgis

class GeometryProcessor:
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den GeometryProcessor mit DB-Verbindung und Verbindungsparametern.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur
        self.kartierung_dd_gesamt_layer = None

    def create_schema_if_not_exists(self):
        """
        Legt die Tabellen 'parcels' und 'building_development' im Schema an, falls sie nicht existieren.
        """
        try:                        
            create_parcels_table_query = """
            CREATE TABLE IF NOT EXISTS "MPSCDresden".parcels (
                id SERIAL PRIMARY KEY,
                geom geometry(Polygon, 4326)
            );
            """
            self.cur.execute(create_parcels_table_query)
            
            create_building_development_table_query = """
            CREATE TABLE IF NOT EXISTS "MPSCDresden".building_development (
                id SERIAL PRIMARY KEY,
                geom geometry(Polygon, 4326)
            );
            """
            self.cur.execute(create_building_development_table_query)
            self.conn.commit()
            QgsMessageLog.logMessage("Tables 'parcels' and 'building_development' created", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error creating tables in CityDB: {str(e)}", level=Qgis.Critical)

    def create_built_up_parcel_table(self):
        """
        Legt die Tabelle 'built_up_parcel' an, falls sie nicht existiert und Daten vorhanden sind.
        """
        try:
            self.cur.execute('SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s)', ('MPSCDresden', 'built_up_parcel'))
            exists = self.cur.fetchone()[0]
            
            if not exists:
                # Prüfen, ob `parcels` Daten hat
                self.cur.execute('SELECT COUNT(*) FROM "MPSCDresden".parcels;')
                parcel_count = self.cur.fetchone()[0]
                
                if parcel_count == 0:
                    QgsMessageLog.logMessage("Parcels table is empty, skipping built_up_parcel creation", level=Qgis.Warning)
                    return
                
                # SRID sicher abrufen
                self.cur.execute('SELECT ST_SRID(geom) FROM "MPSCDresden".parcels WHERE geom IS NOT NULL LIMIT 1;')
                srid = self.cur.fetchone()
                srid = srid[0] if srid else 4326  # Standardwert setzen

                create_table_query = f"""
                CREATE TABLE "MPSCDresden".built_up_parcel (
                    TOPO_ID SERIAL PRIMARY KEY,
                    GUID_ALKIS VARCHAR UNIQUE NOT NULL,
                    blocknr VARCHAR,
                    development_type VARCHAR,
                    development_type_Lv2 VARCHAR,
                    development_type_Lv3 VARCHAR,
                    development_type_Code VARCHAR,
                    geom geometry(Polygon, {srid})
                );
                """
                self.cur.execute(create_table_query)
                self.conn.commit()
                QgsMessageLog.logMessage("built_up_parcel table created in CityDB", level=Qgis.Info)
            else:
                QgsMessageLog.logMessage("built_up_parcel table already exists in CityDB", level=Qgis.Info)

        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error creating built_up_parcel table in CityDB: {e}", level=Qgis.Critical)
            
    def create_indexes(self):
        """
        Stellt sicher, dass GIST-Indizes auf den Geometriespalten existieren.
        """
        try:
            # Indizes für schnellere Geometrieabfragen
            self.cur.execute('CREATE INDEX IF NOT EXISTS idx_parcels_geom ON "MPSCDresden".parcels USING GIST (geom);')
            self.cur.execute('CREATE INDEX IF NOT EXISTS idx_building_development_geom ON "MPSCDresden".building_development USING GIST (geom);')
            self.cur.execute('CREATE INDEX IF NOT EXISTS idx_built_up_parcel_geom ON "MPSCDresden".built_up_parcel USING GIST (geom);')
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
            self.cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'MPSCDresden' AND table_name = 'parcels');")
            parcels_exists = self.cur.fetchone()[0]
            
            self.cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'MPSCDresden' AND table_name = 'building_development');")
            building_development_exists = self.cur.fetchone()[0]
            
            if not parcels_exists or not building_development_exists:
                QgsMessageLog.logMessage("Required tables are not present in the database. Overlapping geometries process skipped.", level=Qgis.Warning)
                return
        
            self.cur.execute("BEGIN;")

            # SRID-Werte abrufen
            self.cur.execute("SELECT Find_SRID('MPSCDresden', 'built_up_parcel', 'geom');")
            target_srid = self.cur.fetchone()[0]

            self.cur.execute("SELECT Find_SRID('MPSCDresden', 'parcels', 'geom');")
            parcels_srid = self.cur.fetchone()[0]

            self.cur.execute("SELECT Find_SRID('MPSCDresden', 'building_development', 'geom');")
            building_development_srid = self.cur.fetchone()[0]

            QgsMessageLog.logMessage(f"SRIDs - Target: {target_srid}, Parcels: {parcels_srid}, Buildings: {building_development_srid}", level=Qgis.Info)

            batch_size = 1000 # Verarbeitung als Batch, um Zeit zu sparen

            # Cursor für große Datenmengen verwenden
            self.cur.execute("""
                DECLARE overlap_cursor CURSOR FOR 
                SELECT 
                    a.id AS GUID_ALKIS,
                    b.blocknr,
                    b.sst_liste AS development_type,
                    b.sst_lv_2_liste AS development_type_Lv2,
                    b.sst_lv_3_liste AS development_type_Lv3,
                    b.desk3 AS development_type_Code,
                    a.geom
                FROM "MPSCDresden".parcels a
                JOIN "MPSCDresden".building_development b 
                    ON ST_Intersects(a.geom, b.geom) 
                    AND NOT ST_Touches(a.geom, b.geom);
            """)

            while True:
                self.cur.execute(f"FETCH {batch_size} FROM overlap_cursor;")
                rows = self.cur.fetchall()

                if not rows:
                    break

                # Batch-Insert für Performance
                self.cur.executemany("""
                    INSERT INTO "MPSCDresden".built_up_parcel 
                    (GUID_ALKIS, blocknr, development_type, development_type_Lv2, development_type_Lv3, development_type_Code, geom)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (GUID_ALKIS) DO NOTHING;
                """, rows)

                QgsMessageLog.logMessage(f"Processed {len(rows)} rows", level=Qgis.Info)

            self.cur.execute("CLOSE overlap_cursor;")  # Cursor schließen

            # Transaktion erst nach allen Inserts committen
            self.cur.execute("COMMIT;")

            # Finales Logging
            self.cur.execute('SELECT COUNT(*) FROM "MPSCDresden".built_up_parcel;')
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
            ALTER TABLE "MPSCDresden".built_up_parcel
            ALTER COLUMN geom TYPE geometry(Polygon, {target_epsg})
            USING ST_Transform(geom, {target_epsg});
            """
            self.cur.execute(transform_query)
            self.conn.commit()
            QgsMessageLog.logMessage(f"Transformed 'built_up_parcel' table to EPSG:{target_epsg}", level=Qgis.Info)
        
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error transforming 'built_up_parcel' table: {str(e)}", level=Qgis.Critical)