import configparser
import os
import pandas as pd
from qgis.core import QgsMessageLog, Qgis, QgsProject
from qgis.PyQt.QtWidgets import QFileDialog

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
        Fügt zusätzliche Spalten für Baualter, Genehmigungsjahr, Quelle und Training zu citydb_filter hinzu.
        """
        try:
            add_columns_query = """
            ALTER TABLE "MPSCDresden".citydb_filter
            ADD COLUMN IF NOT EXISTS building_age VARCHAR,
            ADD COLUMN IF NOT EXISTS baugenehmigung_year INTEGER,
            ADD COLUMN IF NOT EXISTS baujahr DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS building_age_source VARCHAR,
            ADD COLUMN IF NOT EXISTS training VARCHAR(1)
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
        Aktualisiert building_age und building_age_source für alle Gebäude, die mit Neubauten-Geometrien überlappen.
        Setzt zusätzlich baugenehmigung_year für Gebäude mit gültigem Genehmigungsdatum.
        """
        try:
            # Erst alle intersecting buildings mit building_age und source aktualisieren
            update_all_neubauten_query = """
            UPDATE "MPSCDresden".citydb_filter cf
            SET building_age = '7',
                building_age_source = 'neubauten'
            FROM "MPSCDresden".neubauten nb
            WHERE ST_Intersects(cf.geom, nb.geom)
            """
            self.cur.execute(update_all_neubauten_query)
            
            # Dann zusätzlich baugenehmigung_year für die mit gültigem Datum setzen
            update_permit_year_query = """
            UPDATE "MPSCDresden".citydb_filter cf
            SET baugenehmigung_year = EXTRACT(YEAR FROM nb.baugenehmi::DATE)
            FROM "MPSCDresden".neubauten nb
            WHERE ST_Intersects(cf.geom, nb.geom)
                AND nb.baugenehmi IS NOT NULL
            """
            self.cur.execute(update_permit_year_query)
            
            self.conn.commit()
            QgsMessageLog.logMessage("Building age and source updated for all intersecting buildings (neubauten), permit year updated where available", level=Qgis.Info)
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

            # CSV einlesen mit verschiedenen Kodierungen versuchen
            df = None
            encodings_to_try = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(csv_path, dtype=str, sep=';', encoding=encoding)
                    QgsMessageLog.logMessage(f"CSV (monuments) erfolgreich mit Kodierung '{encoding}' gelesen", level=Qgis.Info)
                    break
                except UnicodeDecodeError as e:
                    QgsMessageLog.logMessage(f"Kodierung '{encoding}' fehlgeschlagen: {str(e)}", level=Qgis.Warning)
                    continue
            
            if df is None:
                QgsMessageLog.logMessage(f"Konnte CSV-Datei (monuments) mit keiner der Kodierungen lesen: {encodings_to_try}", level=Qgis.Critical)
                return

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
        Aktualisiert sst/sst_sub aus CSV, markiert Quelle, verteilt 80:20 auf Train/Validation
        und passt den training-Tag ('t'/'v') in citydb_filter entsprechend an.
        """
        try:
            # Projektordner bestimmen
            start_dir = None
            try:
                proj = QgsProject.instance()
                if proj is not None:
                    home = proj.homePath()
                    if home and os.path.isdir(home):
                        start_dir = home
                    else:
                        proj_file = proj.fileName()
                        if proj_file:
                            start_dir = os.path.dirname(proj_file)
            except Exception:
                start_dir = None

            if not start_dir or not os.path.isdir(start_dir):
                start_dir = os.path.expanduser("~")

            import_dir = os.path.join(start_dir, 'import_data')
            if os.path.isdir(import_dir):
                start_dir = import_dir

            csv_path, _ = QFileDialog.getOpenFileName(
                None,
                "CSV für additional_sst auswählen",
                start_dir,
                "CSV-Dateien (*.csv);;Alle Dateien (*.*)"
            )

            if not csv_path:
                QgsMessageLog.logMessage("Auswahl der additional_sst-CSV abgebrochen.", level=Qgis.Warning)
                return

            if not os.path.exists(csv_path):
                QgsMessageLog.logMessage(f"CSV-Datei für additional_sst nicht gefunden: {csv_path}", level=Qgis.Critical)
                return

            # CSV einlesen mit verschiedenen Kodierungen versuchen
            df = None
            encodings_to_try = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp1252']
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(csv_path, dtype={'gml_id': str}, sep=';', encoding=encoding)
                    QgsMessageLog.logMessage(f"CSV erfolgreich mit Kodierung '{encoding}' gelesen: {csv_path}", level=Qgis.Info)
                    break
                except UnicodeDecodeError as e:
                    QgsMessageLog.logMessage(f"Kodierung '{encoding}' fehlgeschlagen: {str(e)}", level=Qgis.Warning)
                    continue

            if df is None:
                QgsMessageLog.logMessage(f"Konnte CSV-Datei mit keiner der Kodierungen lesen: {encodings_to_try}", level=Qgis.Critical)
                return

            if 'gml_id' not in df.columns or 'sst' not in df.columns or 'sst_sub' not in df.columns:
                QgsMessageLog.logMessage("CSV muss die Spalten 'gml_id', 'sst' und 'sst_sub' enthalten.", level=Qgis.Critical)
                return

            csv_ids = set(df['gml_id'].astype(str))
            self.cur.execute('SELECT gml_id FROM "MPSCDresden".citydb_filter')
            db_ids = set(row[0] for row in self.cur.fetchall())
            fehlende = csv_ids - db_ids
            QgsMessageLog.logMessage(f"Nicht in DB gefundene gml_id aus der Nachkartierung: {fehlende}", level=Qgis.Warning)

            update_count = 0
            updated_gml_ids = set()
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
                if self.cur.rowcount > 0:
                    updated_gml_ids.add(gml_id)
                update_count += self.cur.rowcount

            self.conn.commit()
            QgsMessageLog.logMessage(f"{update_count} Gebäude mit sst/sst_sub aus CSV aktualisiert.", level=Qgis.Info)

            # Nur die wirklich aktualisierten Datensätze weiterverteilen
            if not updated_gml_ids:
                QgsMessageLog.logMessage("Keine Datensätze aktualisiert; es erfolgt keine Verteilung in Train/Validation.", level=Qgis.Info)
                return

            # Quelle als 'Nachkartierung' markieren (verifiziert) — nicht 'Kartierung'
            try:
                self.cur.execute("""
                    UPDATE "MPSCDresden".citydb_filter
                    SET classification_source_id = 4,
                        classification_source = 'Nachkartierung'
                    WHERE gml_id = ANY(%s)
                """, (list(updated_gml_ids),))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                QgsMessageLog.logMessage("Konnte Quelle 'Nachkartierung' für importierte Datensätze nicht setzen (fortgesetzt).", level=Qgis.Warning)

            # Kandidaten-IDs (db_filter_id) anhand der aktualisierten gml_id bestimmen, nur mit sst
            self.cur.execute("""
                SELECT db_filter_id
                FROM "MPSCDresden".citydb_filter
                WHERE gml_id = ANY(%s) AND sst IS NOT NULL
            """, (list(updated_gml_ids),))
            candidate_ids = [int(r[0]) for r in self.cur.fetchall()]

            if not candidate_ids:
                QgsMessageLog.logMessage("Keine Kandidaten mit sst gefunden, keine Verteilung in Train/Validation.", level=Qgis.Warning)
                return

            # Deduplikation: betroffene IDs aus Train/Validation entfernen
            try:
                self.cur.execute('DELETE FROM "MPSCDresden".train_data WHERE db_filter_id = ANY(%s)', (candidate_ids,))
                self.cur.execute('DELETE FROM "MPSCDresden".validation_data WHERE db_filter_id = ANY(%s)', (candidate_ids,))
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                QgsMessageLog.logMessage(f"Fehler beim Entfernen existierender Kandidaten aus Train/Validation: {e}", level=Qgis.Warning)

            # Training-Flag für Kandidaten vor Neuverteilung zurücksetzen
            try:
                self.cur.execute(
                    'UPDATE "MPSCDresden".citydb_filter SET training = NULL WHERE db_filter_id = ANY(%s)',
                    (candidate_ids,)
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                QgsMessageLog.logMessage(f"Fehler beim Zurücksetzen des training-Flags: {e}", level=Qgis.Warning)

            # 80:20 Split (robust bei kleinen Mengen)
            cand_df = pd.DataFrame({'db_filter_id': candidate_ids})
            val_df = pd.DataFrame(columns=['db_filter_id'])
            if len(cand_df) >= 5:
                val_df = cand_df.sample(frac=0.2, random_state=42)
                train_df = cand_df.drop(val_df.index)
            else:
                train_df = cand_df

            train_ids = train_df['db_filter_id'].astype(int).tolist()
            val_ids = val_df['db_filter_id'].astype(int).tolist()

            # Spaltenlisten aus Zieltabellen und SELECT-Liste aus citydb_filter bauen (volle Kopien)
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
                    INSERT INTO "MPSCDresden".train_data ({', '.join(f'"{c}"' for c in cols_t)})
                    SELECT {build_select_list(cols_t)}
                    FROM "MPSCDresden".citydb_filter cf
                    WHERE cf.db_filter_id = ANY(%s)
                    ''',
                    (train_ids,)
                )
            if val_ids:
                cols_v = get_columns('validation_data', exclude_cols=['validation_id'])
                self.cur.execute(
                    f'''
                    INSERT INTO "MPSCDresden".validation_data ({', '.join(f'"{c}"' for c in cols_v)})
                    SELECT {build_select_list(cols_v)}
                    FROM "MPSCDresden".citydb_filter cf
                    WHERE cf.db_filter_id = ANY(%s)
                    ''',
                    (val_ids,)
                )
            self.conn.commit()

            # training-Flags in citydb_filter setzen ('t' für Train, 'v' für Validation)
            if train_ids:
                self.cur.execute(
                    'UPDATE "MPSCDresden".citydb_filter SET training = %s WHERE db_filter_id = ANY(%s)',
                    ('t', train_ids)
                )
            if val_ids:
                self.cur.execute(
                    'UPDATE "MPSCDresden".citydb_filter SET training = %s WHERE db_filter_id = ANY(%s)',
                    ('v', val_ids)
                )
            self.conn.commit()

            # Aus classification_data entfernen (falls vorhanden)
            try:
                self.cur.execute('DELETE FROM "MPSCDresden".classification_data WHERE db_filter_id = ANY(%s)', (candidate_ids,))
                self.conn.commit()
            except Exception:
                self.conn.rollback()  # optional, falls classification_data nicht existiert

            QgsMessageLog.logMessage(f"Nachkartierung verteilt: {len(train_ids)} Train, {len(val_ids)} Validation.", level=Qgis.Info)

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