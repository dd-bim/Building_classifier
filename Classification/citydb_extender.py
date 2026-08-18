import os
import pandas as pd
from qgis.core import QgsMessageLog, Qgis, QgsProject
from qgis.PyQt.QtWidgets import QFileDialog
from .config_loader import get_config

class CityDBExtender:
    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den CityDBExtender mit DB-Verbindung und Verbindungsparametern.
        """
        self.connection_params = connection_params
        self.conn = conn
        self.cur = cur

        config = get_config()
        self.schema = config.get('Database', 'schema')
        self.paths = {
            'additional_sst': os.path.join(os.path.dirname(__file__), config.get('Paths', 'additional_sst')),
            'building_age_monuments': os.path.join(os.path.dirname(__file__), config.get('Paths', 'building_age_monuments'))
        }
        
    def add_additional_columns(self):
        """
        Fügt zusätzliche Spalten für Baualter, Genehmigungsjahr, Quelle und Training zu citydb_filter hinzu.
        """
        try:
            add_columns_query = f"""
            ALTER TABLE "{self.schema}".citydb_filter
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
            reset_columns_query = f"""
            UPDATE "{self.schema}".citydb_filter
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
            update_all_neubauten_query = f"""
            UPDATE "{self.schema}".citydb_filter cf
            SET building_age = '7',
                building_age_source = 'neubauten'
            FROM "{self.schema}".neubauten nb
            WHERE ST_Intersects(cf.geom, nb.geom)
            """
            self.cur.execute(update_all_neubauten_query)
            
            # Dann zusätzlich baugenehmigung_year für die mit gültigem Datum setzen
            update_permit_year_query = f"""
            UPDATE "{self.schema}".citydb_filter cf
            SET baugenehmigung_year = EXTRACT(YEAR FROM nb.baugenehmi::DATE)
            FROM "{self.schema}".neubauten nb
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
            update_building_age_query = f"""
            UPDATE "{self.schema}".citydb_filter cf
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
            FROM "{self.schema}".baualter ba
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
                QgsMessageLog.logMessage(f"CSV file for building_age_monuments not found: {csv_path}", level=Qgis.Critical)
                return

            # CSV einlesen mit verschiedenen Kodierungen versuchen
            df = None
            encodings_to_try = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(csv_path, dtype=str, sep=';', encoding=encoding)
                    QgsMessageLog.logMessage(f"CSV (monuments) successfully read with encoding '{encoding}'", level=Qgis.Info)
                    break
                except UnicodeDecodeError as e:
                    QgsMessageLog.logMessage(f"Encoding '{encoding}' failed: {str(e)}", level=Qgis.Warning)
                    continue

            if df is None:
                QgsMessageLog.logMessage(f"Could not read CSV file (monuments) with any of the encodings: {encodings_to_try}", level=Qgis.Critical)
                return

            if 'ID' not in df.columns or 'Baualtersstufe' not in df.columns:
                QgsMessageLog.logMessage("CSV must contain the columns 'ID' and 'Baualtersstufe'.", level=Qgis.Critical)
                return

            update_count = 0
            skipped_gml_ids = []
            for idx, row in df.iterrows():
                gml_id = row.get('ID')
                building_age = row.get('Baualtersstufe')
                if not gml_id or not building_age or pd.isna(gml_id) or pd.isna(building_age):
                    skipped_gml_ids.append(gml_id)
                    continue
                update_query = f"""
                    UPDATE "{self.schema}".citydb_filter
                    SET building_age = %s,
                        building_age_source = 'monument'
                    WHERE gml_id = %s
                """
                self.cur.execute(update_query, (building_age, gml_id))
                update_count += self.cur.rowcount

            self.conn.commit()
            QgsMessageLog.logMessage(f"{update_count} buildings updated with building_age from building_age_monuments.csv.", level=Qgis.Info)
            if skipped_gml_ids:
                QgsMessageLog.logMessage(
                    f"{len(skipped_gml_ids)} rows in building_age_monuments.csv had empty values and were skipped. gml_id: {skipped_gml_ids}",
                    level=Qgis.Warning
                )
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error updating building_age from monument data: {str(e)}", level=Qgis.Critical)

    def update_sst_from_csv(self):
        """
        Importiert Nachkartierungen aus einer oder mehreren CSV-Dateien (Mehrfachauswahl möglich).
        Ablauf:
        1. sst/sst_sub in citydb_mirror aktualisieren (Quelle der Wahrheit)
        2. citydb_filter per UPSERT synchronisieren:
           - UPDATE für bereits vorhandene Gebäude
           - INSERT für bisher herausgefilterte Gebäude (waren nicht in citydb_filter)
        3. Quelle 'Nachkartierung' setzen, 80:20 in Train/Validation verteilen.
        """
        try:
            # --- Startverzeichnis bestimmen ---
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

            # --- Mehrfachauswahl von CSV-Dateien ---
            csv_paths, _ = QFileDialog.getOpenFileNames(
                None,
                "CSV-Dateien für Nachkartierung auswählen (Mehrfachauswahl möglich)",
                start_dir,
                "CSV-Dateien (*.csv);;Alle Dateien (*.*)"
            )

            if not csv_paths:
                QgsMessageLog.logMessage("Selection of the re-survey CSV cancelled.", level=Qgis.Warning)
                return

            encodings_to_try = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp1252']
            all_frames = []

            # --- Schritt 1: Alle CSVs einlesen ---
            for csv_path in csv_paths:
                if not os.path.exists(csv_path):
                    QgsMessageLog.logMessage(f"CSV file not found, skipped: {csv_path}", level=Qgis.Warning)
                    continue

                df = None
                for encoding in encodings_to_try:
                    try:
                        df = pd.read_csv(csv_path, dtype={'gml_id': str}, sep=';', encoding=encoding)
                        QgsMessageLog.logMessage(f"CSV read ({encoding}): {csv_path}", level=Qgis.Info)
                        break
                    except UnicodeDecodeError:
                        continue

                if df is None:
                    QgsMessageLog.logMessage(f"Could not read CSV with any encoding, skipped: {csv_path}", level=Qgis.Critical)
                    continue

                if not {'gml_id', 'sst'}.issubset(df.columns):
                    QgsMessageLog.logMessage(
                        f"CSV must contain at least the columns 'gml_id' and 'sst', skipped: {csv_path}",
                        level=Qgis.Critical
                    )
                    continue

                # sst_sub ist optional (Unterklasse wird in der Praxis meist nicht angegeben)
                if 'sst_sub' not in df.columns:
                    df['sst_sub'] = pd.NA

                all_frames.append(df)

            if not all_frames:
                QgsMessageLog.logMessage("No usable CSV files found.", level=Qgis.Warning)
                return

            # Duplikate über alle CSVs hinweg: letzte Definition gewinnt
            combined_df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset='gml_id', keep='last')

            # --- Schritt 2: citydb_mirror aktualisieren ---
            csv_ids = set(combined_df['gml_id'].astype(str))
            self.cur.execute(f'SELECT gml_id FROM "{self.schema}".citydb_mirror')
            mirror_ids = set(row[0] for row in self.cur.fetchall())
            fehlende = csv_ids - mirror_ids
            if fehlende:
                QgsMessageLog.logMessage(
                    f"gml_id not found in citydb_mirror (will be ignored): {fehlende}",
                    level=Qgis.Warning
                )

            mirror_update_count = 0
            updated_gml_ids = set()
            for _, row in combined_df.iterrows():
                gml_id = str(row['gml_id'])
                sst = row['sst']
                sst_sub = row['sst_sub']
                if pd.isna(sst_sub) or str(sst_sub).strip() == "":
                    self.cur.execute(
                        f'UPDATE "{self.schema}".citydb_mirror SET sst = %s, sst_sub = NULL WHERE gml_id = %s',
                        (sst, gml_id)
                    )
                else:
                    self.cur.execute(
                        f'UPDATE "{self.schema}".citydb_mirror SET sst = %s, sst_sub = %s WHERE gml_id = %s',
                        (sst, sst_sub, gml_id)
                    )
                if self.cur.rowcount > 0:
                    updated_gml_ids.add(gml_id)
                    mirror_update_count += self.cur.rowcount

            self.conn.commit()
            QgsMessageLog.logMessage(
                f"{mirror_update_count} buildings in citydb_mirror updated with sst/sst_sub.",
                level=Qgis.Info
            )

            if not updated_gml_ids:
                QgsMessageLog.logMessage("No buildings in citydb_mirror updated.", level=Qgis.Info)
                return

            updated_list = list(updated_gml_ids)

            # --- Schritt 3: citydb_filter synchronisieren ---
            # UPDATE: Gebäude bereits in citydb_filter vorhanden
            self.cur.execute(f"""
                UPDATE "{self.schema}".citydb_filter cf
                SET sst     = m.sst,
                    sst_sub = m.sst_sub
                FROM "{self.schema}".citydb_mirror m
                WHERE m.gml_id = cf.gml_id
                  AND m.gml_id = ANY(%s)
            """, (updated_list,))
            filter_updated = self.cur.rowcount
            self.conn.commit()

            # INSERT: Gebäude bisher nicht in citydb_filter (wurden ursprünglich herausgefiltert)
            self.cur.execute(f"""
                INSERT INTO "{self.schema}".citydb_filter (
                    cityobject_id, gml_id, function, address, roof_type,
                    storeys_above_ground, building_footprint, length_footprint, width_footprint,
                    roof_ridge_height, eaves_height, storey_height, number_roof_surfaces,
                    roof_slope, sst, sst_sub, geom
                )
                SELECT
                    m.cityobject_id, m.gml_id, m.function, m.address, m.roof_type,
                    m.storeys_above_ground, m.building_footprint, m.length_footprint, m.width_footprint,
                    m.roof_ridge_height, m.eaves_height,
                    CASE
                        WHEN m.eaves_height IS NOT NULL
                         AND m.storeys_above_ground IS NOT NULL
                         AND m.storeys_above_ground > 0
                        THEN m.eaves_height / m.storeys_above_ground
                        ELSE m.storey_height
                    END,
                    m.number_roof_surfaces, m.roof_slope, m.sst, m.sst_sub, m.geom
                FROM "{self.schema}".citydb_mirror m
                WHERE m.gml_id = ANY(%s)
                  AND NOT EXISTS (
                      SELECT 1 FROM "{self.schema}".citydb_filter cf WHERE cf.gml_id = m.gml_id
                  )
                ON CONFLICT (gml_id) DO NOTHING
            """, (updated_list,))
            filter_inserted = self.cur.rowcount
            self.conn.commit()

            QgsMessageLog.logMessage(
                f"citydb_filter: {filter_updated} buildings updated, {filter_inserted} newly inserted.",
                level=Qgis.Info
            )

            # --- Schritt 4: Quelle als 'Nachkartierung' markieren ---
            try:
                self.cur.execute(f"""
                    UPDATE "{self.schema}".citydb_filter
                    SET classification_source_id = 4,
                        classification_source = 'Nachkartierung'
                    WHERE gml_id = ANY(%s)
                """, (updated_list,))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                QgsMessageLog.logMessage(
                    "Could not set source 'Nachkartierung' (continuing).", level=Qgis.Warning
                )

            # --- Schritt 5: db_filter_ids der betroffenen Gebäude mit sst bestimmen ---
            self.cur.execute(f"""
                SELECT db_filter_id
                FROM "{self.schema}".citydb_filter
                WHERE gml_id = ANY(%s) AND sst IS NOT NULL
            """, (updated_list,))
            candidate_ids = [int(r[0]) for r in self.cur.fetchall()]

            if not candidate_ids:
                QgsMessageLog.logMessage(
                    "No candidates with sst found, no distribution into Train/Validation.",
                    level=Qgis.Warning
                )
                return

            # --- Schritt 6: Deduplizierung, training-Flag zurücksetzen ---
            try:
                self.cur.execute(
                    f'DELETE FROM "{self.schema}".train_data WHERE db_filter_id = ANY(%s)', (candidate_ids,)
                )
                self.cur.execute(
                    f'DELETE FROM "{self.schema}".validation_data WHERE db_filter_id = ANY(%s)', (candidate_ids,)
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                QgsMessageLog.logMessage(f"Error removing from Train/Validation: {e}", level=Qgis.Warning)

            try:
                self.cur.execute(
                    f'UPDATE "{self.schema}".citydb_filter SET training = NULL WHERE db_filter_id = ANY(%s)',
                    (candidate_ids,)
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                QgsMessageLog.logMessage(f"Error resetting the training flag: {e}", level=Qgis.Warning)

            # --- Schritt 7: 80:20-Split ---
            cand_df = pd.DataFrame({'db_filter_id': candidate_ids})
            if len(cand_df) >= 5:
                val_df = cand_df.sample(frac=0.2, random_state=42)
                train_df = cand_df.drop(val_df.index)
            else:
                val_df = pd.DataFrame(columns=['db_filter_id'])
                train_df = cand_df

            train_ids = train_df['db_filter_id'].astype(int).tolist()
            val_ids = val_df['db_filter_id'].astype(int).tolist()

            def get_columns(table, exclude_cols):
                self.cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (self.schema, table))
                cols = [r[0] for r in self.cur.fetchall()]
                return [c for c in cols if c not in exclude_cols]

            def build_select_list(cols):
                return ', '.join(
                    "NULL::varchar AS results" if c == 'results' else f'cf."{c}"'
                    for c in cols
                )

            if train_ids:
                cols_t = get_columns('train_data', exclude_cols=['train_id'])
                self.cur.execute(
                    f'INSERT INTO "{self.schema}".train_data ({", ".join(f"{chr(34)}{c}{chr(34)}" for c in cols_t)})'
                    f' SELECT {build_select_list(cols_t)}'
                    f' FROM "{self.schema}".citydb_filter cf WHERE cf.db_filter_id = ANY(%s)',
                    (train_ids,)
                )
            if val_ids:
                cols_v = get_columns('validation_data', exclude_cols=['validation_id'])
                self.cur.execute(
                    f'INSERT INTO "{self.schema}".validation_data ({", ".join(f"{chr(34)}{c}{chr(34)}" for c in cols_v)})'
                    f' SELECT {build_select_list(cols_v)}'
                    f' FROM "{self.schema}".citydb_filter cf WHERE cf.db_filter_id = ANY(%s)',
                    (val_ids,)
                )
            self.conn.commit()

            if train_ids:
                self.cur.execute(
                    f'UPDATE "{self.schema}".citydb_filter SET training = %s WHERE db_filter_id = ANY(%s)',
                    ('t', train_ids)
                )
            if val_ids:
                self.cur.execute(
                    f'UPDATE "{self.schema}".citydb_filter SET training = %s WHERE db_filter_id = ANY(%s)',
                    ('v', val_ids)
                )
            self.conn.commit()

            try:
                self.cur.execute(
                    f'DELETE FROM "{self.schema}".classification_data WHERE db_filter_id = ANY(%s)',
                    (candidate_ids,)
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()

            QgsMessageLog.logMessage(
                f"Re-survey completed: {len(train_ids)} train, {len(val_ids)} validation.",
                level=Qgis.Info
            )

        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error importing the re-survey: {str(e)}", level=Qgis.Critical)

    def set_classification_source_kartierung(self):
        """
        Setzt classification_source und classification_source_id auf 'Kartierung' (1) für alle Datensätze mit sst oder sst_sub,
        sofern noch keine Quelle gesetzt ist.
        """
        try:
            self.cur.execute(f"""
                UPDATE "{self.schema}".citydb_filter
                SET classification_source_id = 1,
                    classification_source = 'Kartierung'
                WHERE (sst IS NOT NULL OR sst_sub IS NOT NULL)
                  AND (classification_source IS NULL OR classification_source = '');
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("classification_source set for Kartierung.", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error setting classification_source Kartierung: {str(e)}", level=Qgis.Critical)