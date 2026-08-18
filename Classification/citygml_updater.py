import pandas as pd
import os
from lxml import etree
from concurrent.futures import ThreadPoolExecutor
import random
from qgis.core import QgsMessageLog, Qgis, QgsVectorLayer, QgsDataSourceUri, QgsProject
from .config_loader import get_config

CITYGML_BATCH_SIZE = 10
CITYGML_MAX_WORKERS = 4

class CityGMLUpdater:
    """
    Aktualisiert die CityGML-Dateien und die citydb_filter-Tabelle mit den Klassifikationsergebnissen.

    Diese Klasse bietet Methoden zum:
    - Laden der Klassifikationsergebnisse aus der Datenbank
    - Übertragen der sst-Werte in die citydb_filter-Tabelle
    - Aktualisieren der CityGML-Dateien mit den neuen sst-Attributen
    """

    def __init__(self, conn, cur, connection_params):
        """
        Initialisiert den CityGMLUpdater mit DB-Verbindung und Konfigurationsparametern.

        :param conn: Datenbankverbindung (z.B. psycopg2 connection)
        :param cur: Datenbank-Cursor
        :param connection_params: Dictionary mit Verbindungsparametern
        """
        self.conn = conn
        self.cur = cur
        self.connection_params = connection_params

        config = get_config()
        self.schema = config.get('Database', 'schema')
        self.input_dir = config.get('Paths', 'input_citygml_dir', fallback=None)
        self.output_dir = config.get('Paths', 'output_citygml_dir', fallback=None)

        if not self.input_dir or not self.output_dir:
            raise ValueError("Die Pfade 'input_citygml_dir' und 'output_citygml_dir' müssen in der config.ini definiert sein.")
        
    def load_classification_results(self):
        """
        Lädt die Klassifikationsergebnisse (gml_id, sst, confidence, classification_source_id, classification_source) aus der Datenbank.
        :return: DataFrame mit den Spalten gml_id, sst, confidence, classification_source_id, classification_source
        """
        query = f'''
            SELECT gml_id, sst, overall_confidence, classification_source_id, classification_source
            FROM "{self.schema}".classification_data
        '''
        self.cur.execute(query)
        rows = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        df = pd.DataFrame(rows, columns=colnames)
        return df

    def update_citydb_filter(self):
        """
        Aktualisiert die citydb_filter-Relation mit den Werten für sst, confidence, classification_source_id und classification_source aus der classification_data-Relation.

        Für jeden Datensatz wird geprüft, ob der sst-Wert bereits existiert oder aktualisiert werden muss.
        """
        try:
            classification_results = self.load_classification_results()
            if classification_results.empty:
                QgsMessageLog.logMessage("No classification results found.", level=Qgis.Warning)
                return

            successful_updates = 0
            mirror_updates = 0

            for index, row in classification_results.iterrows():
                gml_id = row['gml_id']
                sst = row['sst']
                confidence = row['overall_confidence']
                classification_source_id = row['classification_source_id']
                classification_source = row['classification_source']

                # Prüfe, ob das Attribut bereits existiert
                check_query = f"""
                SELECT sst, confidence, classification_source_id, classification_source FROM "{self.schema}".citydb_filter
                WHERE gml_id = %s
                """
                self.cur.execute(check_query, (gml_id,))
                existing_result = self.cur.fetchone()

                if existing_result is None:
                    # Falls Attribut nicht existiert → Einfügen
                    insert_query = f"""
                    INSERT INTO "{self.schema}".citydb_filter (gml_id, sst, confidence, classification_source_id, classification_source)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    self.cur.execute(insert_query, (gml_id, sst, confidence, classification_source_id, classification_source))
                    successful_updates += 1

                elif (existing_result[0] != sst or existing_result[1] != confidence or
                      existing_result[2] != classification_source_id or existing_result[3] != classification_source):
                    # Falls sich das Ergebnis geändert hat → Aktualisieren
                    update_query = f"""
                    UPDATE "{self.schema}".citydb_filter
                    SET sst = %s, confidence = %s, classification_source_id = %s, classification_source = %s
                    WHERE gml_id = %s
                    """
                    self.cur.execute(update_query, (sst, confidence, classification_source_id, classification_source, gml_id))
                    successful_updates += 1

                # citydb_mirror analog zu citydb_filter aktualisieren (Quelle der Wahrheit für sst/sst_sub,
                # gleiches Muster wie citydb_extender.update_sst_from_csv). sst_sub wird zurückgesetzt, da
                # das Modell keine Unterklassen liefert und der neue sst-Wert die alte Kartierungs-Unterklasse ersetzt.
                if pd.notna(sst):
                    mirror_query = f"""
                    UPDATE "{self.schema}".citydb_mirror
                    SET sst = %s, sst_sub = NULL
                    WHERE gml_id = %s AND sst IS DISTINCT FROM %s
                    """
                    self.cur.execute(mirror_query, (sst, gml_id, sst))
                    if self.cur.rowcount > 0:
                        mirror_updates += self.cur.rowcount

            self.conn.commit()
            QgsMessageLog.logMessage(
                f"citydb_filter successfully updated. Number of successful transfers: {successful_updates}. "
                f"citydb_mirror updated analogously: {mirror_updates}.",
                level=Qgis.Success
            )

        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error updating citydb_filter: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            raise e

    def update_citygml_files(self, test_mode=False):
        """
        Aktualisiert die CityGML-Dateien im Eingabeverzeichnis mit den sst-Werten aus der citydb_filter-Tabelle.

        :param test_mode: Wenn True, wird nur eine zufällige Datei verarbeitet (für Tests).
        """
        try:
            # Lade die Zuordnung von gml_id zu sst aus der Datenbank
            query = f"""
            SELECT gml_id, sst
            FROM "{self.schema}".citydb_filter
            WHERE sst IS NOT NULL
            """
            self.cur.execute(query)
            classification_results = self.cur.fetchall()
            sst_mapping = {f"bldg_{row[0]}": row[1] for row in classification_results}

            if not sst_mapping:
                QgsMessageLog.logMessage("No valid classification results found.", level=Qgis.Warning)
                return

            # Verarbeite CityGML-Dateien im Eingabeverzeichnis
            file_names = [f for f in os.listdir(self.input_dir) if f.endswith(".gml")]

            if test_mode:
                # Wähle eine zufällige Datei aus
                if not file_names:
                    QgsMessageLog.logMessage("No CityGML files found in the input directory.", level=Qgis.Warning)
                    return
                file_name = random.choice(file_names)
                QgsMessageLog.logMessage(f"Test mode activated: processing random file {file_name}.", level=Qgis.Info)
                self.process_file(file_name, self.input_dir, self.output_dir, sst_mapping, test_mode)
            else:
                # Verarbeite die Dateien in Batches mit begrenzter Worker-Anzahl,
                # damit nicht zu viele XML-Baeume gleichzeitig im Speicher gehalten werden
                for i in range(0, len(file_names), CITYGML_BATCH_SIZE):
                    batch = file_names[i:i + CITYGML_BATCH_SIZE]
                    with ThreadPoolExecutor(max_workers=CITYGML_MAX_WORKERS) as executor:
                        futures = [executor.submit(self.process_file, file_name, self.input_dir, self.output_dir, sst_mapping) for file_name in batch]
                        for future in futures:
                            future.result()

            QgsMessageLog.logMessage("CityGML files successfully updated.", level=Qgis.Success)

        except Exception as e:
            QgsMessageLog.logMessage(f"Error updating CityGML files: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            raise e
        
    def process_file(self, file_name, input_dir, output_dir, sst_mapping, test_mode=False):
        """
        Verarbeitet eine einzelne CityGML-Datei: Fügt die sst-Werte hinzu und speichert sie im Ausgabeverzeichnis.

        :param file_name: Name der zu verarbeitenden Datei
        :param input_dir: Eingabeverzeichnis
        :param output_dir: Ausgabeverzeichnis
        :param sst_mapping: Dictionary mit gml_id → sst
        :param test_mode: Wenn True, werden Warnungen für fehlende gml_ids unterdrückt
        """
        try:
            input_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)
            
            if os.path.exists(output_file):
                os.remove(output_file)

            # Füge sst-Werte zur CityGML-Datei hinzu
            self.add_sst_to_citygml(input_file, output_file, sst_mapping, test_mode)

            QgsMessageLog.logMessage(f"File {file_name} successfully processed.", level=Qgis.Info)

        except Exception as e:
            QgsMessageLog.logMessage(f"Error processing file {file_name}: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)

    def add_sst_to_citygml(self, input_file, output_file, sst_mapping, test_mode=False):
        """
        Fügt die sst-Werte als generisches Attribut zu bldg:Building-Elementen hinzu.
        BuildingPart-Elemente werden explizit ausgeschlossen.
        Unterstützt CityGML 1.0 und 2.0 (Namespace-Erkennung aus dem Dokument).

        :param input_file: Pfad zur Eingabedatei
        :param output_file: Pfad zur Ausgabedatei
        :param sst_mapping: Dictionary mit gml_id → sst
        :param test_mode: Wenn True, werden Warnungen für fehlende gml_ids unterdrückt
        """
        try:
            tree = etree.parse(input_file)
            root = tree.getroot()

            # Namespace-Erkennung: CityGML 1.0 oder 2.0
            all_ns_uris = set(root.nsmap.values())
            if 'http://www.opengis.net/citygml/building/2.0' in all_ns_uris:
                bldg_ns = 'http://www.opengis.net/citygml/building/2.0'
                gen_ns  = 'http://www.opengis.net/citygml/generics/2.0'
            else:
                bldg_ns = 'http://www.opengis.net/citygml/building/1.0'
                gen_ns  = 'http://www.opengis.net/citygml/generics/1.0'
            gml_ns = 'http://www.opengis.net/gml'

            ns = {'bldg': bldg_ns, 'gen': gen_ns, 'gml': gml_ns}

            # Nur bldg:Building-Elemente erfassen – bldg:BuildingPart explizit ausgeschlossen
            gml_id_map = {
                elem.attrib[f'{{{gml_ns}}}id']: elem
                for elem in root.xpath('.//bldg:Building[@gml:id]', namespaces=ns)
            }

            filtered_sst_mapping = {
                gml_id: sst_value
                for gml_id, sst_value in sst_mapping.items()
                if gml_id in gml_id_map
            }

            for gml_id, sst_value in filtered_sst_mapping.items():
                target_elem = gml_id_map.get(gml_id)
                if target_elem is not None:
                    # Nur direkte Kinder prüfen – kein rekursives Suchen in BuildingParts
                    existing_sst = next(
                        (child for child in target_elem
                         if child.tag == f'{{{gen_ns}}}stringAttribute'
                         and child.get('name') == 'sst'),
                        None
                    )
                    if existing_sst is not None:
                        value_elem = existing_sst.find(f'{{{gen_ns}}}value')
                        if value_elem is not None:
                            value_elem.text = sst_value
                    else:
                        # Neues generisches Attribut erzeugen
                        string_attr = etree.Element(f'{{{gen_ns}}}stringAttribute')
                        string_attr.set('name', 'sst')
                        value_elem = etree.SubElement(string_attr, f'{{{gen_ns}}}value')
                        value_elem.text = sst_value

                        # Einfügeposition: nach dem letzten gen:stringAttribute,
                        # sonst vor dem ersten bldg:-Kindelement, sonst ans Ende
                        children = list(target_elem)
                        last_gen_idx = None
                        first_bldg_idx = None
                        for i, child in enumerate(children):
                            if child.tag == f'{{{gen_ns}}}stringAttribute':
                                last_gen_idx = i
                            if first_bldg_idx is None and child.tag.startswith(f'{{{bldg_ns}}}'):
                                first_bldg_idx = i

                        if last_gen_idx is not None:
                            target_elem.insert(last_gen_idx + 1, string_attr)
                        elif first_bldg_idx is not None:
                            target_elem.insert(first_bldg_idx, string_attr)
                        else:
                            target_elem.append(string_attr)
                else:
                    if not test_mode:
                        QgsMessageLog.logMessage(f"gml_id '{gml_id}' not found as bldg:Building in file.", level=Qgis.Warning)

            try:
                # Schreibe die aktualisierte Datei als formatiertes XML (lxml-eigenes Pretty-Print,
                # vermeidet den speicherintensiven Umweg Ã¼ber minidom)
                tree.write(output_file, encoding='UTF-8', xml_declaration=True, pretty_print=True)

                QgsMessageLog.logMessage(f"File successfully updated: {output_file}", level=Qgis.Info)
            except Exception as e:
                QgsMessageLog.logMessage(f"Error writing file {output_file}: {str(e)}", level=Qgis.Critical)
                import traceback
                QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)

        except Exception as e:
            QgsMessageLog.logMessage(f"Error processing file {input_file}: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            raise e

    def load_and_visualize_citydb_filter(self):
        """
        Lädt die citydb_filter_view als QGIS-Layer mit ausgewählten Attributen.
        Erstellt die View, falls sie noch nicht existiert.
        """
        # View erzeugen, falls nicht vorhanden
        try:
            self.cur.execute(f"""
                CREATE OR REPLACE VIEW "{self.schema}".citydb_filter_view AS
                SELECT 
                    db_filter_id, 
                    gml_id, 
                    sst, 
                    classification_source_id, 
                    classification_source, 
                    confidence, 
                    geom
                FROM "{self.schema}".citydb_filter;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("View citydb_filter_view was (re-)created.", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Error creating view citydb_filter_view: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)
            return

        uri = QgsDataSourceUri()
        uri.setConnection(
            self.connection_params['host'],
            str(self.connection_params['port']),
            self.connection_params['dbname'],
            self.connection_params['user'],
            self.connection_params['password']
        )
        uri.setDataSource(
            self.schema,
            'citydb_filter_view',
            'geom',
            '',
            'db_filter_id'
        )

        layer_name = 'CityDB Filter'
        # Entferne ggf. alten Layer
        existing_layer = QgsProject.instance().mapLayersByName(layer_name)
        if existing_layer:
            QgsProject.instance().removeMapLayer(existing_layer[0])

        layer = QgsVectorLayer(uri.uri(False), layer_name, 'postgres')
        if not layer.isValid():
            QgsMessageLog.logMessage("Layer CityDB Filter is not valid", level=Qgis.Critical)
            return

        QgsProject.instance().addMapLayer(layer)
        QgsMessageLog.logMessage("Layer CityDB Filter successfully loaded.", level=Qgis.Info)