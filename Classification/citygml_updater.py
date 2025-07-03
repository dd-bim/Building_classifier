import pandas as pd
import os
from lxml import etree
from concurrent.futures import ThreadPoolExecutor
import configparser
import random
from qgis.core import QgsMessageLog, Qgis, QgsVectorLayer, QgsDataSourceUri, QgsProject
from xml.dom.minidom import parseString

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
        
        # Lade die Konfiguration aus der config.ini
        self.config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Die Konfigurationsdatei '{config_path}' wurde nicht gefunden.")
        self.config.read(config_path)

        # Speichere die Pfade als Attribute
        self.input_dir = self.config.get('Paths', 'input_citygml_dir', fallback=None)
        self.output_dir = self.config.get('Paths', 'output_citygml_dir', fallback=None)

        if not self.input_dir or not self.output_dir:
            raise ValueError("Die Pfade 'input_citygml_dir' und 'output_citygml_dir' müssen in der config.ini definiert sein.")
        
    def load_classification_results(self):
        """
        Lädt die Klassifikationsergebnisse (gml_id, sst, confidence, classification_source_id, classification_source) aus der Datenbank.
        :return: DataFrame mit den Spalten gml_id, sst, confidence, classification_source_id, classification_source
        """
        query = '''
            SELECT gml_id, sst, overall_confidence, classification_source_id, classification_source
            FROM "MPSCDresden".classification_data
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
                QgsMessageLog.logMessage("Keine Klassifizierungsergebnisse gefunden.", level=Qgis.Warning)
                return

            successful_updates = 0

            for index, row in classification_results.iterrows():
                gml_id = row['gml_id']
                sst = row['sst']
                confidence = row['overall_confidence']
                classification_source_id = row['classification_source_id']
                classification_source = row['classification_source']

                # Prüfe, ob das Attribut bereits existiert
                check_query = """
                SELECT sst, confidence, classification_source_id, classification_source FROM "MPSCDresden".citydb_filter 
                WHERE gml_id = %s
                """
                self.cur.execute(check_query, (gml_id,))
                existing_result = self.cur.fetchone()

                if existing_result is None:
                    # Falls Attribut nicht existiert → Einfügen
                    insert_query = """
                    INSERT INTO "MPSCDresden".citydb_filter (gml_id, sst, confidence, classification_source_id, classification_source)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    self.cur.execute(insert_query, (gml_id, sst, confidence, classification_source_id, classification_source))
                    successful_updates += 1
                
                elif (existing_result[0] != sst or existing_result[1] != confidence or
                      existing_result[2] != classification_source_id or existing_result[3] != classification_source):
                    # Falls sich das Ergebnis geändert hat → Aktualisieren
                    update_query = """
                    UPDATE "MPSCDresden".citydb_filter
                    SET sst = %s, confidence = %s, classification_source_id = %s, classification_source = %s
                    WHERE gml_id = %s
                    """
                    self.cur.execute(update_query, (sst, confidence, classification_source_id, classification_source, gml_id))
                    successful_updates += 1

            self.conn.commit()
            QgsMessageLog.logMessage(f"citydb_filter erfolgreich aktualisiert. Anzahl der erfolgreichen Übertragungen: {successful_updates}", level=Qgis.Success)

        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Aktualisieren der citydb_filter: {str(e)}", level=Qgis.Critical)
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
            query = """
            SELECT gml_id, sst
            FROM "MPSCDresden".citydb_filter
            WHERE sst IS NOT NULL
            """
            self.cur.execute(query)
            classification_results = self.cur.fetchall()
            sst_mapping = {f"bldg_{row[0]}": row[1] for row in classification_results}

            if not sst_mapping:
                QgsMessageLog.logMessage("Keine gültigen Klassifizierungsergebnisse gefunden.", level=Qgis.Warning)
                return

            # Verarbeite CityGML-Dateien im Eingabeverzeichnis
            file_names = [f for f in os.listdir(self.input_dir) if f.endswith(".gml")]

            if test_mode:
                # Wähle eine zufällige Datei aus
                if not file_names:
                    QgsMessageLog.logMessage("Keine CityGML-Dateien im Eingabeverzeichnis gefunden.", level=Qgis.Warning)
                    return
                file_name = random.choice(file_names)
                QgsMessageLog.logMessage(f"Testmodus aktiviert: Verarbeite zufällige Datei {file_name}.", level=Qgis.Info)
                self.process_file(file_name, self.input_dir, self.output_dir, sst_mapping, test_mode)
            else:
                # Verarbeite alle Dateien parallel
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.process_file, file_name, self.input_dir, self.output_dir, sst_mapping) for file_name in file_names]
                    for future in futures:
                        future.result()

            QgsMessageLog.logMessage("CityGML-Dateien erfolgreich aktualisiert.", level=Qgis.Success)

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Aktualisieren der CityGML-Dateien: {str(e)}", level=Qgis.Critical)
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

            QgsMessageLog.logMessage(f"Datei {file_name} erfolgreich verarbeitet.", level=Qgis.Info)

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Verarbeiten der Datei {file_name}: {str(e)}", level=Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)

    def add_sst_to_citygml(self, input_file, output_file, sst_mapping, test_mode=False):
        """
        Fügt die sst-Werte als generisches Attribut zu den Gebäuden in einer CityGML-Datei hinzu.

        :param input_file: Pfad zur Eingabedatei
        :param output_file: Pfad zur Ausgabedatei
        :param sst_mapping: Dictionary mit gml_id → sst
        :param test_mode: Wenn True, werden Warnungen für fehlende gml_ids unterdrückt
        """
        try:
            tree = etree.parse(input_file)
            root = tree.getroot()
            ns = {
                'core': 'http://www.opengis.net/citygml/1.0',
                'gen': 'http://www.opengis.net/citygml/generics/1.0',
                'bldg': 'http://www.opengis.net/citygml/building/1.0',
                'gml': 'http://www.opengis.net/gml'
            }

            # Mappe alle gml:ids auf ihre Elemente
            gml_id_map = {elem.attrib['{http://www.opengis.net/gml}id']: elem for elem in root.xpath(".//*[@gml:id]", namespaces=ns)}
            filtered_sst_mapping = {gml_id: sst_value for gml_id, sst_value in sst_mapping.items() if gml_id in gml_id_map}

            for gml_id, sst_value in filtered_sst_mapping.items():
                target_elem = gml_id_map.get(gml_id)
                if target_elem:
                    # Prüfe, ob das Attribut bereits existiert
                    existing_sst = target_elem.find(".//gen:stringAttribute[@name='sst']", ns)
                    if existing_sst is not None:
                        value_elem = existing_sst.find('.//gen:value', ns)
                        value_elem.text = sst_value
                    else:
                        # Erzeuge neues generisches Attribut
                        string_attr = etree.Element('{http://www.opengis.net/citygml/generics/1.0}stringAttribute')
                        string_attr.set('name', 'sst')
                        value_elem = etree.SubElement(string_attr, '{http://www.opengis.net/citygml/generics/1.0}value')
                        value_elem.text = sst_value

                        # Füge das Attribut an geeigneter Stelle ein
                        last_gen_attr = None
                        for child in target_elem:
                            if child.tag == '{http://www.opengis.net/citygml/generics/1.0}stringAttribute':
                                last_gen_attr = child

                        if last_gen_attr is not None:
                            index = list(target_elem).index(last_gen_attr)
                            target_elem.insert(index + 1, string_attr)
                        else:
                            first_bldg_attr = None
                            for child in target_elem:
                                if child.tag.startswith('{http://www.opengis.net/citygml/building/1.0}'):
                                    first_bldg_attr = child
                                    break

                            if first_bldg_attr is not None:
                                index = list(target_elem).index(first_bldg_attr)
                                target_elem.insert(index, string_attr)
                            else:
                                target_elem.append(string_attr)
                else:
                    if not test_mode:
                        QgsMessageLog.logMessage(f"gml_id '{gml_id}' nicht in Datei gefunden.", level=Qgis.Warning)

            try:
                # Schreibe die aktualisierte Datei als formatiertes XML
                formatted_xml = parseString(etree.tostring(tree, encoding='UTF-8')).toprettyxml(indent="    ")
                formatted_xml = "\n".join([line for line in formatted_xml.splitlines() if line.strip()])

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_xml)

                QgsMessageLog.logMessage(f"Datei erfolgreich aktualisiert: {output_file}", level=Qgis.Info)
            except Exception as e:
                QgsMessageLog.logMessage(f"Fehler beim Schreiben der Datei {output_file}: {str(e)}", level=Qgis.Critical)
                import traceback
                QgsMessageLog.logMessage(traceback.format_exc(), level=Qgis.Critical)

        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Verarbeiten der Datei {input_file}: {str(e)}", level=Qgis.Critical)
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
            self.cur.execute("""
                CREATE OR REPLACE VIEW "MPSCDresden".citydb_filter_view AS
                SELECT 
                    db_filter_id, 
                    gml_id, 
                    sst, 
                    classification_source_id, 
                    classification_source, 
                    confidence, 
                    geom
                FROM "MPSCDresden".citydb_filter;
            """)
            self.conn.commit()
            QgsMessageLog.logMessage("View citydb_filter_view wurde (ggf. erneut) erzeugt.", level=Qgis.Info)
        except Exception as e:
            self.conn.rollback()
            QgsMessageLog.logMessage(f"Fehler beim Erzeugen der View citydb_filter_view: {str(e)}", level=Qgis.Critical)
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
            'MPSCDresden',
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
        QgsMessageLog.logMessage("Layer CityDB Filter erfolgreich geladen.", level=Qgis.Info)