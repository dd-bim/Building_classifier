from qgis.core import QgsMessageLog, Qgis
from .config_loader import get_option


# =============================================================================
# DIAGNOSE-MODUL – Nachverfolgung eines einzelnen Gebäudes durch die Pipeline
#
# Nutzung in citydb_processor.py:
#   OBJEKT_ID in trace_building() eintragen, dann self.trace_building() aufrufen.
#
# Zum Deaktivieren: trace_building_enabled = false in config.ini setzen.
# =============================================================================


def trace_building(cur, schema, objekt_id):
    if not get_option('trace_building_enabled', 'false').lower() == 'true':
        return
    """
    Verfolgt das Gebäude mit der angegebenen OBJEKT_ID (= gml_id aus dem
    Adresslayer) durch jeden Filterschritt der CityDBProcessor-Pipeline und gibt
    das Ergebnis in der QGIS-Konsole aus.

    Aufruf in citydb_processor.py:
        self.trace_building()  # OBJEKT_ID direkt in der Methode eintragen

    Parameter
    ---------
    cur       : psycopg2-Cursor der aktiven DB-Verbindung
    schema    : Datenbankschema (aus config.ini)
    objekt_id : OBJEKT_ID des zu verfolgenden Gebäudes (= gml_id)
    """
    OBJEKT_ID = objekt_id

    def log(msg, level=Qgis.Info):
        QgsMessageLog.logMessage(f"[TRACE {OBJEKT_ID}] {msg}", level=level)

    SEP = "=" * 60
    log(SEP)
    log("BUILDING TRACE STARTED")
    log(SEP)

    # ------------------------------------------------------------------
    # SCHRITT 1: Ist das Gebäude überhaupt in der CityDB (objectclass_id = 901)?
    # ------------------------------------------------------------------
    cur.execute("""
        SELECT f.id
        FROM citydb.feature f
        JOIN citydb.property p ON f.id = p.feature_id
        WHERE f.objectclass_id = 901
          AND p.name = 'OBJEKT_ID'
          AND p.val_string = %s
        LIMIT 1;
    """, (OBJEKT_ID,))
    row = cur.fetchone()
    if not row:
        log("STEP 1 [FAILED]: No building with this OBJEKT_ID found in citydb.feature"
            " (objectclass_id=901) → building does not exist in the CityDB.", Qgis.Critical)
        log("Possible causes: OBJEKT_ID incorrect, building has objectclass_id ≠ 901,"
            " or the OBJEKT_ID property is missing.", Qgis.Warning)
        log(SEP)
        return
    cityobject_id = row[0]
    log(f"STEP 1 [OK]: Building found in CityDB (cityobject_id={cityobject_id})")

    # ------------------------------------------------------------------
    # SCHRITT 2: Hat das Gebäude eine gml_id (externalReference)?
    # ------------------------------------------------------------------
    cur.execute("""
        SELECT val_string
        FROM citydb.property
        WHERE feature_id = %s
          AND name = 'OBJEKT_ID'
          AND val_string IS NOT NULL
        LIMIT 1;
    """, (cityobject_id,))
    row = cur.fetchone()
    if not row:
        log("STEP 2 [FAILED]: Property 'OBJEKT_ID' not present or val_string is NULL"
            " → gml_id cannot be set.", Qgis.Critical)
    else:
        log(f"STEP 2 [OK]: gml_id (OBJEKT_ID) = '{row[0]}'.")

    # ------------------------------------------------------------------
    # SCHRITT 3: Hat das Gebäude eine gültige Adresse in der CityDB?
    # ------------------------------------------------------------------
    cur.execute("""
        SELECT CONCAT(a.street, ' ', COALESCE(a.house_number, '')) AS addr,
               a.street, a.house_number
        FROM citydb.property p
        JOIN citydb.address a ON p.val_address_id = a.id
        WHERE p.feature_id = %s
          AND p.name = 'address'
          AND p.val_address_id IS NOT NULL
          AND a.street IS NOT NULL
          AND a.street != ''
          AND a.street != '0'
        LIMIT 1;
    """, (cityobject_id,))
    row = cur.fetchone()
    if not row:
        log("STEP 3 [FAILED]: No valid address found in citydb.address"
            " (street NULL, empty, or '0'). Address completion via SHP layer required.", Qgis.Warning)
        cur.execute("""
            SELECT a.street, a.house_number, a.id
            FROM citydb.property p
            JOIN citydb.address a ON p.val_address_id = a.id
            WHERE p.feature_id = %s AND p.name = 'address'
            LIMIT 3;
        """, (cityobject_id,))
        raw = cur.fetchall()
        if raw:
            for r in raw:
                log(f"  → Raw address data: street='{r[0]}', house_number='{r[1]}', address_id={r[2]}", Qgis.Warning)
        else:
            log("  → No address property present in citydb.property.", Qgis.Warning)
    else:
        log(f"STEP 3 [OK]: Address from CityDB = '{row[0]}' (street='{row[1]}', hnr='{row[2]}')")

    # ------------------------------------------------------------------
    # SCHRITT 4: Hat das Gebäude eine Funktion?
    # ------------------------------------------------------------------
    cur.execute("""
        SELECT val_string
        FROM citydb.property
        WHERE feature_id = %s AND name = 'function'
        LIMIT 1;
    """, (cityobject_id,))
    row = cur.fetchone()
    citydb_function = row[0] if row else None
    if not row:
        log("STEP 4 [INFO]: No 'function' attribute in the CityDB. May be overwritten from CSV.", Qgis.Warning)
    else:
        log(f"STEP 4 [OK]: function = '{citydb_function}'"
            f"{'  ← allowed (1000/1100)' if citydb_function in ('1000', '1100', '31001_1000', '31001_1100') else '  ← NOT 1000/1100'}")

    # ------------------------------------------------------------------
    # SCHRITT 5: Ist das Gebäude aktuell in citydb_filter?
    # ------------------------------------------------------------------
    cur.execute(f"""
        SELECT db_filter_id, gml_id, address, function, sst, sst_sub,
               building_footprint, geom IS NOT NULL AS has_geom
        FROM "{schema}".citydb_filter
        WHERE gml_id = %s OR cityobject_id = %s
        LIMIT 1;
    """, (OBJEKT_ID, cityobject_id))
    row = cur.fetchone()
    if not row:
        log("STEP 5 [FAILED]: Building is NOT contained in citydb_filter.", Qgis.Critical)
        log("The building was either never inserted or has already been removed by a filter step.", Qgis.Warning)
        cur.execute(f"""
            SELECT COUNT(*) FROM "{schema}".citydb_filter WHERE cityobject_id = %s;
        """, (cityobject_id,))
        if cur.fetchone()[0] == 0:
            log("  → fill_table(): Building was never inserted into citydb_filter."
                " Check whether TRUNCATE has already been executed or whether cityobject_id is correct.", Qgis.Warning)
        log(SEP)
        return

    db_filter_id, gml_id, address, func, sst, sst_sub, footprint, has_geom = row
    log(f"STEP 5 [OK]: Building is in citydb_filter (db_filter_id={db_filter_id})")
    log(f"  gml_id='{gml_id}' | address='{address}' | function='{func}'"
        f" | sst='{sst}' | sst_sub='{sst_sub}' | footprint={footprint} | has_geom={has_geom}")

    # ------------------------------------------------------------------
    # SCHRITT 6: Adressfilter (filter_table entfernt address IS NULL)
    # ------------------------------------------------------------------
    if not address:
        log("STEP 6 [FAILED]: address IS NULL → building will be removed by the address filter"
            " (filter_table).", Qgis.Critical)
    else:
        log(f"STEP 6 [OK]: Address present ('{address}') → address filter passed.")

    # ------------------------------------------------------------------
    # SCHRITT 7: Funktionsfilter (function IN ('1000','1100','31001_1000','31001_1100') ODER in kartierung_dd_gesamt)
    # ------------------------------------------------------------------
    if func in ('1000', '1100', '31001_1000', '31001_1100'):
        log(f"STEP 7 [OK]: function='{func}' → function filter passed directly.")
    else:
        cur.execute(f"""
            SELECT k.id, k.sst, k.str, k.hnr, k.id_alkis
            FROM "{schema}".kartierung_dd_gesamt k
            WHERE k.id_alkis = %s
               OR (
                   k.id_alkis IS NULL
                   AND trim(lower(k.str || ' ' || k.hnr)) = trim(lower(%s))
               )
            LIMIT 1;
        """, (gml_id, address or ''))
        kart_row = cur.fetchone()
        if kart_row:
            log(f"STEP 7 [OK]: function='{func}' (not 1000/1100), but present in kartierung_dd_gesamt"
                f" (id={kart_row[0]}, sst='{kart_row[1]}', id_alkis='{kart_row[4]}') → function filter passed.")
        else:
            log(f"STEP 7 [FAILED]: function='{func}' (not 1000/1100)"
                " and NOT in kartierung_dd_gesamt → building will be removed by filter_table().", Qgis.Critical)
            cur.execute(f"""
                SELECT str, hnr, id_alkis FROM "{schema}".kartierung_dd_gesamt
                WHERE trim(lower(str)) = trim(lower(split_part(%s, ' ', 1)))
                LIMIT 3;
            """, (address or '',))
            addr_candidates = cur.fetchall()
            if addr_candidates:
                log("  → Street found in kartierung_dd_gesamt, but house number does not match:", Qgis.Warning)
                for c in addr_candidates:
                    log(f"     str='{c[0]}', hnr='{c[1]}', id_alkis='{c[2]}'", Qgis.Warning)
            else:
                log("  → Street not found in kartierung_dd_gesamt.", Qgis.Warning)

    # ------------------------------------------------------------------
    # SCHRITT 8: Geometrie (fill_remaining_attributes_and_geometry)
    # ------------------------------------------------------------------
    if not has_geom:
        cur.execute("""
            SELECT COUNT(*) FROM citydb.property p
            JOIN citydb.feature f ON p.val_feature_id = f.id
            WHERE p.feature_id = %s
              AND p.name = 'boundary'
              AND f.objectclass_id = 710;
        """, (cityobject_id,))
        ground_count = cur.fetchone()[0]
        log(f"STEP 8 [FAILED]: No geometry in citydb_filter."
            f" Ground Surfaces (objectclass=710) in CityDB: {ground_count}", Qgis.Critical)
    else:
        log("STEP 8 [OK]: Geometry present.")

    # ------------------------------------------------------------------
    # SCHRITT 9: Footprint-Filter (calculate_footprint entfernt < 30 m²)
    # ------------------------------------------------------------------
    if footprint is None:
        log("STEP 9 [INFO]: Footprint not calculated (no geometry yet, or step"
            " not yet executed).", Qgis.Warning)
    elif footprint < 30:
        log(f"STEP 9 [FAILED]: Footprint = {footprint:.2f} m² below 30 m²"
            " → building will be removed by calculate_footprint().", Qgis.Critical)
    else:
        log(f"STEP 9 [OK]: Footprint = {footprint:.2f} m² ≥ 30 m² → area filter passed.")

    # ------------------------------------------------------------------
    # SCHRITT 10: Kartierungsverknüpfung (intersect_and_update_citydb_filter)
    # ------------------------------------------------------------------
    cur.execute(f"""
        SELECT sst, sst_sub, development_type_code, mapping_id, classification_source
        FROM "{schema}".citydb_filter
        WHERE gml_id = %s;
    """, (OBJEKT_ID,))
    kart_info = cur.fetchone()
    if kart_info:
        log(f"STEP 10 [INFO]: Survey linkage:"
            f" sst='{kart_info[0]}', sst_sub='{kart_info[1]}',"
            f" dev_code='{kart_info[2]}', mapping_id={kart_info[3]}, source='{kart_info[4]}'")
        if kart_info[0] is None:
            log("  → SST is NULL: building has no survey assignment"
                " (neither via id_alkis nor address match).", Qgis.Warning)

    log(SEP)
    log("TRACE COMPLETED")
    log(SEP)
