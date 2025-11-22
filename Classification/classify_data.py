import os
import json
import numpy as np
import pandas as pd
import joblib
import configparser
from typing import List, Dict, Tuple, Optional
from qgis.core import QgsMessageLog, Qgis, QgsDataSourceUri, QgsVectorLayer, QgsProject

from .validate_data import ValidateData

class Classifier:
    """
    Klassifiziert die verbleibenden Gebäude in classification_data mit derselben Logik wie validate_data:
    - sequentielle Level-Klassifikation mit Filterung nach vorherigem Ergebnis
    - building_age als Feature, Constraint und direkte Zuweisung in Endlevels (nur atomare Ages)
    - Level 11 und 1111 deterministisch (regelbasiert)
    - per-Level Confidence und Gesamtconfidence je Gebäude
    - Confidence-Report wie in classify_data.py
    """

    def __init__(self, conn, cur, connection_params):
        self.conn = conn
        self.cur = cur
        self.connection_params = connection_params

        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self.model_dir = os.path.join(os.path.dirname(__file__), config.get('Paths', 'model_dir'))
        self.vis_path = os.path.join(os.path.dirname(__file__), config.get('Paths', 'vis_path'))
        self.conf_report_path = os.path.join(os.path.dirname(__file__), config.get('Paths', 'confidence_report'))

        # Reuse validator helpers to keep logic consistent
        self.validator = ValidateData(conn, cur, connection_params)
        self.LEVELS: List[Tuple[str, List[str]]] = self.validator.LEVELS
        self.ENDLEVELS_DIRECT_AGE = self.validator.ENDLEVELS_DIRECT_AGE
        self.SKIP_METRIC_LEVELS = getattr(self.validator, 'SKIP_METRIC_LEVELS', {'11', '1111'})

    # -----------------------------
    # DB utilities
    # -----------------------------
    def ensure_level_columns(self):
        """
        Stellt sicher, dass Level-, Confidence- und Metadaten-Spalten existieren.
        Metadaten: confidence, classification_source, classification_source_id
        """
        level_columns = [lvl for (lvl, _) in self.LEVELS]
        for column in level_columns:
            self.cur.execute(f'''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_schema = 'MPSCDresden'
                          AND table_name = 'classification_data' 
                          AND column_name = '{column}'
                    ) THEN
                        ALTER TABLE "MPSCDresden".classification_data 
                        ADD COLUMN "{column}" VARCHAR;
                    END IF;
                END $$;
            ''')
            confidence_column = f"{column}_confidence"
            self.cur.execute(f'''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_schema = 'MPSCDresden'
                          AND table_name = 'classification_data' 
                          AND column_name = '{confidence_column}'
                    ) THEN
                        ALTER TABLE "MPSCDresden".classification_data 
                        ADD COLUMN "{confidence_column}" FLOAT;
                    END IF;
                END $$;
            ''')
        self.conn.commit()

        # Gesamtconfidence
        self.cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'MPSCDresden'
                      AND table_name = 'classification_data'
                      AND column_name = 'overall_confidence'
                ) THEN
                    ALTER TABLE "MPSCDresden".classification_data
                    ADD COLUMN overall_confidence FLOAT;
                END IF;
            END $$;
        """)
        self.conn.commit()

        # Metadaten-Spalten (correct column name: classification_source_id)
        self.cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'MPSCDresden'
                      AND table_name = 'classification_data'
                      AND column_name = 'confidence'
                ) THEN
                    ALTER TABLE "MPSCDresden".classification_data
                    ADD COLUMN confidence FLOAT;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'MPSCDresden'
                      AND table_name = 'classification_data'
                      AND column_name = 'classification_source'
                ) THEN
                    ALTER TABLE "MPSCDresden".classification_data
                    ADD COLUMN classification_source VARCHAR;
                END IF;
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'MPSCDresden'
                      AND table_name = 'classification_data'
                      AND column_name = 'classification_source_id'
                ) THEN
                    ALTER TABLE "MPSCDresden".classification_data
                    ADD COLUMN classification_source_id INTEGER;
                END IF;
            END $$;
        """)
        self.conn.commit()

    def load_classification_data(self) -> pd.DataFrame:
        query = 'SELECT * FROM "MPSCDresden".classification_data'
        self.cur.execute(query)
        rows = self.cur.fetchall()
        colnames = [desc[0] for desc in self.cur.description]
        return pd.DataFrame(rows, columns=colnames)

    def reset_level_columns(self, df: pd.DataFrame):
        """
        Setzt Level- und Confidence-Spalten auf NULL vor der neuen Klassifikation.
        """
        level_columns = [lvl for (lvl, _) in self.LEVELS]
        set_null_sql = ', '.join([f'"{col}" = NULL, "{col}_confidence" = NULL' for col in level_columns])
        self.cur.execute(f'''
            UPDATE "MPSCDresden".classification_data
            SET {set_null_sql},
                "sst" = NULL,
                "overall_confidence" = NULL,
                "confidence" = NULL,
                "classification_source" = NULL,
                "classification_source_id" = NULL
        ''')
        self.conn.commit()
        QgsMessageLog.logMessage("classification_data Level-, Confidence- und Metadaten-Spalten zurückgesetzt.", level=Qgis.Info)

    def batch_update_level(self, df: pd.DataFrame, level: str, indices: pd.Index, preds: pd.Series, confs: pd.Series):
        """
        Batch-Update der Levelspalten und Confidence-Spalten.
        Schreibt jetzt zusätzlich die Werte ins DataFrame (df), nicht nur in die DB,
        sodass write_final_results die sst/overall_confidence korrekt berechnen kann.
        """
        if len(indices) == 0:
            return

        # Sicherstellen, dass preds/confs Series mit passenden Indizes sind
        if not isinstance(preds, pd.Series):
            preds = pd.Series(preds, index=indices)
        else:
            preds = preds.reindex(indices)

        if not isinstance(confs, pd.Series):
            confs = pd.Series(confs, index=indices, dtype=float)
        else:
            confs = confs.reindex(indices).astype(float)

        # Update DataFrame (ersetze nur für die übergebenen Indizes)
        try:
            df.loc[indices, level] = preds.values
            df.loc[indices, f"{level}_confidence"] = confs.values
        except Exception:
            # fallback: elementweise (robuster bei ungewöhnlichen Index-Typen)
            for idx in indices:
                df.at[idx, level] = preds.get(idx)
                df.at[idx, f"{level}_confidence"] = confs.get(idx)

        # Persist in DB (nur für nicht-null Vorhersagen)
        update_rows = []
        for idx in indices:
            cls = preds.get(idx)
            conf = confs.get(idx)
            if pd.notna(cls):
                update_rows.append((str(cls), float(conf) if pd.notna(conf) else None, int(df.at[idx, 'db_filter_id'])))
        if update_rows:
            self.cur.executemany(
                f'UPDATE "MPSCDresden".classification_data SET "{level}" = %s, "{level}_confidence" = %s WHERE db_filter_id = %s',
                update_rows
            )
            self.conn.commit()

    def write_final_results(self, df: pd.DataFrame):
        """
        Finalisiert sst + overall_confidence und setzt classification_source / classification_source_id:
          age_direct -> Baualter (2)
          model / rule -> Modell (3)
        """
        level_columns = [lvl for (lvl, _) in self.LEVELS]

        # final sst by walking reverse order
        final_sst = []
        final_conf = []
        for i, row in df.iterrows():
            sst_val = None
            confs_path = []
            for lvl in reversed(level_columns):
                v = row.get(lvl)
                c = row.get(f"{lvl}_confidence")
                if pd.notna(v) and pd.notna(c):
                    confs_path.append(float(c))
                if pd.notna(v) and sst_val is None:
                    sst_val = v
            if confs_path:
                # geometric mean of confidences along the path
                gm = float(np.prod(confs_path) ** (1.0 / len(confs_path)))
            else:
                gm = None
            final_sst.append(sst_val)
            final_conf.append(gm)

        df['sst'] = final_sst
        df['overall_confidence'] = final_conf
        df['confidence'] = final_conf  # Alias der Gesamtconfidence

        # Nur ID 2 (Baualter) und 3 (Modell) behandeln
        source_label_map = {'model': 'Modell', 'age_direct': 'Baualter', 'rule': 'Modell'}
        source_id_map = {'model': 3, 'age_direct': 2, 'rule': 3}
        df['classification_source'] = df.get('source', pd.Series(index=df.index)).map(source_label_map)
        df['classification_source_id'] = df.get('source', pd.Series(index=df.index)).map(source_id_map)

        # persist sst, confidences und Quelle
        update_rows = []
        for i, row in df.iterrows():
            if pd.notna(row.get('sst')):
                update_rows.append((
                    str(row['sst']),
                    float(row['overall_confidence']) if pd.notna(row['overall_confidence']) else None,
                    float(row['confidence']) if pd.notna(row['confidence']) else None,
                    str(row['classification_source']) if pd.notna(row.get('classification_source')) else None,
                    int(row['classification_source_id']) if pd.notna(row.get('classification_source_id')) else None,
                    int(row['db_filter_id'])
                ))
        if update_rows:
            self.cur.executemany(
                '''
                UPDATE "MPSCDresden".classification_data
                SET "sst" = %s,
                    "overall_confidence" = %s,
                    "confidence" = %s,
                    "classification_source" = %s,
                    "classification_source_id" = %s
                WHERE db_filter_id = %s
                ''',
                update_rows
            )
            self.conn.commit()

    # -----------------------------
    # Confidence report (same as classify_data)
    # -----------------------------
    def generate_confidence_report(self, df: pd.DataFrame):
        try:
            confidence_ranges = [
                (0.9, 1.0, "> 0.9"),
                (0.8, 0.9, "0.8 - 0.9"),
                (0.7, 0.8, "0.7 - 0.8"),
                (0.6, 0.7, "0.6 - 0.7"),
                (0.5, 0.6, "0.5 - 0.6"),
                (0.0, 0.5, "< 0.5")
            ]

            report_lines = []
            report_lines.append("Confidence Report\n")
            report_lines.append("=" * 50 + "\n")

            level_columns = [lvl for (lvl, _) in self.LEVELS]
            for level in level_columns:
                confidence_column = f"{level}_confidence"
                if confidence_column not in df.columns or level not in df.columns:
                    continue

                report_lines.append(f"Level: {level}\n")
                report_lines.append("-" * 50 + "\n")

                targets = df[level].dropna().unique().tolist()
                for target_value in targets:
                    report_lines.append(f"  Target value: {target_value}\n")
                    filtered = df[df[level] == target_value]
                    for lower, upper, label in confidence_ranges:
                        cnt = filtered[(filtered[confidence_column] > lower) & (filtered[confidence_column] <= upper)].shape[0]
                        report_lines.append(f"    {label}: {cnt}\n")
                report_lines.append("\n")

            with open(self.conf_report_path, "w", encoding="utf-8") as f:
                f.writelines(report_lines)

            QgsMessageLog.logMessage(f"Confidence-Report erstellt: {self.conf_report_path}", level=Qgis.Info)
        except Exception as e:
            QgsMessageLog.logMessage(f"Fehler beim Confidence-Report: {str(e)}", level=Qgis.Critical)

    # -----------------------------
    # Main classification flow (consistent with ValidateData)
    # -----------------------------
    def classify(self):
        # Prepare (keine Kartierung-Übernahme notwendig)
        self.ensure_level_columns()
        all_data = self.load_classification_data()
        if all_data.empty:
            QgsMessageLog.logMessage("Keine Daten in classification_data.", level=Qgis.Critical)
            return
        self.reset_level_columns(all_data)

        # working frame with results + source to drive the pipeline
        all_data['results'] = None
        all_data['source'] = None  # 'model' | 'age_direct' | 'rule'

        # loop through levels like in ValidateData, write per-level predictions + confidences
        for level, target_names in self.LEVELS:
            # indices eligible for current level
            if level == '1':
                current_idx = all_data.index
            elif level == '11':
                current_idx = all_data.index  # rule-based from level 1 result
            elif level == '12':
                current_idx = all_data.index[all_data['results'] == 'Other']
            elif level == '121':
                current_idx = all_data.index[all_data['results'] == 'HH']
            elif level == '122':
                current_idx = all_data.index[all_data['results'] == 'LW']
            elif level == '112':
                current_idx = all_data.index[all_data['results'] == 'ME']
            elif level == '113':
                current_idx = all_data.index[all_data['results'] == 'ER']
            elif level == '114':
                current_idx = all_data.index[all_data['results'] == 'EE']
            elif level == '111':
                current_idx = all_data.index[all_data['results'] == 'MR']
            elif level == '1111':
                mask = all_data['results'].astype(str).str.startswith('MR', na=False) & (~all_data['results'].isin(['MR5','MR6']))
                current_idx = all_data.index[mask]
            else:
                current_idx = all_data.index

            if len(current_idx) == 0:
                continue

            # rule-based levels (no model)
            if level == '11':
                y_pred = self.validator.predict_level_11(all_data.loc[current_idx])
                conf = pd.Series(1.0, index=current_idx, dtype=float)
                all_data.loc[current_idx, 'results'] = y_pred.values
                all_data.loc[current_idx, 'source'] = 'rule'
                self.batch_update_level(all_data, level, current_idx, y_pred, conf)
                continue

            if level == '1111':
                y_pred = self.validator.predict_level_1111(all_data.loc[current_idx])
                conf = pd.Series(1.0, index=current_idx, dtype=float)
                all_data.loc[current_idx, 'results'] = y_pred.values
                all_data.loc[current_idx, 'source'] = 'rule'
                self.batch_update_level(all_data, level, current_idx, y_pred, conf)
                continue

            # end-level direct assignment by atomic building_age
            direct_assigned = pd.Series(index=current_idx, dtype=object)
            conf_assigned = pd.Series(index=current_idx, dtype=float)
            direct_mask = pd.Series(False, index=current_idx)
            if level in self.ENDLEVELS_DIRECT_AGE:
                assigned = []
                for idx in current_idx:
                    row = all_data.loc[idx]
                    ages = self.validator.parse_building_age(row.get('building_age'))
                    cls = self.validator.direct_assign_endlevel(level, ages, target_names)
                    assigned.append(cls)
                assigned_series = pd.Series(assigned, index=current_idx)
                assignable = assigned_series.notna()
                if assignable.any():
                    all_data.loc[current_idx[assignable], 'results'] = assigned_series[assignable].values
                    all_data.loc[current_idx[assignable], 'source'] = 'age_direct'
                    direct_assigned.loc[current_idx[assignable]] = assigned_series[assignable]
                    conf_assigned.loc[current_idx[assignable]] = 1.0
                    direct_mask.loc[current_idx[assignable]] = True

                # write direct assignments to DB for this level
                self.batch_update_level(all_data, level, current_idx[assignable], direct_assigned, conf_assigned)

            # remaining rows -> model inference
            model_idx = current_idx[~direct_mask]
            if len(model_idx) > 0:
                X, model, _ = self.validator.prepare_features_for_level(all_data.loc[model_idx], level)
                if model is not None and not X.empty:
                    prob = model.predict_proba(X)
                    classes = model.classes_
                    base_indices = np.argmax(prob, axis=1)
                    base_pred = classes[base_indices]

                    # apply constraints (including level 1 and level 12)
                    constrained_labels = []
                    constrained_conf = []
                    for i, idx in enumerate(model_idx):
                        new_label = self.validator.restrict_with_constraints(
                            level,
                            all_data.loc[idx],
                            target_names,
                            classes,
                            prob[i],
                            base_pred[i]
                        )
                        if new_label is None:
                            # block transition -> keep previous result; no update for this level
                            constrained_labels.append(None)
                            constrained_conf.append(None)
                        else:
                            constrained_labels.append(new_label)
                            # confidence for chosen label
                            if new_label in classes:
                                new_conf = float(prob[i][np.where(classes == new_label)[0][0]])
                            else:
                                # in case constrained label not in model.classes_ (should not happen)
                                new_conf = float(prob[i][base_indices[i]])
                            constrained_conf.append(new_conf)

                    y_pred_model = pd.Series(constrained_labels, index=model_idx, dtype=object)
                    conf_model = pd.Series(constrained_conf, index=model_idx, dtype=float)

                    # update results only for non-None predictions
                    valid_mask = y_pred_model.notna()
                    if valid_mask.any():
                        valid_idx = model_idx[valid_mask]
                        all_data.loc[valid_idx, 'results'] = y_pred_model.loc[valid_idx].values
                        all_data.loc[valid_idx, 'source'] = 'model'
                        self.batch_update_level(all_data, level, valid_idx, y_pred_model, conf_model)

        # finalize sst + overall confidence + metadata (nur ID 2/3)
        self.write_final_results(all_data)

        # generate confidence report
        self.generate_confidence_report(self.load_classification_data())

        # load layer in QGIS (optional)
        self.load_and_visualize_classification_data()

    # -----------------------------
    # Visualization
    # -----------------------------
    def load_and_visualize_classification_data(self):
        try:
            uri = QgsDataSourceUri()
            uri.setConnection(
                self.connection_params['host'],
                str(self.connection_params['port']),
                self.connection_params['dbname'],
                self.connection_params['user'],
                self.connection_params['password']
            )
            uri.setDataSource('MPSCDresden', 'classification_data', 'geom', '', 'db_filter_id')
            layer_name = 'Classification Data'

            existing = QgsProject.instance().mapLayersByName(layer_name)
            if existing:
                QgsProject.instance().removeMapLayer(existing[0])

            layer = QgsVectorLayer(uri.uri(False), layer_name, 'postgres')
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                QgsMessageLog.logMessage("Classification Data layer loaded.", level=Qgis.Info)
            else:
                QgsMessageLog.logMessage("Classification Data layer invalid.", level=Qgis.Warning)
        except Exception as e:
            QgsMessageLog.logMessage(f"Error loading classification layer: {e}", level=Qgis.Warning)
