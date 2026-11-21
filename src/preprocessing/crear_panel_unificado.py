#!/usr/bin/env python3
"""
crear_panel_unificado.py
Consolidación y armonización de las Encuestas Nacionales de Salud Mental de Colombia
(ENSM 2003, ENSM 2015 y ENSM 2025) en un solo archivo Parquet apilado (Pooled Cross-Section).

Arquitectura y Rendimiento:
1. Extrae automáticamente archivos comprimidos (.rar) usando unrar si no están extraídos.
2. Procesa y armoniza cada ronda por separado guardando un archivo temporal intermedio,
   liberando memoria inmediatamente (gc.collect()) para garantizar un consumo de RAM controlado.
3. Lee formatos heterogéneos: SAS (.sas7bdat), CSV con separador pipe (|) y SPSS (.sav).
4. Armoniza un núcleo de variables sociodemográficas, geográficas, ponderadores muestrales
   e indicadores de salud mental (depresión, conducta suicida).
5. Conserva el 100% de las columnas originales con prefijos únicos:
   'ensm2003_', 'ensm2015_', 'ensm2025_'.
6. Unifica los esquemas mediante PyArrow y escribe el archivo final por bloques (streaming)
   en un único archivo Parquet comprimido con Snappy.
"""

import os
import gc
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
import pyarrow as pa
import pyarrow.parquet as pq

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"

PATH_2003 = DATOS_DIR / "estudio-nacional-salud-mental-2003-interno" / "adultosfinal.sas7bdat"
DIR_2015 = DATOS_DIR / "base-datos-salud-mental-2015-msps" / "Base de datos y diccionario"
RAR_2015 = DIR_2015 / "ENSM_2015_csv.rar"
CSV_2015_DIR = DIR_2015 / "csv_extracted"
DIR_2025 = DATOS_DIR / "microdatos-salud-mental-2025deyd"

OUTPUT_PARQUET = DATOS_DIR / "panel_salud_mental_colombia.parquet"

TEMP_2003 = DATOS_DIR / "_temp_ensm_2003.parquet"
TEMP_2015 = DATOS_DIR / "_temp_ensm_2015.parquet"
TEMP_2025 = DATOS_DIR / "_temp_ensm_2025.parquet"


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on, how="left") -> pd.DataFrame:
    """Combina dos DataFrames evitando sufijos duplicados _x, _y para columnas repetidas."""
    cols_to_use = [c for c in right.columns if c not in left.columns or c in on]
    return pd.merge(left, right[cols_to_use], on=on, how=how)


def asegurar_extraccion_2015():
    """Descomprime ENSM_2015_csv.rar usando unrar si no existe la carpeta extraída."""
    if not CSV_2015_DIR.exists() or not any(CSV_2015_DIR.glob("*.csv")):
        print(f"[EXTRACCIÓN] Descomprimiendo {RAR_2015.name} con unrar...")
        CSV_2015_DIR.mkdir(parents=True, exist_ok=True)
        cmd = ["unrar", "x", "-o+", str(RAR_2015), str(CSV_2015_DIR) + "/"]
        subprocess.run(cmd, check=True)
        print("[EXTRACCIÓN] Extracción completada exitosamente.")
    else:
        print("[EXTRACCIÓN] Archivos CSV 2015 ya disponibles en csv_extracted/.")


def crear_columnas_armonizadas(df_harm: pd.DataFrame) -> pd.DataFrame:
    """Asegura dtypes consistentes y uniformes para las columnas armonizadas en todas las rondas."""
    tipos = {
        "ola_anio": "Int32",
        "encuesta": "string",
        "id_hogar": "string",
        "id_persona": "string",
        "edad": "Float32",
        "sexo": "string",
        "sexo_cod": "Float32",
        "es_adulto": "boolean",
        "tipo_poblacion": "string",
        "estrato": "Float32",
        "nivel_educativo": "string",
        "cod_departamento": "string",
        "cod_municipio": "string",
        "region": "Float32",
        "zona": "string",
        "fex": "Float32",
        "depresion_tristeza_2sem": "Float32",
        "depresion_perdida_interes": "Float32",
        "suicidio_ideacion": "Float32",
        "suicidio_plan": "Float32",
        "suicidio_intento": "Float32",
    }
    for col, dtype in tipos.items():
        if col not in df_harm.columns:
            df_harm[col] = pd.Series([None] * len(df_harm), index=df_harm.index, dtype=dtype)
        if dtype == "boolean":
            df_harm[col] = df_harm[col].astype("boolean")
        elif dtype == "string":
            df_harm[col] = df_harm[col].fillna("").astype("string")
        elif dtype.startswith("Float"):
            df_harm[col] = pd.to_numeric(df_harm[col], errors="coerce").astype("Float32")
        elif dtype.startswith("Int"):
            df_harm[col] = pd.to_numeric(df_harm[col], errors="coerce").astype("Int32")
        else:
            df_harm[col] = df_harm[col].astype(dtype)
    return df_harm[list(tipos.keys())]


def procesar_2003():
    """Lee y armoniza el Estudio Nacional de Salud Mental 2003 (SAS) y guarda a parquet temporal."""
    print("\n" + "="*50)
    print("[2003] Leyendo adultosfinal.sas7bdat...")
    df_raw, meta = pyreadstat.read_sas7bdat(str(PATH_2003))
    print(f"[2003] Filas: {len(df_raw):,}, Columnas originales: {len(df_raw.columns):,}")

    df_harm = pd.DataFrame(index=df_raw.index)
    df_harm["ola_anio"] = 2003
    df_harm["encuesta"] = "ENSM_2003"
    
    int_mun = df_raw["INT_MUN"].fillna(0).astype(int).astype(str)
    int_seg = df_raw["INT_SEG"].fillna(0).astype(int).astype(str)
    int_fol = df_raw["INT_FOL"].fillna(0).astype(int).astype(str)
    int_hog = df_raw["INT_HOG"].fillna(0).astype(int).astype(str)
    sample_id = df_raw["SampleID"].astype(str)
    
    df_harm["id_hogar"] = "2003_H_" + int_mun + "_" + int_seg + "_" + int_fol + "_" + int_hog
    df_harm["id_persona"] = "2003_P_" + sample_id

    edad = pd.to_numeric(df_raw["Age"], errors="coerce")
    if "SC1" in df_raw.columns:
        edad = edad.fillna(pd.to_numeric(df_raw["SC1"], errors="coerce"))
    df_harm["edad"] = edad

    sexo_num = pd.to_numeric(df_raw["Sex"], errors="coerce")
    df_harm["sexo_cod"] = sexo_num
    df_harm["sexo"] = np.where(sexo_num == 1, "Hombre", np.where(sexo_num == 2, "Mujer", "No especificado"))

    df_harm["es_adulto"] = df_harm["edad"] >= 18
    df_harm["tipo_poblacion"] = "adulto"

    df_harm["region"] = pd.to_numeric(df_raw.get("region"), errors="coerce")
    df_harm["cod_municipio"] = int_mun
    df_harm["cod_departamento"] = ""
    df_harm["estrato"] = np.nan
    df_harm["nivel_educativo"] = df_raw.get("DE20", "").astype(str)
    df_harm["zona"] = "No especificado"
    df_harm["fex"] = np.nan

    df_harm["depresion_tristeza_2sem"] = pd.to_numeric(df_raw.get("D24a"), errors="coerce")
    df_harm["depresion_perdida_interes"] = pd.to_numeric(df_raw.get("D25"), errors="coerce")
    df_harm["suicidio_ideacion"] = pd.to_numeric(df_raw.get("D26cc"), errors="coerce")
    df_harm["suicidio_plan"] = pd.to_numeric(df_raw.get("D26dd"), errors="coerce")
    df_harm["suicidio_intento"] = pd.to_numeric(df_raw.get("D26ee"), errors="coerce")

    df_harm = crear_columnas_armonizadas(df_harm)

    df_raw_renamed = df_raw.copy()
    df_raw_renamed.columns = [f"ensm2003_{c}" for c in df_raw.columns]

    for col in df_raw_renamed.select_dtypes(include=["object", "string"]).columns:
        df_raw_renamed[col] = df_raw_renamed[col].astype(str)

    df_final = pd.concat([df_harm, df_raw_renamed], axis=1)
    table_2003 = pa.Table.from_pandas(df_final)
    pq.write_table(table_2003, str(TEMP_2003), compression="snappy")
    print(f"[2003] Guardado temporal: {TEMP_2003.name} | Shape: {table_2003.shape}")
    del df_raw, df_raw_renamed, df_harm, df_final, table_2003
    gc.collect()


def procesar_2015():
    """Lee y armoniza la Encuesta Nacional de Salud Mental 2015 (CSV) y guarda a parquet temporal."""
    print("\n" + "="*50)
    print("[2015] Leyendo tablas CSV de 2015...")

    path_ad1 = CSV_2015_DIR / "Adultos_I.csv"
    path_ad2 = CSV_2015_DIR / "Adultos_II.csv"
    path_hog = CSV_2015_DIR / "Hogares.csv"
    path_per = CSV_2015_DIR / "Personas.csv"

    ad1 = pd.read_csv(path_ad1, sep="|", low_memory=False, na_values=["NULL", "null", "NA"])
    ad2 = pd.read_csv(path_ad2, sep="|", low_memory=False, na_values=["NULL", "null", "NA"])
    hog = pd.read_csv(path_hog, sep="|", low_memory=False, na_values=["NULL", "null", "NA"])
    per = pd.read_csv(path_per, sep="|", low_memory=False, na_values=["NULL", "null", "NA"])

    print(f"[2015] Registros leídos -> Adultos_I: {len(ad1):,}, Adultos_II: {len(ad2):,}, Personas: {len(per):,}, Hogares: {len(hog):,}")

    df_adultos = safe_merge(ad1, ad2, on=["identificador_hogar", "identificador_persona"])
    del ad1, ad2
    gc.collect()

    df_adultos = safe_merge(df_adultos, per, on=["identificador_hogar", "identificador_persona"])
    df_merged = safe_merge(df_adultos, hog, on=["identificador_hogar"])
    del df_adultos, per, hog
    gc.collect()

    print(f"[2015] Adultos consolidado: {len(df_merged):,} filas y {len(df_merged.columns):,} columnas")

    df_harm = pd.DataFrame(index=df_merged.index)
    df_harm["ola_anio"] = 2015
    df_harm["encuesta"] = "ENSM_2015"

    df_harm["id_hogar"] = "2015_H_" + df_merged["identificador_hogar"].astype(str)
    df_harm["id_persona"] = "2015_P_" + df_merged["identificador_persona"].astype(str)

    edad = pd.to_numeric(df_merged.get("age"), errors="coerce")
    df_harm["edad"] = edad

    sex_num = pd.to_numeric(df_merged.get("sex"), errors="coerce")
    df_harm["sexo_cod"] = sex_num
    df_harm["sexo"] = np.where(sex_num == 1, "Hombre", np.where(sex_num == 2, "Mujer", "No especificado"))

    df_harm["es_adulto"] = df_harm["edad"] >= 18
    df_harm["tipo_poblacion"] = "adulto"

    df_harm["estrato"] = pd.to_numeric(df_merged.get("m1_c4_estrato"), errors="coerce")
    df_harm["nivel_educativo"] = df_merged.get("NivelEducativo", "").astype(str)

    df_harm["region"] = pd.to_numeric(df_merged.get("region"), errors="coerce")
    df_harm["cod_municipio"] = ""
    df_harm["cod_departamento"] = ""
    area_val = pd.to_numeric(df_merged.get("AREA"), errors="coerce")
    df_harm["zona"] = np.where(area_val == 1, "Cabecera", np.where(area_val == 2, "Rural", "No especificado"))

    fex = pd.to_numeric(df_merged.get("FEX").astype(str).str.replace(",", "."), errors="coerce")
    df_harm["fex"] = fex

    df_harm["depresion_tristeza_2sem"] = pd.to_numeric(df_merged.get("d24a"), errors="coerce")
    df_harm["depresion_perdida_interes"] = pd.to_numeric(df_merged.get("d25"), errors="coerce")
    df_harm["suicidio_ideacion"] = pd.to_numeric(df_merged.get("d26cc"), errors="coerce")
    df_harm["suicidio_plan"] = pd.to_numeric(df_merged.get("d26dd"), errors="coerce")
    df_harm["suicidio_intento"] = pd.to_numeric(df_merged.get("d26ee"), errors="coerce")

    df_harm = crear_columnas_armonizadas(df_harm)

    df_raw_renamed = df_merged.copy()
    df_raw_renamed.columns = [f"ensm2015_{c}" for c in df_merged.columns]

    for col in df_raw_renamed.select_dtypes(include=["object", "string"]).columns:
        df_raw_renamed[col] = df_raw_renamed[col].astype(str)

    df_final = pd.concat([df_harm, df_raw_renamed], axis=1)
    table_2015 = pa.Table.from_pandas(df_final)
    pq.write_table(table_2015, str(TEMP_2015), compression="snappy")
    print(f"[2015] Guardado temporal: {TEMP_2015.name} | Shape: {table_2015.shape}")
    del df_merged, df_raw_renamed, df_harm, df_final, table_2015
    gc.collect()


def procesar_2025():
    """Lee y armoniza la Encuesta Nacional de Salud Mental 2025 (SPSS) y guarda a parquet temporal."""
    print("\n" + "="*50)
    print("[2025] Leyendo archivos SPSS desarrollo_ensm_d.sav y hogares_ensm_h.sav...")

    path_des = DIR_2025 / "desarrollo_ensm_d.sav"
    path_hog = DIR_2025 / "hogares_ensm_h.sav"

    df_des, meta_des = pyreadstat.read_sav(str(path_des))
    df_hog, meta_hog = pyreadstat.read_sav(str(path_hog))

    print(f"[2025] Desarrollo filas: {len(df_des):,}, Hogares filas: {len(df_hog):,}")

    cols_hog = ["KEY", "estrato", "cod_dpto", "zona", "FFINALHOGAR"]
    cols_hog_avail = [c for c in cols_hog if c in df_hog.columns]
    df_hog_sub = df_hog[cols_hog_avail].rename(columns={"KEY": "KEY_HOGAR"})
    del df_hog
    gc.collect()

    df_merged = safe_merge(df_des, df_hog_sub, on="KEY_HOGAR")
    del df_des, df_hog_sub
    gc.collect()

    print(f"[2025] Registros consolidados: {len(df_merged):,} columnas: {len(df_merged.columns):,}")

    df_harm = pd.DataFrame(index=df_merged.index)
    df_harm["ola_anio"] = 2025
    df_harm["encuesta"] = "ENSM_2025"

    df_harm["id_hogar"] = "2025_H_" + df_merged["KEY_HOGAR"].astype(str)
    df_harm["id_persona"] = "2025_P_" + df_merged["KEY"].astype(str)

    edad = pd.to_numeric(df_merged.get("c1_p4"), errors="coerce")
    df_harm["edad"] = edad

    sexo_str = df_merged.get("c1_p3").astype(str).str.strip().str.capitalize()
    df_harm["sexo"] = sexo_str
    df_harm["sexo_cod"] = np.where(sexo_str == "Hombre", 1.0, np.where(sexo_str == "Mujer", 2.0, np.nan))

    df_harm["es_adulto"] = df_harm["edad"] >= 18
    df_harm["tipo_poblacion"] = np.where(df_harm["edad"] >= 18, "adulto", np.where(df_harm["edad"] >= 12, "adolescente", "nino"))

    estrato_raw = df_merged.get("estrato").astype(str).str.extract(r'(\d+)')[0]
    df_harm["estrato"] = pd.to_numeric(estrato_raw, errors="coerce")
    df_harm["nivel_educativo"] = ""

    df_harm["region"] = np.nan
    df_harm["cod_municipio"] = ""
    df_harm["cod_departamento"] = df_merged.get("cod_dpto", "").astype(str)
    zona_val = pd.to_numeric(df_merged.get("zona"), errors="coerce")
    df_harm["zona"] = np.where(zona_val == 1, "Cabecera", np.where(zona_val == 2, "Rural", "No especificado"))

    fex = pd.to_numeric(df_merged.get("FPER_DES"), errors="coerce")
    df_harm["fex"] = fex

    df_harm["depresion_tristeza_2sem"] = pd.to_numeric(df_merged.get("i2_1_p3_a"), errors="coerce")
    df_harm["depresion_perdida_interes"] = pd.to_numeric(df_merged.get("i2_1_p3_d"), errors="coerce")
    df_harm["suicidio_ideacion"] = pd.to_numeric(df_merged.get("i2_2_p20"), errors="coerce")
    df_harm["suicidio_plan"] = pd.to_numeric(df_merged.get("i2_2_p22"), errors="coerce")
    df_harm["suicidio_intento"] = pd.to_numeric(df_merged.get("i2_2_p23"), errors="coerce")

    df_harm = crear_columnas_armonizadas(df_harm)

    df_raw_renamed = df_merged.copy()
    df_raw_renamed.columns = [f"ensm2025_{c}" for c in df_merged.columns]

    for col in df_raw_renamed.select_dtypes(include=["object", "string"]).columns:
        df_raw_renamed[col] = df_raw_renamed[col].astype(str)

    df_final = pd.concat([df_harm, df_raw_renamed], axis=1)
    table_2025 = pa.Table.from_pandas(df_final)
    pq.write_table(table_2025, str(TEMP_2025), compression="snappy")
    print(f"[2025] Guardado temporal: {TEMP_2025.name} | Shape: {table_2025.shape}")
    del df_merged, df_raw_renamed, df_harm, df_final, table_2025
    gc.collect()


def fusionar_en_parquet_final():
    """Lee esquemas, unifica y escribe en streaming al Parquet final."""
    print("\n" + "="*50)
    print("[UNIFICACIÓN] Leyendo esquemas de archivos temporales...")

    sch_2003 = pq.read_schema(str(TEMP_2003))
    sch_2015 = pq.read_schema(str(TEMP_2015))
    sch_2025 = pq.read_schema(str(TEMP_2025))

    unified_schema = pa.unify_schemas([sch_2003, sch_2015, sch_2025])
    print(f"[UNIFICACIÓN] Esquema unificado construido con {len(unified_schema)} columnas.")

    print(f"[STREAMING] Escribiendo archivo Parquet consolidado en {OUTPUT_PARQUET}...")
    total_filas = 0
    with pq.ParquetWriter(str(OUTPUT_PARQUET), unified_schema, compression="snappy") as writer:
        for temp_file in [TEMP_2003, TEMP_2015, TEMP_2025]:
            print(f"  -> Transfiriendo bloques desde {temp_file.name}...")
            pf = pq.ParquetFile(str(temp_file))
            for batch in pf.iter_batches(batch_size=10000):
                tbl = pa.Table.from_batches([batch])
                arrays = []
                for field in unified_schema:
                    if field.name in tbl.column_names:
                        arrays.append(tbl[field.name])
                    else:
                        arrays.append(pa.nulls(len(tbl), type=field.type))
                conformed_tbl = pa.Table.from_arrays(arrays, schema=unified_schema)
                writer.write_table(conformed_tbl)
                total_filas += len(tbl)
                del tbl, arrays, conformed_tbl

    print(f"[STREAMING] Finalizado. Total de filas escritas: {total_filas:,}")

    # Limpieza de temporales
    for temp_file in [TEMP_2003, TEMP_2015, TEMP_2025]:
        if temp_file.exists():
            temp_file.unlink()
    print("[LIMPIEZA] Archivos temporales eliminados.")


def main():
    print("="*60)
    print("CONSOLIDACIÓN DE PANELES DE SALUD MENTAL COLOMBIA (POOLED CROSS-SECTION)")
    print("="*60)

    # 1. Asegurar descompresión
    asegurar_extraccion_2015()

    # 2. Procesar cada ronda independientemente
    procesar_2003()
    procesar_2015()
    procesar_2025()

    # 3. Fusión por streaming en Parquet
    fusionar_en_parquet_final()

    # 4. Estadísticas finales
    size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
    print(f"\n[GUARDADO] ¡Archivo Parquet creado exitosamente!")
    print(f"  * Ruta: {OUTPUT_PARQUET}")
    print(f"  * Tamaño final comprimido: {size_mb:.2f} MB")

    print("\n" + "="*50)
    print("[VALIDACIÓN RÁPIDA DE CONTENIDOS]")
    core_cols = ["ola_anio", "encuesta", "edad", "sexo", "tipo_poblacion", "estrato", "suicidio_ideacion", "suicidio_intento"]
    df_core = pq.read_table(str(OUTPUT_PARQUET), columns=core_cols).to_pandas()
    print("\nConteo de observaciones por ola/encuesta y tipo de población:")
    print(df_core.groupby(["ola_anio", "encuesta", "tipo_poblacion"]).size())
    print("\nDistribución por Sexo:")
    print(pd.crosstab(df_core["ola_anio"], df_core["sexo"]))
    print("\n" + "="*60)
    print("PIPELINE COMPLETADO CON ÉXITO")
    print("="*60)


if __name__ == "__main__":
    main()
