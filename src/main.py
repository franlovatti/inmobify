import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from apify import Actor

from .geo_features import construir_features_inferencia
from .model_defs import cargar_modelo_seguro

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_MODELOS = BASE_DIR / "modelos_entrenados"
RUTA_META = BASE_DIR / "modelos_ranking_meta.json"

COLS_MODELO = [
    "log_superficie_m2",
    "superficie_m2",
    "es_gran_superficie",
    "tiene_expensas",
    "log_trafico_resenas_200m",
    "densidad_comercial_200m",
    "diversidad_rubros_200m",
    "dist_polo_mas_cercano_m",
    "barrio",
    "tipo_inmueble",
]


def _clasificar_diagnostico(precio_pub: float | None, precio_pred: float) -> tuple[float | None, str]:
    if precio_pub is None or pd.isna(precio_pub) or precio_pub <= 0 or precio_pred <= 0:
        return None, "Tasación de Referencia"

    diff_pct = ((precio_pub - precio_pred) / precio_pred) * 100.0
    if diff_pct <= -15.0:
        return round(diff_pct, 1), "Oportunidad (≥15% bajo tasación)"
    if diff_pct >= 15.0:
        return round(diff_pct, 1), "Sobrevaluado (≥15% sobre tasación)"
    return round(diff_pct, 1), "Precio de Mercado"


async def main() -> None:
    async with Actor:
        actor_input = await Actor.get_input() or {}

        kv_store_name = actor_input.get("kv_store_name", "inmobify-storage")
        modelo_elegido = actor_input.get("modelo_elegido", "Ensamble Top-3")
        modo = actor_input.get("modo_ejecucion", "individual")
        propiedades_lote = actor_input.get("propiedades") or []

        # 1. Resolver registros de entrada (soporta Lote o Individual)
        if isinstance(propiedades_lote, list) and len(propiedades_lote) > 0:
            registros = propiedades_lote
            modo_real = "lote" if len(registros) > 1 else "individual"
        elif modo == "individual" or actor_input.get("superficie_m2") is not None:
            registros = [{
                "id_consulta": actor_input.get("id_consulta", "IND-001"),
                "barrio": actor_input.get("barrio", "Centro"),
                "tipo_inmueble": actor_input.get("tipo_inmueble", "Local comercial"),
                "superficie_m2": actor_input.get("superficie_m2", 65),
                "latitud": actor_input.get("latitud", -32.9468),
                "longitud": actor_input.get("longitud", -60.6393),
                "tiene_expensas": actor_input.get("tiene_expensas", 1),
                "precio_publicado_usd": actor_input.get("precio_publicado_usd", None),
            }]
            modo_real = "individual"
        else:
            await Actor.fail(status_message="No se recibieron propiedades válidas en 'propiedades' ni en campos individuales.")
            return

        Actor.log.info(f"Iniciando inferencia | Modo: {modo_real} ({len(registros)} registros) | Modelo: {modelo_elegido}")

        # 2. Leer catálogo local empaquetado en el Actor
        if not RUTA_META.exists():
            await Actor.fail(status_message=f"No se encontró '{RUTA_META}' dentro de la imagen del Actor.")
            return

        metadata = json.loads(RUTA_META.read_text(encoding="utf-8"))
        modelos_disp = metadata.get("modelos_disponibles", {})

        if modelo_elegido not in modelos_disp:
            modelo_elegido = metadata.get("modelo_recomendado", "Ensamble Top-3")
            Actor.log.warning(f"Modelo no reconocido; usando recomendado: {modelo_elegido}")

        info_modelo = modelos_disp[modelo_elegido]
        archivo_pkl = info_modelo["archivo_kv"]
        sha_esperado = info_modelo.get("sha256", "")
        ruta_pkl = DIR_MODELOS / archivo_pkl

        # 3. Cargar .pkl local y verificar integridad SHA-256
        if not ruta_pkl.exists():
            await Actor.fail(status_message=f"No se encontró el archivo de modelo local: '{ruta_pkl}'.")
            return

        model_bytes = ruta_pkl.read_bytes()
        sha_actual = hashlib.sha256(model_bytes).hexdigest()
        if sha_esperado and sha_actual != sha_esperado:
            await Actor.fail(status_message=f"Fallo de integridad SHA-256 para '{archivo_pkl}'.")
            return

        modelo = cargar_modelo_seguro(model_bytes)
        Actor.log.info(f"Modelo local '{modelo_elegido}' ({archivo_pkl}) verificado y cargado OK.")

        # 4. Cargar maestro de comercios desde Key-Value Store para cálculo Haversine
        kv = await Actor.open_key_value_store(name=kv_store_name)
        comercios_raw = await kv.get_value("maestro_comercios_rosario.csv")
        df_comercios = None
        if comercios_raw:
            txt_c = comercios_raw.decode("utf-8") if isinstance(comercios_raw, bytes) else str(comercios_raw)
            df_comercios = pd.read_csv(io.StringIO(txt_c))

        # 5. Ingeniería de features y predicción
        df_in = pd.DataFrame(registros)
        df_feat = construir_features_inferencia(df_in, df_comercios)
        X_infer = df_feat[COLS_MODELO]

        preds_usd = modelo.predict(X_infer)
        preds_m2 = modelo.predict_m2(X_infer)

        # 6. Empaquetar resultados y publicar en Default Dataset
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        metricas_mod = info_modelo.get("metricas", {})
        resultados = []

        for idx, row in df_feat.iterrows():
            p_usd = round(float(preds_usd[idx]), 2)
            p_m2 = round(float(preds_m2[idx]), 2)
            p_pub = pd.to_numeric(row.get("precio_publicado_usd"), errors="coerce")
            p_pub_val = None if pd.isna(p_pub) or p_pub <= 0 else round(float(p_pub), 2)
            diff_pct, diag = _clasificar_diagnostico(p_pub_val, p_usd)

            resultados.append({
                "row_number": int(row["row_number"]) if pd.notna(row.get("row_number")) else None,
                "id_consulta": str(row.get("id_consulta", f"PROP-{idx+1:03d}")),
                "barrio": row["barrio"],
                "tipo_inmueble": row["tipo_inmueble"],
                "superficie_m2": round(float(row["superficie_m2"]), 1),
                "tiene_expensas": int(row["tiene_expensas"]),
                "densidad_comercial_200m": int(row["densidad_comercial_200m"]),
                "diversidad_rubros_200m": int(row["diversidad_rubros_200m"]),
                "dist_polo_mas_cercano_m": round(float(row["dist_polo_mas_cercano_m"]), 1),
                "precio_publicado_usd": p_pub_val,
                "precio_predicho_usd": p_usd,
                "precio_m2_predicho_usd": p_m2,
                "diferencia_vs_publicado_pct": diff_pct,
                "diagnostico_mercado": diag,
                "modelo_utilizado": modelo_elegido,
                "modelo_cv_mape_pct": metricas_mod.get("CV_MAPE (%)"),
                "modelo_test_mae_m2_usd": metricas_mod.get("Test_MAE_m2 (USD)"),
                "fecha_inferencia_utc": timestamp_iso,
            })

        await Actor.push_data(resultados)
        await kv.set_value("ultima_inferencia_inmobify.json", json.dumps(resultados, indent=2, ensure_ascii=False))
        Actor.log.info(f"Inferencia finalizada con éxito: {len(resultados)} tasaciones publicadas.")