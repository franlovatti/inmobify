import io
import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from apify import Actor
from apify_client import ApifyClient

# ==============================================================================
# CONSTANTES DE CONFIGURACIÓN
# ==============================================================================
KV_STORE_NAME = "inmobify-storage"
TIPO_CAMBIO_USD = 1600
TTL_DIAS_MAPS = 45
MIN_VECINOS_CUBIERTOS = 5
PRESUPUESTO_INICIAL_MAPS = 2100
PRESUPUESTO_SEMANAL_MAPS = 700

RUBROS_MAPS = [
    "tienda", "restaurante", "cafeteria", "farmacia",
    "supermercado", "banco", "ferreteria", "panaderia",
    "peluqueria", "gimnasio", "libreria", "heladeria"
]

# ==============================================================================
# FUNCIONES DE CÁLCULO GEOESPACIAL Y PARSEO
# ==============================================================================
def haversine_m(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000.0
    lat1, lon1, lat2_arr, lon2_arr = map(np.radians, [lat1, lon1, lat2_arr, lon2_arr])
    dlat = lat2_arr - lat1
    dlon = lon2_arr - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2_arr) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def parse_price_info(val):
    if isinstance(val, str):
        try:
            data = ast.literal_eval(val)
        except Exception:
            return np.nan, np.nan, np.nan
    elif isinstance(val, list):
        data = val
    else:
        return np.nan, np.nan, np.nan

    if len(data) > 0 and isinstance(data[0], dict):
        op = data[0].get('operation_type', {}).get('name', np.nan)
        prices = data[0].get('prices', [])
        if len(prices) > 0 and isinstance(prices[0], dict):
            return op, prices[0].get('amount', np.nan), prices[0].get('currency', np.nan)
    return np.nan, np.nan, np.nan

def calcular_entorno_puntual(lat: float, lon: float, df_comercios: pd.DataFrame) -> dict:
    if df_comercios.empty:
        return {
            "densidad_comercial_200m": 0,
            "trafico_resenas_200m": 0.0,
            "densidad_comercial_350m": 0,
            "trafico_resenas_350m": 0.0,
            "resenas_por_comercio_350m": 0.0
        }
    d = haversine_m(lat, lon, df_comercios['lat'].values, df_comercios['lon'].values)
    res_arr = df_comercios['resenas'].values
    m200 = d <= 200
    m350 = d <= 350
    d350 = int(m350.sum())
    r350 = float(res_arr[m350].sum())
    return {
        "densidad_comercial_200m": int(m200.sum()),
        "trafico_resenas_200m": float(res_arr[m200].sum()),
        "densidad_comercial_350m": d350,
        "trafico_resenas_350m": r350,
        "resenas_por_comercio_350m": round(r350 / (d350 + 1), 1)
    }

# ==============================================================================
# BLOQUE ETL + CACHÉ MAPS + EDA (RAMA ANALYTICS)
# ==============================================================================
def ejecutar_etl_y_eda(apify_token: str, df_maestro_previo: pd.DataFrame, usar_ultimo_run_zp: bool = False):
    client = ApifyClient(apify_token)
    hoy = pd.Timestamp.now().normalize()

    # 1. EXTRACCIÓN Y LIMPIEZA DE ZONAPROP
    if usar_ultimo_run_zp:
        Actor.log.info("[1/4] Recuperando último run exitoso de Zonaprop...")
        run_zp = client.actor("ecomscrape/zonaprop-property-listings-scraper").last_run(status="SUCCEEDED").get()
    else:
        Actor.log.info("[1/4] Ejecutando scrapeo semanal en Zonaprop...")
        categorias_urls = [
            "https://www.zonaprop.com.ar/locales-comerciales-alquiler-rosario",
            "https://www.zonaprop.com.ar/oficinas-comerciales-alquiler-rosario",
            "https://www.zonaprop.com.ar/locales-comerciales-alquiler-gran-rosario",
            "https://www.zonaprop.com.ar/oficinas-comerciales-alquiler-gran-rosario"
        ]
        urls_zp = [f"{base}.html" for base in categorias_urls] + [
            f"{base}-pagina-{p}.html" for base in categorias_urls for p in range(2, 10)
        ]
        run_zp = client.actor("ecomscrape/zonaprop-property-listings-scraper").call(
            run_input={
                "ignore_url_failures": True,
                "max_items_per_url": 30,
                "max_retries_per_url": 1,
                "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"], "apifyProxyCountry": "AR"},
                "urls": urls_zp
            }
        )

    df_zp_raw = pd.json_normalize(list(client.dataset(run_zp.default_dataset_id).iterate_items()), sep='/')
    parsed = df_zp_raw['price_operation_types'].apply(parse_price_info)

    zp_clean = pd.DataFrame({
        'posting_id': df_zp_raw.get('posting_id', pd.Series(range(len(df_zp_raw)))),
        'lat': pd.to_numeric(df_zp_raw['posting_location/posting_geolocation/geolocation/latitude'], errors='coerce'),
        'lon': pd.to_numeric(df_zp_raw['posting_location/posting_geolocation/geolocation/longitude'], errors='coerce'),
        'barrio': df_zp_raw['posting_location/location/name'],
        'ciudad': df_zp_raw['posting_location/location/parent/name'],
        'tipo_inmueble': df_zp_raw['real_estate_type/name'],
        'tipo_operacion': [x[0] for x in parsed],
        'precio_crudo': pd.to_numeric([x[1] for x in parsed], errors='coerce'),
        'moneda': [x[2] for x in parsed],
        'expensas_ars': pd.to_numeric(df_zp_raw.get('expenses/amount', 0), errors='coerce').fillna(0),
        'superficie_m2': pd.to_numeric(df_zp_raw['main_features/c_f_t100/value'], errors='coerce')
    }).drop_duplicates(subset=['posting_id'])

    zp_clean = zp_clean.dropna(subset=['lat', 'lon', 'precio_crudo', 'superficie_m2', 'moneda', 'tipo_operacion'])
    zp_clean = zp_clean[
        (zp_clean['lat'].between(-33.05, -32.70)) &
        (zp_clean['lon'].between(-60.95, -60.60)) &
        (zp_clean['tipo_operacion'] == 'Alquiler') &
        (zp_clean['superficie_m2'].between(10, 2000))
    ].copy()

    es_pesos = zp_clean['moneda'].isin(['$', 'ARS'])
    zp_clean['precio_usd'] = np.where(es_pesos, zp_clean['precio_crudo'] / TIPO_CAMBIO_USD, zp_clean['precio_crudo'])
    zp_clean['precio_ars'] = np.where(~es_pesos, zp_clean['precio_crudo'] * TIPO_CAMBIO_USD, zp_clean['precio_crudo'])
    zp_clean = zp_clean[(zp_clean['precio_usd'] > 20) & (zp_clean['precio_ars'] <= zp_clean['precio_ars'].quantile(0.98))].copy()
    zp_clean['precio_m2_usd'] = (zp_clean['precio_usd'] / zp_clean['superficie_m2']).round(2)

    # 2. ACTUALIZACIÓN INCREMENTAL DE GOOGLE MAPS (TTL 45 DÍAS + KMEANS + FLAGS DE AHORRO)
    Actor.log.info("[2/4] Verificando cobertura comercial vigente en maestro de Maps...")
    if df_maestro_previo is not None and not df_maestro_previo.empty:
        df_maestro_previo['fecha_extraccion'] = pd.to_datetime(df_maestro_previo['fecha_extraccion'])
        df_gm_vigente = df_maestro_previo[(hoy - df_maestro_previo['fecha_extraccion']).dt.days <= TTL_DIAS_MAPS].copy()
    else:
        df_gm_vigente = pd.DataFrame(columns=['placeId', 'lat', 'lon', 'nombre', 'rubro', 'resenas', 'fecha_extraccion'])

    huerfanas = []
    for _, row in zp_clean.iterrows():
        if df_gm_vigente.empty:
            huerfanas.append(row)
        else:
            dists = haversine_m(row['lat'], row['lon'], df_gm_vigente['lat'].values, df_gm_vigente['lon'].values)
            if (dists <= 350).sum() < MIN_VECINOS_CUBIERTOS:
                huerfanas.append(row)

    df_huerfanas = pd.DataFrame(huerfanas)
    if not df_huerfanas.empty:
        es_carga_inicial = df_gm_vigente.empty
        n_clusters = min(6 if es_carga_inicial else 3, len(df_huerfanas))
        presupuesto = PRESUPUESTO_INICIAL_MAPS if es_carga_inicial else PRESUPUESTO_SEMANAL_MAPS

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_huerfanas['cluster'] = kmeans.fit_predict(df_huerfanas[['lat', 'lon']])

        nuevos_items = []
        for cid in range(n_clusters):
            grupo = df_huerfanas[df_huerfanas['cluster'] == cid]
            c_lat, c_lon = kmeans.cluster_centers_[cid]
            dists_km = haversine_m(c_lat, c_lon, grupo['lat'].values, grupo['lon'].values) / 1000.0
            radio_km = float(np.clip(np.percentile(dists_km, 90) + 0.4, 1.5, 4.0))

            proporcion = len(grupo) / len(df_huerfanas)
            cupo = max(40, int(presupuesto * proporcion))
            max_rubro = max(4, int(cupo / len(RUBROS_MAPS)))

            run_gm = client.actor("compass/crawler-google-places").call(
                run_input={
                    "customGeolocation": {"type": "Point", "coordinates": [float(c_lon), float(c_lat)], "radiusKm": round(radio_km, 2)},
                    "searchStringsArray": RUBROS_MAPS,
                    "maxCrawledPlacesPerSearch": max_rubro,
                    "language": "es",
                    "deeperCityScrape": False,
                    "scrapePlaceDetailPage": False,
                    "maxReviews": 0,
                    "maxImages": 0,
                    "scrapeContacts": False
                }
            )
            nuevos_items.extend(list(client.dataset(run_gm.default_dataset_id).iterate_items()))

        if nuevos_items:
            df_nuevos_raw = pd.json_normalize(nuevos_items, sep='/')
            if 'permanentlyClosed' in df_nuevos_raw.columns:
                df_nuevos_raw = df_nuevos_raw[df_nuevos_raw['permanentlyClosed'] != True]

            df_nuevos = pd.DataFrame({
                'placeId': df_nuevos_raw['placeId'],
                'lat': pd.to_numeric(df_nuevos_raw['location/lat'], errors='coerce'),
                'lon': pd.to_numeric(df_nuevos_raw['location/lng'], errors='coerce'),
                'nombre': df_nuevos_raw['title'],
                'rubro': df_nuevos_raw.get('categoryName', 'Comercio'),
                'resenas': pd.to_numeric(df_nuevos_raw['reviewsCount'], errors='coerce').fillna(0),
                'fecha_extraccion': hoy
            }).dropna(subset=['placeId', 'lat', 'lon'])

            df_gm_vigente = pd.concat([df_gm_vigente, df_nuevos], ignore_index=True)
            df_gm_vigente = df_gm_vigente.drop_duplicates(subset=['placeId'], keep='last').tail(2500)

    # 3. CRUCE GEOESPACIAL HAVERSINE
    Actor.log.info("[3/4] Calculando métricas de densidad y tráfico comercial...")
    metricas = [calcular_entorno_puntual(r['lat'], r['lon'], df_gm_vigente) for _, r in zp_clean.iterrows()]
    df_metricas = pd.DataFrame(metricas, index=zp_clean.index)
    df_dataset_final = pd.concat([zp_clean, df_metricas], axis=1)

    # 4. GENERACIÓN DE TABLAS EDA PARA LOOKER STUDIO (BLOQUE APIFY EDA)
    Actor.log.info("[4/4] Generando tablas agregadas EDA para Looker Studio...")
    csv1_barrios = (
        df_dataset_final.groupby(['barrio', 'tipo_inmueble'], as_index=False)
        .agg(
            cantidad_inmuebles=('posting_id', 'count'),
            alquiler_promedio_usd=('precio_usd', 'mean'),
            alquiler_mediana_usd=('precio_usd', 'median'),
            alquiler_promedio_ars=('precio_ars', 'mean'),
            precio_m2_promedio_usd=('precio_m2_usd', 'mean'),
            superficie_promedio_m2=('superficie_m2', 'mean'),
            densidad_comercial_200m_prom=('densidad_comercial_200m', 'mean'),
            trafico_resenas_200m_prom=('trafico_resenas_200m', 'mean')
        )
        .sort_values(by='cantidad_inmuebles', ascending=False)
        .round(2)
    )

    csv2_mapa = df_dataset_final[[
        'posting_id', 'barrio', 'ciudad', 'tipo_inmueble',
        'superficie_m2', 'precio_usd', 'precio_ars', 'precio_m2_usd',
        'densidad_comercial_200m', 'trafico_resenas_200m',
        'densidad_comercial_350m', 'trafico_resenas_350m', 'resenas_por_comercio_350m'
    ]].copy()
    csv2_mapa['coordenadas_geo'] = df_dataset_final['lat'].astype(str) + "," + df_dataset_final['lon'].astype(str)
    csv2_mapa['entorno_comercial'] = pd.cut(
        csv2_mapa['densidad_comercial_200m'],
        bins=[-1, 4, 15, 30, 999],
        labels=['1. Bajo tránsito (<5)', '2. Tránsito medio (5-15)', '3. Corredor comercial (16-30)', '4. Polo Prime (>30)']
    )

    df_temp = df_dataset_final.copy()
    df_temp['rango_superficie'] = pd.cut(
        df_temp['superficie_m2'],
        bins=[0, 50, 120, 300, 2500],
        labels=['1. Chico (10-50 m²)', '2. Mediano (51-120 m²)', '3. Grande (121-300 m²)', '4. Gran Superficie (>300 m²)']
    )
    df_temp['entorno_comercial'] = csv2_mapa['entorno_comercial']

    csv3_segmentos = (
        df_temp.groupby(['tipo_inmueble', 'entorno_comercial', 'rango_superficie'], observed=True, as_index=False)
        .agg(
            cantidad=('posting_id', 'count'),
            precio_promedio_usd=('precio_usd', 'mean'),
            precio_m2_usd=('precio_m2_usd', 'mean'),
            resenas_promedio_200m=('trafico_resenas_200m', 'mean')
        )
        .round(2)
    )

    return df_dataset_final, df_gm_vigente, csv1_barrios, csv2_mapa, csv3_segmentos

# ==============================================================================
# ENTRYPOINT PRINCIPAL DEL ACTOR DE APIFY
# ==============================================================================
async def main():
    async with Actor:
        actor_input = await Actor.get_input() or {}
        operacion = actor_input.get("operation", "analytics")
        usar_ultimo_run_zp = bool(actor_input.get("usar_ultimo_run_zp", False))

        # Token de la propia cuenta donde corre el Actor (Apify lo inyecta automáticamente)
        apify_token = actor_input.get("apify_token") or Actor.config.token
        store = await Actor.open_key_value_store(name=KV_STORE_NAME)

        # ----------------------------------------------------------------------
        # RAMA 1: ACTUALIZACIÓN SEMANAL (ETL + CACHÉ MAPS + EDA)
        # ----------------------------------------------------------------------
        if operacion == "analytics":
            Actor.log.info("Ejecutando flujo semanal de Inmobify (ETL + EDA)...")
            csv_anterior = await store.get_value("maestro_comercios_rosario.csv")
            df_maestro = pd.read_csv(io.StringIO(csv_anterior)) if csv_anterior else pd.DataFrame()

            df_limpio, df_maestro_act, csv1, csv2, csv3 = ejecutar_etl_y_eda(
                apify_token=apify_token,
                df_maestro_previo=df_maestro,
                usar_ultimo_run_zp=usar_ultimo_run_zp
            )

            # Guardar todos los artefactos en el Key-Value Store para Make y para Inference
            await store.set_value("maestro_comercios_rosario.csv", df_maestro_act.to_csv(index=False))
            await store.set_value("dataset_inmobiliario_rosario.csv", df_limpio.to_csv(index=False))
            await store.set_value("eda_csv1_resumen_barrios.csv", csv1.to_csv(index=False))
            await store.set_value("eda_csv2_mapa_propiedades.csv", csv2.to_csv(index=False))
            await store.set_value("eda_csv3_segmentos_densidad.csv", csv3.to_csv(index=False))

            await Actor.push_data({
                "status": "SUCCESS",
                "operation": "analytics",
                "propiedades_limpias": len(df_limpio),
                "comercios_maestro": len(df_maestro_act),
                "barrios_analizados": len(csv1)
            })

        # ----------------------------------------------------------------------
        # RAMA 2: TASACIÓN ON-DEMAND (USA MODELO .PKL SI EXISTE O BASELINE)
        # ----------------------------------------------------------------------
        elif operacion == "inference":
            Actor.log.info("Ejecutando tasación puntual (inference)...")
            csv_comercios = await store.get_value("maestro_comercios_rosario.csv")
            csv_dataset = await store.get_value("dataset_inmobiliario_rosario.csv")
            modelo_bytes = await store.get_value("modelo_inmobify.pkl")

            if not csv_comercios or not csv_dataset:
                raise ValueError("Faltan datos base en 'inmobify-storage'. Ejecutá primero {'operation': 'analytics'}.")

            df_comercios = pd.read_csv(io.StringIO(csv_comercios))
            df_dataset = pd.read_csv(io.StringIO(csv_dataset))

            lat = float(actor_input["lat"])
            lon = float(actor_input["lon"])
            sup = float(actor_input["superficie_m2"])
            tipo = actor_input.get("tipo_inmueble", "Locales Comerciales")
            barrio = actor_input.get("barrio", "Centro")
            precio_pub = actor_input.get("precio_publicado_usd")

            entorno = calcular_entorno_puntual(lat, lon, df_comercios)

            if modelo_bytes:
                modelo = joblib.load(io.BytesIO(modelo_bytes))
                X_pred = pd.DataFrame([{
                    "superficie_m2": sup,
                    "tipo_inmueble": tipo,
                    "barrio": barrio,
                    **entorno
                }])
                precio_estimado = float(modelo.predict(X_pred)[0])
                metodo = "ML_MODEL"
            else:
                # Fallback estadístico hasta subir el modelo entrenado
                subset = df_dataset[(df_dataset["tipo_inmueble"] == tipo) & (df_dataset["barrio"] == barrio)]
                if subset.empty:
                    subset = df_dataset[df_dataset["tipo_inmueble"] == tipo]
                precio_m2_ref = float(subset["precio_m2_usd"].median())
                precio_estimado = round(precio_m2_ref * sup, 2)
                metodo = "MEDIANA_BARRIO_BASELINE"

            resultado = {
                "status": "SUCCESS",
                "operation": "inference",
                "metodo": metodo,
                "barrio": barrio,
                "tipo_inmueble": tipo,
                "superficie_m2": sup,
                "precio_estimado_usd": round(precio_estimado, 2),
                "precio_m2_estimado_usd": round(precio_estimado / sup, 2),
                **entorno
            }

            if precio_pub is not None:
                precio_pub = float(precio_pub)
                diff_pct = round(((precio_pub - precio_estimado) / precio_estimado) * 100, 1)
                resultado["precio_publicado_usd"] = precio_pub
                resultado["diferencia_pct"] = diff_pct
                resultado["diagnostico"] = (
                    "OPORTUNIDAD" if diff_pct <= -10 else ("SOBREVALUADO" if diff_pct >= 12 else "EN PRECIO")
                )

            await Actor.push_data(resultado)

        else:
            raise ValueError(f"Operación inválida: {operacion}. Usá 'analytics' o 'inference'.")