import time
import unicodedata
import numpy as np
import pandas as pd
import requests

RADIO_TIERRA_M = 6_371_000.0

POLOS_ROSARIO = {
    "monumento": (-32.9477, -60.6305),
    "pichincha": (-32.9367, -60.6547),
    "pellegrini": (-32.9553, -60.6436),
    "alto_rosario": (-32.9272, -60.6683),
    "portal_rosario": (-32.9125, -60.6789),
}

CENTROIDES_BARRIOS = {
    "Centro": (-32.9468, -60.6393),
    "Pichincha": (-32.9367, -60.6547),
    "Pellegrini": (-32.9553, -60.6436),
    "Martin": (-32.9512, -60.6295),
    "Abasto": (-32.9615, -60.6408),
    "Echesortu": (-32.9442, -60.6745),
    "Lourdes": (-32.9478, -60.6585),
    "Alberdi": (-32.9055, -60.6865),
    "Arroyito": (-32.9175, -60.6812),
    "Republica de la Sexta": (-32.9635, -60.6265),
    "Fisherton": (-32.9278, -60.7315),
    "Refineria": (-32.9265, -60.6635),
    "Cinco Esquinas": (-32.9545, -60.6765),
    "Bella Vista": (-32.9620, -60.6710),
    "Remedios de Escalada": (-32.9375, -60.6820),
    "Agote": (-32.9385, -60.6665),
    "Triangulo": (-32.9715, -60.6850),
    "Parque": (-32.9640, -60.6550),
    "Hospitales": (-32.9685, -60.6435),
    "Tablada": (-32.9745, -60.6285),
    "La Guardia": (-32.9810, -60.6350),
    "Tiro Suizo": (-32.9915, -60.6485),
    "Azcuenaga": (-32.9440, -60.6945),
    "Belgrano": (-32.9380, -60.7085),
}


def _normalizar_texto(texto: str) -> str:
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return s.strip().lower()


_MAPA_BARRIOS_NORM = {_normalizar_texto(k): (k, v) for k, v in CENTROIDES_BARRIOS.items()}


def _haversine_matriz(lats_p: np.ndarray, lons_p: np.ndarray, lats_c: np.ndarray, lons_c: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lats_p)[:, None]
    lon1 = np.radians(lons_p)[:, None]
    lat2 = np.radians(lats_c)[None, :]
    lon2 = np.radians(lons_c)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * RADIO_TIERRA_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _geocodificar_direccion_rosario(direccion: str, google_api_key: str | None = None) -> tuple[float, float] | None:
    query = f"{direccion.strip()}, Rosario, Santa Fe, Argentina"

    # 1. Google Maps Geocoding API (si se proveyó API Key)
    if google_api_key:
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": query, "key": google_api_key, "region": "ar"},
                timeout=6,
            )
            data = resp.json()
            if data.get("status") == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return float(loc["lat"]), float(loc["lng"])
        except Exception:
            pass

    # 2. Fallback gratuito: OpenStreetMap Nominatim
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "countrycodes": "ar",
                "viewbox": "-60.78,-32.85,-60.58,-33.03",
                "bounded": 1,
            },
            headers={"User-Agent": "InmobifyRosarioCapstone/1.0"},
            timeout=6,
        )
        items = resp.json()
        if items:
            return float(items[0]["lat"]), float(items[0]["lon"])
    except Exception:
        pass

    return None


def _barrio_mas_cercano(lat: float, lon: float) -> str:
    nombres = list(CENTROIDES_BARRIOS.keys())
    coords = np.array(list(CENTROIDES_BARRIOS.values()))
    dists = _haversine_matriz(np.array([lat]), np.array([lon]), coords[:, 0], coords[:, 1])[0]
    return nombres[int(np.argmin(dists))]


def construir_features_inferencia(
    df_props: pd.DataFrame,
    df_comercios: pd.DataFrame | None,
    google_api_key: str | None = None,
) -> pd.DataFrame:
    df = df_props.copy()

    df["superficie_m2"] = pd.to_numeric(df["superficie_m2"], errors="coerce").fillna(50.0).clip(lower=5.0)
    df["tiene_expensas"] = pd.to_numeric(df.get("tiene_expensas", 0), errors="coerce").fillna(0).astype(int)
    df["tipo_inmueble"] = df.get("tipo_inmueble", "Local comercial").fillna("Local comercial").astype(str).str.strip()

    df["log_superficie_m2"] = np.log1p(df["superficie_m2"])
    df["es_gran_superficie"] = (df["superficie_m2"] > 200.0).astype(int)

    lats_resueltas, lons_resueltas, barrios_resueltos, fuentes_geo = [], [], [], []
    cache_geo: dict[str, tuple[float, float] | None] = {}

    for _, row in df.iterrows():
        raw_dir = str(row.get("direccion", "") or "").strip()
        raw_barrio = str(row.get("barrio", "") or "").strip()
        lat_val = pd.to_numeric(row.get("latitud"), errors="coerce")
        lon_val = pd.to_numeric(row.get("longitud"), errors="coerce")

        lat_ok = pd.notna(lat_val) and -33.15 <= float(lat_val) <= -32.80
        lon_ok = pd.notna(lon_val) and -60.85 <= float(lon_val) <= -60.55

        # Prioridad 1: Coordenadas explícitas válidas
        if lat_ok and lon_ok:
            lat_f, lon_f = float(lat_val), float(lon_val)
            fuente = "Coordenadas exactas"
        # Prioridad 2: Geocodificación por dirección
        elif raw_dir and raw_dir.lower() not in ("nan", "none", ""):
            if raw_dir not in cache_geo:
                cache_geo[raw_dir] = _geocodificar_direccion_rosario(raw_dir, google_api_key)
                if not google_api_key:
                    time.sleep(0.35)  # Respetar rate-limit de Nominatim en lotes
            coords_dir = cache_geo[raw_dir]
            if coords_dir:
                lat_f, lon_f = coords_dir
                fuente = "Geocodificada por dirección"
            else:
                # Si la dirección no matcheó, cae al centroide del barrio
                clave_b = _normalizar_texto(raw_barrio)
                if clave_b in _MAPA_BARRIOS_NORM:
                    _, (lat_f, lon_f) = _MAPA_BARRIOS_NORM[clave_b]
                    fuente = "Centroide de barrio (fallback dirección)"
                else:
                    lat_f, lon_f = -32.9468, -60.6393
                    fuente = "Centroide Rosario (default)"
        # Prioridad 3: Centroide del barrio informado
        elif raw_barrio and _normalizar_texto(raw_barrio) in _MAPA_BARRIOS_NORM:
            _, (lat_f, lon_f) = _MAPA_BARRIOS_NORM[_normalizar_texto(raw_barrio)]
            fuente = "Centroide de barrio"
        else:
            lat_f, lon_f = -32.9468, -60.6393
            fuente = "Centroide Rosario (default)"

        # Resolver barrio si venía vacío o normalizar capitalización
        if raw_barrio and raw_barrio.lower() not in ("nan", "none", ""):
            clave_b = _normalizar_texto(raw_barrio)
            barrio_f = _MAPA_BARRIOS_NORM[clave_b][0] if clave_b in _MAPA_BARRIOS_NORM else raw_barrio
        else:
            barrio_f = _barrio_mas_cercano(lat_f, lon_f)

        lats_resueltas.append(lat_f)
        lons_resueltas.append(lon_f)
        barrios_resueltos.append(barrio_f)
        fuentes_geo.append(fuente)

    df["latitud_resuelta"] = np.round(lats_resueltas, 6)
    df["longitud_resuelta"] = np.round(lons_resueltas, 6)
    df["barrio"] = barrios_resueltos
    df["origen_coordenadas"] = fuentes_geo

    lats_p = np.array(lats_resueltas)
    lons_p = np.array(lons_resueltas)

    # Distancia al polo comercial más cercano
    dists_polos = [
        _haversine_matriz(lats_p, lons_p, np.array([plat]), np.array([plon]))[:, 0]
        for plat, plon in POLOS_ROSARIO.values()
    ]
    df["dist_polo_mas_cercano_m"] = np.min(np.column_stack(dists_polos), axis=1)

    # Entorno comercial en radio de 200m
    if df_comercios is not None and not df_comercios.empty:
        col_lat_c = next((c for c in ["latitud", "latitude", "lat"] if c in df_comercios.columns), None)
        col_lon_c = next((c for c in ["longitud", "longitude", "lon", "lng"] if c in df_comercios.columns), None)
        col_res = next((c for c in ["total_resenas", "resenas", "reviews_count", "user_ratings_total"] if c in df_comercios.columns), None)
        col_rubro = next((c for c in ["rubro", "categoria", "category", "tipo"] if c in df_comercios.columns), None)

        if col_lat_c and col_lon_c:
            c_valid = df_comercios.dropna(subset=[col_lat_c, col_lon_c]).copy()
            lats_c = pd.to_numeric(c_valid[col_lat_c], errors="coerce").values
            lons_c = pd.to_numeric(c_valid[col_lon_c], errors="coerce").values
            resenas_c = pd.to_numeric(c_valid[col_res], errors="coerce").fillna(0.0).values if col_res else np.zeros(len(c_valid))
            rubros_c = c_valid[col_rubro].fillna("Otro").astype(str).values if col_rubro else np.array(["Otro"] * len(c_valid))

            dist_mat = _haversine_matriz(lats_p, lons_p, lats_c, lons_c)
            mask_200 = dist_mat <= 200.0

            densidad_200, trafico_200, diversidad_200 = [], [], []
            for i in range(len(df)):
                idx_c = np.where(mask_200[i])[0]
                densidad_200.append(int(len(idx_c)))
                trafico_200.append(float(np.sum(resenas_c[idx_c])) if len(idx_c) > 0 else 0.0)
                diversidad_200.append(int(len(set(rubros_c[idx_c]))) if len(idx_c) > 0 else 0)

            df["densidad_comercial_200m"] = densidad_200
            df["log_trafico_resenas_200m"] = np.log1p(trafico_200)
            df["diversidad_rubros_200m"] = diversidad_200
        else:
            df["densidad_comercial_200m"] = 12
            df["log_trafico_resenas_200m"] = np.log1p(1500.0)
            df["diversidad_rubros_200m"] = 8
    else:
        df["densidad_comercial_200m"] = 12
        df["log_trafico_resenas_200m"] = np.log1p(1500.0)
        df["diversidad_rubros_200m"] = 8

    return df