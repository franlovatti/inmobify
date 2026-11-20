import numpy as np
import pandas as pd

RADIO_TIERRA_M = 6_371_000.0

POLOS_ROSARIO = {
    "monumento": (-32.9477, -60.6305),
    "pichincha": (-32.9367, -60.6547),
    "pellegrini": (-32.9553, -60.6436),
    "alto_rosario": (-32.9272, -60.6683),
    "portal_rosario": (-32.9125, -60.6789),
}


def _haversine_matriz(lats_p: np.ndarray, lons_p: np.ndarray, lats_c: np.ndarray, lons_c: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lats_p)[:, None]
    lon1 = np.radians(lons_p)[:, None]
    lat2 = np.radians(lats_c)[None, :]
    lon2 = np.radians(lons_c)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * RADIO_TIERRA_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def construir_features_inferencia(df_props: pd.DataFrame, df_comercios: pd.DataFrame | None) -> pd.DataFrame:
    df = df_props.copy()

    # Limpieza básica de tipos
    df["superficie_m2"] = pd.to_numeric(df["superficie_m2"], errors="coerce").fillna(50.0).clip(lower=5.0)
    df["tiene_expensas"] = pd.to_numeric(df.get("tiene_expensas", 0), errors="coerce").fillna(0).astype(int)
    df["barrio"] = df.get("barrio", "Centro").fillna("Centro").astype(str).str.strip()
    df["tipo_inmueble"] = df.get("tipo_inmueble", "Local comercial").fillna("Local comercial").astype(str).str.strip()

    # Features lineales derivadas
    df["log_superficie_m2"] = np.log1p(df["superficie_m2"])
    df["es_gran_superficie"] = (df["superficie_m2"] > 200.0).astype(int)

    # Coordenadas (fallback al microcentro de Rosario si no se envían)
    lats_p = pd.to_numeric(df.get("latitud", -32.9468), errors="coerce").fillna(-32.9468).values
    lons_p = pd.to_numeric(df.get("longitud", -60.6393), errors="coerce").fillna(-60.6393).values

    # Distancia al polo comercial más cercano
    dists_polos = []
    for plat, plon in POLOS_ROSARIO.values():
        d = _haversine_matriz(lats_p, lons_p, np.array([plat]), np.array([plon]))[:, 0]
        dists_polos.append(d)
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