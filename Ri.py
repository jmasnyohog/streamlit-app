# -*- coding: utf-8 -*-
"""
Streamlit版（MSM / LFM 切替対応）
Ri（1000–975hPa）マップ
表示領域切替 + アメダスON/OFF
"""

from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io import shapereader
import streamlit as st
from concurrent.futures import ThreadPoolExecutor


# ==================================================
# ページ設定
# ==================================================
st.set_page_config(page_title="MSM/LFM Ri", layout="centered")


# ==================================================
# モデル選択
# ==================================================
st.sidebar.header("モデル選択")

model = st.sidebar.radio(
    "モデル",
    ["MSM", "LFM"],
    horizontal=True
)

st.title(f"{model} Ri（1000–975hPa）")


# ==================================================
# 表示領域
# ==================================================
AREA = {

    "宮城": (37.8, 39.0, 140.3, 141.7),

    "岩手": (38.6, 40.3, 140.5, 142.1),

    "福島": (36.8, 38.3, 139.3, 141.1),

    "山形": (37.7, 39.4, 139.2, 140.7),

    "秋田": (38.7, 40.3, 139.3, 140.6),

    "青森": (40.1, 41.6, 139.3, 141.8),

    "東北全体": (36.5, 41.6, 139.0, 142.8)
}

st.sidebar.header("表示領域")

area_name = st.sidebar.selectbox(
    "領域選択",
    list(AREA.keys()),
    index=0
)


# ==================================================
# アメダスON/OFF
# ==================================================
st.sidebar.header("表示設定")

show_amedas = st.sidebar.checkbox(
    "アメダス表示",
    value=True
)


# ==================================================
# セッションFT
# ==================================================
if "ft" not in st.session_state:
    st.session_state.ft = 0


# ==================================================
# requests Session
# ==================================================
@st.cache_resource
def get_session():
    s = requests.Session()
    s.verify = False
    s.trust_env = True
    return s


session = get_session()


# ==================================================
# Natural Earth
# ==================================================
NE_BASE = r"C:\natural_earth"


@st.cache_resource
def load_shapes():

    land = list(
        shapereader.Reader(
            os.path.join(NE_BASE, "ne_10m_land", "ne_10m_land.shp")
        ).geometries()
    )

    coast = list(
        shapereader.Reader(
            os.path.join(NE_BASE, "ne_10m_coastline", "ne_10m_coastline.shp")
        ).geometries()
    )

    border = list(
        shapereader.Reader(
            os.path.join(
                NE_BASE,
                "ne_10m_admin_0_boundary_lines_land",
                "ne_10m_admin_0_boundary_lines_land.shp"
            )
        ).geometries()
    )

    return land, coast, border


LAND, COAST, BORDER = load_shapes()


# ==================================================
# BASE URL
# ==================================================
def get_base_url(model):

    if model == "MSM":
        return (
            "http://diag.comdev.naps.kishou.go.jp:8080/"
            "NUSDAS/nwp/Msm2/fcst_p_ll.nus/_MSMLLPP.FCSV.ADS2/"
        )

    else:
        return (
            "http://diag.comdev.naps.kishou.go.jp:8080/"
            "NUSDAS/nwp/Lfm3/fcst_p_ll.nus/_LFMLLPP.FCSV.ADS2/"
        )


BASE_URL = get_base_url(model)


# ==================================================
# 最新初期値
# ==================================================
def latest_init_utc(now, model):

    if model == "MSM":
        hours = [0,3,6,9,12,15,18,21]
    else:
        hours = list(range(24))

    for h in reversed(hours):
        if now.hour >= h:
            return now.replace(hour=h, minute=0, second=0, microsecond=0)

    return (now - timedelta(days=1)).replace(
        hour=hours[-1], minute=0, second=0, microsecond=0
    )


# ==================================================
# MSM格子
# ==================================================
LON0, LAT0 = 116.0, 50.5
DLON, DLAT = 0.0625, 0.05


def lonlat_to_index(lon, lat):

    ix = int(round((lon - LON0) / DLON))
    iy = int(round((LAT0 - lat) / DLAT))

    return ix, iy


# ==================================================
# URL生成
# ==================================================
def make_url(base, init, valid, level, elem, grid):

    return (
        base
        + f"{init:%Y-%m-%dt%H}00/none/{valid:%Y-%m-%dt%H}00/min1/"
        + f"{level}/{level}/{elem}{grid}data.txt"
    )


# ==================================================
# 並列取得
# ==================================================
def fetch_one(args):

    init, valid, level, elem, grid = args

    url = make_url(BASE_URL, init, valid, level, elem, grid)

    try:

        r = session.get(url, timeout=(10, 20))

        if r.status_code != 200:
            return None

        rows = []

        for line in BeautifulSoup(r.text, "html.parser").get_text().splitlines():

            try:
                rows.append([float(x) for x in line.split()])
            except:
                pass

        return np.array(rows)

    except:
        return None


def fetch_parallel(init, valid, grid):

    tasks = []

    for lev in [1000, 975]:
        for elem in ["U", "V", "T", "Z"]:
            tasks.append((init, valid, lev, elem, grid))

    with ThreadPoolExecutor(max_workers=4) as exe:
        results = list(exe.map(fetch_one, tasks))

    return results


# ==================================================
# Ri計算
# ==================================================
def calc_ri(U1000, V1000, T1000, Z1000,
            U975, V975, T975, Z975):

    theta975 = T975 * (1000 / 975) ** 0.286

    dtheta_dz = (theta975 - T1000) / (Z975 - Z1000)

    shear = np.sqrt((U975 - U1000) ** 2 + (V975 - V1000) ** 2) / (Z975 - Z1000)

    Ri = (9.8 / ((T1000 + theta975) / 2)) * (dtheta_dz / shear ** 2)

    return np.where(Ri <= 0.25, Ri, np.nan)


# ==================================================
# UI
# ==================================================
now_utc = datetime.utcnow()
default_init = latest_init_utc(now_utc, model)

st.sidebar.header("設定")

init_date = st.sidebar.date_input("初期値（日付）", default_init.date())

if model == "MSM":
    init_hours = [0,3,6,9,12,15,18,21]
else:
    init_hours = list(range(24))   # ← 0～23


init_hour = st.sidebar.selectbox(
    "初期値（UTC）",
    init_hours,
    index=init_hours.index(default_init.hour) 
          if default_init.hour in init_hours else 0
)

def shift_ft(d):
    st.session_state.ft = int(np.clip(st.session_state.ft + d, 0, 78))

st.sidebar.markdown("### FT操作")

c1,c2,c3,c4,c5,c6,c7,c8 = st.sidebar.columns(8)
c1.button("≪12", on_click=shift_ft, args=(-12,))
c2.button("≪6", on_click=shift_ft, args=(-6,))
c3.button("≪3", on_click=shift_ft, args=(-3,))
c4.button("≪1", on_click=shift_ft, args=(-1,))
c5.button("1≫", on_click=shift_ft, args=(1,))
c6.button("3≫", on_click=shift_ft, args=(3,))
c7.button("6≫", on_click=shift_ft, args=(6,))
c8.button("12≫", on_click=shift_ft, args=(12,))

ft_max = 78 if model=="MSM" else 18
st.sidebar.slider("FT直接指定", 0, ft_max, key="ft")

ft = st.session_state.ft
# ==================================================
# 時刻
# ==================================================
init = datetime(init_date.year, init_date.month, init_date.day, init_hour)

valid = init + timedelta(hours=ft)

valid_jst = valid + timedelta(hours=9)

st.markdown(
    f"**MODEL** {model}  |  "
    f"**INIT** {init:%Y-%m-%d %H} UTC  |  "
    f"**FT** {ft:02d}  |  "
    f"**JST** {valid_jst:%Y-%m-%d %H}"
)


# ==================================================
# 領域
# ==================================================
lat_min, lat_max, lon_min, lon_max = AREA[area_name]


ix1, iy1 = lonlat_to_index(lon_min, lat_max)

ix2, iy2 = lonlat_to_index(lon_max, lat_min)

grid = f"/{iy1+1},{iy2+1}/{ix1+1},{ix2+1}/"


lons = np.linspace(lon_min, lon_max, ix2 - ix1 + 1)

lats = np.linspace(lat_max, lat_min, iy2 - iy1 + 1)


# ==================================================
# データ取得
# ==================================================
results = fetch_parallel(init, valid, grid)

if any(r is None for r in results):

    st.error("データ取得失敗")

    st.stop()


U1000, V1000, T1000, Z1000, U975, V975, T975, Z975 = results


Ri_plot = calc_ri(
    U1000, V1000, T1000, Z1000,
    U975, V975, T975, Z975
)


# ==================================================
# 作図
# ==================================================
fig = plt.figure(figsize=(4.2, 4.2), dpi=100)

ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([lon_min, lon_max, lat_min, lat_max])


pcm = ax.pcolormesh(
    lons,
    lats,
    Ri_plot,
    cmap="Reds_r",
    vmin=0,
    vmax=0.25,
    shading="auto",
    zorder=1
)


ax.add_geometries(
    COAST,
    ccrs.PlateCarree(),
    facecolor="none",
    edgecolor="black",
    linewidth=0.7,
    zorder=3
)


# ==================================================
# アメダス
# ==================================================
amedas = {
#miyagi
    "Yoneyama": (141.17, 38.62),
    "Furukawa": (140.91, 38.57),
    "Onagawa": (141.45, 38.45),
    "Shiroishi": (140.62, 38.00),
    "Watari": (140.86, 38.03),
    "Ishinomaki": (141.30, 38.43),
    "Sendai": (140.90, 38.26),
#fukushima
    "fukushima":(140.4708, 37.7594),
    "koriyama":(140.3303, 37.3697),
    "shirakawa":(140.2161, 37.1317),
    "soma":(140.9258, 37.7847),
    "onahama":(140.9033, 36.9475),
    "wakamatsu":(139.9106, 37.4883),
#yamagata
    "yamagata":(140.3461, 38.2556),
    "sakata":(139.8433, 38.9086),
    "shinzyo":(140.3122, 38.7596),
    "yonezawa":(140.1439, 37.9128),
#akita
    "akita":(140.0994, 39.7172),
    "noshiro":(140.0332, 40.1983),
    "nikaho":(139.9144, 39.2553),
    "yuwa":(140.2186, 39.6156),
    "yuzawa":(140.4633, 39.1872),
#iwate
    "morioka":(141.1658, 39.6986),
    "shiwa":(141.1272, 39.5478),
    "wakayanagi":(141.0642, 39.1403),
    "kuzi":(141.7486, 40.1686),
    "oohunato":(141.7147, 39.0647),
#aomori
    "aomori":(140.7689, 40.8219),
    "fukaura":(139.9331, 40.6464),
    "mutsu":(141.2114, 41.2833),
    "hachinohe":(141.5219, 40.5275),
}


if show_amedas:

    for name, (lon, lat) in amedas.items():

        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:

            ax.plot(lon, lat, "ko", markersize=4, zorder=10)

            ax.text(lon + 0.02, lat, name, fontsize=7)


plt.colorbar(pcm, ax=ax, shrink=0.75)

ax.set_title(
    f"{model} Ri ≤ 0.25 ({area_name}) | {valid_jst:%Y-%m-%d %H} JST"
)

plt.tight_layout()

st.pyplot(fig, use_container_width=False)

plt.close(fig)