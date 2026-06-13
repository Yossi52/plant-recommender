"""
app.py — Streamlit 식물 추천 대시보드
════════════════════════════════════════════════════════════════════
GAT(+cosine decoder) 기반 이분 그래프 링크 예측 모델을 백엔드로,
사용자가 환경 조건(광도/장소/온도/습도/물주기/관리요구도/독성/꽃)을
선택하면 적합한 식물을 추천한다.

실행 방법
─────────────────────────────────────────────────────────────────
    streamlit run app.py
════════════════════════════════════════════════════════════════════
"""

import os

# torch / numpy / matplotlib 가 각자 OpenMP 런타임을 들고 와서 충돌하는
# Windows 환경 문제(OMP Error #15) 방지. torch import 전에 설정해야 함.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")

# ══════════════════════════════════════════════════════════
# 0. 페이지 설정 & 그린/식물 테마
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="식물 추천 대시보드",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root, html, body {
        color-scheme: light !important;
    }
    :root {
        --leaf:        #4f7942;
        --leaf-dark:   #355e3b;
        --leaf-light:  #e8f0e3;
        --soil:        #6b4423;
        --cream:       #f6f8f1;
        --text:        #2d2d2d;
    }
    .stApp {
        background-color: var(--cream) !important;
        color: var(--text) !important;
    }
    /* 시스템/브라우저 다크모드에서도 본문 텍스트가 보이도록 강제 */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stVerticalBlock"],
    .main, .block-container,
    p, span, label, div {
        color: var(--text);
    }
    h1, h2, h3, h4 {
        color: var(--leaf-dark) !important;
    }
    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background-color: var(--leaf-light) !important;
        border-right: 1px solid #cdded2;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    /* 입력 위젯(라디오/셀렉트/멀티셀렉트 등) 배경/텍스트 */
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"],
    ul[data-baseweb="menu"],
    li[role="option"] {
        background-color: #ffffff !important;
        color: var(--text) !important;
    }
    /* 버튼 */
    div.stButton > button {
        background-color: var(--leaf);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: var(--leaf-dark);
        color: white;
    }
    /* 카드 느낌 컨테이너 */
    .plant-card {
        background-color: white;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #d8e6d2;
        box-shadow: 0 1px 3px rgba(53, 94, 59, 0.08);
        margin-bottom: 0.8rem;
    }
    .rank-badge {
        display: inline-block;
        background-color: var(--leaf);
        color: white;
        border-radius: 999px;
        width: 28px; height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        margin-right: 8px;
    }
    .score-pill {
        background-color: var(--leaf-light);
        color: var(--leaf-dark);
        border-radius: 999px;
        padding: 2px 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .tag {
        display: inline-block;
        background-color: #eef3ea;
        color: #4a5d44;
        border-radius: 999px;
        padding: 1px 10px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    .tag-warn {
        background-color: #fbe9e7;
        color: #b3441e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════
# 1. 모델 / 데이터 로드 (캐시)
# ══════════════════════════════════════════════════════════
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Korean → English (시각화용, 한글 폰트 깨짐 방지)
CAT_EN = {
    "광도": "Light", "장소": "Location", "생육온도": "Grow Temp",
    "겨울최저온도": "Winter Min Temp", "습도": "Humidity",
    "물주기": "Watering", "관리요구도": "Mgmt Level",
    "독성": "Toxicity", "꽃": "Flowering",
}


@st.cache_resource(show_spinner="GAT 모델을 불러오는 중입니다...")
def load_resources():
    model_mod = _load_module("model_def", BASE_DIR / "model.py")
    build_model = model_mod.build_model

    with open(BASE_DIR / "graph_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    n_plants = meta["n_plants"]
    num_nodes = meta["n_total"]
    plant_names = meta["plant_names"]
    cond_to_gidx = meta["cond_name_to_global_idx"]
    cond_nodes = meta["condition_nodes"]

    df_plant = pd.read_csv(BASE_DIR / "data" / "plant_data.csv")

    model = build_model("gat", num_nodes=num_nodes, decoder="cosine").to(DEVICE)
    model.load_state_dict(
        torch.load(BASE_DIR / "model_weights.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()

    train_edge_index = torch.load(
        BASE_DIR / "train_split.pt", weights_only=False
    )["train_edge_index"].to(DEVICE)

    with torch.no_grad():
        z = model.encode(train_edge_index)

    z_plant = z[:n_plants]
    z_cond = z[n_plants:]

    return dict(
        model=model,
        plant_names=plant_names,
        cond_to_gidx=cond_to_gidx,
        cond_nodes=cond_nodes,
        df_plant=df_plant,
        n_plants=n_plants,
        z_plant=z_plant,
        z_cond=z_cond,
    )


res = load_resources()


# ══════════════════════════════════════════════════════════
# 2. 추천 함수 (14_recommend.py 로직과 동일)
# ══════════════════════════════════════════════════════════
VERIFY_COLS = [
    "독성_여부(1=있음)", "관리요구도",
    "광도_낮음", "광도_중간", "광도_높음",
    "장소_발코니 창측", "장소_발코니 내측", "장소_거실 창측", "장소_거실 내측", "장소_실내 어두운 곳",
    "꽃피는계절_봄", "꽃피는계절_여름", "꽃피는계절_가을", "꽃피는계절_겨울",
]


def recommend(include_conditions, exclude_columns=None, top_k=10):
    model = res["model"]
    z_plant = res["z_plant"]
    z_cond = res["z_cond"]
    n_plants = res["n_plants"]
    cond_to_gidx = res["cond_to_gidx"]
    df_plant = res["df_plant"]
    plant_names = res["plant_names"]

    condition_contrib = {}

    with torch.no_grad():
        if include_conditions:
            local_idxs = [cond_to_gidx[c] - n_plants for c in include_conditions]
            selected = z_cond[local_idxs]
            scores_matrix = model.score(z_plant, selected)  # [216, K]
            scores_raw = scores_matrix.mean(dim=1)

            # 조건별 영향도(이번 추천에 한정): 각 조건의 식물별 점수와
            # 최종(평균) 점수 간 상관관계. 높을수록 이번 순위에 해당 조건이
            # 더 크게 기여했음을 의미. 매 요청마다 새로 계산됨.
            final_np = scores_raw.cpu().numpy()
            for i, cname in enumerate(include_conditions):
                col_np = scores_matrix[:, i].cpu().numpy()
                if col_np.std() == 0 or final_np.std() == 0:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(col_np, final_np)[0, 1])
                condition_contrib[cname] = corr
        else:
            scores_matrix = model.score(z_plant, z_cond)
            scores_raw = scores_matrix.mean(dim=1)

        scores = torch.sigmoid(scores_raw)

    result_df = pd.DataFrame({"식물명": plant_names, "추천점수": scores.cpu().numpy()})
    result_df = result_df.join(df_plant[VERIFY_COLS])

    if exclude_columns:
        for col in exclude_columns:
            result_df = result_df[result_df[col] != 1]

    result_df = (
        result_df.sort_values("추천점수", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    result_df.index += 1
    return result_df, condition_contrib


def light_label(row):
    for key, label in [("광도_낮음", "낮음"), ("광도_중간", "중간"), ("광도_높음", "높음")]:
        if row.get(key) == 1:
            return label
    return "-"


def place_label(row):
    mapping = {
        "장소_발코니 창측": "발코니 창가",
        "장소_발코니 내측": "발코니 안쪽",
        "장소_거실 창측": "거실 창가",
        "장소_거실 내측": "거실 안쪽",
        "장소_실내 어두운 곳": "어두운 실내",
    }
    labels = [v for k, v in mapping.items() if row.get(k) == 1]
    return ", ".join(labels) if labels else "-"


def flower_label(row):
    mapping = {
        "꽃피는계절_봄": "봄",
        "꽃피는계절_여름": "여름",
        "꽃피는계절_가을": "가을",
        "꽃피는계절_겨울": "겨울",
    }
    labels = [v for k, v in mapping.items() if row.get(k) == 1]
    return ", ".join(labels) if labels else "없음"


# ══════════════════════════════════════════════════════════
# 3. 헤더
# ══════════════════════════════════════════════════════════
st.title("🌿 식물 추천 대시보드")
st.caption(
    "GAT(Graph Attention Network) 기반 식물–환경조건 이분 그래프 링크 예측 모델로, "
    "원하는 환경 조건만 골라도 적합한 식물을 추천해줍니다. 조건은 일부만 선택해도 됩니다."
)
st.divider()


# ══════════════════════════════════════════════════════════
# 4. 조건 입력 (상단)
# ══════════════════════════════════════════════════════════
NONE = "선택 안 함"

with st.form("condition_form"):
    st.subheader("🏡 우리 집 환경 입력")
    st.caption("해당하는 항목만 선택하세요. 비워두면 추천에 영향을 주지 않습니다.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**☀️ 광도 / 장소**")
        light = st.radio("광도", [NONE, "낮음 (그늘)", "중간", "높음 (직사광)"], index=0)
        place = st.radio(
            "두는 위치",
            [NONE, "발코니 창가", "발코니 안쪽", "거실 창가", "거실 안쪽", "어두운 실내"],
            index=0,
        )

    with c2:
        st.markdown("**🌡️ 온도 / 습도**")
        grow_temp = st.radio("생육 적정 온도", [NONE, "낮음", "중간", "높음"], index=0)
        winter_temp = st.radio(
            "겨울철 최저 온도",
            [NONE, "0℃ 이하도 가능", "5℃ 이상", "7℃ 이상", "10℃ 이상", "13℃ 이상 (열대성)"],
            index=0,
            help="겨울에 식물이 견뎌야 할 최소 온도 수준입니다.",
        )
        humidity = st.radio("습도", [NONE, "낮음", "중간", "높음"], index=0)

    with c3:
        st.markdown("**🪴 관리 / 추가 조건**")
        mgmt = st.radio(
            "관리 난이도",
            [NONE, "아주 쉬움 (초보)", "쉬움", "보통", "손이 많이 감"],
            index=0,
        )
        no_toxic = st.checkbox("반려동물/아이가 있어요 (독성 식물 제외)")
        allergy_seasons = st.multiselect(
            "꽃가루 알러지 계절 (해당 계절에 꽃 피는 식물 제외)",
            ["봄", "여름", "가을", "겨울"],
        )

    with st.expander("💧 계절별 물주기 빈도 (선택)"):
        watering = {}
        wcols = st.columns(4)
        for wc, (season, max_level) in zip(wcols, [("봄", 4), ("여름", 4), ("가을", 4), ("겨울", 3)]):
            with wc:
                opts = [NONE] + [f"{i} (1=적게 ~ {max_level}=자주)" if i == 1 else str(i) for i in range(1, max_level + 1)]
                watering[season] = st.selectbox(f"{season} 물주기", opts, index=0, key=f"water_{season}")

    bottom_l, bottom_r = st.columns([3, 1])
    with bottom_l:
        top_k = st.slider("추천 개수", 5, 20, 10)
    with bottom_r:
        st.write("")
        submitted = st.form_submit_button("🔍 식물 추천 받기", width="stretch")

st.divider()


# ══════════════════════════════════════════════════════════
# 5. 입력 → 조건 노드 매핑
# ══════════════════════════════════════════════════════════
def build_conditions():
    include = []

    light_map = {"낮음 (그늘)": "광도_낮음", "중간": "광도_중간", "높음 (직사광)": "광도_높음"}
    if light in light_map:
        include.append(light_map[light])

    place_map = {
        "발코니 창가": "장소_발코니 창측",
        "발코니 안쪽": "장소_발코니 내측",
        "거실 창가": "장소_거실 창측",
        "거실 안쪽": "장소_거실 내측",
        "어두운 실내": "장소_실내 어두운 곳",
    }
    if place in place_map:
        include.append(place_map[place])

    grow_temp_map = {"낮음": "생육온도_1", "중간": "생육온도_2", "높음": "생육온도_3"}
    if grow_temp in grow_temp_map:
        include.append(grow_temp_map[grow_temp])

    winter_temp_map = {
        "0℃ 이하도 가능": "겨울최저온도_1",
        "5℃ 이상": "겨울최저온도_2",
        "7℃ 이상": "겨울최저온도_3",
        "10℃ 이상": "겨울최저온도_4",
        "13℃ 이상 (열대성)": "겨울최저온도_5",
    }
    if winter_temp in winter_temp_map:
        include.append(winter_temp_map[winter_temp])

    humidity_map = {"낮음": "습도_1", "중간": "습도_2", "높음": "습도_3"}
    if humidity in humidity_map:
        include.append(humidity_map[humidity])

    mgmt_map = {
        "아주 쉬움 (초보)": "관리요구도_1",
        "쉬움": "관리요구도_2",
        "보통": "관리요구도_3",
        "손이 많이 감": "관리요구도_4",
    }
    if mgmt in mgmt_map:
        include.append(mgmt_map[mgmt])

    for season, val in watering.items():
        if val != NONE:
            level = val.split(" ")[0]
            include.append(f"물주기_{season}_{level}")

    exclude = []
    if no_toxic:
        exclude.append("독성_여부(1=있음)")
    for season in allergy_seasons:
        exclude.append(f"꽃피는계절_{season}")

    # 어떤 카테고리가 선택되었는지 (attention 시각화용)
    selected_categories = set()
    for cname in include:
        for cn in res["cond_nodes"]:
            if cn["name"] == cname:
                selected_categories.add(cn["category"])
    if exclude:
        if "독성_여부(1=있음)" in exclude:
            selected_categories.add("독성")
        if any(e.startswith("꽃피는계절") for e in exclude):
            selected_categories.add("꽃")

    return include, exclude, selected_categories


# ══════════════════════════════════════════════════════════
# 6. 결과 출력
# ══════════════════════════════════════════════════════════
def render_results(include, exclude):
    df, condition_contrib = recommend(include, exclude_columns=exclude, top_k=top_k)

    if df.empty:
        st.warning("조건에 맞는 식물을 찾지 못했습니다. 조건을 조금 완화해보세요.")
        return

    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"🌱 추천 식물 Top {len(df)}")
        for rank, row in df.iterrows():
            tags = [f"<span class='tag'>광도 {light_label(row)}</span>"]
            place_txt = place_label(row)
            if place_txt != "-":
                tags.append(f"<span class='tag'>{place_txt}</span>")
            tags.append(f"<span class='tag'>관리요구도 {row['관리요구도']:.1f}</span>")
            if row["독성_여부(1=있음)"] == 1:
                tags.append("<span class='tag tag-warn'>⚠️ 독성 있음</span>")
            else:
                tags.append("<span class='tag'>무독성</span>")
            tags.append(f"<span class='tag'>🌸 {flower_label(row)}</span>")

            st.markdown(
                f"""
                <div class="plant-card">
                    <span class="rank-badge">{rank}</span>
                    <b style="font-size:1.05rem;">{row['식물명']}</b>
                    <span class="score-pill" style="float:right;">적합도 {row['추천점수']*100:.1f}%</span>
                    <div style="margin-top:8px;">{''.join(tags)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("🔎 이번 추천에 대한 조건별 영향도")
        if not condition_contrib:
            st.info("조건을 선택하면, 어떤 조건이 이번 추천 순위에 더 크게 기여했는지 보여드려요.")
        else:
            name_to_cat = {cn["name"]: cn["category"] for cn in res["cond_nodes"]}

            cat_scores = {}
            for cname, corr in condition_contrib.items():
                cat = name_to_cat.get(cname, cname)
                cat_scores.setdefault(cat, []).append(corr)
            cat_mean = {cat: float(np.mean(vs)) for cat, vs in cat_scores.items()}

            cats_sorted = sorted(cat_mean, key=lambda c: cat_mean[c], reverse=True)

            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.barh(
                [CAT_EN.get(c, c) for c in cats_sorted],
                [cat_mean[c] for c in cats_sorted],
                color="#4f7942",
            )
            ax.invert_yaxis()
            ax.set_xlim(-1, 1)
            ax.set_xlabel("Contribution to this ranking (correlation)")
            ax.set_title("Selected Conditions' Influence on This Result")
            fig.patch.set_facecolor("#f6f8f1")
            ax.set_facecolor("#f6f8f1")
            plt.tight_layout()
            st.pyplot(fig, width="stretch")

            st.caption(
                "값이 클수록 해당 조건의 선호 패턴이 이번 추천 순위와 더 비슷함을 의미합니다 "
                "(= 그 조건이 결과에 더 크게 반영됨). 매 추천마다 새로 계산됩니다."
            )

    with st.expander("📋 상세 결과 표"):
        show_cols = ["식물명", "추천점수", "관리요구도", "독성_여부(1=있음)"]
        st.dataframe(
            df[show_cols].rename(columns={"추천점수": "추천점수(0~1)", "독성_여부(1=있음)": "독성여부"}),
            width="stretch",
        )


# ══════════════════════════════════════════════════════════
# 7. 메인 영역
# ══════════════════════════════════════════════════════════
include, exclude, _selected_categories = build_conditions()

if submitted:
    if not include and not exclude:
        st.info("위에서 환경 조건을 하나 이상 선택한 뒤 '식물 추천 받기'를 눌러주세요. "
                "조건을 입력하지 않으면 전체 평균 기준 추천을 보여줍니다.")
    render_results(include, exclude)
else:
    st.markdown(
        """
        ### 👆 위에서 우리 집 환경을 선택하고 '식물 추천 받기'를 눌러주세요
        - 광도, 두는 위치, 온도, 습도, 관리 난이도 등 일부만 선택해도 추천이 가능합니다.
        - 반려동물/아이가 있다면 '독성 식물 제외'를 체크하세요.
        - 꽃가루 알러지가 있다면 해당 계절을 선택하면 그 계절에 꽃 피는 식물을 제외합니다.
        """
    )