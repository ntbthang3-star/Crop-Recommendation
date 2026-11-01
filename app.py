"""
app.py - Ứng dụng web Streamlit: Hệ thống hỗ trợ lựa chọn cây trồng theo điều kiện đất.
Chạy:  streamlit run app.py
Yêu cầu: các file sau nằm CÙNG THƯ MỤC với app.py
    thuat_toan.py   (bắt buộc: chứa lớp MyRandomForest mà crop_models.pkl cần)
    crop_models.pkl (mô hình đã huấn luyện)
    Tap_Test.csv    (để hiển thị đánh giá mô hình)
    Tap_Train.csv   (để xem dữ liệu)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import thuat_toan as tt  # PHẢI import để joblib tìm thấy lớp MyRandomForest

THU_MUC = Path(__file__).parent

TEN_VI = {
    "apple": "Táo", "banana": "Chuối", "blackgram": "Đậu đen", "chickpea": "Đậu gà",
    "coconut": "Dừa", "coffee": "Cà phê", "cotton": "Bông", "grapes": "Nho",
    "jute": "Đay", "kidneybeans": "Đậu thận", "lentil": "Đậu lăng", "maize": "Ngô",
    "mango": "Xoài", "mothbeans": "Đậu mốc", "mungbean": "Đậu xanh",
    "muskmelon": "Dưa lưới", "orange": "Cam", "papaya": "Đu đủ",
    "pigeonpeas": "Đậu triều", "pomegranate": "Lựu", "rice": "Lúa", "watermelon": "Dưa hấu",
}


def ten(c):
    return f"{TEN_VI.get(c, c)} ({c})"


st.set_page_config(page_title="Gợi ý cây trồng", page_icon="🌱", layout="wide")

# ---------- GIAO DIỆN MẪU A - ĐỒNG LÚA ----------
st.markdown(
    """
    <style>
    .stApp { background: #F1F8EE; }
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .banner {
        background: #2E7D32; color: #fff; padding: 26px 36px; border-radius: 16px;
        display: flex; gap: 16px; align-items: center; margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(46,125,50,.25);
    }
    .banner h1 { color: #fff; margin: 0; padding: 0; font-size: 1.8rem; }
    .banner p  { color: #E8F5E9; margin: 4px 0 0 0; font-size: 1rem; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff; border: none; border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
    }
    h2, h3 { color: #1B5E20; }
    div[data-testid="stMetric"] {
        background: #fff; border-left: 6px solid #43A047; border-radius: 12px;
        padding: 14px 18px; box-shadow: 0 2px 8px rgba(0,0,0,.07);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: #E3F1DD; border-radius: 10px 10px 0 0; padding: 8px 18px; }
    .stTabs [aria-selected="true"] { background: #2E7D32 !important; color: #fff !important; }
    .stButton > button[kind="primary"] {
        background: #2E7D32; border: none; border-radius: 10px; padding: 10px 28px; color: #fff;
    }
    .stButton > button[kind="primary"]:hover { background: #1B5E20; color: #fff; }
    .kq { margin-bottom: 18px; }
    .kq b { display: flex; justify-content: space-between; font-size: 1.05rem; color: #1B3A1F; }
    .kq .bar { height: 12px; background: #E3F1DD; border-radius: 6px; margin-top: 6px; }
    .kq .bar i { display: block; height: 12px; background: #43A047; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def tai_mo_hinh():
    return joblib.load(THU_MUC / "crop_models.pkl")


@st.cache_data
def tai_csv(ten_file):
    return pd.read_csv(THU_MUC / ten_file)


data = tai_mo_hinh()
mo_hinh, X_train, nhan = data["models"], data["X_train"], data["labels"]

LA = (
    '<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M5 19c0-8 5-13 14-14 0 9-5 14-14 14z"/><path d="M5 19 13 11"/></svg>'
)
st.markdown(
    f"""
    <div class="banner">{LA}
        <div><h1>Hệ thống hỗ trợ lựa chọn cây trồng</h1>
        <p>Nhập chỉ số đất, nhận gợi ý loại cây phù hợp nhất</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔮 Dự đoán", "📊 Đánh giá mô hình", "🗂️ Dữ liệu"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    trai, phai = st.columns(2, gap="large")
    cau_hinh = {
        "N": dict(label="N (Nitơ)", min_value=0.0, max_value=300.0, value=90.0, step=1.0),
        "P": dict(label="P (Photpho)", min_value=0.0, max_value=300.0, value=42.0, step=1.0),
        "K": dict(label="K (Kali)", min_value=0.0, max_value=400.0, value=43.0, step=1.0),
        "pH": dict(label="pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1),
    }
    mau = {}
    with trai:
        with st.container(border=True):
            st.subheader("Chỉ số đất")
            st.caption("Bỏ tích ô nào bạn chưa đo được; hệ thống tự chọn mô hình phù hợp.")
            for c in tt.THU_TU_CHI_SO:
                lo, hi = X_train[c].min(), X_train[c].max()
                dung = st.checkbox(f"Dùng {c}  (dữ liệu huấn luyện: {lo:.1f} - {hi:.1f})",
                                   value=True, key=f"dung_{c}")
                gia_tri = st.number_input(cau_hinh[c]["label"], **{k: v for k, v in cau_hinh[c].items() if k != "label"},
                                          key=f"val_{c}", disabled=not dung)
                if dung:
                    mau[c] = gia_tri
            so_goi_y = st.slider("Số gợi ý hiển thị", 1, 5, 3)
            bam = st.button("Gợi ý cây trồng", type="primary")

    with phai:
        with st.container(border=True):
            st.subheader("Kết quả gợi ý")
            if not bam:
                st.info("Nhập chỉ số đất ở bên trái rồi bấm **Gợi ý cây trồng**.")
            elif not mau:
                st.error("Cần chọn ít nhất một chỉ số đất.")
            else:
                kq = tt.du_doan_mau_moi(mo_hinh, mau, X_train, top=so_goi_y)
                st.caption(f"Mô hình sử dụng: {kq['mô hình dùng']}")
                html = ""
                for i, (c, p) in enumerate(kq["gợi ý"], 1):
                    html += (f'<div class="kq"><b><span>{i}. {ten(c)}</span><span>{p:.0%}</span></b>'
                             f'<div class="bar"><i style="width:{max(p * 100, 1):.0f}%"></i></div></div>')
                st.markdown(html, unsafe_allow_html=True)
                for w in kq["cảnh báo"]:
                    st.warning(w)

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Kết quả trên tập kiểm tra (mô hình đủ 4 chỉ số)")
    try:
        test = tai_csv("Tap_Test.csv")
    except FileNotFoundError:
        st.error("Không tìm thấy Tap_Test.csv trong thư mục dự án (hãy đổi tên từ Tập_Test.csv).")
        st.stop()
    day_du = mo_hinh[tuple(tt.THU_TU_CHI_SO)]
    y_that = test["Crop"].values
    y_dd = day_du.predict(test[tt.THU_TU_CHI_SO])

    ct = tt.tong_hop_chi_so(y_that, y_dd, nhan)
    m = st.columns(4)
    m[0].metric("Accuracy", f"{ct['Accuracy']:.2%}")
    m[1].metric("F1 (macro)", f"{ct['F1 (macro)']:.2%}")
    m[2].metric("Precision (macro)", f"{ct['Precision (macro)']:.2%}")
    m[3].metric("Recall (macro)", f"{ct['Recall (macro)']:.2%}")

    cm = tt.ma_tran_nham_lan(y_that, y_dd, nhan)
    cA, cB = st.columns([3, 2])
    with cA:
        st.markdown("**Ma trận nhầm lẫn (chuẩn hoá theo hàng)**")
        chuan = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(chuan, cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(len(nhan))); ax.set_xticklabels(nhan, rotation=90, fontsize=7)
        ax.set_yticks(range(len(nhan))); ax.set_yticklabels(nhan, fontsize=7)
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        st.pyplot(fig)
    with cB:
        st.markdown("**Độ quan trọng của từng chỉ số**")
        dq = pd.Series(day_du.feature_importances_, index=tt.THU_TU_CHI_SO)
        st.bar_chart(dq, color="#2E7D32")
        st.markdown("**Các cặp hay bị nhầm nhất**")
        st.dataframe(tt.cap_nham_nhieu_nhat(cm, nhan, top=8), hide_index=True, use_container_width=True)

    st.markdown("**Precision / Recall / F1 theo từng loại cây**")
    st.dataframe(tt.bao_cao_theo_lop(cm, nhan).style.format(
        {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}), use_container_width=True)

    with st.expander("Tham số mô hình"):
        st.json(day_du.get_params())
        st.write(f"OOB score (trên tập train): {day_du.oob_score_:.2%}")

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Dữ liệu huấn luyện / kiểm tra")
    try:
        train = tai_csv("Tap_Train.csv")
        test = tai_csv("Tap_Test.csv")
    except FileNotFoundError:
        st.error("Thiếu Tap_Train.csv hoặc Tap_Test.csv trong thư mục dự án.")
        st.stop()
    st.write(f"Train: {train.shape[0]} dòng | Test: {test.shape[0]} dòng | {train['Crop'].nunique()} loại cây")
    chon = st.radio("Xem tập", ["Train", "Test"], horizontal=True)
    df = train if chon == "Train" else test
    loc = st.multiselect("Lọc theo cây trồng", sorted(df["Crop"].unique()))
    if loc:
        df = df[df["Crop"].isin(loc)]
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("**Phân bố một chỉ số theo cây trồng**")
    cs = st.selectbox("Chỉ số", tt.THU_TU_CHI_SO)
    st.bar_chart(train.groupby("Crop")[cs].mean(), color="#66BB6A")
