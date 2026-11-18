import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.io import MemoryFile


st.set_page_config(
    page_title="Demo bản đồ nhiệt độ bề mặt Hà Nội",
    page_icon="🌡️",
    layout="wide",
)

st.title("🌡️ HỆ THỐNG DEMO BẢN ĐỒ NHIỆT ĐỘ BỀ MẶT (LST)")
st.caption("Dữ liệu Landsat 8/9 Collection 2 Level-2 · ST_B10 · QA_PIXEL · ST_QA tùy chọn · MTL")

with st.sidebar:
    st.header("Cấu hình xử lý")
    min_lst = st.number_input("Nhiệt độ thấp nhất (°C)", value=0.0, min_value=-50.0, max_value=80.0, step=1.0)
    max_lst = st.number_input("Nhiệt độ cao nhất (°C)", value=60.0, min_value=-20.0, max_value=100.0, step=1.0)
    use_stqa = st.checkbox("Sử dụng ST_QA để lọc độ không chắc chắn", value=False)
    max_stqa_k = st.number_input("Ngưỡng ST_QA (Kelvin)", value=5.0, min_value=0.1, max_value=20.0, step=0.5)
    display_size = st.select_slider("Kích thước cạnh lớn nhất khi hiển thị", options=[600, 800, 1000, 1200, 1400, 1600], value=1200)

st.subheader("1. Tải dữ liệu đầu vào")
c1, c2 = st.columns(2)
with c1:
    st_file = st.file_uploader("Ảnh ST_B10 (.TIF)", type=["tif", "tiff"], key="st")
    qa_file = st.file_uploader("Ảnh QA_PIXEL (.TIF)", type=["tif", "tiff"], key="qa")
with c2:
    mtl_file = st.file_uploader("Metadata MTL (.txt) – tùy chọn", type=["txt"], key="mtl")
    stqa_file = st.file_uploader("Ảnh ST_QA (.TIF) – tùy chọn", type=["tif", "tiff"], key="stqa")

st.info(
    "Chọn ST_B10 và QA_PIXEL thuộc cùng một cảnh Landsat. "
    "Nếu bật ST_QA, hãy tải thêm đúng ảnh ST_QA cùng cảnh."
)

def parse_mtl(raw_bytes):
    if not raw_bytes:
        return {}
    text = raw_bytes.decode("utf-8", errors="ignore")
    wanted = [
        "UTM_ZONE",
        "CORNER_UL_PROJECTION_X_PRODUCT", "CORNER_UL_PROJECTION_Y_PRODUCT",
        "CORNER_UR_PROJECTION_X_PRODUCT", "CORNER_UR_PROJECTION_Y_PRODUCT",
        "CORNER_LL_PROJECTION_X_PRODUCT", "CORNER_LL_PROJECTION_Y_PRODUCT",
        "CORNER_LR_PROJECTION_X_PRODUCT", "CORNER_LR_PROJECTION_Y_PRODUCT",
    ]
    result = {}
    for key in wanted:
        match = re.search(r"(?m)^\s*" + re.escape(key) + r"\s*=\s*\"?([-+]?\d+(?:\.\d+)?)", text)
        if match:
            result[key] = float(match.group(1))
    return result


def open_tiff(raw_bytes):
    mem = MemoryFile(raw_bytes)
    ds = mem.open()
    return mem, ds


def process_scene(st_bytes, qa_bytes, mtl_bytes, stqa_bytes, min_c, max_c, use_qa, qa_limit, max_side):
    st_mem, st_ds = open_tiff(st_bytes)
    qa_mem, qa_ds = open_tiff(qa_bytes)
    stqa_mem = stqa_ds = None
    try:
        if st_ds.count < 1 or qa_ds.count < 1:
            raise ValueError("Không tìm thấy band raster hợp lệ.")
        if (st_ds.height, st_ds.width) != (qa_ds.height, qa_ds.width):
            raise ValueError("Kích thước QA_PIXEL không trùng với ST_B10.")
        if use_qa:
            if not stqa_bytes:
                raise ValueError("Bạn đã bật ST_QA nhưng chưa tải ảnh ST_QA.")
            stqa_mem, stqa_ds = open_tiff(stqa_bytes)
            if (st_ds.height, st_ds.width) != (stqa_ds.height, stqa_ds.width):
                raise ValueError("Kích thước ST_QA không trùng với ST_B10.")

        rows, cols = st_ds.height, st_ds.width
        total = rows * cols
        keys = ["fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow", "snow"]
        bit_counts = {k: 0 for k in keys}
        valid_dn_count = 0
        bad_qa_count = 0
        valid_qa_count = 0
        valid_combined_count = 0
        abnormal_count = 0
        valid_count = 0
        sum_temp = 0.0
        sum_sq = 0.0
        tmin = np.inf
        tmax = -np.inf
        hist_edges = np.linspace(min_c, max_c, 61)
        hist_counts = np.zeros(len(hist_edges) - 1, dtype=np.int64)

        # Đọc theo block để không nạp toàn bộ ảnh 59 triệu pixel vào RAM.
        for _, window in st_ds.block_windows(1):
            dn = st_ds.read(1, window=window).astype(np.float32)
            qa = qa_ds.read(1, window=window).astype(np.uint16)

            valid_dn = np.isfinite(dn) & (dn > 0) & (dn < 65535)
            flags = [
                (qa & (1 << 0)) != 0,
                (qa & (1 << 1)) != 0,
                (qa & (1 << 2)) != 0,
                (qa & (1 << 3)) != 0,
                (qa & (1 << 4)) != 0,
                (qa & (1 << 5)) != 0,
            ]
            for key, flag in zip(keys, flags):
                bit_counts[key] += int(np.count_nonzero(flag))

            bad = np.logical_or.reduce(flags)
            valid_qa = ~bad
            valid_dn_count += int(valid_dn.sum())
            bad_qa_count += int(bad.sum())
            valid_qa_count += int(valid_qa.sum())

            valid = valid_dn & valid_qa
            if use_qa and stqa_ds is not None:
                stqa = stqa_ds.read(1, window=window).astype(np.float32)
                valid_stqa = np.isfinite(stqa) & (stqa >= 0) & ((stqa * 0.01) <= qa_limit)
                valid &= valid_stqa

            lst = dn * 0.00341802 + 149.0 - 273.15
            lst[~valid] = np.nan
            valid_combined_count += int(valid.sum())

            in_range = valid & (lst >= min_c) & (lst <= max_c)
            abnormal_count += int((valid & ~((lst >= min_c) & (lst <= max_c))).sum())
            values = lst[in_range]
            if values.size:
                valid_count += int(values.size)
                sum_temp += float(values.sum(dtype=np.float64))
                sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
                tmin = min(tmin, float(values.min()))
                tmax = max(tmax, float(values.max()))
                hist_counts += np.histogram(values, bins=hist_edges)[0]

        if valid_count == 0:
            raise ValueError("Không còn pixel hợp lệ. Hãy kiểm tra dữ liệu hoặc nới ngưỡng nhiệt độ.")

        mean = sum_temp / valid_count
        variance = max(0.0, sum_sq / valid_count - mean * mean)
        std = float(np.sqrt(variance))
        median_bin = int(np.searchsorted(np.cumsum(hist_counts), valid_count / 2, side="left"))
        median_bin = min(median_bin, len(hist_edges) - 2)
        median_approx = float((hist_edges[median_bin] + hist_edges[median_bin + 1]) / 2)

        # Tạo ảnh xem trước có giới hạn kích thước.
        scale = min(1.0, max_side / max(rows, cols))
        out_h = max(1, int(round(rows * scale)))
        out_w = max(1, int(round(cols * scale)))
        dn_small = st_ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
        qa_small = qa_ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.uint16)
        valid_small = np.isfinite(dn_small) & (dn_small > 0) & (dn_small < 65535)
        bad_small = np.zeros((out_h, out_w), dtype=bool)
        for bit in range(6):
            bad_small |= (qa_small & (1 << bit)) != 0
        valid_small &= ~bad_small

        if use_qa and stqa_ds is not None:
            stqa_small = stqa_ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.float32)
            valid_small &= np.isfinite(stqa_small) & (stqa_small >= 0) & ((stqa_small * 0.01) <= qa_limit)

        lst_small = dn_small * 0.00341802 + 149.0 - 273.15
        valid_temp_small = valid_small & (lst_small >= min_c) & (lst_small <= max_c)
        lst_small[~valid_temp_small] = np.nan
        bad_map = (~valid_temp_small).astype(np.uint8)

        # Ảnh ST_B10 grayscale stretch percentile.
        dn_valid_small = dn_small[valid_small]
        if dn_valid_small.size:
            lo, hi = np.percentile(dn_valid_small, [2, 98])
        else:
            lo, hi = 0, 1
        if hi <= lo:
            hi = lo + 1
        st_gray = np.clip((dn_small - lo) / (hi - lo), 0, 1)
        st_gray[~valid_small] = 0

        mtl = parse_mtl(mtl_bytes)
        utm_keys = [
            "CORNER_UL_PROJECTION_X_PRODUCT", "CORNER_UL_PROJECTION_Y_PRODUCT",
            "CORNER_UR_PROJECTION_X_PRODUCT", "CORNER_UR_PROJECTION_Y_PRODUCT",
            "CORNER_LL_PROJECTION_X_PRODUCT", "CORNER_LL_PROJECTION_Y_PRODUCT",
        ]
        has_utm = all(k in mtl for k in utm_keys)

        stats = {
            "Tổng số pixel": total,
            "Pixel DN hợp lệ": valid_dn_count,
            "Fill": bit_counts["fill"],
            "Dilated Cloud": bit_counts["dilated_cloud"],
            "Cirrus": bit_counts["cirrus"],
            "Cloud": bit_counts["cloud"],
            "Cloud Shadow": bit_counts["cloud_shadow"],
            "Snow": bit_counts["snow"],
            "Pixel QA bị loại": bad_qa_count,
            "Pixel QA hợp lệ": valid_qa_count,
            "Pixel hợp lệ sau kết hợp": valid_combined_count,
            "Pixel nhiệt độ ngoài ngưỡng": abnormal_count,
            "Pixel hợp lệ cuối cùng": valid_count,
            "LST nhỏ nhất (°C)": tmin,
            "LST lớn nhất (°C)": tmax,
            "LST trung bình (°C)": mean,
            "LST trung vị xấp xỉ (°C)": median_approx,
            "Độ lệch chuẩn (°C)": std,
        }
        csv_data = pd.DataFrame({"Chỉ tiêu": list(stats.keys()), "Giá trị": list(stats.values())}).to_csv(index=False).encode("utf-8-sig")

        # Sinh PNG ở bộ nhớ để người dùng tải xuống.
        def fig_png(fig):
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            buffer.seek(0)
            return buffer.getvalue()

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(st_gray, cmap="gray", origin="upper")
        ax.set_title("Ảnh ST_B10 (ảnh xem trước)")
        ax.set_xlabel("Cột pixel")
        ax.set_ylabel("Hàng pixel")
        fig.colorbar(im, ax=ax, label="DN đã co giãn hiển thị")
        st_png = fig_png(fig)

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(lst_small, cmap="turbo", vmin=min_c, vmax=max_c, origin="upper")
        ax.set_title("Bản đồ nhiệt độ bề mặt LST sau lọc QA")
        ax.set_xlabel("Cột pixel")
        ax.set_ylabel("Hàng pixel")
        fig.colorbar(im, ax=ax, label="Nhiệt độ (°C)")
        lst_png = fig_png(fig)

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(bad_map, cmap="gray", vmin=0, vmax=1, origin="upper")
        ax.set_title("Bản đồ pixel không hợp lệ / bị loại")
        ax.set_xlabel("Cột pixel")
        ax.set_ylabel("Hàng pixel")
        bad_png = fig_png(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        ax.bar(centers, hist_counts, width=np.diff(hist_edges) * 0.9, color="#4C9BD6", edgecolor="black")
        ax.set_title("Histogram phân bố nhiệt độ LST")
        ax.set_xlabel("Nhiệt độ (°C)")
        ax.set_ylabel("Số lượng pixel")
        ax.grid(alpha=0.25)
        hist_png = fig_png(fig)

        utm_png = None
        if has_utm:
            ulx = mtl["CORNER_UL_PROJECTION_X_PRODUCT"]
            uly = mtl["CORNER_UL_PROJECTION_Y_PRODUCT"]
            urx = mtl["CORNER_UR_PROJECTION_X_PRODUCT"]
            lly = mtl["CORNER_LL_PROJECTION_Y_PRODUCT"]
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(lst_small, cmap="turbo", vmin=min_c, vmax=max_c,
                           extent=[ulx, urx, lly, uly], origin="upper", aspect="auto")
            ax.set_title("Bản đồ LST theo phạm vi tọa độ UTM trong MTL")
            ax.set_xlabel("Tọa độ UTM X (m)")
            ax.set_ylabel("Tọa độ UTM Y (m)")
            fig.colorbar(im, ax=ax, label="Nhiệt độ (°C)")
            zone = mtl.get("UTM_ZONE")
            if zone is not None:
                ax.text(0.02, 0.02, f"UTM Zone {zone:g}", transform=ax.transAxes,
                        color="white", bbox={"facecolor": "black", "alpha": 0.55})
            utm_png = fig_png(fig)

        return {
            "stats": stats,
            "csv": csv_data,
            "st_png": st_png,
            "lst_png": lst_png,
            "bad_png": bad_png,
            "hist_png": hist_png,
            "utm_png": utm_png,
            "hist_edges": hist_edges,
            "hist_counts": hist_counts,
            "preview_shape": f"{out_h} × {out_w}",
            "original_shape": f"{rows} × {cols}",
        }
    finally:
        st_ds.close()
        qa_ds.close()
        st_mem.close()
        qa_mem.close()
        if stqa_ds is not None:
            stqa_ds.close()
        if stqa_mem is not None:
            stqa_mem.close()


run = st.button("🚀 BẮT ĐẦU XỬ LÝ VÀ TẠO BẢN ĐỒ", type="primary", use_container_width=True)
if run:
    if st_file is None or qa_file is None:
        st.error("Bạn cần tải lên cả ST_B10 và QA_PIXEL.")
    elif min_lst >= max_lst:
        st.error("Nhiệt độ thấp nhất phải nhỏ hơn nhiệt độ cao nhất.")
    else:
        try:
            with st.spinner("Đang đọc GeoTIFF, lọc QA và tính nhiệt độ LST. Ảnh lớn có thể mất vài phút..."):
                result = process_scene(
                    st_file.getvalue(),
                    qa_file.getvalue(),
                    mtl_file.getvalue() if mtl_file else None,
                    stqa_file.getvalue() if stqa_file else None,
                    float(min_lst), float(max_lst), bool(use_stqa),
                    float(max_stqa_k), int(display_size),
                )
            st.session_state["lst_demo_result"] = result
            st.success("Đã xử lý xong dữ liệu và tạo các sản phẩm bản đồ!")
        except Exception as exc:
            st.error(f"Không thể xử lý dữ liệu: {exc}")
            st.exception(exc)

result = st.session_state.get("lst_demo_result")
if result:
    st.divider()
    st.subheader("2. Kết quả thống kê")
    st.caption(f"Kích thước ảnh gốc: {result['original_shape']} · Kích thước ảnh xem trước: {result['preview_shape']}")
    s = result["stats"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LST trung bình", f"{s['LST trung bình (°C)']:.2f} °C")
    k2.metric("LST nhỏ nhất", f"{s['LST nhỏ nhất (°C)']:.2f} °C")
    k3.metric("LST lớn nhất", f"{s['LST lớn nhất (°C)']:.2f} °C")
    k4.metric("Pixel hợp lệ", f"{s['Pixel hợp lệ cuối cùng']:,}")

    st.dataframe(pd.DataFrame({"Chỉ tiêu": list(s.keys()), "Giá trị": list(s.values())}), use_container_width=True, hide_index=True)

    st.subheader("3. Bản đồ và biểu đồ")
    tabs = st.tabs(["Bản đồ LST", "Ảnh ST_B10", "Pixel bị loại", "Histogram", "Bản đồ UTM"])
    with tabs[0]:
        st.image(result["lst_png"], use_container_width=True)
        st.download_button("Tải bản đồ LST PNG", result["lst_png"], "Ban_do_LST_HaNoi.png", "image/png")
    with tabs[1]:
        st.image(result["st_png"], use_container_width=True)
        st.download_button("Tải ảnh ST_B10 PNG", result["st_png"], "Anh_goc_ST_B10.png", "image/png")
    with tabs[2]:
        st.image(result["bad_png"], use_container_width=True)
        st.download_button("Tải bản đồ pixel bị loại PNG", result["bad_png"], "Ban_do_pixel_bi_loai.png", "image/png")
    with tabs[3]:
        st.image(result["hist_png"], use_container_width=True)
        st.download_button("Tải Histogram PNG", result["hist_png"], "Histogram_LST_HaNoi.png", "image/png")
    with tabs[4]:
        if result["utm_png"]:
            st.image(result["utm_png"], use_container_width=True)
            st.download_button("Tải bản đồ UTM PNG", result["utm_png"], "Ban_do_LST_UTM.png", "image/png")
        else:
            st.warning("Không tìm thấy đủ tọa độ góc trong MTL. Bản đồ UTM chưa được tạo; các bản đồ theo pixel vẫn có.")
    st.download_button("⬇️ Tải bảng thống kê CSV", result["csv"], "ThongKe_LST_HaNoi.csv", "text/csv")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Ban_do_LST_HaNoi.png", result["lst_png"])
        zf.writestr("Anh_goc_ST_B10.png", result["st_png"])
        zf.writestr("Ban_do_pixel_bi_loai.png", result["bad_png"])
        zf.writestr("Histogram_LST_HaNoi.png", result["hist_png"])
        if result["utm_png"]:
            zf.writestr("Ban_do_LST_UTM.png", result["utm_png"])
        zf.writestr("ThongKe_LST_HaNoi.csv", result["csv"])
    st.download_button(
        "📦 Tải toàn bộ kết quả (.ZIP)",
        zip_buffer.getvalue(),
        "KetQua_LST_HaNoi.zip",
        "application/zip",
        use_container_width=True,
    )

st.divider()
st.caption("Lưu ý: Ảnh hiển thị được giảm mẫu để tăng tốc; thống kê pixel được tính theo từng block trên dữ liệu raster gốc. Trung vị được ước lượng từ histogram.")
