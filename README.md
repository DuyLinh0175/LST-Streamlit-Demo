# Demo bản đồ nhiệt độ bề mặt Hà Nội (LST)

## Chức năng
- Tải lên ST_B10, QA_PIXEL, MTL và ST_QA tùy chọn.
- Tính LST theo công thức Landsat Collection 2 Level-2:
  `ST_K = DN * 0.00341802 + 149.0`
  `ST_C = ST_K - 273.15`
- Giải mã QA_PIXEL: Fill, Dilated Cloud, Cirrus, Cloud, Cloud Shadow, Snow.
- Lọc ngưỡng nhiệt độ và tùy chọn lọc ST_QA.
- Hiển thị bản đồ LST, ảnh ST_B10, bản đồ pixel bị loại, histogram và bản đồ UTM (nếu MTL có đủ tọa độ).
- Tải PNG, CSV hoặc ZIP kết quả.

## Chạy trên Windows
1. Cài Python 3.10 hoặc mới hơn.
2. Giải nén thư mục chứa `app.py` và `requirements.txt`.
3. Mở CMD/PowerShell tại thư mục đó.
4. Chạy:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

5. Trình duyệt sẽ mở giao diện tại `http://localhost:8501`.

## Dữ liệu đầu vào
- Bắt buộc: ST_B10 và QA_PIXEL dạng GeoTIFF của cùng một cảnh Landsat.
- Tùy chọn: MTL.txt.
- Tùy chọn: ST_QA. Nếu bật tùy chọn ST_QA trong sidebar thì phải tải ảnh ST_QA.

## Lưu ý
- Ảnh xem trước được giảm mẫu để tránh quá tải RAM; thống kê được cộng dồn theo từng block trên raster gốc.
- Bản đồ UTM trong demo dùng các tọa độ góc sản phẩm trong MTL để gắn phạm vi hiển thị. Khi cần bản đồ GIS có độ chính xác tọa độ đầy đủ, nên dùng transform/CRS GeoTIFF trong phần mềm GIS.
- Trung vị báo cáo là giá trị xấp xỉ theo histogram.
