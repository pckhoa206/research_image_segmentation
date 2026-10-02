# 🔬 Research in Medical Image Segmentation

Kho lưu trữ tài liệu, tổng hợp kiến thức và nghiên cứu khoa học chuyên sâu về **Phân đoạn Ảnh Y sinh (Biomedical Image Segmentation)**.

---

## 📚 Danh sách các nghiên cứu & bài báo (Papers & Summaries)

| STT | Bài báo / Mô hình | Năm | Tài liệu tóm tắt | Trạng thái |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **U-Net: Convolutional Networks for Biomedical Image Segmentation** | 2015 | [📖 Đọc U-Net.md](./U-Net.md) | ✅ Hoàn thành |
| 2 | **Attention U-Net: Learning Where to Look for the Pancreas** | 2018 | *Đang cập nhật* | ⏳ Sắp có |
| 3 | **Transformers in Medical Imaging: A Survey** | 2021+ | *Đang cập nhật* | ⏳ Sắp có |

---

## 🌟 Điểm nhấn về U-Net

Bài báo nền tảng mở đầu cho toàn bộ hướng nghiên cứu:
- **Kiến trúc:** Dạng chữ U đối xứng (Encoder - Decoder) với các đường nối tắt (Skip Connections) giúp truyền nguyên vẹn đặc trưng độ phân giải cao sang decoder.
- **Ưu điểm vượt trội:**
  - Hoạt động hiệu quả vượt bậc ngay cả khi tập dữ liệu rất nhỏ (chỉ vài chục ảnh y tế).
  - Tách các tế bào tiếp xúc dính liền nhờ hàm mất mát có trọng số biên (Weighted Loss).
  - Xử lý ảnh siêu lớn bằng chiến lược gạch đè (Overlap-Tile) và đối xứng gương (Mirroring).
  - Tăng cường dữ liệu bằng biến dạng đàn hồi ngẫu nhiên (Elastic Deformation).

👉 Chi tiết toàn bộ phương pháp, công thức toán và phân tích chuyên sâu có tại file: **[U-Net.md](./U-Net.md)**.
