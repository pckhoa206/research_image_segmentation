# 🔬 Research in Medical Image Segmentation

Kho lưu trữ tài liệu, tổng hợp kiến thức và nghiên cứu khoa học chuyên sâu về **Phân đoạn Ảnh Y sinh (Biomedical Image Segmentation)**.

---

## 📚 Danh sách các nghiên cứu & bài báo (Papers & Summaries)

| STT | Bài báo / Mô hình | Năm | Tài liệu tóm tắt | Trạng thái |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **U-Net: Convolutional Networks for Biomedical Image Segmentation** | 2015 | [📖 Đọc U-Net.md](./U-Net.md) | ✅ Hoàn thành |
| 2 | **Attention U-Net: Learning Where to Look for the Pancreas** | 2018 | [📖 Đọc Attention_U-Net.md](./Attention_U-Net.md) | ✅ Hoàn thành |
| 3 | **Transformers in Medical Imaging: A Survey** | 2021+ | *Đang cập nhật* | ⏳ Sắp có |

---

## 🌟 Điểm nhấn các nghiên cứu tiêu biểu

### 1. U-Net (Ronneberger et al., 2015)
- **Kiến trúc:** Dạng chữ U đối xứng (Encoder - Decoder) với các đường nối tắt (Skip Connections) giúp truyền nguyên vẹn đặc trưng độ phân giải cao sang decoder.
- **Ưu điểm vượt trội:**
  - Hoạt động hiệu quả vượt bậc ngay cả khi tập dữ liệu rất nhỏ (chỉ vài chục ảnh y tế).
  - Tách các tế bào tiếp xúc dính liền nhờ hàm mất mát có trọng số biên (Weighted Loss).
  - Xử lý ảnh siêu lớn bằng chiến lược gạch đè (Overlap-Tile) và đối xứng gương (Mirroring).
  - Tăng cường dữ liệu bằng biến dạng đàn hồi ngẫu nhiên (Elastic Deformation).
👉 Chi tiết toàn bộ phương pháp, công thức toán và phân tích chuyên sâu: **[U-Net.md](./U-Net.md)**.

### 2. Attention U-Net (Oktay et al., 2018)
- **Kiến trúc:** Tích hợp các **Cổng chú ý (Attention Gates - AGs)** vào trực tiếp các đường Skip Connections của U-Net 3D.
- **Ưu điểm vượt trội:**
  - Tự động tập trung vào các cơ quan nhỏ, khó phân đoạn (như tuyến tụy trên ảnh CT bụng) và triệt tiêu nhiễu nền.
  - Loại bỏ hoàn toàn sự cồng kềnh của hệ thống 2 mạng (Cascaded CNNs: dò ROI thô + phân đoạn tinh), chạy mô hình End-to-End duy nhất.
  - Lọc gradient trong lan truyền ngược giúp các tầng nông của Encoder tập trung tối ưu hóa vật thể đích.
  - Tăng vọt độ nhạy (Recall) và điểm Dice (+2.6% DSC trên CT-150) trong khi chỉ tốn thêm ~8% tham số.
👉 Chi tiết toàn bộ phương pháp, công thức toán và phân tích chuyên sâu: **[Attention_U-Net.md](./Attention_U-Net.md)**.
