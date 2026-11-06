# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** Python: 3.13.18, PyTorch: 2.11.0+cu130, Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` 

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`= 1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Phạm vi của record: Record này mô tả toàn bộ bức ảnh (Image-level). Lý do: Bài toán Image Classification chỉ gán một nhãn phân loại tổng thể cho toàn khung hình, không định vị tọa độ hay phân tách từng đối tượng riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: Danh sách lớp đi kèm theo taxonomy ImageNet-1K (1.000 lớp cố định) được định nghĩa sẵn trong checkpoint `yolo11n-cls.pt`, mô hình không tự suy nghĩ hoặc tự sinh ra tên lớp mới.
4. Lý do cần ghi đủ id, name và taxonomy: Tránh hiểu lầm và xung đột nhãn giữa các hệ thống, vì cùng một tên lớp có thể mang ID khác nhau ở các taxonomy khác nhau (ví dụ: COCO vs ImageNet vs VOC), hoặc cùng ID nhưng biểu thị vật thể khác nhau nếu không kèm theo tên taxonomy cụ thể.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần nêu rõ quy tắc ưu tiên chọn lớp: ưu tiên chủ thể chính của bức ảnh, đối tượng chiếm diện tích khung hình lớn nhất ở tiền cảnh, hoặc quy tắc theo mục tiêu nghiệp vụ (ví dụ: ưu tiên phương tiện giao thông hơn phông nền đường phố).
6. Vì sao không được lấy `score` của model làm ground truth? Vì `score` (confidence) chỉ là giá trị xác suất toán học do mô hình ước lượng nội bộ dựa trên trọng số huấn luyện; nó không bảo đảm tính đúng đắn thực tế khách quan. Ground truth bắt buộc phải do con người kiểm chứng và gán theo guideline chính xác.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`= [385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: Bounding box này bao quanh người đầu bếp đang đứng quay lưng. Điểm góc trên-bên trái nằm tại tọa độ pixel (x=385.33, y=69.24) (ngay đỉnh đầu/vai), điểm góc dưới-bên phải nằm tại tọa độ pixel (x=498.92, y=348.92) (dưới gót chân). Hộp có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel (hệ tọa độ có gốc (0, 0) ở góc trên cùng bên trái ảnh).
3. Chọn hai ngưỡng (vd. 0.35 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? 
   - Ngưỡng 0.35: có 11 predictions (gồm 2 person, 2 oven, 5 bowl, 2 cup).
   - Ngưỡng 0.60: có 6 predictions (chỉ giữ lại các vật thể có độ tin cậy cao: 2 person, 2 bowl, 2 oven; các vật nhỏ và khó nhận biết như cup hay bowl nhỏ bị lọc bỏ).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Reviewer phải rà soát và loại bỏ nhiều dự đoán sai / dương tính giả (false positives), đồng thời xử lý nhiều hộp bao bị trùng lặp hoặc bao sai vị trí, làm tăng đáng kể khối lượng và thời gian kiểm định (QC/rework).
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này): "Bounding box phải bao trọn khít viền ngoài cùng của đối tượng (tight bounding box), không cắt lẹm vào phần thân đối tượng và khoảng cách thừa giữa mép hộp với biên đối tượng không vượt quá 3-5 pixel. Đối với vật thể bị che khuất một phần, vẽ hộp bao quanh toàn bộ phần nhìn thấy được."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? 
   - Đối tượng quan sát: Ở góc dưới bên trái có một cánh tay/bàn tay người thò vào bị mép ảnh cắt cụt (được model phát hiện là `person 0.61`, box `[0.12, 263.18, 61.55, 310.78]`).
   - Quyết định: Cần đưa lên hỏi Lab Coach (**Escalation**). Vì đây là trường hợp bộ phận cơ thể bị cắt rời rạc không thấy thân người; cần sự thống nhất từ Lab Coach/Lead xem dự án có quy ước gắn nhãn một bàn tay đơn lẻ là `person` hay phải bỏ qua, tránh việc annotator tự phỏng đoán gây nhiễu dữ liệu.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? Polygon cung cấp chính xác đường biên ranh giới theo từng pixel bám sát hình dáng thực tế (contour) của vật thể, tách biệt hoàn toàn vật thể khỏi phông nền. Trong khi bounding box chỉ là hình chữ nhật chứa cả khoảng không gian thừa (ví dụ: khoảng trống giữa hai chân hoặc vùng nách của người), polygon loại bỏ hoàn toàn các pixel nền này.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào? 
   - `instance_id` là mã định danh duy nhất để phân biệt từng cá thể/đối tượng riêng lẻ trong cùng một bức ảnh (ví dụ: `kitchen-002` và `kitchen-003` đều là lớp `bowl` nhưng là 2 chiếc bát khác nhau).
   - Nó không phải class ID vì nó không đại diện cho tên phân loại đối tượng (class ID là 0 cho người, 45 cho bát).
   - Nó không phải tracking ID vì nó chỉ có phạm vi trong bức ảnh tĩnh của bài lab, không có chức năng theo dõi cùng một đối tượng qua chuỗi khung hình video liên tiếp theo thời gian.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo: "Đường biên đa giác (polygon) phải bám sát theo đường viền thực tế của đối tượng với sai số không quá 2-3 pixel; không vẽ lấn sang vật thể tiền cảnh khác; và phải khoét rỗng (donut polygon) các vùng nền lọt vào giữa các bộ phận đối tượng (như khoảng trống giữa hai chân)."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - Đối tượng quan sát: Chiếc bàn bếp lớn (`dining table 0.58`) có bề mặt tiếp xúc và bị đè lên bởi nhiều vật thể khác (bát `bowl 0.70`, khuôn bột, khăn lau); hoặc chùm lá khô treo trên trần bị model phân đoạn nhầm thành `potted plant 0.63`.
   - Quyết định: Với các vật tiếp xúc chạm nhau thông thường trên mặt bàn, dùng **Guideline** (quy định vẽ mask trọn vẹn cho vật thể tiền cảnh phía trên, vật thể bị đè bên dưới sẽ dừng biên tại mép tiếp xúc). Với trường hợp chùm lá khô treo bếp bị gán nhãn là cây cảnh (`potted plant`), cần **Escalation** (hỏi Lab Coach) để xác định rõ taxonomy của bài toán có gán nhãn đồ trang trí/gia vị treo bếp hay không.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn danh mục (Class ID + Class Name) gán cho toàn bộ bức ảnh | Ảnh phức tạp có nhiều vật thể; model ImageNet đoán ảnh kitchen thành `gong` (score 0.42) hoặc `dining_table` (score 0.07) | Đọc kỹ guideline để chọn duy nhất một nhãn ưu tiên theo chủ thể chính hoặc diện tích lớn nhất | Kiểm tra xem nhãn được gán có đúng taxonomy quy định và đúng quy tắc ưu tiên chủ thể không |
| Phát hiện vật thể | Tập hợp các Bounding Box `[x_min, y_min, x_max, y_max]` kèm Class Name cho từng vật thể | Bàn tay bị cắt mép ở góc trái bị model đoán là `person 0.61`; bỏ sót nhiều chai lọ nhỏ trên giá treo | Vẽ bounding box bao khít từng vật thể nhìn thấy, không cắt lẹm vào đối tượng | Soát lỗi dương tính giả (hộp vẽ thừa/sai), kiểm tra độ khít của hộp và phát hiện các vật bị bỏ sót |
| Instance segmentation | Tập hợp các đa giác khép kín (`polygon_xy`) kèm Class Name và `instance_id` cho từng cá thể | Chiếc bàn `dining table` bị mask đè lấn lên các bát đĩa trên mặt bàn; chùm lá khô treo bị gán nhầm là `potted plant 0.63` | Chấm các điểm polygon men theo đường viền thực tế của từng cá thể đối tượng | Kiểm tra độ mịn và độ chính xác của đường biên mask, đảm bảo các instance chạm nhau không bị gộp chung |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
Chỉ sử dụng hình ảnh công khai có giấy phép mở được cung cấp sẵn trong bài lab; tuyệt đối không tải lên dữ liệu cá nhân (khuôn mặt, biển số, thông tin nhạy cảm) hoặc dữ liệu nội bộ của doanh nghiệp lên Colab và GitHub công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
Dừng ngay việc xử lý và báo cáo trực tiếp cho Giảng viên hướng dẫn / Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
