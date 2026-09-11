# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
Record hạng 1 là: class_id = 468 <br>
class_name = "cab" <br>
rank = 1 <br>
score = 0.510915 <br>
taxonomy_name = "ImageNet-1K" <br>
Nó có nghĩa là: Model dự đoán toàn bộ ảnh traffic thuộc lớp cab (taxi) với score khoảng 51.09%, và đây là dự đoán có score cao nhất trong các lớp được trả về. Lưu ý: Đây là image classification, nên prediction nói về toàn bộ ảnh, không nói vị trí của từng vật thể. <br>
- Record này mô tả toàn ảnh như thế nào? <br>
class_name = "cab" → model dự đoán toàn ảnh thuộc lớp "cab" (taxi). <br>
score = 0.510915 → model có mức điểm khoảng 51.09% cho dự đoán này. <br>
rank = 1 → đây là lớp có score cao nhất trong các prediction được trả về. <br>
class_id = 468 → ID của lớp cab trong taxonomy ImageNet-1K. <br>
taxonomy_name = "ImageNet-1K" → cho biết class cab được hiểu theo bộ nhãn ImageNet-1K. <br>
- Ai định nghĩa class list mà checkpoint có thể dự đoán? <br>
ImageNet-1K có 1.000 lớp. Checkpoint yolo11n-cls.pt được train cho bài toán classification với class mapping tương ứng. <br>
Vì vậy: <br>
Model không tự tạo ra class list. Class list đã được xác định bởi taxonomy/dataset dùng để train checkpoint.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? <br>
 Field           : Ý nghĩa                                        <br>
 class_id        : ID định danh ổn định của lớp                   <br>
 taxonomy_name   : Cho biết ID/tên đó thuộc hệ thống phân loại nào<br>

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? <br>
Đây là điểm rất quan trọng với image classification.

    Guideline cần quy định:

    Nếu ảnh có nhiều chủ thể, label phải đại diện cho cái gì?

    Ví dụ ảnh traffic có:

    taxi + bus + motorcycle + car

    Thì cần thống nhất:

    Chọn chủ thể chính?
    Chọn đối tượng nổi bật nhất?
    Chọn toàn cảnh?
    Hay cho phép multi-label classification?

    Trong dữ liệu hiện tại, model đang làm single-label ranking: đưa ra nhiều class có score nhưng mỗi prediction vẫn là một class cho toàn ảnh.

    Vì vậy guideline nên quy định rõ:

    Một ảnh có nhiều đối tượng thì ground truth phải xác định tiêu chí chọn nhãn chính, hoặc phải chuyển sang multi-label nếu muốn ghi nhận nhiều đối tượng cùng lúc.

    Nếu không có guideline, người gán nhãn có thể hiểu khác nhau → ground truth không nhất quán → đánh giá model bị sai.
- Vì sao model score không phải ground truth? <br>

    Vì:
    Model score = niềm tin/dự đoán của model

    còn:

    Ground truth = nhãn đúng được xác định bởi dataset/guideline/con người.

    Ví dụ:

    Model:
    cab = 0.510915

    Không có nghĩa:

    "Ảnh này chắc chắn là cab."

    Nó chỉ có nghĩa:

    Model đánh giá cab là khả năng/dự đoán đứng đầu với score 0.510915.

    Ground truth có thể là:

    ground_truth = minibus

    khi đó model đã dự đoán sai, dù cab có score cao nhất.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): <br>
class_name = "person" → model phát hiện một người. <br>
score = 0.912625 → model có score khoảng 91.26% cho prediction này. <br>
bbox_xyxy = [385.33, 69.24, 498.92, 348.92] → tọa độ bounding box. <br>
bbox_width = 113.58 → chiều rộng của box khoảng 113.58 pixel. <br>
bbox_height = 279.68 → chiều cao của box khoảng 279.68 pixel. <br>
- Diễn giải vị trí box bằng lời:
Bounding box của người nằm ở khu vực bên phải của ảnh, bắt đầu khoảng từ x = 385, y = 69 và kéo xuống đến x = 499, y = 349.

- So sánh số prediction ở hai threshold:
Tăng threshold từ 0.35 lên 0.50 làm số prediction giảm từ 11 xuống 6.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? <br>
Threshold thấp: 0.35 <br>
Ưu điểm: <br>
Giữ được nhiều object hơn. <br>
Độ bao phủ (recall) cao hơn. <br>
Có cơ hội giữ lại những object khó phát hiện hoặc có score thấp. <br>
Nhược điểm: <br>
Có nhiều prediction hơn. <br>
Có thể xuất hiện nhiều false positive hơn. <br>
Reviewer phải kiểm tra nhiều box hơn. <br>
Threshold cao: 0.50 <br>
Ưu điểm: <br>
Ít prediction hơn. <br>
Ít prediction yếu cần reviewer kiểm tra. <br>
Giảm khối lượng review. <br>
Thường giúp tăng precision. <br>
Nhược điểm: <br>
Có thể loại bỏ những object thật nhưng model chỉ cho score thấp. <br>
Độ bao phủ (recall) có thể giảm. <br>
- Đề xuất một quy tắc box chặt: <br>
Guideline có thể quy định: <br>
Bounding box phải bao sát object cần phát hiện, nhưng không được cắt mất phần nhìn thấy của object. <br>
Cụ thể: <br>
Box phải bao phủ toàn bộ phần object nhìn thấy. <br>
Không lấy quá nhiều background. <br>
Không bao gồm các object khác nếu không thuộc object đang label. <br>
Các cạnh box nên nằm sát biên của object. <br>
Nếu object có hình dạng không phải hình chữ nhật, vẫn dùng bounding rectangle bao quanh phần object nhìn thấy. <br>
Không mở rộng box chỉ để làm box "đẹp" hoặc dễ nhìn hơn. <br>

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Nếu một phần object bị object khác che: <br>
Box nên bao quanh phần object có thể nhìn thấy, thay vì đoán phần bị che phía sau. <br>
Tuy nhiên guideline phải quy định thêm: <br>
Che khuất bao nhiêu phần thì vẫn được label? <br>
Khi nào object còn quá ít để xác định? <br>
Có cần đánh dấu occluded = true hay không? <br>
Có cần escalation nếu không chắc class? <br>

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

1. Một record (instance_id, class_name, score, số điểm và một phần polygon_xy)

    instance_id = "kitchen-001": ID riêng của instance/object cụ thể trong ảnh. <br>
    class_name = "person": object được dự đoán là người. <br>
    score = 0.899318: điểm dự đoán của model. <br>
    polygon_point_count = 348: polygon được biểu diễn bằng 348 điểm. <br>
    polygon_xy: danh sách các điểm [x, y] tạo thành đường biên của mask. <br>
    Ảnh có kích thước 640 × 427 pixel và taxonomy là COCO-80. 

    Ví dụ khác, kitchen-005 là dining table, score 0.578416 và có tới 598 điểm polygon, cho thấy số điểm có thể khác nhau tùy hình dạng mask.

2. Polygon bổ sung chi tiết gì so với box?

    Bounding box chỉ cho biết hình chữ nhật bao quanh object.

    Ví dụ:

    bbox = [x1, y1, x2, y2]

    Trong khi polygon mô tả đường biên của object bằng nhiều điểm:

    polygon = [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        ...
    ]

    Vì vậy:

    Box: biết object nằm trong vùng hình chữ nhật nào.
    Polygon: biết phần pixel/hình dạng cụ thể của object nằm ở đâu.
    Polygon có thể bám theo các cạnh cong, góc, hình dạng không chữ nhật.
    Polygon giúp phân biệt phần object với background tốt hơn box.

    Ví dụ trong evidence, kitchen-001 có bbox khoảng [385.45, 66.44, 498.02, 348.58] nhưng mask được mô tả chi tiết bằng 348 điểm polygon.

3. instance_id dùng để làm gì và không phải loại ID nào?

    instance_id dùng để phân biệt từng object cụ thể trong cùng một ảnh.

    Ví dụ:

    kitchen-001 → person
    kitchen-002 → bowl
    kitchen-003 → bowl
    kitchen-004 → potted plant

    Hai object đều có thể cùng class_name = "bowl" nhưng phải có instance_id khác nhau.

    Ví dụ evidence có:

    kitchen-002 → bowl
    kitchen-003 → bowl
    kitchen-006 → bowl
    kitchen-010 → bowl

    Các object này cùng class nhưng là những instance khác nhau.

    instance_id không phải:
    ❌ class_id: ID của loại object trong taxonomy.
    ❌ coco_image_id: ID của ảnh.
    ❌ score: độ tin cậy của prediction.
    ❌ tracking_id: ID dùng để theo dõi một object qua nhiều frame của video.

    Có thể hiểu đơn giản:

    class_id    → object thuộc loại gì?
    instance_id → object cụ thể nào?

    Ví dụ:

    class_id = 45
    class_name = bowl

    instance_id = kitchen-002 → cái bát thứ 1
    instance_id = kitchen-003 → cái bát thứ 2
4. Đề xuất một quy tắc biên mask

    Có thể quy định:

    Mask phải bám sát đường biên nhìn thấy của object, bao phủ phần object có thể xác định được và hạn chế tối đa background.

    Cụ thể:

    Điểm polygon phải nằm trên hoặc sát biên của object.
    Không đưa vùng background rõ ràng vào mask.
    Không bỏ sót phần object đang nhìn thấy rõ.
    Không mở rộng mask ra ngoài object chỉ để tạo hình đẹp.
    Nếu object có đường cong hoặc hình dạng phức tạp, polygon nên follow theo hình dạng thực tế.
    Polygon phải nằm trong phạm vi ảnh:
    0 <= x <= image_width
    0 <= y <= image_height

    Ví dụ ảnh kitchen có kích thước 640 × 427, nên mask không được vượt ra ngoài vùng ảnh.

    Mục tiêu: mask phải đại diện cho phần object thực sự nhìn thấy, không phải vùng background bao quanh object.

5. Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

    Đây là các trường hợp cần guideline rõ ràng vì ranh giới object không còn chắc chắn.

    Vùng mờ

    Nếu biên object bị mờ:

    Nếu vẫn xác định được biên → vẽ mask theo biên nhìn thấy.
    Nếu không xác định được chính xác → cần quy định mức sai lệch cho phép.
    Nếu quá khó xác định → escalate cho reviewer.
    Hai object tiếp xúc nhau

    Nếu hai object chạm nhau:

    Không gộp hai object thành một mask.
    Mỗi object phải có instance riêng.
    Biên giữa hai object cần được xác định theo guideline.
    Nếu không thể xác định object nào thuộc pixel nào → reviewer quyết định.
    Object bị che khuất

    Nếu object bị một object khác che:

    Guideline cần quyết định:

    Mask chỉ bao phần nhìn thấy hay được phép suy đoán phần bị che?
    Khi nào đánh dấu occluded?
    Mức độ che khuất nào thì vẫn label?
    Khi nào phải escalation?

    Đề xuất an toàn:

    Nhìn rõ → tạo mask bình thường
    Bị che nhưng vẫn nhận dạng được → mask phần nhìn thấy + đánh dấu occluded
    Biên không rõ → reviewer kiểm tra
    Quá mờ/che khuất → escalation
    Không thể xác định → không tự đoán

    Lưu ý: file evidence hiện có các trường như instance_id, class_name, score, bbox và polygon, nhưng không định nghĩa sẵn quy tắc occluded hay blur. Vì vậy các quy tắc xử lý vùng mờ/tiếp xúc/che khuất ở trên là guideline đề xuất, cần được project xác nhận trước khi áp dụng.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

# Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                    | Đơn vị/định dạng ground truth                                | Lỗi hoặc điểm mơ hồ quan sát được                                                                   | Annotator làm gì?                                                                                                       | Reviewer xem gì?                                                                                             |
| ------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Phân loại ảnh**         | 1 ảnh → 1 `class` (hoặc nhiều class nếu guideline cho phép). | Ảnh có nhiều chủ thể; không rõ class chính; các class gần giống nhau.                               | Gán class theo guideline; nếu không chắc chắn thì đánh dấu để review.                                                   | Kiểm tra class có phù hợp với toàn ảnh và guideline không; xử lý trường hợp mơ hồ.                           |
| **Phát hiện vật thể**     | Mỗi object → 1 `class` + 1 bounding box `bbox_xyxy`.         | Box quá rộng/hẹp; box chứa background; object bị che khuất hoặc cắt mép ảnh.                        | Gán class và vẽ box chặt quanh phần object nhìn thấy; không tự đoán phần bị che.                                        | Kiểm tra class, vị trí và độ chặt của box; kiểm tra object bị bỏ sót và các trường hợp occlusion/truncation. |
| **Instance segmentation** | Mỗi instance → 1 `class` + 1 polygon/mask `polygon_xy`.      | Mask lệch biên; ăn vào background; bỏ sót object; hai object tiếp xúc bị gộp; biên bị mờ/che khuất. | Gán class và vẽ polygon bám sát biên object nhìn thấy; giữ các instance riêng biệt; trường hợp không rõ thì escalation. | Kiểm tra class, từng instance và đường biên mask; kiểm tra mask bị gộp/tách sai và vùng occlusion/blur.      |


## 5. An toàn dữ liệu

* **Một quy tắc bảo vệ dữ liệu:** Không chia sẻ, sao chép hoặc sử dụng dữ liệu/ảnh ngoài phạm vi được cấp quyền; chỉ truy cập và xử lý dữ liệu cần thiết cho công việc.

* **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:** **Reviewer hoặc người phụ trách dự án/data owner** để xác nhận và xử lý.


## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
