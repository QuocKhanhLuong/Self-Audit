# A-3D — đặc tả triển khai và review

Ngày: 10.11.2026. Trạng thái: **ĐỀ XUẤT ĐỂ TRIỂN KHAI; CHƯA XÁC NHẬN HIỆU QUẢ.**

Tài liệu này lưu phần thiết kế cập nhật sau cuộc trao đổi chọn teacher và kiểm tra độ phù hợp với student. Đây là yêu cầu cho một lần triển khai tiếp theo, không phải báo cáo rằng code đã đáp ứng các yêu cầu. Chưa có kết quả chứng minh đạt 96 Dice, SOTA, hay novelty.

## 1. Tên gọi và nguồn gốc

- **A** là toàn bộ phương án: một teacher có nhánh vùng/biên → hướng dẫn chọn phép đo → mã do student tạo → decoder đọc mã.
- **A1.tex, A2.tex, A3.tex là ba trang của cùng A**: tổng thể, representation, cách học. `A2.tex` không có nghĩa là “kiến trúc A phiên bản 2”.
- **A-3D** là tên dùng trong tài liệu này cho bản đề xuất cập nhật: cụ thể hóa teacher 3D, student đọc thông tin cục bộ 3D và quy trình so sánh teacher trên cùng student. Nó kế thừa cơ chế mã có kiểu và K1 của A.

Nguồn thiết kế đang có trên local:

- [A.1 — tổng thể](sections-paper-final/A1.tex)
- [A.2 — representation](sections-paper-final/A2.tex)
- [A.3 — cách học](sections-paper-final/A3.tex)
- [Nguồn LaTeX đầy đủ](main.tex)
- [Phạm vi khoa học của bản hình/LaTeX](README-final.md)

Repository triển khai để kiểm tra: `/Users/alvinluong/Self-Audit`. Trước khi sửa code phải đọc hướng dẫn của checkout đang dùng và kiểm tra lại trạng thái hiện tại; không giả định tài liệu này phản ánh mọi thay đổi về sau.

Snapshot đã kiểm tra khi lưu tài liệu: nhánh `main`, HEAD `f01add12524cf73bb15675ecfeadc1014e420210`. Code tham chiếu có trên local, nhưng chưa có triển khai A-3D theo đặc tả này:

- `self_audit/candidate_a/model.py`: student 2D/2.5D, Conv2d; head giá trị dùng `features.mean(dim=(-2,-1))`. Decoder đã có hợp đồng chỉ đọc mã.
- `self_audit/candidate_a/teacher.py`: teacher đầu vào `[B,3,H,W]`, nhánh vùng/biên riêng; chưa phải teacher volume 3D.
- `self_audit/candidate_a/bank.py`: hình học phép đo trong mặt phẳng, chưa có khoảng cách 3D theo z-spacing.
- `configs/candidate_a/teacher_pretrain.json`: backbone mặc định `dinov3_convnext_tiny`, không phải MedNeXt/RefineFormer.

Đây là kiểm tra cấu trúc và config, chưa phải một lần review toàn bộ implementation hoặc chạy lại test suite. Repository có file untracked ngoài phạm vi; lần này không sửa chúng.

## 2. A-3D khác bản đang có ở đâu?

| Thành phần | A trong tài liệu / code tham chiếu | Yêu cầu A-3D |
|---|---|---|
| Teacher | A mô tả một teacher tách nhánh, chưa khóa backbone. Code tham chiếu chưa chứng minh teacher mạnh trên cardiac MRI. | Dùng MedNeXt-L 3D k3 làm cấu hình tham chiếu ban đầu; RefineFormer3D Standard là ứng viên thách thức hiệu năng, cần xác minh/reproduce trước khi xếp hạng. |
| Ngữ cảnh ảnh | Code student tham chiếu dùng Conv2d và đầu vào nhiều lát như kênh. | Teacher và student nhận cùng ngữ cảnh volume/patch 3D, có spacing và biến đổi tọa độ rõ ràng. Giữ kết quả 2D/2.5D thành track riêng. |
| Đọc feature để tạo mã | A cho phép head phụ thuộc ID; code tham chiếu dùng global mean pooling cho mọi ID. | Query vùng đọc feature của chính vùng đó; query biên đọc feature tại điểm/lân cận đó; cả hai được bổ sung ngữ cảnh toàn cục. |
| Mã và decoder | ID có nghĩa cố định, giá trị student, decoder chỉ đọc mã. | Giữ nguyên hợp đồng này, mở rộng tọa độ và phép đo sang 3D; đo đường cong chất lượng theo dung lượng mã. |
| Cách chọn teacher | Chưa có so sánh thực nghiệm công bằng cho hệ A. | Teacher phải vừa segment tốt vừa cải thiện student so với đối chứng không có teacher, dưới cùng budget và protocol. |

**Điểm không thay đổi:** teacher chỉ gợi ý ID; giá trị ghi vào mã vẫn do student dự đoán. A-3D không tự động trở thành fusion feature của hai encoder. Ý tưởng ban đầu về representation chung mang trực tiếp thông tin từ cả teacher và student vẫn là câu hỏi nghiên cứu riêng, chưa được phương án này giải quyết đầy đủ.

## 3. Hợp đồng kiến trúc

### 3.1 Một teacher, hai nhánh xử lý

Luồng teacher: `MRI 3D → trunk chung → nhánh vùng / nhánh biên`.

- Giữ đường segmentation gốc của backbone làm đường vùng. Không thay decoder gốc bằng một head yếu rồi tiếp tục gán cho model điểm Dice công bố.
- Nhánh biên có block xử lý riêng, đọc được feature độ phân giải cao và ngữ cảnh; xuất segmentation và trường khoảng cách có dấu. Không coi hai linear head trên cùng tensor là đã tạo hai chuyên môn bổ sung.
- MedNeXt-L k3 là điểm xuất phát để triển khai, chưa phải người thắng. Adapter phải cho phép so sánh RefineFormer mà không đổi student, bank, bitrate hoặc chế độ nhãn.
- Ghi phiên bản upstream, cấu hình, nguồn/hash checkpoint và cách khởi tạo. Thiếu checkpoint phải báo thiếu; model khởi tạo ngẫu nhiên chỉ được ghi là chưa huấn luyện.

Teacher học bằng:

$$
\mathcal L_T=\mathcal L_{seg}(P_R,y)+\mathcal L_{seg}(P_B,y)
+\lambda_d\,\operatorname{mean}|\bar d_B-\bar d(y)|,
\qquad \mathcal L_{seg}=\mathcal L_{CE}+\mathcal L_{Dice}.
$$

Đánh giá chất lượng vùng/biên trước và sau khi thêm nhánh. Pretrain teacher, làm nóng nhánh mới, sau đó mới thử cập nhật teacher và student xen kẽ. Snapshot teacher cố định trong mỗi lượt tạo target; loss student không cập nhật teacher. Phải có đối chứng teacher đóng băng.

### 3.2 Student đọc cục bộ trước khi tạo mã

Student encoder tạo feature 3D cục bộ $F_{loc}$ và ngữ cảnh $g$. Head được phân biệt bằng kiểu phép đo:

$$
\hat q^R_{S,j}=h_R\big(\operatorname{Pool}_{V_j}(F_{loc}),g,e(j)\big),
\qquad
\hat q^B_{S,j}=h_B\big(\operatorname{Sample}_{p_j}(F_{loc}),g,e(j)\big).
$$

$e(j)$ mô tả kiểu đo, lớp, tọa độ và tỉ lệ. `Pool` tổng hợp feature đúng vùng; `Sample` lấy tại điểm hoặc lân cận đã khai báo. Không dùng chỉ một global mean vector giống nhau cho mọi vùng rồi coi như đã giữ thông tin không gian.

Kiến trúc student là encoder nhỏ + head phép đo + policy + decoder tọa độ. Không cần encoder giống teacher. Không có đường skip từ feature ảnh vào decoder. Số tầng, width, patch size, bank size và số bit là tham số thí nghiệm phải ghi trong config; chưa có giá trị được xác nhận tối ưu.

### 3.3 Mã có kiểu và hình học

$$
j=(\mathrm{type},c,\mathrm{location/region},\mathrm{scale}),
\qquad R_t=\{(j,Q(\hat q_{S,j}))\}_{j\in J_t}.
$$

$Q$ là lượng tử hóa với số bit khai báo. Cùng ID phải có cùng nghĩa ở teacher/student, không chứa tên encoder. Với TRAIN label hợp lệ:

$$
q^{R*}_j=\frac{\sum_{v\in V_j}w_v\mathbf1[y_v=c]}{\sum_{v\in V_j}w_v},
\qquad q^{B*}_j=\operatorname{clip}\!\left(d_c(y,p_j)/r,-1,1\right).
$$

- $w_v$ là trọng số thể tích/tích phân đã xác định; $d_c$ tính bằng mm, âm bên trong, dương bên ngoài; $r>0$ cố định trước đánh giá.
- Lớp vắng: target vùng bằng 0, target khoảng cách bằng +1. Phân biệt lớp vắng thật với cấu trúc bị cắt ngoài patch; phải khai báo cách tạo target từ volume và quy tắc query ngoài miền hợp lệ.
- Vùng rỗng hoặc query ngoài miền phải bị loại/mask theo quy tắc rõ ràng, không âm thầm chia cho 0 hay đổi tọa độ.
- Theo dõi affine, spacing, crop, resampling và augmentation. Chuẩn hóa tọa độ để đưa vào mạng không được làm mất đơn vị vật lý của phép đo.
- Tính đủ bit ID, giá trị và metadata phụ thuộc mẫu, kể cả mã khởi đầu $R_0$. Nếu có vector latent bổ sung, phải tính dung lượng vector đó và coi là biến thể được báo riêng.

Decoder chỉ nhận mã, tọa độ truy vấn 3D và hình học công khai cần thiết:

$$\hat y(p)=D(R_{final},p,\mathrm{geometry}).$$

Không truyền $F_{loc}$, $g$, feature teacher, ảnh gốc hoặc cache ẩn vào decoder. Global context có thể giúp encoder/head/policy nhưng không được đi vòng qua mã. Phải hỗ trợ decode theo chunk và đo chi phí toàn hệ; chia chunk giảm bộ nhớ, không mặc nhiên giảm tổng phép tính.

## 4. K1 và loss student

Teacher đề xuất ID từ sai khác teacher/student được chuẩn hóa theo kiểu đo. Trộn đề xuất teacher, student và thăm dò; luôn có STOP. Chỉ thử ID chưa dùng, hợp lệ và còn đủ bit. Target policy được tạo bằng cách thử **giá trị student đã lượng tử hóa**, cùng decoder và cùng trạng thái:

$$
R'_{t,j}=R_t\cup\{(j,Q(\hat q_{S,j}))\},
\quad U_{t,j}=\mathcal L_{seg}(D(R_t),y)-\mathcal L_{seg}(D(R'_{t,j}),y)-\lambda_c C_j.
$$

$C_j$ là chi phí bit tăng thêm. STOP giữ nguyên mã, utility bằng 0. Khi tính trial, giữ các điều kiện ngẫu nhiên nhất quán để không biến nhiễu decoder thành lợi ích query.

$$
\pi_t^*=\operatorname{softmax}_{\mathcal C_t}(\operatorname{stopgrad}(U_t)/\tau),
\quad\mathcal L_\pi=CE(\pi_t^*,\pi_S|_{\mathcal C_t}),
$$

$$
\mathcal L_S=\mathcal L_{seg}+\lambda_R\mathcal L_{q,R}
+\lambda_B\mathcal L_{q,B}+\lambda_\pi\mathcal L_\pi.
$$

Hai loss $q$ là sai số tuyệt đối trung bình, chuẩn hóa theo kiểu và so với phép đo từ **TRAIN GT**, không phải mặc định matching giá trị teacher. Policy phải được chuẩn hóa trên đúng tập ứng viên đã trial; target utility bị ngắt gradient. Quá trình học phải có rollout bằng policy student để giảm chênh lệch train/inference. Nếu dùng straight-through estimator cho lượng tử hóa khi học, phải khai báo; đánh giá và tính utility sử dụng giá trị lượng tử hóa thật.

Inference chỉ dùng student để chọn query, điền mã và decode; không teacher, không GT, không tính utility bằng nhãn. STOP hợp lệ từ bước 0; không lặp ID hoặc vượt budget.

## 5. Phạm vi dữ liệu và tiêu chí khoa học

Đọc `AGENTS.md`, `ACTIVE_PIPELINE.json`, `docs/METHOD.md` tại checkout trước khi tích hợp. Canonical pipeline của Self-Audit có giới hạn nhãn riêng; specification này không cấp quyền đọc nhãn ngoài giới hạn đó. Nếu thực nghiệm teacher full-supervised khác với regime hiện hành, phải khai báo thành protocol riêng, không trộn số liệu.

ACDC và **M&Ms gốc** là hai tập mục tiêu. Ghi split theo bệnh nhân, lớp, ED/ES, 2D/3D, preprocessing, metric aggregation, postprocessing/TTA, external pretraining và ngân sách nhãn. Không thay M&Ms bằng M&Ms-2 hoặc một điểm trung bình gộp nhiều bộ dữ liệu.

Thứ tự kiểm tra đề xuất:

1. **Mã có đủ khả năng giữ hình dạng không?** Dùng giá trị phép đo từ TRAIN GT để làm chẩn đoán tái dựng với các budget khác nhau. Đây là diagnostic có thông tin ưu tiên; không phải kết quả triển khai hay cận trên toán học. Cố định protocol chọn ID và phân biệt ảnh hưởng của chọn ID với chất lượng giá trị.
2. **Teacher có đủ tốt không?** Đo Dice theo lớp, chỉ số biên theo mm và kết quả từng vendor trước/sau tách nhánh. Chọn metric chính và quy tắc xử lý lớp vắng trước khi chạy. Điểm công bố khác split không phải kết quả reproduce.
3. **Teacher có giúp student không?** Cùng student, bank, bitrate và ngân sách huấn luyện: không teacher; MedNeXt; RefineFormer nếu đã có triển khai được kiểm chứng. Đối chứng không teacher vẫn học vùng/biên và dùng cùng cách chấm K1; chỉ bỏ nguồn đề xuất teacher, giữ số ứng viên trial tương đương.
4. **Cơ chế nào có ích?** So global-only với local+global; chỉ vùng với vùng+biên ở cùng bitrate; teacher đóng băng với cập nhật xen kẽ. Báo chi phí inference toàn student và kết quả theo seed/bệnh nhân, không chỉ tham số encoder.

Điểm cần đo thêm: lợi ích của ứng viên teacher so với ứng viên student/thăm dò, tỉ lệ chọn từng kiểu, STOP và chất lượng theo bit. Hai nhánh cùng đạt loss thấp chưa chứng minh chúng bổ sung nhau. Student nén tốt chưa chứng minh teacher đóng góp.

## 6. Checklist review code

Mỗi mục phải trả lời **PASS / FAIL / CHƯA KIỂM TRA**, kèm file:dòng và test/log nếu có. PASS test tổng hợp chỉ xác nhận hành vi phần mềm tương ứng.

| ID | Yêu cầu review | Bằng chứng tối thiểu |
|---|---|---|
| A3D-01 | Track 3D thật và cùng ngữ cảnh teacher/student | Shape contract, data flow, config; không đổi tên Conv2d thành 3D. |
| A3D-02 | Giữ đường vùng teacher và có xử lý biên riêng | Forward path, init/checkpoint mapping, output contract; đánh giá chất lượng là bước riêng. |
| A3D-03 | Query đọc đúng vùng/điểm | Test thay đổi cục bộ tại các vị trí khác nhau; xác nhận sampling/pooling với affine/spacing. |
| A3D-04 | Target vùng/biên đúng nghĩa | Hình tổng hợp biết trước thể tích và khoảng cách; test anisotropic spacing, lớp vắng, crop và vùng rỗng. |
| A3D-05 | Mã chứa giá trị student và tính đủ dung lượng | Kiểm tra nguồn giá trị, lượng tử hóa, cost và metadata; không đưa teacher q vào trial. |
| A3D-06 | Decoder thật sự chỉ đọc mã | Sau khi giữ mã và geometry cố định, đổi/xóa feature cache không đổi kết quả; decoder chạy độc lập. |
| A3D-07 | K1 thử cùng trạng thái và đúng ứng viên | Test utility cải thiện/xấu đi/STOP, cost, detach và policy renormalization. |
| A3D-08 | Không rò nhãn/gradient | Phân quyền split và target source; backward student không tạo gradient teacher. |
| A3D-09 | Rollout và export không cần teacher/GT | Test STOP bước 0, hết budget, ID trùng; inference với teacher và label loader bị loại. |
| A3D-10 | So sánh teacher công bằng | Config diff chỉ ra các phần cố định và phần được đổi; cùng student/bank/regime/bitrate. |
| A3D-11 | Tính đúng chi phí | Profile encoder + query + policy + toàn bộ decoder, shape và phần cứng được ghi. |
| A3D-12 | Báo cáo đúng mức bằng chứng | Tách implementation, test tổng hợp, trained checkpoint và số liệu bệnh nhân; không tự suy ra SOTA/96/novelty. |

## 7. Prompt giao Codex

### Triển khai

> Đọc file đặc tả này cùng hướng dẫn tại `/Users/alvinluong/Self-Audit`. Kiểm tra checkout, `ACTIVE_PIPELINE.json`, `docs/METHOD.md` và code `self_audit/candidate_a` hiện tại. Lập bảng khoảng cách giữa code hiện tại và A3D-01 đến A3D-12, rồi triển khai phần còn thiếu trong worktree riêng, giữ tương thích bản tham chiếu. Ưu tiên MedNeXt-L 3D k3, giao diện teacher thay được, student local-query 3D, code-only decoder và K1 đúng hợp đồng. Không âm thầm đổi phương án thành feature fusion hoặc teacher-value matching. Viết/chạy test hành vi cần thiết; ghi rõ phần chưa có weights, dữ liệu hoặc thực nghiệm. Không tự mở chiến dịch GPU, tải dữ liệu bệnh nhân hoặc thay job đang chạy. Giao code, config, hướng dẫn chạy, kết quả test và các điểm còn thiếu; chưa có số liệu thì không kết luận hiệu quả.

### Review độc lập

> Review code so với file đặc tả này và hướng dẫn repository, không chỉ đọc README hoặc báo cáo của người triển khai. Với A3D-01 đến A3D-12, dẫn file:dòng và bằng chứng test/log; ưu tiên lỗi leakage, feature bypass, target/geometry, lượng tử hóa/budget, K1/gradient và train–inference mismatch. Phân biệt bug, khác biệt thiết kế và câu hỏi cần thực nghiệm. Không đánh dấu PASS cho chất lượng teacher/student chỉ vì test tổng hợp chạy qua. Trả findings theo mức ảnh hưởng và danh sách chưa kiểm tra.

## 8. Giới hạn của lần lưu tài liệu này

Lần này chỉ tạo đặc tả local để giao việc và review. Không sửa source LaTeX đang mở, không triển khai A-3D, không chạy huấn luyện và không xác nhận toàn bộ code tham chiếu đạt checklist. Các chi tiết chưa khóa như kích thước mạng, bank, bitrate và lịch học phải được ghi rõ trong config thí nghiệm tiếp theo; không gọi chúng là lựa chọn đã được chứng minh tốt nhất.
