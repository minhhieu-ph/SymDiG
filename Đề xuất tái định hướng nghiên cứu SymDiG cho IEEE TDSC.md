# Đề xuất tái định hướng nghiên cứu SymDiG cho IEEE TDSC

Oct 29, 2026 · @Hieu Pham

## Tóm tắt điều hành

Đề xuất: bỏ hướng "ghép thêm thành phần" và viết lại bài thành một nghiên cứu **chẩn đoán + khắc phục tối giản + đánh giá đáng tin cậy** về GNN đa đồ thị có hướng trên lưu lượng NetFlow dày đặc. Hướng này trả lời trực tiếp mọi điểm editor TIFS đã nêu.

- **Vì sao phải đổi hướng.** DiG-RSB bị desk reject vì "ghép kỹ thuật có sẵn". Bản SymDiG chỉ đổi tên các thành phần đó (L\_ortho, L\_MAC) và dựa vào số liệu giả định, nên sẽ gặp đúng lời từ chối cũ, thêm rủi ro liêm chính.
- **Tài sản thật mạnh nhất.** Câu chuyện Multi-GNN \~45–50% → loader đối xứng \~70% → dual-stream + attention \~94% macro-F1 trên NF-ToN-IoT-v2. Thêm vào đó là lỗi mất seed trong code Multi-GNN, đã xác nhận.
- **Đóng góp mới.** (1) Chẩn đoán định lượng vì sao pipeline GNN đa đồ thị thất bại trên đồ thị dày; (2) cách khắc phục tối giản với bậc thang ablation tách bạch từng bước; (3) đánh giá trong một hệ quy chiếu duy nhất, có XGBoost tinh chỉnh, chia theo host và đo khả năng triển khai.
- **Loại bỏ.** GCL/RSM, DWB/MAC, L\_ortho và mọi con số chưa chạy thật.
- **Kỳ vọng.** Nếu ưu thế ở bước kiến trúc vẫn giữ được khi kiểm soát đầy đủ, khả năng được TDSC chấp nhận sau revision ước tính 25–35% (ước lượng chủ quan). Thời gian thực nghiệm khoảng 6–8 tuần trước khi viết lại.

## Hiện trạng

Bài có nền tảng thật, nhưng phần đóng góp được viết không khớp với phần có bằng chứng. Bảng dưới phân loại theo mức độ tin cậy.

| Thành phần | Nguồn | Mức chắc chắn | Giữ / bỏ |
| --- | --- | --- | --- |
| Kết quả chính DiG-RSB (macro-F1: BoT 95.33, ToN 94.12, IDS2018 82.09, UNSW 67.47) | Chạy thật | Cao, nhưng chưa rõ độ phủ tập test | Giữ, chạy lại trong pipeline mới |
| Ablation loss: CE 95.71 và WCE 95.76 > DWB 94.12 trên ToN | Chạy thật (Bảng IV DiG-RSB) | Cao | Giữ đúng số; dùng làm lý do bỏ phần loss |
| Tài nguyên IP so với IP+PORT trên UNSW (Hình 5 DiG-RSB) | Đo thật | Cao | Giữ, mở rộng |
| Multi-GNN \~45–50% → loader đối xứng \~70% → dual-stream \~94% (ToN) | Trí nhớ | Trung bình | Trục chính của bài, chạy lại có kiểm soát |
| Code Multi-GNN chỉ tính loss trên seed được lấy mẫu; bù seed thiếu khi test chỉ cho Small\_J/Small\_Q | Đọc code (training.py, train\_util.py) | Cao | Giữ, định lượng độ phủ |
| XGBoost macro-F1 \~90% (đủ dữ liệu), >80% (2%) trên ToN | Trí nhớ | Trung bình | Chạy lại, kiểm tra cổng |
| GCL/RSM (pretraining) | Chạy thật, không cải thiện rõ; LR Scatter Encoder 2e-7 | — | Bỏ |
| L\_ortho, L\_MAC, S\_cos/Erank, Bảng 3/7/9/10/11 của SymDiG | Giả định | — | Bỏ |
| Baseline E-GraphSAGE / Anomal-E / E-ResGAT / FeCoGraph | Chép từ Bảng III FeCoGraph (protocol 2%, chia 30/70) | Không cùng hệ quy chiếu | Thay bằng chạy lại |

**Bài học từ desk reject TIFS:** editor không phản đối hướng đồ thị chỉ dùng IP. Họ phản đối việc mỗi claim không có thí nghiệm riêng chứng minh, và baseline không khớp với bài gốc. Họ còn gợi ý sẵn hướng đi: triển khai, thời gian thực, khả năng mở rộng, drift, thay đổi topo.

**Việc phải sửa ngay trong draft:** Bảng 8 của SymDiG đã hạ CE (95.71 → 91.71) và WCE (95.76 → 92.76) so với số thật. Phải khôi phục về số gốc.

## Câu hỏi nghiên cứu và giả thuyết

Câu hỏi trung tâm: **trên lưu lượng NetFlow dày đặc, khi được lấy mẫu và đánh giá đúng, cấu trúc đa đồ thị có hướng có đem lại khả năng phát hiện vượt trên đặc trưng của từng flow không, và lựa chọn thiết kế nào quyết định điều đó?**

Bốn giả thuyết dưới đây được chốt trước khi chạy. Tiêu chí bác bỏ không được sửa sau khi đã thấy kết quả test.

| Giả thuyết | Phát biểu | Đo bằng | Bác bỏ nếu |
| --- | --- | --- | --- |
| H1: Độ tin cậy của đánh giá | Trên đa đồ thị dày, mất seed và lấy mẫu bất đối xứng làm sai lệch cả huấn luyện lẫn đánh giá | Độ phủ seed (train/test); tỉ lệ cặp cạnh xuôi–ngược cùng có mặt; F1 trên toàn bộ test so với F1 trên phần seed được lấy mẫu | Độ phủ > 95% hoặc chênh lệch F1 nằm trong 1 std |
| H2: Kiến trúc | Khi độ phủ đạt 100%, tách hai luồng và hợp nhất muộn tốt hơn gộp trung bình theo từng tầng | Bậc thang ablation, bậc 2 → 3 → 4 | Chênh lệch macro-F1 nằm trong 1 std giữa 5 seed |
| H3: Giá trị của đồ thị | GNN vượt XGBoost đã tinh chỉnh, ít nhất ở các lớp phụ thuộc topo (MITM, Injection, Ransomware, Scanning) | Macro-F1 và F1 từng lớp, cùng split và cùng ngân sách tinh chỉnh | XGBoost bằng hoặc hơn GNN ở mọi lớp |
| H4: Không học tắt | Ưu thế ở H2/H3 vẫn còn khi chia theo host | Chia theo host (IP kẻ tấn công ở test không có trong train); xáo trộn danh tính node | Macro-F1 giảm về mức XGBoost hoặc thấp hơn |

H1 là đóng góp **không phụ thuộc vào việc mô hình của bạn thắng**: nó vẫn có giá trị ngay cả khi H2 yếu. H4 là câu hỏi đầu tiên mà phản biện security sẽ đặt ra.

## Phương pháp tối giản đề xuất

Phương pháp chỉ giữ những gì có bằng chứng cải thiện. Mỗi thành phần ứng với một bậc trong bậc thang ablation.

1. **Đồ thị:** đa đồ thị có hướng, node là IP, mỗi flow là một cạnh mang đặc trưng NetFlow (cổng nằm trong đặc trưng cạnh). Không gộp cạnh song song.
2. **Sampler giữ seed (bậc 1):** các cạnh đích của batch luôn nằm trong subgraph tính toán, cả khi train lẫn khi test. Độ phủ seed = 100% theo thiết kế.
3. **Ghép cặp xuôi–ngược (bậc 2):** hai hướng dùng chung một tập cạnh đã lấy mẫu, cạnh nào cũng có bản đảo ngược của nó.
4. **Hai luồng tách biệt, hợp nhất muộn (bậc 3–4):** luồng xuôi và luồng ngược lan truyền độc lập qua L tầng. Chỉ ở tầng cuối mới hợp nhất: concat ở bậc 3, MHA ở bậc 4.
5. **Readout cạnh và loss:** nối biểu diễn hai đầu mút với biểu diễn cạnh, rồi qua MLP. Dùng CE hoặc CE có trọng số, chọn trên validation.

**Loại khỏi phương pháp:** GCL/RSM, DWB/MAC, L\_ortho. Có thể nhắc đến trong mục "Negative results" nếu muốn minh bạch, kèm số liệu thật.

**Câu hỏi mở cần trả lời trong bài:** khi bỏ hoàn toàn walk trộn hướng, hợp nhất muộn có còn phát hiện được motif cần cả hai hướng (scatter-gather, nút trung gian vừa nhận vừa gửi) không? Trả lời bằng benchmark motif tổng hợp ở phần thực nghiệm. Nếu không phát hiện được, cần nói rõ trong mục giới hạn.

**Lý thuyết:** viết lại Proposition 1–2 ở dạng trừu tượng (lấy mẫu độc lập theo từng quan hệ; gộp trung bình theo từng tầng), không dùng tên hàm PyG. Nêu giả thiết rõ ràng; chứng minh đầy đủ chuyển sang phụ lục. Dùng "we characterize" thay cho "we prove". Kiểm chứng thực nghiệm bằng benchmark motif.

## Thiết kế thực nghiệm

Mọi mô hình chạy trong **một pipeline duy nhất**, cùng split, cùng đặc trưng, cùng ngân sách tinh chỉnh, và luôn đánh giá trên 100% tập test.

### Protocol chốt trước khi chạy

- **Dữ liệu:** 4 bộ NetFlow v2. IDS2018 gộp thành 6 họ tấn công + Benign.
- **Split A, ngẫu nhiên:** 75/25 phân tầng theo lớp, seed cố định. Validation là 10% tập train.
- **Split B, theo host:** chia IP nguồn của kẻ tấn công theo tỉ lệ \~75/25, nên IP tấn công ở test không xuất hiện trong train. Nếu một lớp chỉ có 1–2 IP tấn công, ghi rõ và loại lớp đó khỏi macro-F1 của split B.
- **Đặc trưng:** 39 cột NetFlow (bỏ IP). Cổng là biến thí nghiệm: chạy cả có và không có cổng.
- **Tinh chỉnh:** 30 lần thử Optuna cho mỗi mô hình trên validation, kể cả XGBoost.
- **Thống kê:** 5 seed, báo cáo mean ± std. Kiểm định Wilcoxon giữa các bậc liền kề.
- **Metric:** macro-F1 (chính), F1 từng lớp, weighted-F1, độ phủ seed train/test.
- **Ghi lại mọi lần chạy,** kể cả lần thất bại. Bảng trong bài được sinh tự động từ file kết quả.

### Các thí nghiệm

| # | Thí nghiệm | Trả lời | Đầu ra trong bài |
| --- | --- | --- | --- |
| E1 | Tái hiện Multi-GNN gốc (code IBM, chỉ sửa để chạy multiclass); đo độ phủ seed và F1 trên toàn bộ test so với F1 trên phần seed được lấy mẫu | H1 | Bảng chẩn đoán + hình độ phủ theo bậc node |
| E2 | Bậc thang ablation 0–4 (mỗi bậc thay đúng một yếu tố), cả 4 dataset | H1, H2 | Bảng chính của bài |
| E3 | Benchmark motif tổng hợp: fan-in/out, relay 2 bước, chu trình, scatter-gather; bội số cạnh M tăng dần | H1, H2, Proposition 1–2 | Hình: xác suất motif còn nguyên và F1 theo M |
| E4 | Baseline cùng hệ quy chiếu: XGBoost, Random Forest, MLP, E-GraphSAGE, Dir-GNN, Multi-GNN (bản sửa độ phủ) | H3 | Bảng so sánh + F1 từng lớp |
| E5 | Split theo host; xáo trộn danh tính node; huấn luyện trên một dataset, kiểm thử trên dataset khác (nếu đặc trưng tương thích) | H4 | Bảng độ bền |
| E6 | Triển khai: VRAM, thông lượng (flow/giây), độ trễ suy luận; so với đồ thị IP+PORT và line graph; trên GPU 16 GB phổ thông | Góp ý của editor | Bảng + hình tài nguyên |
| E7 (tuỳ chọn) | Tấn công chèn chaff có chủ đích, kẻ tấn công biết kiến trúc hai luồng | Góc độ security của TDSC | Bảng evasion |

E6 và E7 chỉ làm sau khi E1–E5 cho kết quả đủ mạnh để quyết định loại bài.

## Trả lời từng nhận xét của editor

Mỗi điểm editor TIFS nêu đều có một thí nghiệm trả lời trực tiếp. Đây là khác biệt chính so với bản SymDiG, vốn trả lời bằng cách đổi tên và viết lại câu chuyện.

| Nhận xét của editor | Cách trả lời | Thí nghiệm |
| --- | --- | --- |
| Tính mới hạn chế, chỉ ghép kỹ thuật có sẵn | Đổi đóng góp chính thành chẩn đoán định lượng vì sao GNN đa đồ thị thất bại trên đồ thị dày, kèm cách khắc phục tối giản; bỏ các thành phần ghép (GCL/RSM, DWB) | E1, E2, E3 |
| F1 của baseline không khớp với bài gốc | Chạy lại mọi baseline trong cùng pipeline. Nếu trích số từ bài khác, đặt trong nhóm riêng và ghi rõ protocol gốc | E4 |
| F1 end-to-end không chứng minh từng claim | Bậc thang ablation, mỗi bậc thay đúng một yếu tố, 5 seed + kiểm định | E2 |
| Biểu diễn chỉ dùng IP cần đánh giá bộ nhớ và khả năng mở rộng so với baseline | So với IP+PORT và line graph trên cùng dữ liệu: số node, số cạnh, VRAM, thông lượng, độ trễ | E6 |
| DWB không có lợi | Thừa nhận và bỏ; dùng CE. Có thể báo cáo số thật (CE 95.71 > DWB 94.12 trên ToN) | — |
| RSM chưa giải thích được vai trò | Bỏ | — |
| Cần chứng minh lợi thế với giao tiếp bất đối xứng | Benchmark motif có kiểm soát + F1 từng lớp ở các lớp relay/scatter (MITM, Injection, Ransomware) | E2, E3 |
| Gợi ý: triển khai, thời gian thực, khả năng mở rộng, drift, thay đổi topo | Thông lượng, độ trễ trên GPU phổ thông; split theo host như mô phỏng host mới xuất hiện | E5, E6 |

## Cấu trúc bài mới và các kịch bản kết quả

Bài được viết **sau** khi có kết quả E1–E5; loại bài do kết quả quyết định.

| Kết quả | Loại bài | Tiêu đề gợi ý | Nơi nộp |
| --- | --- | --- | --- |
| H1–H4 đều đúng | Chẩn đoán + phương pháp + đánh giá chặt chẽ | Why Provably Expressive Multigraph GNNs Fail on Dense Network Traffic, and How to Fix Them | TDSC |
| H1 đúng, H2 yếu | Bài về các cái bẫy khi đánh giá GNN-NIDS | Pitfalls in Evaluating Graph-Based Intrusion Detection on Dense Multigraphs | TDSC (góc dependable) hoặc Computers & Security |
| XGBoost ≥ GNN ở mọi lớp | Bài kết quả âm | Do Graphs Help NetFlow Intrusion Detection? A Controlled Re-evaluation | Tạp chí hoặc hội nghị ưa nghiên cứu tái lập |

### Bố cục cho kịch bản thứ nhất (12 trang)

1. **Introduction** (1 trang): vấn đề, khoảng cách giữa lý thuyết Multi-GNN và thực tế trên NetFlow, 3 đóng góp.
2. **Background và Related Work** (1.5 trang): GNN-NIDS theo cách dựng đồ thị; GNN có hướng (Dir-GNN, MagNet, DiGCN); Multi-GNN; các cạm bẫy ML trong security (Arp et al., USENIX Security 2022).
3. **Diagnosis** (2 trang): mô hình hóa trừu tượng, Proposition 1–2, số đo độ phủ từ E1.
4. **Method** (1.5 trang): sampler giữ seed, ghép cặp, hai luồng hợp nhất muộn.
5. **Evaluation protocol** (1 trang): split A/B, tinh chỉnh, seed, metric.
6. **Results** (3.5 trang): bậc thang ablation, benchmark motif, so với baseline, độ bền theo host, triển khai.
7. **Discussion và Limitations** (0.5 trang): bao gồm các kết quả âm (GCL/RSM, DWB) nếu muốn minh bạch.
8. **Conclusion** + references + tiểu sử (1 trang). Chứng minh đầy đủ và bảng phụ đưa vào supplemental.

## Kế hoạch, rủi ro và khả năng accept

Khoảng 8 tuần, với một cổng quyết định sau tuần 6: chỉ viết lại bài khi đã biết kịch bản kết quả nào xảy ra.

&#91;embedded content: lộ trình 8 tuần · 4 giai đoạn, 1 cổng quyết định\]

Các giai đoạn không vẽ theo tỉ lệ thời gian; thời lượng phụ thuộc GPU sẵn có, BoT-IoT (37.8M flow) là bộ tốn thời gian nhất.

### Rủi ro

| Rủi ro | Phát hiện ở | Phương án |
| --- | --- | --- |
| Pipeline DiG-RSB không đánh giá đủ 100% test, số 94.12 giảm khi đo lại | Tuần 1 | Dùng số mới; chính sự chênh lệch là bằng chứng cho H1 |
| XGBoost ngang hoặc hơn GNN | E4 | Chuyển sang kịch bản 2 hoặc 3 |
| Macro-F1 sụp khi chia theo host | E5 | Báo cáo như một phát hiện về giới hạn của bộ dữ liệu |
| Lớp hiếm chỉ có 1–2 IP tấn công, không chia theo host được | Thiết lập E5 | Báo cáo riêng, nêu rõ trong Limitations |
| Hợp nhất muộn không bắt được scatter-gather | E3 | Nêu trong Limitations; thử thêm một tầng trộn hướng cuối cùng |
| Thiếu thời gian GPU | Tuần 3 | Ưu tiên ToN và UNSW đủ 5 seed; BoT và IDS2018 3 seed |

### Khả năng được chấp nhận (ước lượng chủ quan)

| Kịch bản | Ước lượng tại TDSC |
| --- | --- |
| Nộp bản SymDiG hiện tại | ≤ 5%, kèm rủi ro liêm chính |
| Kịch bản 1: H1–H4 đúng, làm đủ E1–E6 | 25–35% sau 1–2 vòng major revision |
| Kịch bản 2: chỉ H1 mạnh | 10–15%; nơi khác phù hợp hơn |
| Kịch bản 3: kết quả âm | Thấp tại TDSC; có giá trị ở nơi ưa nghiên cứu tái lập |

Tôi không tìm được tỉ lệ chấp nhận chính thức của TDSC, nên các con số trên không dựa trên một mức nền công bố.

## Việc cần làm ngay và nguyên tắc liêm chính

### Tuần này

- [ ] Khôi phục Bảng 8 của SymDiG về số thật (CE 95.71, WCE 95.76 trên ToN multiclass)
- [ ] Thay mọi con số chưa chạy thật trong draft bằng placeholder hiển thị màu đỏ (\\tbd{})
- [ ] Ghi lại môi trường, commit code và cấu hình đã cho ra các số DiG-RSB
- [ ] Kiểm tra pipeline DiG-RSB có đánh giá đủ 100% cạnh test không
- [ ] Chạy lại XGBoost trên ToN: có và không có cổng, 5 seed, macro-F1 + F1 từng lớp
- [ ] Chốt protocol ở mục Thiết kế thực nghiệm thành một file trong repo, có ngày
- [ ] Mở issue trên GitHub Multi-GNN về đoạn bù seed chỉ cho Small\_J/Small\_Q

### Nguyên tắc liêm chính

- Mọi con số trong bài đều truy ngược được về một file kết quả và một commit.
- Không sửa tiêu chí hay protocol sau khi đã nhìn kết quả test; nếu buộc phải đổi, ghi rõ lý do trong bài.
- Baseline trích từ bài khác phải ghi rõ nguồn và protocol gốc, không trộn vào bảng xếp hạng chung.
- Kết quả âm được báo cáo đúng, không đổi tên thành phần để tránh nó.
- Viết công bằng với Multi-GNN và FeCoGraph: nêu điều kiểm chứng được, không suy diễn ý định của tác giả.
- Công bố code (TDSC hỗ trợ Code Ocean).
