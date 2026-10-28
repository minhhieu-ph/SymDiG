# SymDiG — mã nguồn thực nghiệm E1–E7

Mã nguồn chạy toàn bộ thực nghiệm trong *Đề xuất tái định hướng nghiên cứu SymDiG cho IEEE TDSC*. Mỗi thực nghiệm được tách thành các **job độc lập**, đặt trong một hàng đợi dùng chung. Mỗi GPU chạy một worker; nhiều pod Kubeflow có thể dùng chung một hàng đợi. Bảng và hình được sinh tự động từ file kết quả, nên mọi con số đều truy ngược được về một lần chạy cụ thể.

## 1. Cài đặt

```bash
cd SymDiG                                   # folder đã có dataset/ chứa 4 file CSV NetFlow v2
conda create -n symdig python=3.11 -y && conda activate symdig
pip install torch --index-url https://download.pytorch.org/whl/cu121   # chọn đúng bản CUDA của máy
pip install -r requirements.txt
git clone https://github.com/minhhieu-ph/LineGraph external/LineGraph  # FeCoGraph (chỉ dùng cho E4-p2)
python scripts/selfcheck.py                 # 1–3 phút, phải in "SELF-CHECK PASSED"
```

Yêu cầu Python ≥ 3.9. PyTorch Geometric chỉ cần bản thuần Python, **không** cần `torch-scatter` hay `torch-sparse`.

## 2. Chạy

```bash
# (0) Smoke test: mọi thực nghiệm ở cấu hình tí hon, 200k dòng đầu của ToN và UNSW (~10–20 phút)
python run.py smoke --datasets ton unsw
python make_tables.py --results results_smoke && python make_figures.py --results results_smoke

# (1) Lập kế hoạch (an toàn khi chạy lại: job trùng chỉ được tạo một lần)
python run.py plan E1 E2            # cốt lõi của bài: chẩn đoán + bậc thang ablation
python launch.py                    # 1 worker / GPU, tự chia CPU; worker tự thoát khi hết việc
python run.py status                # tiến độ, job lỗi, job bị chặn

python run.py plan tune E4 E5 E7    # tinh chỉnh (Optuna, cùng 30 trial cho mọi mô hình) -> so sánh -> độ bền -> evasion
python run.py plan E3 E6            # motif tổng hợp, chi phí triển khai
python launch.py

# (2) Bảng và hình
python make_tables.py               # results/tables/summary.md + E*.tex
python make_figures.py              # results/figures/*.pdf|png
```

- **Nhiều pod:** chạy `python launch.py` trên mỗi pod, trỏ `--results` và `paths.cache` vào cùng một PVC. Job được nhận bằng thao tác rename nguyên tử, nên mỗi job chỉ chạy một lần.
- **Pod bị khởi động lại:** chạy `python run.py requeue` (chuyển các job đang "running" về lại hàng đợi), rồi chạy `launch.py` lại.
- **Job lỗi:** sửa nguyên nhân, rồi chạy `python run.py retry`.
- **Chạy riêng một job để debug:** `python run.py run <job-id>`.

### Chia tài nguyên (4 × A100 40 GB, 11 core, 128 GiB)

`launch.py` mặc định chạy 1 worker cho mỗi GPU. Mỗi worker nhận `11 // 4 = 2` luồng CPU và 1 tiến trình lấy mẫu subgraph.

- Lấy mẫu chạy trên CPU, nên với 11 core cho cả 4 GPU, **CPU thường là nút thắt** của các job GNN.
- Nếu mỗi pod/GPU có riêng 11 core (như README của LineGraph mô tả), dùng `python launch.py --cpus 44` để mỗi worker có ~10 luồng lấy mẫu.
- Mảng dữ liệu được memory-map, nên 4 worker dùng chung page cache. BoT-IoT (lớn nhất) cần khoảng 11 GB cho đặc trưng cộng vài GB cho mỗi worker, nằm gọn trong 128 GiB.

## 3. Thực nghiệm → job → bảng/hình

| Thực nghiệm | Giả thuyết | Job | Bảng / hình |
|---|---|---|---|
| **E1** Chẩn đoán | H1 | `coverage`: độ phủ seed và độ phủ cặp xuôi–ngược của 3 kiểu sampler, fan-out 20/50/100, chia theo bậc node (không huấn luyện). `gnn` ladder0 + siêu tham số gốc của Multi-GNN: F1 trên phần seed được lấy mẫu (cách code gốc đo) so với F1 trên 100% test | `E1_coverage`, `E1_multignn`, `fig_e1_coverage` |
| **E2** Bậc thang ablation | H1, H2 | `gnn` ladder0–4 × 4 dataset × 5 seed (BoT 3 seed), cùng backbone và siêu tham số | `E2_ladder` (kèm Δ và p-value cặp), `E2_perclass_*`, `fig_e2_ladder` |
| **E3** Motif tổng hợp | H1, H2, Prop. 1–2 | `survival`: xác suất chu trình còn nguyên, đo thực nghiệm so với công thức. `gnn` ladder0–4 trên nhãn chỉ phụ thuộc cấu trúc, M = 1…256 | `E3_survival`, `E3_ladder`, `E3_perclass`, `fig_e3_*` |
| **E4** Cùng hệ quy chiếu | H3 | `tune` + 8 mô hình × 4 dataset × 5 seed; bản không có cổng; protocol FeCoGraph p2 (2%, 30% nhãn) với XGBoost, RF, E-GraphSAGE, SymDiG, FeCoGraph (repo của bạn) | `E4_main`, `E4_noports`, `E4_p2`, `E4_perclass_*`, `fig_e4_perclass` |
| **E5** Độ bền | H4 | Split theo host p1B (IP nguồn tấn công ở test không xuất hiện trong train) cho SymDiG, Multi-GNN, XGBoost, MLP; chuyển giao chéo dataset (nhị phân) | `E5_host`, `E5_transfer` |
| **E6** Triển khai | Góp ý editor | `sizes`: số node IP / IP:PORT / cạnh line graph. `deploy`: VRAM, targets/s, flows/s, độ trễ p95, có và không có giới hạn 16 GB | `E6_sizes`, `E6_deploy`, `fig_e6_cost` |
| **E7** Evasion | Góc security | Chèn chaff theo 3 chiến lược (pair / reverse / flood), K = 0…100, trên checkpoint của E4 | `E7_evasion`, `fig_e7_evasion` |

**Bậc thang ablation:** mỗi bậc thay đúng một yếu tố so với bậc trước.

| Bậc | arch | coupled | force | fusion | Ý nghĩa |
|---|---|---|---|---|---|
| 0 | hetero | false | false | — | Pipeline Multi-GNN: `LinkNeighborLoader` + `to_hetero(aggr='mean')`; loss chỉ tính trên seed được lấy mẫu |
| 1 | hetero | false | true | — | + cạnh đích luôn có trong subgraph |
| 2 | hetero | true | true | — | + xuôi/ngược dùng chung một tập cạnh |
| 3 | dual | true | true | concat | + hai luồng tách biệt, hợp nhất muộn |
| 4 | dual | true | true | mha | + hợp nhất bằng attention (SymDiG) |

**Số job với cấu hình mặc định:** khoảng 670, trong đó khoảng 365 job huấn luyện GNN. Thời gian phụ thuộc chủ yếu vào tốc độ lấy mẫu, nên hãy chạy E1 + E2 trước, xem `sec_per_step` trong `results/runs/*.json`, rồi mới quyết định số seed và số epoch cho các thực nghiệm còn lại.

## 4. Các lựa chọn cài đặt cần ghi trong bài

1. **Split.**
   - Split cố định với `split_seed = 42`. Seed của mô hình chỉ ảnh hưởng đến khởi tạo và thứ tự batch.
   - **p1A** chia ngẫu nhiên phân tầng 75/25. Validation là 10% tập train (tối đa 200k), chỉ dùng để chọn epoch và tinh chỉnh. Lúc train, đồ thị chỉ gồm flow của train; lúc test, đồ thị gồm mọi flow, nhưng không dùng nhãn test.
   - **p1B** chia theo IP nguồn tấn công.
     - Lớp có ít hơn 2 IP nguồn không chia được: lớp đó chỉ nằm trong train và bị loại khỏi macro-F1 của p1B. Các flow của lớp này xuất phát từ IP thuộc tập test bị bỏ hẳn để tránh rò rỉ.
     - `split_meta` ghi lại số IP, tỉ lệ test thực tế của từng lớp, và chỉ số `attack_ip_overlap` (phải bằng 0).
   - **p2** đúng theo protocol của FeCoGraph và là split transductive. Mọi mô hình trong E4-p2 dùng **cùng** tập con và cùng các tập train/val/test.
2. **Đặc trưng.**
   - log1p cho các cột số, tách 8 bit cho mỗi cột TCP flag, bỏ IP.
   - Cổng (giá trị gốc + 3 dải mỗi phía) là một biến thí nghiệm: `data.ports`.
   - Chuẩn hoá bằng thống kê tính trên các cạnh của đồ thị train.
3. **Đánh giá trên 100% tập test.** Với bậc 0, cạnh test không được lấy mẫu sẽ được đưa vào lại khi tính F1 "full". F1 theo cách của code gốc được báo cáo riêng (`metrics_covered`), kèm độ phủ.
4. **Sampler.**
   - Lấy mẫu đều, có hoàn lại, mỗi node B cạnh theo mỗi chiều mỗi hop. Đây là xấp xỉ của `LinkNeighborLoader` (không hoàn lại) khi bậc node lớn hơn nhiều so với B.
   - Mỗi hop giới hạn `max_hop_edges = 400k` cạnh để giữ an toàn bộ nhớ GPU; giới hạn này áp dụng như nhau cho mọi bậc.
5. **Multi-GNN.**
   - Được cài lại theo đúng ngữ nghĩa trong repo IBM: `to_hetero` gộp trung bình theo từng tầng, edge update riêng cho từng quan hệ, ego ID.
   - Không dùng port numbering, vì kỹ thuật này cần thứ tự thời gian mà NetFlow v2 không có.
   - Có thêm job E1 với siêu tham số gốc của họ (lr 0.006, hidden 66, fan-out 100, batch 8192).
6. **Tinh chỉnh.**
   - 30 trial Optuna (TPE, seed 0) cho **mọi** mô hình, tối ưu macro-F1 trên validation của p1A.
   - Mô hình bảng được tinh chỉnh trên cả 4 dataset. GNN được tinh chỉnh trên ToN và UNSW; IDS2018 và BoT dùng lại tham số của ToN (`tuning.gnn_datasets`).
   - Các bậc của ablation **không** được tinh chỉnh riêng, có chủ đích, để mỗi bậc chỉ khác đúng một yếu tố.
7. **Thống kê.** Mean ± std qua các seed. Giữa hai bậc liền kề dùng kiểm định hoán vị dấu cho cặp seed (với 5 seed, p nhỏ nhất đạt được là 0.0625).
8. **Evasion.**
   - Đặc trưng của chaff được sao từ flow benign thật trong tập test.
   - Precision được hiệu chỉnh khi lấy mẫu con các target không bị tấn công (trọng số 1/q).
   - XGBoost và MLP không bị chaff ảnh hưởng theo định nghĩa, vì chúng phân loại từng flow độc lập. Đây là một nhận xét trung thực cần nêu trong bài.

## 5. Gửi kết quả cho tôi

Sau mỗi đợt chạy, gửi:

```bash
python make_tables.py
tar czf symdig_results.tgz results/tables results/figures results/runs results/tuned results/index results/config.yaml \
    $(python run.py status | grep -o 'results/logs/[a-f0-9]*\.log')
```

Không cần gửi `results/ckpt` (checkpoint lớn) hay `cache/`.

## 6. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| Lỗi liên quan bf16 trong `GATv2Conv` hoặc scatter | `python run.py plan ... --set train.amp=false` (job mới có id mới) |
| CUDA OOM khi huấn luyện | `--set train.batch_size=2048 sampler.max_hop_edges=200000` |
| GPU dùng thấp, CPU 100% | Lấy mẫu là nút thắt: tăng `--cpus`, hoặc giảm `--workers-per-gpu` |
| Thiếu RAM | Giữ `data.in_memory=false` (mặc định); giảm số worker |
| `p1B` loại bỏ hầu hết các lớp | Ghi nhận như một phát hiện: dataset có quá ít IP tấn công để kiểm chứng tổng quát hoá theo host |

## 7. Cấu trúc

```
configs/default.yaml      mọi siêu tham số + định nghĩa bậc thang + danh sách thực nghiệm
run.py / launch.py        CLI (plan, worker, status, smoke...) / khởi chạy worker theo GPU
symdig/data.py            CSV -> cache, đặc trưng, split p1A/p1B/p2, CSR đa đồ thị (cache memmap)
symdig/sampler.py         sampler cho bậc thang (coupled / force), thống kê độ phủ
symdig/loader.py          DataLoader: lấy mẫu song song trong worker
symdig/models.py          hetero (Multi-GNN), dual (SymDiG), dir (Dir-GNN), egsage (E-GraphSAGE)
symdig/trainer.py         huấn luyện, đánh giá full / covered, checkpoint
symdig/tabular.py         XGBoost, Random Forest (XGBRF), MLP trên GPU
symdig/fecograph_adapter.py  chạy repo LineGraph của bạn trên đúng split p2 (không sửa repo)
symdig/motifs.py          sinh đa đồ thị có motif + xác suất sống sót (Prop. 1)
symdig/evasion.py         chèn chaff (pair / reverse / flood)
symdig/tuning.py          Optuna, cùng ngân sách cho mọi mô hình
symdig/planner.py         E1–E7 -> job; symdig/jobs.py hàng đợi; symdig/executors.py thực thi từng loại job
make_tables.py / make_figures.py   bảng (Markdown + LaTeX) và hình (PDF vector)
scripts/selfcheck.py      kiểm tra môi trường + mọi kiến trúc trên dữ liệu tí hon
tests/                    test phần numpy (không cần GPU)
```
