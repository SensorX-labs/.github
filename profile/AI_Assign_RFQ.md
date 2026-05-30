# CHƯƠNG 5. MÔ HÌNH PHÂN BỔ ĐỘNG SENSORX

## 5.1. Kiến trúc tổng thể

Mô hình phân bổ SensorX áp dụng kiến trúc Học máy trực tuyến (Online Machine Learning) với hai pha xử lý độc lập:

* **Mô hình suy luận (Forward Pass):** Tính toán điểm số, ra quyết định phân bổ RFQ và lưu vết dự báo.
* **Mô hình học trực tuyến (Backward Pass):** So sánh dự báo với kết quả thực tế để cập nhật năng lực nhân sự và tinh chỉnh siêu tham số hệ thống.

---

## 5.2. Mô hình suy luận (Forward Pass)

### 5.2.1. Phương trình tổng quát

Hệ thống xếp hạng nhân sự dựa trên phương trình tuyến tính kết hợp:

$$FinalScore = (AggregatedSkillScore \times Penalty_{workload}) + Boost_{idle}$$

*Trong đó:*

* $FinalScore$: Điểm quyết định phân bổ cuối cùng của nhân viên đối với RFQ hiện tại.
* $AggregatedSkillScore$: Điểm kỹ năng chuyên môn tổng hợp của nhân sự đối với cấu trúc sản phẩm của RFQ.
* $Penalty_{workload}$: Hệ số điều chỉnh tải công việc nhằm phòng ngừa trạng thái thắt cổ chai.
* $Boost_{idle}$: Hệ số ưu tiên thời gian nhàn rỗi nhằm hỗ trợ cân bằng tải.

### 5.2.2. Điểm kỹ năng tổng hợp ($AggregatedSkillScore$)

Đánh giá mức độ phù hợp chuyên môn của nhân sự trên toàn bộ RFQ bằng phương pháp trung bình cộng có trọng số:

$$AggregatedSkillScore = \frac{\sum_{j=1}^{n} \left( ExpectedPerformanceScore_{c(j)} \times ItemWeight_j \right)}{\sum_{j=1}^{n} ItemWeight_j}$$

*Trong đó:*

* $n$: Tổng số lượng mặt hàng độc lập có trong RFQ.
* $j$: Chỉ số đại diện cho mặt hàng thứ $j$ trong RFQ.
* $c(j)$: Hàm ánh xạ mặt hàng $j$ sang danh mục quản lý tương ứng.
* $ExpectedPerformanceScore_{c(j)}$: Điểm hiệu suất kỳ vọng của nhân sự tại danh mục $c(j)$.
* $ItemWeight_j$: Trọng số quy mô (sức nặng tài chính) của mặt hàng $j$.

Các biến thành phần được bóc tách chi tiết như sau:

**a) Trọng số quy mô ($ItemWeight_j$):** Định hướng AI ưu tiên các mặt hàng giá trị cao.


$$ItemWeight_j = Quantity_j \times BasePrice_j$$

* $Quantity_j$: Số lượng yêu cầu của mặt hàng $j$.
* $BasePrice_j$: Đơn giá vốn nội bộ của mặt hàng $j$ tại thời điểm nhận RFQ.

**b) Biên lợi nhuận lịch sử ($AverageMargin_i$):** Tỷ suất sinh lời trung bình trong quá khứ tại danh mục $i$.


$$AverageMargin_i = \begin{cases} \frac{TotalMarginAccumulated_i}{SuccessCount_i} & \text{nếu } SuccessCount_i > 0 \\ 0 & \text{nếu } SuccessCount_i = 0 \end{cases}$$

* $i$: Chỉ số đại diện cho danh mục sản phẩm ($i = c(j)$).
* $TotalMarginAccumulated_i$: Tổng biên lợi nhuận tích lũy từ các báo giá đã được khách hàng chấp nhận tại danh mục $i$.
* $SuccessCount_i$: Tổng số báo giá được khách hàng chấp nhận tại danh mục $i$.

**c) Xác suất khách hàng chấp nhận báo giá ($Probability_i$):** Cân bằng giữa Khai thác (nhân sự cũ giỏi) và Khám phá (nhân sự mới) bằng thuật toán Thompson Sampling.

$$Probability_i = \text{Beta.Sample}(SuccessCount_i + 1,\ FailureCount_i + 1)$$

* $\text{Beta.Sample}$: Hàm lấy mẫu ngẫu nhiên từ phân phối xác suất Beta.
* $FailureCount_i$: Tổng số báo giá bị khách hàng từ chối tại danh mục $i$.

**d) Điểm hiệu suất kỳ vọng ($ExpectedPerformanceScore_i$):** Tích hợp xác suất và hiệu quả kinh tế của nhân sự tại danh mục $i$.


$$ExpectedPerformanceScore_i = Probability_i \times (1 + AverageMargin_i)$$

### 5.2.3. Các siêu tham số vận hành

Lượng hóa mức độ ảnh hưởng của trạng thái công việc hiện tại đối với từng nhân sự:

$$Penalty_{workload} = \frac{1}{(CurrentWorkload + 1)^{k_{old}}}$$

$$Boost_{idle} = IdleHours \times IdleWeight_{old}$$

*Trong đó:*

* $CurrentWorkload$: Số lượng yêu cầu báo giá đang được nhân sự xử lý tại thời điểm tính toán.
* $IdleHours$: Số giờ rảnh rỗi liên tục của nhân sự tính từ lần cuối cùng được hệ thống phân bổ RFQ.
* $k_{old}$: Siêu tham số điều chỉnh tải công việc hiện tại.
* $IdleWeight_{old}$: Siêu tham số ưu tiên thời gian nhàn rỗi hiện tại.

### 5.2.4. Lưu vết dự báo

Chuẩn hóa $FinalScore$ thành xác suất báo giá được khách hàng chấp nhận $\hat{y} \in (0, 1)$ phục vụ pha học trực tuyến.

$$\hat{y} = \frac{1}{1 + e^{-FinalScore}}$$

* $\hat{y}$: Giá trị xác suất chốt báo giá thành công do AI dự báo.
* $e$: Hằng số Euler ($e \approx 2.71828$).

---

## 5.3. Mô hình học trực tuyến (Backward Pass)

### 5.3.1. Hàm mất mát nhị phân (Binary Cross-Entropy Loss)

Mô hình tối ưu hóa dựa trên hàm mất mát Log-Likelihood đảo dấu tiêu chuẩn, định nghĩa mức độ sai lệch giữa dự báo và thực tế:

$$L = - \left[ y \cdot \ln(\hat{y}) + (1 - y) \cdot \ln(1 - \hat{y}) \right]$$

* $L$: Giá trị hàm mất mát (độ lỗi toàn cục của lượt dự báo).
* $y$: Nhãn thực tế từ phản hồi của khách hàng ($y = 1$ nếu khách hàng chấp nhận báo giá, $y = 0$ nếu khách hàng từ chối báo giá).
* $\ln$: Hàm logarit tự nhiên (cơ số $e$).

**Mục tiêu tối ưu hóa:** Mục tiêu của mô hình là áp dụng thuật toán Gradient Descent nhằm **cực tiểu hóa (tối thiểu hóa)** giá trị của hàm mất mát $L$ về cận dưới bằng $0$. Quá trình cực tiểu hóa này ép xác suất dự báo $\hat{y}$ đạt độ hội tụ sát nhất với nhãn thực tế $y$, làm nền tảng giải tích để định hình cấu trúc toán Gradient và điều hướng luồng cập nhật tham số ở các bước tiếp theo.

### 5.3.2. Cập nhật trạng thái chuyên môn (Parameter Adaptation)

Hồ sơ năng lực của nhân sự tại danh mục $i$ được tái đánh giá dựa trên kết quả phản hồi thực tế của báo giá. Biên lợi nhuận thực tế và các chỉ số tích lũy tuân theo luật cập nhật logic sau:

* **Trường hợp báo giá được khách hàng chấp nhận ($y = 1$):**
Hệ thống tiến hành tính toán Biên lợi nhuận thực tế ($Margin_{Q, i}$) của báo giá $Q$ tại danh mục $i$:

$$Margin_{Q, i} = \frac{\sum_{g=1}^{m} (QuotedPrice_g \times Quantity_g) - \sum_{g=1}^{m} (BasePrice_g \times Quantity_g)}{\sum_{g=1}^{m} (BasePrice_g \times Quantity_g)}$$



*(Trong đó: $m$ là số lượng mặt hàng thuộc danh mục $i$ trong báo giá $Q$; $QuotedPrice_g$ là đơn giá bán thực tế chốt với khách hàng).*
Sau đó, cập nhật dữ liệu tích lũy chuyên môn:

$$SuccessCount_i = SuccessCount_i + 1$$


$$TotalMarginAccumulated_i = TotalMarginAccumulated_i + Margin_{Q, i}$$


* **Trường hợp báo giá bị khách hàng từ chối ($y = 0$):**
Biên lợi nhuận thực tế của giao dịch không được thiết lập. Hệ thống giữ nguyên chỉ số lợi nhuận tích lũy và chỉ cập nhật tần suất thất bại của nhân sự tại danh mục $i$:

$$FailureCount_i = FailureCount_i + 1$$



### 5.3.3. Cập nhật siêu tham số (Online Gradient Update)

Hệ thống điều chỉnh siêu tham số toàn cục dựa trên Sai số dự báo và Đạo hàm hướng.

**Luật cập nhật siêu tham số trừng phạt ($k$):**


$$k_{new} = k_{old} + \alpha \cdot \underbrace{(y - \hat{y})}_{\text{Thành phần A: Sai số dự báo}} \cdot \underbrace{\left[ -AggregatedSkillScore \cdot Penalty_{workload} \cdot \ln(CurrentWorkload + 1) \right]}_{\text{Thành phần B: Đạo hàm riêng } \frac{\partial FinalScore}{\partial k_{old}}}$$

**Luật cập nhật siêu tham số khuyến khích ($IdleWeight$):**


$$IdleWeight_{new} = IdleWeight_{old} + \alpha \cdot \underbrace{(y - \hat{y})}_{\text{Thành phần A: Sai số dự báo}} \cdot \underbrace {IdleHours}_{\text{Thành phần B: Đạo hàm riêng } \frac{\partial FinalScore}{\partial IdleWeight_{old}}}$$

*Trong đó:*

* $k_{new}, IdleWeight_{new}$: Giá trị mới của siêu tham số sau khi được AI tối ưu hóa.
* $\alpha$: Tốc độ học (Learning Rate), cấu hình cố định ở mức $0.01$.

---

## 5.4. Phân tích Toán học Gradient

### 5.4.1. Đạo hàm theo siêu tham số tải trọng ($k$)

Mục này khai triển chứng minh giải tích cho Thành phần B trong luật cập nhật hệ số trừng phạt $k$. Tiến hành lấy đạo hàm riêng của hàm số $FinalScore$ (định nghĩa tại Mục 5.2.1) theo biến số $k_{old}$:

$$\frac{\partial FinalScore}{\partial k_{old}} = \frac{\partial}{\partial k_{old}} \left[ AggregatedSkillScore \cdot (CurrentWorkload + 1)^{-k_{old}} + IdleHours \cdot IdleWeight_{old} \right]$$

Do thành phần bổ trợ thời gian nhàn rỗi ($IdleHours \cdot IdleWeight_{old}$) hoàn toàn độc lập với biến số $k_{old}$, đạo hàm riêng của cụm này bằng 0:


$$\frac{\partial FinalScore}{\partial k_{old}} = \frac{\partial}{\partial k_{old}} \left[ AggregatedSkillScore \cdot (CurrentWorkload + 1)^{-k_{old}} \right] + 0$$

Áp dụng quy tắc đạo hàm hàm số mũ dạng $(a^{-x})' = -a^{-x} \cdot \ln(a)$ với cơ số hằng số là $a = (CurrentWorkload + 1)$ and biến số là $x = k_{old}$:


$$\frac{\partial FinalScore}{\partial k_{old}} = AggregatedSkillScore \cdot \left[ -(CurrentWorkload + 1)^{-k_{old}} \cdot \ln(CurrentWorkload + 1) \right]$$

Đưa dấu âm ra phía trước biểu thức và thu gọn thành phần cấu trúc mũ về dạng hàm gốc $Penalty_{workload}$:


$$\frac{\partial FinalScore}{\partial k_{old}} = -AggregatedSkillScore \cdot \underbrace{(CurrentWorkload + 1)^{-k_{old}}}_{Penalty_{workload}} \cdot \ln(CurrentWorkload + 1)$$

$$\frac{\partial FinalScore}{\partial k_{old}} = -AggregatedSkillScore \cdot Penalty_{workload} \cdot \ln(CurrentWorkload + 1)$$

**Ý nghĩa học máy:** Cụm $\ln(CurrentWorkload + 1)$ đóng vai trò là **Hệ số khuếch đại độ nhạy**.

* Tải thấp ($W=1 \rightarrow \ln(2) \approx 0.69$): Mức độ điều chỉnh diễn ra tiệm tiến.
* Tải cao ($W=10 \rightarrow \ln(11) \approx 2.40$): Khuếch đại biên độ điều chỉnh lên gấp 3.5 lần, thực hiện điều chỉnh mạnh hơn đối với trạng thái quá tải của hệ thống.

### 5.4.2. Đạo hàm theo siêu tham số thời gian rảnh ($IdleWeight$)

Mục này khai triển chứng minh giải tích cho Thành phần B trong luật cập nhật trọng số ưu tiên thời gian nhàn rỗi. Tiến hành lấy đạo hàm riêng của hàm số $FinalScore$ theo biến số $IdleWeight_{old}$:

$$\frac{\partial FinalScore}{\partial IdleWeight_{old}} = \frac{\partial}{\partial IdleWeight_{old}} \left[ AggregatedSkillScore \cdot Penalty_{workload} + IdleHours \cdot IdleWeight_{old} \right]$$

Do thành phần năng lực và tải trọng ($AggregatedSkillScore \cdot Penalty_{workload}$) hoàn toàn đóng vai trò là hằng số đối với biến $IdleWeight_{old}$, đạo hàm riêng của cụm này bị triệt tiêu:


$$\frac{\partial FinalScore}{\partial IdleWeight_{old}} = 0 + \frac{\partial}{\partial IdleWeight_{old}} \left[ IdleHours \cdot IdleWeight_{old} \right]$$

Áp dụng quy tắc đạo hàm hàm bậc nhất tuyến tính dạng $(c \cdot x)' = c$ với hằng số hệ số là $c = IdleHours$ và biến số là $x = IdleWeight_{old}$:


$$\frac{\partial FinalScore}{\partial IdleWeight_{old}} = IdleHours$$

**Ý nghĩa học máy:** Mối liên hệ đồng biến bậc nhất phản ánh rằng mức độ điều chỉnh hệ số ưu tiên thời gian nhàn rỗi luôn tỷ lệ thuận tuyệt đối với thời gian chờ việc thực tế của nhân sự.

### 5.4.3. Chứng minh toán học quy trình triệt tiêu Gradient tổng thể

Mục này chứng minh nguồn gốc toán học của Thành phần A - Sai số dự báo $(y - \hat{y})$ dùng trong các công thức cập nhật tại Mục 5.3.3. Theo nguyên lý Gradient Descent, hướng cập nhật tối ưu nhằm tối thiểu hóa hàm mục tiêu được dẫn dắt bởi đạo hàm riêng của Hàm mất mát ($L$) theo $FinalScore$.

Áp dụng quy tắc chuỗi toán giải tích đối với hàm hợp, ta tách biệt bài toán thành hai phân đoạn:


$$\frac{\partial L}{\partial FinalScore} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial FinalScore}$$

**Bước 1: Tính đạo hàm riêng của Hàm mất mát $L$ theo biến dự báo $\hat{y}$**
Từ phương trình gốc tại Mục 5.3.1: $L = - [y \cdot \ln(\hat{y}) + (1 - y) \cdot \ln(1 - \hat{y})]$. Tiến hành lấy đạo hàm riêng theo biến $\hat{y}$:


$$\frac{\partial L}{\partial \hat{y}} = - \left[ y \cdot \frac{1}{\hat{y}} + (1 - y) \cdot \frac{-1}{1 - \hat{y}} \right] = - \left[ \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right]$$

Quy đồng mẫu số và rút gọn biểu thức bên trong dấu ngoặc:


$$\frac{\partial L}{\partial \hat{y}} = - \left[ \frac{y(1 - \hat{y}) - \hat{y}(1 - y)}{\hat{y}(1 - \hat{y})} \right] = - \left[ \frac{y - \hat{y}}{\hat{y}(1 - \hat{y})} \right] = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

**Bước 2: Tính đạo hàm riêng của hàm Sigmoid theo biến $FinalScore$**
Từ phương trình hàm kích hoạt: $\hat{y} = \frac{1}{1 + e^{-FinalScore}}$. Đạo hàm riêng theo biến $FinalScore$ được biểu diễn qua chính hàm gốc $\hat{y}$:


$$\frac{\partial \hat{y}}{\partial FinalScore} = \hat{y}(1 - \hat{y})$$

**Bước 3: Tổng hợp và thực hiện phép triệt tiêu cơ số**
Thay thế kết quả từ Bước 1 và Bước 2 vào phương trình quy tắc chuỗi ban đầu:


$$\frac{\partial L}{\partial FinalScore} = \left[ \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right] \cdot \left[ \hat{y}(1 - \hat{y}) \right]$$

Triệt tiêu đại lượng $\hat{y}(1 - \hat{y})$ đồng nhất giữa tử số và mẫu số:


$$\frac{\partial L}{\partial FinalScore} = \hat{y} - y = - (y - \hat{y})$$

**Kết luận:** Phép chứng minh giải tích trên khẳng định rằng sự xuất hiện của cụm tuyến tính $(y - \hat{y})$ là hệ quả tất yếu của việc triệt tiêu toán học giữa hàm toán học Log-Likelihood và hàm Sigmoid. Kết quả thu gọn này cho phép Gradient của toàn bộ hệ thống được biểu diễn dưới dạng tuyến tính $(y - \hat{y})$, từ đó đơn giản hóa đáng kể quá trình cập nhật tham số trong môi trường học trực tuyến khi vận hành thực tế.
