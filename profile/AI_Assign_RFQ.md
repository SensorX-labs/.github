# CHƯƠNG 5. ỨNG DỤNG TRÍ TUỆ NHÂN TẠO

## 5.1. Ứng dụng ML trong phân bổ yêu cầu báo giá

### 5.1.1. Kiến trúc tổng thể

Mô hình phân bổ SensorX áp dụng kiến trúc Học máy trực tuyến (Online Machine Learning) dựa trên cấu trúc của một mạng nơ-ron nhân tạo đơn tầng tùy biến (Custom Single-Layer Perceptron). Kiến trúc thuật toán được chia thành hai pha xử lý độc lập:

* **Mô hình suy luận (Forward Pass):** Tính toán điểm số đầu vào thông qua hàm tổng tuyến tính, ra quyết định phân bổ yêu cầu báo giá (RFQ) và sử dụng hàm kích hoạt để lưu vết dự báo.
* **Mô hình học trực tuyến (Backward Pass):** Đo lường sai số dự báo sau khi tiếp nhận kết quả thực tế, áp dụng thuật toán lan truyền ngược để cập nhật năng lực chuyên môn cá nhân và tối ưu hóa hệ thống siêu tham số toàn cục.

---

### 5.1.2. Mô hình suy luận (Forward Pass)

#### a) Phương trình tổng quát

Hệ thống xếp hạng và chỉ định nhân sự dựa trên phương trình tổng nơ-ron tuyến tính kết hợp:

$$FinalScore = (AggregatedSkillScore \times Penalty_{workload}) + Boost_{idle}$$

*Trong đó:*

* $FinalScore$: Điểm quyết định phân bổ cuối cùng của nhân viên đối với RFQ hiện tại.
* $AggregatedSkillScore$: Điểm kỹ năng chuyên môn tổng hợp của nhân sự đối với cấu trúc sản phẩm của RFQ.
* $Penalty_{workload}$: Hệ số điều chỉnh tải công việc nhằm phòng ngừa trạng thái thắt cổ chai.
* $Boost_{idle}$: Hệ số ưu tiên thời gian nhàn rỗi nhằm hỗ trợ cân bằng tải.

#### b) Điểm kỹ năng tổng hợp ($AggregatedSkillScore$)

Đánh giá mức độ phù hợp chuyên môn của nhân sự trên toàn bộ RFQ bằng phương pháp trung bình cộng có trọng số:

$$AggregatedSkillScore = \frac{\sum_{j=1}^{n} \left( ExpectedPerformanceScore_{c(j)} \times ItemWeight_j \right)}{\sum_{j=1}^{n} ItemWeight_j}$$

*Trong đó:*

* $n$: Tổng số lượng mặt hàng độc lập có trong RFQ.
* $j$: Chỉ số đại diện cho mặt hàng thứ $j$ trong RFQ.
* $c(j)$: Hàm ánh xạ mặt hàng $j$ sang danh mục quản lý tương ứng.
* $ExpectedPerformanceScore_{c(j)}$: Điểm hiệu suất kỳ vọng của nhân sự tại danh mục $c(j)$.
* $ItemWeight_j$: Trọng số quy mô (sức nặng tài chính) của mặt hàng $j$.

Các biến thành phần được bóc tách chi tiết như sau:

* **Trọng số quy mô ($ItemWeight_j$):** Định hướng AI ưu tiên các mặt hàng giá trị cao.

$$ItemWeight_j = Quantity_j \times BasePrice_j$$

*(Với $Quantity_j$ là số lượng yêu cầu và $BasePrice_j$ là đơn giá vốn nội bộ của mặt hàng $j$).*

* **Biên lợi nhuận lịch sử chuẩn hóa ($\widetilde{AverageMargin}_i$):** Tỷ suất sinh lời trung bình trong quá khứ tại danh mục $i$. Nhằm đảm bảo tính ổn định toán học, bảo vệ mô hình khỏi các dữ liệu tồn dư hoặc dữ liệu mẫu sai lệch đơn vị (như lưu dạng số phần trăm nguyên nguyên thể thay vì số thực tỷ lệ), hệ thống áp dụng bộ tiền xử lý và nắn dòng dữ liệu lịch sử về không gian giải tích chuẩn $[0, 1]$:

$$AverageMargin_i^{thô} = \begin{cases} \frac{TotalMarginAccumulated_i}{SuccessCount_i} & \text{nếu } SuccessCount_i > 0 \\ 0 & \text{nếu } SuccessCount_i = 0 \end{cases}$$

$$\widetilde{AverageMargin}_i = \begin{cases} \frac{AverageMargin_i^{thô}}{100} & \text{nếu } AverageMargin_i^{thô} > 1.0 \\ AverageMargin_i^{thô} & \text{nếu } AverageMargin_i^{thô} \le 1.0 \end{cases}$$

*(Với $i = c(j)$, $TotalMarginAccumulated_i$ là tổng biên lợi nhuận tích lũy từ các báo giá thành công và $SuccessCount_i$ là tổng số báo giá được khách hàng chấp nhận tại danh mục $i$).*

* **Xác suất khách hàng chấp nhận báo giá ($Probability_i$):** Cân bằng giữa Khai thác và Khám phá bằng thuật toán Thompson Sampling.

$$Probability_i = \text{Beta.Sample}(SuccessCount_i + 1,\ FailureCount_i + 1)$$

*(Với $\text{Beta.Sample}$ là hàm lấy mẫu ngẫu nhiên từ phân phối Beta; $FailureCount_i$ là tổng số báo giá bị khách hàng từ chối tại danh mục $i$).*

* **Điểm hiệu suất kỳ vọng ($ExpectedPerformanceScore_i$):** Tích hợp xác suất và hiệu quả kinh tế đã được chuẩn hóa của nhân sự tại danh mục $i$.

$$ExpectedPerformanceScore_i = Probability_i \times (1 + \widetilde{AverageMargin}_i)$$

#### c) Các siêu tham số vận hành

Lượng hóa mức độ ảnh hưởng của trạng thái công việc hiện tại đối với từng nhân sự:

$$Penalty_{workload} = \frac{1}{(CurrentWorkload + 1)^{k_{old}}}$$

$$Boost_{idle} = \tanh\left(\frac{IdleHours}{24}\right) \times IdleWeight_{old}$$

*Trong đó:*

* $CurrentWorkload`: Số lượng yêu cầu báo giá đang được nhân sự xử lý tại thời điểm tính toán.
* $IdleHours`: Số giờ rảnh rỗi liên tục của nhân sự tính từ lần cuối cùng được hệ thống phân bổ RFQ. Hàm tang Hyperbolic ($\tanh$) được áp dụng nhằm bão hòa phi tuyến tính thời gian nhàn rỗi về dải $[0, 1)$, ngăn chặn việc bùng nổ điểm số khi nhân sự nghỉ phép hoặc không nhận đơn kéo dài. Phép chia cho $24$ đóng vai trò quy chuẩn hóa đơn vị (Unit Scaling) theo chu kỳ ngày để đạt trạng thái hội tụ mịn màng nhất.
* $k_{old}$: Siêu tham số điều chỉnh tải công việc hiện tại.
* $IdleWeight_{old}$: Siêu tham số ưu tiên thời gian nhàn rỗi hiện tại.

#### d) Lưu vết dự báo

Chuẩn hóa $FinalScore$ thông qua hàm kích hoạt Sigmoid nhằm thiết lập dự báo xác suất báo giá được khách hàng chấp nhận $\hat{y} \in (0, 1)$ phục vụ pha học trực tuyến.

$$\hat{y} = \frac{1}{1 + e^{-FinalScore}}$$

*(Với $\hat{y}$ là giá trị xác suất chốt báo giá thành công do AI dự báo và $e$ là hằng số Euler).*

---

### 5.1.3. Mô hình học trực tuyến (Backward Pass)

#### a) Hàm mất mát nhị phân (Binary Cross-Entropy Loss)

Mô hình tối ưu hóa dựa trên hàm mất mát Log-Likelihood đảo dấu tiêu chuẩn, định nghĩa mức độ sai lệch giữa dự báo và thực tế:

$$L = - \left[ y \cdot \ln(\hat{y}) + (1 - y) \cdot \ln(1 - \hat{y}) \right]$$

* $L$: Giá trị hàm mất mát (độ lỗi toàn cục của lượt dự báo).
* $y$: Nhãn thực tế từ phản hồi của khách hàng ($y = 1$ nếu khách hàng chấp nhận báo giá, $y = 0$ nếu khách hàng từ chối báo giá).
* $\ln$: Hàm logarit tự nhiên (cơ số $e$).

**Mục tiêu tối ưu hóa:** Mục tiêu của mô hình là áp dụng thuật toán Gradient Descent nhằm **cực tiểu hóa (tối thiểu hóa)** giá trị của hàm mất mát $L$ về cận dưới bằng $0$. Quá trình cực tiểu hóa này ép xác suất dự báo $\hat{y}$ đạt độ hội tụ sát nhất với nhãn thực tế $y$, làm nền tảng giải tích để định hình cấu trúc toán Gradient và điều hướng luồng cập nhật tham số ở các bước tiếp theo.

#### b) Cập nhật trạng thái chuyên môn (Parameter Adaptation)

Hồ sơ năng lực của nhân sự tại danh mục $i$ được tái đánh giá dựa trên kết quả phản hồi thực tế của báo giá theo luật cập nhật logic sau:

* **Trường hợp báo giá được khách hàng chấp nhận ($y = 1$):**
Hệ thống tiến hành tính toán Biên lợi nhuận thực tế ($Margin_{Q, i}$) của báo giá $Q$ tại danh mục $i$ dựa trên tỷ suất sinh lời vượt sàn (Markup):

$$Margin_{Q, i}^{thô} = \frac{\sum_{g=1}^{m} (QuotedPrice_g \times Quantity_g) - \sum_{g=1}^{m} (BasePrice_g \times Quantity_g)}{\sum_{g=1}^{m} (BasePrice_g \times Quantity_g)}$$

*(Trong đó: $m$ là số lượng mặt hàng thuộc danh mục $i$ trong báo giá $Q$; $QuotedPrice_g$ là đơn giá bán thực tế chốt với khách hàng).*

Để bảo vệ mô hình trực tuyến khỏi hiện tượng bùng nổ cục bộ khi nhân sự chốt được các đơn hàng siêu lợi nhuận (Markup vượt quá 100% giá vốn), hoặc khi hệ thống tiếp nhận các sai lệch đơn vị đo lường thô, một bộ gác cổng dữ liệu phi tuyến (Data Sanitization Guard) được thiết lập để chuẩn hóa biên độ của biến số trước khi đưa vào bộ nhớ tích lũy:

$$\widetilde{Margin}_{Q, i} = \begin{cases} \frac{Margin_{Q, i}^{thô}}{100} & \text{nếu } 1.0 < Margin_{Q, i}^{thô} \le 100.0 \\ 0.3 & \text{nếu } Margin_{Q, i}^{thô} > 100.0 \lor Margin_{Q, i}^{thô} < 0 \\ Margin_{Q, i}^{thô} & \text{nếu } 0 \le Margin_{Q, i}^{thô} \le 1.0 \end{cases}$$

*(Với mốc $0.3$ là hằng số thiết lập an toàn mặc định ứng với biên lợi nhuận trung bình ngành đạt 30%).*

Sau đó, tiến hành cập nhật dữ liệu tích lũy chuyên môn:

$$SuccessCount_i = SuccessCount_i + 1$$

$$TotalMarginAccumulated_i = TotalMarginAccumulated_i + \widetilde{Margin}_{Q, i}$$

* **Trường hợp báo giá bị khách hàng từ chối ($y = 0$):**
Biên lợi nhuận thực tế của giao dịch không được thiết lập. Hệ thống giữ nguyên chỉ số lợi nhuận tích lũy và chỉ cập nhật tần suất thất bại của nhân sự tại danh mục $i$:

$$FailureCount_i = FailureCount_i + 1$$

#### c) Cập nhật siêu tham số

Hệ thống điều chỉnh các siêu tham số vận hành toàn cục dựa trên nguyên lý toán học của Vector Gradient (tập hợp các đạo hàm riêng đa biến để tìm hướng dốc cực tiểu). Tiến trình lan truyền ngược thực hiện tính toán giá trị biến thiên Gradient thô ($\Delta k$ và $\Delta IdleWeight$) cho từng chu kỳ dựa trên đạo hàm riêng của hàm mục tiêu theo từng siêu tham số:

$$\Delta k = (y - \hat{y}) \cdot \left[ -AggregatedSkillScore \cdot Penalty_{workload} \cdot \ln(CurrentWorkload + 1) \right]$$

$$\Delta IdleWeight = (y - \hat{y}) \cdot \tanh\left(\frac{IdleHours}{24}\right)$$

Luật cập nhật chính thức tích hợp các toán tử điều hướng, tốc độ học và cơ chế kìm hãm biên độ bước nhảy được xác định nhất quán như sau:

$$k_{new} = \max\left(0.0,\ k_{old} + \alpha \cdot \text{clip}(\Delta k, -1.0, 1.0)\right)$$

$$IdleWeight_{new} = \max\left(0.0,\ IdleWeight_{old} + \alpha \cdot \text{clip}(\Delta IdleWeight, -1.0, 1.0)\right)$$

*(Trong đó $\alpha$ là tốc độ học - Learning Rate được thiết lập cố định ở ngưỡng $0.01$).*

#### d) Cơ chế kiểm soát và ổn định không gian tham số

Do đặc thù luồng dữ liệu RFQ trong môi trường B2B có thể xuất hiện các chuỗi biến động dị biệt (Outliers) hoặc nhiễu hệ thống, hai hàm kiểm soát toán học $\text{clip}$ và $\max$ được lồng ghép trực tiếp vào quy trình cập nhật tại Mục 5.1.3.c nhằm bảo vệ mô hình:

* **Hàm cắt cụm toán Gradient ($\text{clip}$):** Hoạt động dựa trên nguyên lý giới hạn biên độ bước nhảy (Gradient Clipping) trong không gian dải an toàn $[-1.0, 1.0]$:

$$\text{clip}(x, -1.0, 1.0) = \begin{cases} -1.0 & \text{nếu } x < -1.0 \\ 1.0 & \text{nếu } x > 1.0 \\ x & \text{nếu } -1.0 \le x \le 1.0 \end{cases}$$

Cơ chế này chặn đứng rủi ro bùng nổ Gradient (Gradient Explosion) khi các giá trị đặc trưng đầu vào tăng trưởng lớn. Nó bắt buộc AI chỉ được thực hiện những bước đi ngắn và mịn màng, giúp quỹ đạo tối ưu hội tụ ổn định thay vì nhảy vọt một cách hỗn loạn qua điểm cực trị.

* **Hàm hình chiếu chặn dưới ($\max$):** Áp đặt điều kiện biên nghiêm ngặt nhằm bảo toàn tính logic nghiệp vụ của hệ thống:

$$\max(0.0, \theta_{new}) = \begin{cases} 0.0 & \text{nếu } \theta_{new} < 0.0 \\ \theta_{new} & \text{nếu } \theta_{new} \ge 0.0 \end{cases}$$

Phép toán này triệt tiêu hoàn toàn nguy cơ đảo ngược nghiệm toán học (ngăn không cho hằng số $k$ và $IdleWeight$ rơi vào miền số âm). Từ đó, bảo vệ mô hình khỏi các trạng thái biến tướng sai lệch (như hệ số trừng phạt tải trọng chuyển dịch thành khuyến khích quá tải, hoặc trọng số nhàn rỗi biến đổi thành trừng phạt nhân sự đang chờ việc).

---

### 5.1.4. Phân tích Toán học Gradient

#### a) Đạo hàm theo siêu tham số tải trọng ($k$)

Mục này khai triển chứng minh giải tích cho Thành phần toán học trong luật cập nhật hệ số trừng phạt $k$. Tiến hành lấy đạo hàm riêng của hàm số $FinalScore$ (định nghĩa tại Mục 5.1.2.a) theo biến số $k_{old}$:

$$\frac{\partial FinalScore}{\partial k_{old}} = \frac{\partial}{\partial k_{old}} \left[ AggregatedSkillScore \cdot (CurrentWorkload + 1)^{-k_{old}} + Boost_{idle} \right]$$

Do thành phần bổ trợ thời gian nhàn rỗi ($Boost_{idle}$) hoàn toàn độc lập với biến số $k_{old}$, đạo hàm riêng của cụm này bằng 0. Áp dụng quy tắc đạo hàm hàm số mũ dạng $(a^{-x})' = -a^{-x} \cdot \ln(a)$ với $a = (CurrentWorkload + 1)$ and biến số $x = k_{old}$:

$$\frac{\partial FinalScore}{\partial k_{old}} = AggregatedSkillScore \cdot \left[ -(CurrentWorkload + 1)^{-k_{old}} \cdot \ln(CurrentWorkload + 1) \right]$$

Thu gọn thành phần cấu trúc mũ về dạng hàm gốc $Penalty_{workload}$:

$$\frac{\partial FinalScore}{\partial k_{old}} = -AggregatedSkillScore \cdot Penalty_{workload} \cdot \ln(CurrentWorkload + 1)$$

**Ý nghĩa học máy:** Cụm $\ln(CurrentWorkload + 1)$ đóng vai trò là **Hệ số khuếch đại độ nhạy**. Tải thấp ($W=1 \rightarrow \ln(2) \approx 0.69$) mức độ điều chỉnh diễn ra tiệm tiến; tải cao ($W=10 \rightarrow \ln(11) \approx 2.40$) biên độ điều chỉnh được khuếch đại mạnh hơn đối với trạng thái quá tải của hệ thống.

#### b) Đạo hàm theo siêu tham số thời gian rảnh ($IdleWeight$)

Tiến hành lấy đạo hàm riêng của hàm số $FinalScore$ theo biến số $IdleWeight_{old}$:

$$\frac{\partial FinalScore}{\partial IdleWeight_{old}} = \frac{\partial}{\partial IdleWeight_{old}} \left[ AggregatedSkillScore \cdot Penalty_{workload} + \tanh\left(\frac{IdleHours}{24}\right) \times IdleWeight_{old} \right]$$

Do thành phần năng lực và tải trọng đóng vai trò là hằng số đối với biến $IdleWeight_{old}$, đạo hàm riêng của cụm này bị triệt tiêu. Áp dụng quy tắc đạo hàm hàm bậc nhất tuyến tính dạng $f(x) = A \cdot x \implies f'(x) = A$ với hằng số $A = \tanh\left(\frac{IdleHours}{24}\right)$ và biến số $x = IdleWeight_{old}$:

$$\frac{\partial FinalScore}{\partial IdleWeight_{old}} = \tanh\left(\frac{IdleHours}{24}\right)$$

**Ý nghĩa học máy:** Việc lấy đạo hàm riêng theo chính siêu tham số giúp hướng điều chỉnh của véc-tơ Gradient luôn tỷ lệ thuận với mức độ nhàn rỗi thực tế của nhân sự ($\tanh$). Nhân sự rảnh càng lâu, trọng lượng đóng góp của sai số phản hồi vào siêu tham số $IdleWeight$ càng mạnh mẽ, giúp hệ thống tự động bám sát và đưa ra quyết định "bù ga" điều phối chính xác, tránh hiện tượng trơ lì tham số của các công thức lỗi cũ.

#### c) Chứng minh toán học quy trình triệt tiêu Gradient tổng thể

Mục này chứng minh nguồn gốc toán học của Thành phần toán học dùng trong các công thức cập nhật thô tại Mục 5.1.3.c. Áp dụng quy tắc chuỗi toán giải tích đối với hàm hợp để tính đạo hàm riêng của Hàm mất mát ($L$) theo $FinalScore$:

$$\frac{\partial L}{\partial FinalScore} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial FinalScore}$$

* **Bước 1: Tính đạo hàm riêng của Hàm mất mát $L$ theo biến dự báo $\hat{y}$**
Từ phương trình gốc tại Mục 5.1.3.a: $L = - [y \cdot \ln(\hat{y}) + (1 - y) \cdot \ln(1 - \hat{y})]$:

$$\frac{\partial L}{\partial \hat{y}} = - \left[ \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right] = - \left[ \frac{y(1 - \hat{y}) - \hat{y}(1 - y)}{\hat{y}(1 - \hat{y})} \right] = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$

* **Bước 2: Tính đạo hàm riêng của hàm kích hoạt Sigmoid theo biến $FinalScore$**
Từ phương trình hàm kích hoạt: $\hat{y} = \frac{1}{1 + e^{-FinalScore}}$. Đạo hàm riêng biểu diễn qua chính hàm gốc $\hat{y}$:

$$\frac{\partial \hat{y}}{\partial FinalScore} = \hat{y}(1 - \hat{y})$$

* **Bước 3: Tổng hợp và thực hiện phép triệt tiêu cơ số**
Thay thế kết quả từ Bước 1 và Bước 2 vào phương trình quy tắc chuỗi ban đầu:

$$\frac{\partial L}{\partial FinalScore} = \left[ \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})} \right] \cdot \left[ \hat{y}(1 - \hat{y}) \right]$$

Triệt tiêu đại lượng $\hat{y}(1 - \hat{y})$ đồng nhất giữa tử số và mẫu số:

$$\frac{\partial L}{\partial FinalScore} = \hat{y} - y = - (y - \hat{y})$$

**Kết luận:** Phép chứng minh giải tích khẳng định hướng dịch chuyển của Gradient để giảm thiểu hàm mất mát là $- (y - \hat{y})$. Khi áp dụng thuật toán dịch chuyển ngược hướng Gradient (Gradient Descent) để cập nhật tham số ($\theta_{new} = \theta_{old} - \alpha \cdot \frac{\partial L}{\partial \theta}$), dấu trừ của thuật toán và dấu trừ của cấu trúc Gradient tự động triệt tiêu lẫn nhau.

Hệ quả là Gradient tổng thể được biểu diễn trực tiếp dưới dạng tuyến tính cộng thuận với Sai số dự báo $(y - \hat{y})$, giúp đơn giản hóa đáng kể số lượng phép toán cần thực hiện trong quá trình cập nhật tham số của mô hình Học máy trực tuyến khi vận hành thực tế.
