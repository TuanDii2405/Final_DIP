# TIEU LUAN

## De tai

Xay dung he thong ngan hang co tich hop AI trong nhan dien khuon mat.

## Thong tin sinh vien

- Ho va ten: [Dien ho ten]
- MSSV: [Dien MSSV]
- Lop: [Dien lop]
- Mon hoc: [Dien ten mon]
- Giang vien huong dan: [Dien ten GV]
- Ngay nop: [Dien ngay]

## Tom tat

[Tom tat 150-250 tu ve muc tieu, phuong phap, ket qua va dong gop cua de tai.]

## Tu khoa

- Ngan hang so
- Nhan dien khuon mat
- AI
- OCR CCCD
- Xac thuc sinh trac hoc

---

## Chương 1. Giới thiệu

### 1.1 Lý do chọn dự án

Trong những năm gần đây, chuyển đổi số trong lĩnh vực tài chính - ngân hàng diễn ra rất nhanh.
Người dùng có xu hướng thực hiện phần lớn giao dịch trên kênh số thay vì đến quầy truyền thống.
Khi lưu lượng giao dịch trực tuyến tăng, bài toán xác thực danh tính trở thành yêu cầu cốt lõi.
Nếu xác thực không tốt, hệ thống dễ bị gian lận, chiếm đoạt tài khoản và lộ dữ liệu nhạy cảm.
Nếu xác thực quá phức tạp, người dùng dễ bỏ cuộc vì trải nghiệm không thuận tiện.
Do đó, cần có giải pháp vừa an toàn vừa dễ sử dụng.

Phương thức xác thực bằng mật khẩu vẫn phổ biến nhưng có nhiều hạn chế.
Mật khẩu có thể bị đoán, bị lộ, bị tái sử dụng giữa nhiều nền tảng.
Người dùng thường đặt mật khẩu dễ nhớ, dẫn đến giảm mức độ bảo mật.
Trong bối cảnh tấn công mạng ngày càng tinh vi, chỉ dùng mật khẩu là không đủ.

Xác thực sinh trắc học, đặc biệt là nhận diện khuôn mặt, mở ra hướng tiếp cận mới.
Khuôn mặt là đặc điểm gắn liền với mỗi người, khó chia sẻ hơn so với mật khẩu.
Khi kết hợp với kiểm tra liveness, hệ thống có thể giảm nguy cơ giả mạo bằng ảnh hoặc video.
Ngoài ra, với người dùng cuối, việc đưa mặt vào camera thường nhanh và tự nhiên hơn việc nhập mật khẩu dài.

Bên cạnh xác thực khuôn mặt, bài toán onboarding/KYC cũng là phần quan trọng.
Nếu nhập tay thông tin từ CCCD, tỉ lệ sai sót cao và tốn nhiều thời gian xử lý.
OCR có thể hỗ trợ đọc thông tin từ giấy tờ tự động, giảm công việc thủ công.
Khi kết hợp OCR với đối chiếu khuôn mặt CCCD và ảnh chụp live, độ tin cậy của đăng ký được nâng cao.

Từ những lý do trên, dự án "Xây dựng hệ thống ngân hàng có tích hợp AI trong nhận diện khuôn mặt" được lựa chọn.
dự án nhằm giải quyết đồng thời ba bài toán.
Một là xác thực người dùng an toàn hơn.
Hai là tối ưu trải nghiệm đăng ký và đăng nhập.
Ba là xây dựng khung hệ thống có tính thực tiễn để mở rộng.

Ngoài giá trị học thuật, dự án còn có ý nghĩa thực hành kỹ thuật.
Sinh viên có cơ hội kết hợp kiến thức backend, frontend, AI/CV, cơ sở dữ liệu và an toàn thông tin.
Qua đó, sản phẩm cuối cùng không chỉ là mô tả lý thuyết mà là một hệ thống có thể chạy thử và đánh giá được.

### 1.2 Mục tiêu dự án

Mục tiêu tổng quát của dự án là xây dựng một hệ thống mô phỏng ngân hàng số có tích hợp AI.
Hệ thống phải thể hiện được quy trình nghiệp vụ rõ ràng, có khả năng vận hành ổn định trong môi trường thử nghiệm.
Đồng thời, hệ thống phải chứng minh được giá trị của thành phần AI trong bài toán xác thực.

Từ mục tiêu tổng quát, dự án đặt ra các mục tiêu cụ thể như sau.

1. Xây dựng được bộ chức năng cốt lõi của một hệ thống ngân hàng mô phỏng.
2. Hỗ trợ đăng ký tài khoản, đăng nhập, quản lý hồ sơ, quản trị người dùng.
3. Hỗ trợ giao dịch chuyển khoản nội địa và lưu vết lịch sử giao dịch.
4. Tích hợp nhận diện khuôn mặt vào luồng đăng nhập.
5. Tích hợp kiểm tra liveness để giảm nguy cơ gian lận bằng hình ảnh giả.
6. Tích hợp OCR CCCD để trích xuất thông tin và hỗ trợ KYC.
7. Thiết kế hệ thống theo hướng có thể mở rộng trong các giai đoạn tiếp theo.

Mục tiêu kỹ thuật của dự án được cụ thể hóa thành các nhóm.

Nhóm 1 là mục tiêu về kiến trúc hệ thống.
Hệ thống cần tách được lớp giao diện, lớp xử lý nghiệp vụ và lớp dữ liệu.
Cần có cơ chế session và phân quyền để đảm bảo đúng vai trò user/admin.

Nhóm 2 là mục tiêu về AI/CV.
Hệ thống cần trích xuất được đặc trưng khuôn mặt từ ảnh đầu vào.
Hệ thống cần so khớp embedding với ngưỡng xác định để quyết định chấp nhận hay từ chối.
Hệ thống cần có bước kiểm tra sống (liveness) trước khi thực hiện verify.

Nhóm 3 là mục tiêu về nghiệp vụ và trải nghiệm.
Người dùng phải có luồng thao tác rõ ràng, dễ hiểu, ít bước dư thừa.
Trạng thái tài khoản (pending, approved, rejected, locked) phải được thể hiện minh bạch.
Admin phải có công cụ duyệt và quản trị dữ liệu thuận tiện.

Nhóm 4 là mục tiêu về đánh giá.
Hệ thống cần được kiểm thử thông qua các tình huống nghiệp vụ chính.
Kết quả cần cho thấy các chức năng vận hành đúng và dữ liệu được lưu trữ nhất quán.

Với các mục tiêu trên, dự án hướng đến một sản phẩm hoàn chỉnh ở mức học tập - ứng dụng.
Sản phẩm không thay thế hệ thống core banking thực tế.
Tuy nhiên, sản phẩm đủ để chứng minh hướng tiếp cận tích hợp AI trong xác thực ngân hàng số.

### 1.3 Phạm vi dự án

Phạm vi dự án được xác định rõ để đảm bảo tính khả thi trong thời gian thực hiện.
Việc xác định phạm vi giúp tránh tình trạng mở rộng quá lớn, dẫn đến không hoàn thành được sản phẩm.

Trong phạm vi "có thực hiện", dự án bao gồm:

1. Xây dựng giao diện đăng ký, đăng nhập, dashboard user, dashboard admin.
2. Hỗ trợ đăng ký với thông tin cơ bản và ảnh khuôn mặt.
3. Hỗ trợ upload CCCD mặt trước, mặt sau và OCR trích xuất thông tin.
4. Kiểm tra trùng tài khoản và trùng thông tin CCCD ở mức nghiệp vụ cần thiết.
5. Hỗ trợ đăng nhập bằng mật khẩu.
6. Hỗ trợ đăng nhập bằng khuôn mặt kết hợp kiểm tra liveness.
7. Hỗ trợ cập nhật khuôn mặt sau đăng nhập.
8. Hỗ trợ cập nhật KYC và đổi mật khẩu.
9. Hỗ trợ chuyển khoản nội địa giữa hai tài khoản trong hệ thống.
10. Hỗ trợ xem lịch sử giao dịch và thống kê cơ bản.
11. Hỗ trợ admin duyệt tài khoản, khóa/mở khóa, reset dữ liệu khuôn mặt, xóa tài khoản.
12. Hỗ trợ lưu vết hồ sơ đã xóa để phục vụ kiểm tra.

Trong phạm vi "không thực hiện" hoặc "chưa đạt mức độ sẵn sàng":

1. Không tích hợp trực tiếp với core banking thực tế.
2. Không kết nối liên ngân hàng, không xử lý chuyển khoản liên ngân hàng.
3. Không triển khai hạ tầng production quy mô lớn.
4. Không bao gồm đầy đủ yêu cầu tuân thủ pháp lý như một tổ chức tài chính thật.
5. Không bao gồm quy trình vận hành 24/7 với SLA chuẩn doanh nghiệp lớn.
6. Không bao gồm hệ thống phòng thủ bảo mật đa lớp ở mức cao nhất.
7. Không thay thế các cơ chế xác thực bắt buộc của ngân hàng thương mại.

Phạm vi trên phù hợp với đặc thù dự án học phần/tiểu luận.
Mục tiêu là xây dựng prototype có giá trị chuyên môn, không phải triển khai thương mại ngay lập tức.

Ngoài ra, dự án tập trung vào tính đúng đắn về nghiệp vụ và luồng xử lý dữ liệu.
Do đó, các tiêu chí giao diện sẽ được ưu tiên ở mức "dễ sử dụng" thay vì "tối ưu mỹ thuật".
Tiêu chí hiệu năng được đánh giá trên dữ liệu và lưu lượng tải hợp lý cho môi trường học tập.

Việc quy định rõ phạm vi cũng giúp người đọc đánh giá đúng kết quả dự án.
Nếu đánh giá theo tiêu chí production banking, dự án sẽ chưa đầy đủ.
Nếu đánh giá theo tiêu chí hệ thống mô phỏng có tích hợp AI và có khả năng chạy thực tế, dự án là phù hợp.

### 1.4 Cấu trúc báo cáo

Báo cáo được tổ chức thành các chương theo trình tự từ lý thuyết đến triển khai.
Mỗi chương có vai trò riêng và liên kết logic với nhau.

Chương 1 trình bày bối cảnh, tính cấp thiết, mục tiêu, phạm vi và hướng tiếp cận của dự án.
Đây là nền tảng để người đọc hiểu vì sao dự án được hình thành và dự án giải quyết vấn đề gì.

Chương 2 trình bày phân tích yêu cầu và nghiệp vụ của hệ thống.
Nội dung bao gồm xác định các bên liên quan, yêu cầu chức năng theo từng vai trò, yêu cầu phi chức năng và các luồng nghiệp vụ chính.
Chương này là cơ sở để thiết kế và triển khai ở các chương sau.

Chương 3 trình bày phân tích và thiết kế hệ thống.
Người đọc sẽ thấy được yêu cầu chức năng, yêu cầu phi chức năng, kiến trúc tổng thể, thiết kế dữ liệu và các luồng nghiệp vụ chính.
Mục đích của chương này là xác lập bản thiết kế rõ ràng trước khi vào triển khai.

Chương 4 trình bày quá trình triển khai hệ thống.
Nội dung gồm công nghệ sử dụng, mô tả module, API tiêu biểu và cách chạy thử.
Chương này cho thấy sản phẩm được hiện thực như thế nào trong thực tế.

Chương 5 trình bày đánh giá kết quả.
Bao gồm kết quả đạt được, ưu điểm, hạn chế và hướng phát triển.
Đây là phần kết nối giữa sản phẩm hiện tại và các mở rộng trong tương lai.

Phần kết luận tổng hợp lại đóng góp chính của dự án.
Phần tài liệu tham khảo và phụ lục cung cấp bằng chứng và nguồn thông tin bổ trợ.

Với cấu trúc trên, người đọc có thể theo dõi dự án theo hướng từ "tại sao" đến "làm gì", rồi đến "làm như thế nào", và cuối cùng là "đạt được gì".

### 1.5 Đối tượng nghiên cứu và đối tượng áp dụng

dự án hướng đến đối tượng nghiên cứu là mô hình xác thực người dùng trong ứng dụng ngân hàng số.
Trong mô hình này, hệ thống phải cân bằng giữa ba yếu tố.
Một là mức độ an toàn.
Hai là trải nghiệm người dùng.
Ba là khả năng triển khai thực tế ở quy mô vừa và nhỏ.

Đối tượng áp dụng trực tiếp của hệ thống bao gồm:

1. Người dùng cuối có nhu cầu đăng ký và giao dịch trên nền tảng số.
2. Quản trị viên cần công cụ duyệt và quản lý tài khoản.
3. Nhóm phát triển cần một bộ mẫu hệ thống để nghiên cứu mở rộng.

Trong bối cảnh học tập, đối tượng áp dụng có thể là lớp học, phòng thí nghiệm hoặc nhóm nghiên cứu nhỏ.
Trong bối cảnh nghiên cứu, hệ thống có thể làm mẫu đối sánh với các phương án xác thực khác.

dự án không chỉ tập trung vào một thủ tục kỹ thuật đơn lẻ.
Thay vào đó, dự án xem toàn bộ vòng đời sử dụng tài khoản.
Từ đăng ký, duyệt, đăng nhập, cập nhật thông tin, đến giao dịch và giám sát.

Cần tiếp cận theo hướng hệ thống như vậy để thấy được giá trị thực sự của AI.
Nếu chỉ tách riêng module nhận diện khuôn mặt, ta khó đánh giá được tác động lên nghiệp vụ tổng thể.

### 1.6 Phương pháp thực hiện dự án

dự án được thực hiện theo phương pháp kết hợp giữa nghiên cứu lý thuyết và xây dựng sản phẩm.

Bước 1 là khảo sát vấn đề.
Nhóm nghiên cứu tổng hợp các hạn chế của xác thực truyền thống.
Đồng thời tìm hiểu các hướng tiếp cận sinh trắc học trong ứng dụng tài chính.

Bước 2 là định nghĩa yêu cầu.
Từ bối cảnh thực tế, nhóm xác lập danh sách chức năng cần có.
Các chức năng được phân thành nhóm user, admin và hệ thống AI.

Bước 3 là thiết kế kiến trúc.
Nhóm lựa chọn mô hình frontend - backend - database để đảm bảo dễ triển khai và dễ mở rộng.
Thành phần AI được tích hợp thành các endpoint riêng để dễ bảo trì.

Bước 4 là triển khai module.
Tiến hành lập trình backend xử lý nghiệp vụ.
Lập trình frontend cho các màn hình và luồng thao tác.
Tích hợp OCR, face verification và liveness vào các bước cần thiết.

Bước 5 là kiểm thử và hiệu chỉnh.
Nhóm xây dựng các tình huống test cho từng luồng nghiệp vụ.
Kiểm tra lỗi đầu vào, phân quyền, tính nhất quán dữ liệu và kết quả phản hồi.

Bước 6 là đánh giá kết quả.
Tổng hợp những gì đạt được.
Đối chiếu với mục tiêu ban đầu.
Xác định hạn chế và đề xuất hướng phát triển.

Phương pháp trên phù hợp với dự án ứng dụng.
Nó đảm bảo sản phẩm cuối cùng vừa có nền tảng lý thuyết, vừa có bằng chứng triển khai cụ thể.

### 1.7 Đóng góp chính của dự án

dự án có các đóng góp chính sau đây.

Đóng góp thứ nhất là xây dựng được một hệ thống mô phỏng nghiệp vụ ngân hàng có khả năng chạy thực tế.
Hệ thống không dừng ở mức mô tả ý tưởng, mà đã hiện thực thành các luồng thao tác cụ thể.

Đóng góp thứ hai là chứng minh khả năng tích hợp AI vào bài toán xác thực.
AI được áp dụng đúng vị trí nghiệp vụ.
Không chỉ để trình diễn kỹ thuật, mà để nâng cao chất lượng xác thực.

Đóng góp thứ ba là kết hợp OCR CCCD với quy trình đăng ký.
Điều này giúp giảm nhập liệu thủ công và tăng tính nhất quán thông tin.

Đóng góp thứ tư là bổ sung vai trò quản trị viên rõ ràng.
Admin có thể duyệt tài khoản, khóa/mở khóa và can thiệp dữ liệu khi cần.
Thành phần này phản ánh đúng đặc thù quản lý trong hệ thống tài chính.

Đóng góp thứ năm là tạo ra bộ nền để mở rộng.
Với kiến trúc hiện tại, có thể tiếp tục thêm MFA, thông báo, audit, bảo mật nâng cao và tối ưu hiệu năng.

Ngoài đóng góp sản phẩm, dự án còn đóng góp về mặt học tập.
Nó giúp kết nối kiến thức đa lĩnh vực thành một bài toán liên ngành.
Quá trình này tạo kinh nghiệm thực tiễn cho người thực hiện.

### 1.8 Hạn chế và định hướng nghiên cứu tiếp theo

Mặc dù đạt được nhiều kết quả tích cực, dự án vẫn có các giới hạn.

Thứ nhất, hệ thống mới ở mức prototype.
Khả năng đáp ứng tải cao và khả năng chống tấn công nâng cao chưa được kiểm chứng đầy đủ.

Thứ hai, chất lượng nhận diện khuôn mặt phụ thuộc vào chất lượng camera và điều kiện ánh sáng.
Trong môi trường thực tế phức tạp, độ chính xác có thể dao động.

Thứ ba, OCR CCCD có thể gặp khó với ảnh mờ, lệch hoặc bị che mất thông tin.
Cần có cơ chế kiểm tra và can thiệp bổ sung để đảm bảo dữ liệu đúng.

Thứ tư, dự án chưa bao gồm đầy đủ các tiêu chuẩn tuân thủ ngành tài chính ở mức thương mại.
Ví dụ, quản trị khóa mã hóa tập trung, audit chi tiết theo chuẩn doanh nghiệp và quy trình vận hành sự cố.

Từ các hạn chế trên, một số định hướng tiếp theo được đề xuất:

1. Tích hợp xác thực đa yếu tố (MFA) với OTP hoặc token.
2. Nâng cấp mô hình anti-spoofing chuyên sâu hơn.
3. Bổ sung mã hóa dữ liệu nhạy cảm ở mức trường thông tin và kênh truyền.
4. Xây dựng hệ thống log và audit đầy đủ cho mục đích truy vết.
5. Thiết lập bộ test tự động và bộ benchmark hiệu năng.
6. Nghiên cứu mở rộng sang mô hình triển khai microservices khi quy mô tăng.

Nhìn chung, hạn chế là điều thường gặp trong dự án học tập.
Quan trọng là dự án đã tạo được khung nền đúng để tiếp tục nâng cấp.
Đó cũng là giá trị có ý nghĩa nhất của hướng nghiên cứu này.

### 1.9 Tiểu kết chương

Chương 1 đã trình bày bối cảnh, tính cấp thiết, mục tiêu, phạm vi và hướng tiếp cận của dự án.
Nội dung chương xác lập rõ ràng rằng dự án hướng đến một hệ thống ngân hàng mô phỏng có tích hợp AI nhận diện khuôn mặt.
Đồng thời, chương này cũng chỉ ra giới hạn và hướng mở rộng để tránh kỳ vọng sai lệch.
Trên cơ sở đó, các chương tiếp theo sẽ đi sâu vào lý thuyết nền, phân tích thiết kế, triển khai và đánh giá kết quả.

---

## Chương 2. Phân tích yêu cầu và nghiệp vụ

### 2.1 Xác định các bên liên quan

Trước khi phân tích yêu cầu, cần xác định rõ các bên liên quan (stakeholders) và kỳ vọng của từng bên đối với hệ thống.

**Bảng 2.1 – Danh sách các bên liên quan**

| STT | Bên liên quan          | Vai trò                             | Kỳ vọng chính                                                   |
| --- | ---------------------- | ----------------------------------- | --------------------------------------------------------------- |
| 1   | Khách vãng lai (Guest) | Truy cập trang đăng ký, đăng nhập   | Giao diện rõ ràng, đăng ký nhanh chóng                          |
| 2   | Người dùng (User)      | Thực hiện giao dịch, quản lý hồ sơ  | Đăng nhập an toàn, chuyển khoản dễ dàng, thông tin minh bạch    |
| 3   | Quản trị viên (Admin)  | Duyệt tài khoản, quản lý người dùng | Công cụ quản lý đầy đủ, dữ liệu rõ ràng, thao tác nhanh         |
| 4   | Hệ thống AI            | Xử lý nhận diện khuôn mặt, OCR CCCD | Tốc độ phản hồi tốt, độ chính xác phù hợp môi trường thử nghiệm |
| 5   | Nhóm phát triển        | Xây dựng và bảo trì hệ thống        | Kiến trúc rõ ràng, dễ mở rộng và bảo trì                        |

Mỗi bên liên quan có ràng buộc và ưu tiên khác nhau.
Người dùng cuối ưu tiên trải nghiệm mượt mà và phản hồi nhanh.
Quản trị viên ưu tiên tính đầy đủ và độ tin cậy của dữ liệu.
Nhóm phát triển ưu tiên kiến trúc module hóa để dễ bảo trì và nâng cấp.

---

### 2.2 Biểu đồ phân rã chức năng (BFD)

Biểu đồ phân rã chức năng thể hiện toàn bộ các chức năng của hệ thống được tổ chức theo cấp bậc.
Mức cao nhất là tên hệ thống, các mức dưới phân rã theo nhóm chức năng và chức năng cụ thể.

```
HỆ THỐNG NGÂN HÀNG TÍCH HỢP AI
│
├── 1. Quản lý tài khoản
│   ├── 1.1 Đăng ký tài khoản
│   ├── 1.2 Duyệt tài khoản (pending → approved)
│   ├── 1.3 Từ chối tài khoản (pending → rejected)
│   ├── 1.4 Khóa tài khoản (approved → locked)
│   ├── 1.5 Mở khóa tài khoản (locked → approved)
│   └── 1.6 Xóa tài khoản (lưu archive)
│
├── 2. Xác thực người dùng
│   ├── 2.1 Đăng nhập bằng mật khẩu
│   ├── 2.2 Đăng nhập bằng khuôn mặt
│   │   ├── 2.2.1 Kiểm tra liveness
│   │   └── 2.2.2 Trích xuất và so khớp embedding
│   └── 2.3 Đăng xuất
│
├── 3. KYC & Hồ sơ cá nhân
│   ├── 3.1 Upload CCCD (mặt trước / mặt sau)
│   ├── 3.2 OCR trích xuất thông tin CCCD
│   ├── 3.3 Đối chiếu khuôn mặt CCCD với ảnh live
│   ├── 3.4 Cập nhật khuôn mặt mới
│   ├── 3.5 Đổi mật khẩu
│   └── 3.6 Xem thông tin hồ sơ và số dư
│
├── 4. Giao dịch tài chính
│   ├── 4.1 Tra cứu tài khoản nhận
│   ├── 4.2 Chuyển khoản nội địa
│   └── 4.3 Xem lịch sử giao dịch
│
└── 5. Quản trị hệ thống
    ├── 5.1 Xem danh sách toàn bộ người dùng
    ├── 5.2 Lọc theo trạng thái tài khoản
    ├── 5.3 Reset dữ liệu khuôn mặt
    └── 5.4 Xem danh sách hồ sơ đã xóa
```

**Mô tả các nhóm chức năng chính:**

**Nhóm 1 – Quản lý tài khoản** bao gồm toàn bộ vòng đời của một tài khoản người dùng, từ lúc đăng ký, chờ duyệt, được kích hoạt, đến khi bị khóa hoặc xóa. Đây là nhóm chức năng nền tảng của hệ thống.

**Nhóm 2 – Xác thực người dùng** xử lý các phương thức xác minh danh tính. Hệ thống hỗ trợ hai phương thức độc lập: mật khẩu truyền thống và khuôn mặt kết hợp liveness. Mỗi phương thức có luồng xử lý riêng nhưng đều dẫn đến cùng kết quả là cấp hoặc từ chối phiên làm việc.

**Nhóm 3 – KYC & Hồ sơ cá nhân** hỗ trợ người dùng hoàn thiện thông tin định danh. OCR giảm thao tác nhập liệu thủ công. Đối chiếu ảnh CCCD với ảnh chụp live tăng độ tin cậy của quá trình đăng ký.

**Nhóm 4 – Giao dịch tài chính** cung cấp chức năng chuyển khoản cốt lõi và tra cứu lịch sử giao dịch. Chuyển khoản chỉ thực hiện được khi tài khoản đã được duyệt và số dư đủ điều kiện.

**Nhóm 5 – Quản trị hệ thống** tập hợp các công cụ dành riêng cho admin. Admin có quyền can thiệp toàn bộ vòng đời tài khoản và dữ liệu sinh trắc học.

---

### 2.3 Sơ đồ Use Case

#### 2.3.1 Danh sách Use Case

**Bảng 2.2 – Danh sách Use Case theo Actor**

| Mã UC | Tên Use Case                    | Actor chính         |
| ----- | ------------------------------- | ------------------- |
| UC-01 | Đăng ký tài khoản               | Khách vãng lai      |
| UC-02 | Đăng nhập bằng mật khẩu         | Khách vãng lai      |
| UC-03 | Đăng nhập bằng khuôn mặt        | Khách vãng lai + AI |
| UC-04 | Xem thông tin tài khoản         | Người dùng          |
| UC-05 | Cập nhật KYC (upload CCCD, OCR) | Người dùng + AI     |
| UC-06 | Cập nhật khuôn mặt              | Người dùng + AI     |
| UC-07 | Đổi mật khẩu                    | Người dùng          |
| UC-08 | Chuyển khoản nội địa            | Người dùng          |
| UC-09 | Xem lịch sử giao dịch           | Người dùng          |
| UC-10 | Đăng xuất                       | Người dùng          |
| UC-11 | Xem danh sách người dùng        | Admin               |
| UC-12 | Duyệt tài khoản                 | Admin               |
| UC-13 | Từ chối tài khoản               | Admin               |
| UC-14 | Khóa / Mở khóa tài khoản        | Admin               |
| UC-15 | Reset dữ liệu khuôn mặt         | Admin               |
| UC-16 | Xóa tài khoản                   | Admin               |
| UC-17 | Xem hồ sơ đã xóa                | Admin               |

#### 2.3.2 Sơ đồ Use Case – Tổng quát

```
+================================================================+
|           HỆ THỐNG NGÂN HÀNG TÍCH HỢP AI                      |
|                                                                |
|  +--[UC-01] Đăng ký tài khoản                                 |
|  +--[UC-02] Đăng nhập mật khẩu                                |
|  +--[UC-03] Đăng nhập khuôn mặt                               |
|       |--- <<include>> [Kiểm tra liveness]                    |
|       |--- <<include>> [Trích xuất & so khớp embedding]       |
|                                                                |
|  +--[UC-04] Xem thông tin tài khoản                           |
|  +--[UC-05] Cập nhật KYC                                      |
|       |--- <<include>> [OCR trích xuất CCCD]                  |
|  +--[UC-06] Cập nhật khuôn mặt                                |
|  +--[UC-07] Đổi mật khẩu                                      |
|  +--[UC-08] Chuyển khoản nội địa                              |
|       |--- <<include>> [Tra cứu tài khoản nhận]               |
|  +--[UC-09] Xem lịch sử giao dịch                             |
|  +--[UC-10] Đăng xuất                                         |
|                                                                |
|  +--[UC-11] Xem danh sách người dùng                          |
|  +--[UC-12] Duyệt tài khoản                                   |
|  +--[UC-13] Từ chối tài khoản                                 |
|  +--[UC-14] Khóa / Mở khóa tài khoản                         |
|  +--[UC-15] Reset dữ liệu khuôn mặt                           |
|  +--[UC-16] Xóa tài khoản                                     |
|  +--[UC-17] Xem hồ sơ đã xóa                                  |
+================================================================+
        |                    |                      |
  Khách vãng lai        Người dùng           Quản trị viên
  (UC-01, 02, 03)    (UC-04 → UC-10)       (UC-11 → UC-17)
                                                    |
                                             Hệ thống AI
                                         (UC-03, UC-05, UC-06)
```

#### 2.3.3 Sơ đồ Use Case – Nhóm Khách vãng lai

Khách vãng lai là người chưa có tài khoản hoặc chưa đăng nhập.
Các use case dành cho nhóm này tập trung vào onboarding và xác thực đầu vào.

```
Khách vãng lai -----> [UC-01] Đăng ký tài khoản
                       |--- <<include>> Trích xuất embedding khuôn mặt (AI)

Khách vãng lai -----> [UC-02] Đăng nhập mật khẩu

Khách vãng lai -----> [UC-03] Đăng nhập khuôn mặt
                       |--- <<include>> [Kiểm tra liveness] (AI)
                       |--- <<include>> [So khớp embedding] (AI)
```

Quan hệ `<<include>>` với AI nghĩa là mỗi khi UC-03 được thực thi, hệ thống AI bắt buộc phải tham gia vào xử lý.
Nếu hệ thống AI không phản hồi, use case sẽ thất bại.

#### 2.3.4 Sơ đồ Use Case – Nhóm Người dùng

Người dùng đã đăng nhập thành công và tài khoản đang ở trạng thái `approved`.
Nhóm này có quyền truy cập toàn bộ chức năng dành cho end-user.

```
Người dùng -----> [UC-04] Xem thông tin tài khoản và số dư
Người dùng -----> [UC-05] Cập nhật KYC
                   |--- <<include>> [OCR trích xuất CCCD] (AI)
Người dùng -----> [UC-06] Cập nhật khuôn mặt mới
                   |--- <<include>> [Trích xuất embedding mới] (AI)
Người dùng -----> [UC-07] Đổi mật khẩu
Người dùng -----> [UC-08] Chuyển khoản nội địa
                   |--- <<include>> [Tra cứu tài khoản nhận]
Người dùng -----> [UC-09] Xem lịch sử giao dịch
Người dùng -----> [UC-10] Đăng xuất
```

#### 2.3.5 Sơ đồ Use Case – Nhóm Quản trị viên

Quản trị viên đăng nhập bằng tài khoản có trường `role = admin`.
Admin kế thừa toàn bộ quyền người dùng và có thêm quyền quản trị.

```
Admin -----> [UC-11] Xem danh sách toàn bộ người dùng
              |--- <<extend>> Lọc theo trạng thái (pending/approved/locked/rejected)

Admin -----> [UC-12] Duyệt tài khoản (pending → approved)
Admin -----> [UC-13] Từ chối tài khoản (pending → rejected)
Admin -----> [UC-14] Khóa tài khoản   (approved → locked)
              |--- <<extend>> Mở khóa tài khoản (locked → approved)
Admin -----> [UC-15] Reset dữ liệu khuôn mặt
Admin -----> [UC-16] Xóa tài khoản
              |--- <<include>> Lưu archive vào deleted_profiles
Admin -----> [UC-17] Xem danh sách hồ sơ đã xóa
```

---

### 2.4 Đặc tả Use Case

#### UC-01: Đăng ký tài khoản

| Trường             | Nội dung                                                         |
| ------------------ | ---------------------------------------------------------------- |
| **Mã UC**          | UC-01                                                            |
| **Tên**            | Đăng ký tài khoản                                                |
| **Actor**          | Khách vãng lai                                                   |
| **Mục tiêu**       | Tạo tài khoản mới và gửi yêu cầu chờ admin duyệt                 |
| **Tiền điều kiện** | Người dùng chưa có tài khoản trong hệ thống                      |
| **Hậu điều kiện**  | Tài khoản được tạo với trạng thái `pending`, chờ admin xét duyệt |

**Luồng chính (Basic Flow):**

| Bước | Actor                                       | Hệ thống                                                   |
| ---- | ------------------------------------------- | ---------------------------------------------------------- |
| 1    | Truy cập trang đăng ký                      | Hiển thị form đăng ký                                      |
| 2    | Nhập họ tên, email, số điện thoại, mật khẩu | —                                                          |
| 3    | Chụp hoặc upload ảnh khuôn mặt              | —                                                          |
| 4    | —                                           | Gọi AI trích xuất embedding khuôn mặt                      |
| 5    | —                                           | Kiểm tra email chưa tồn tại trong hệ thống                 |
| 6    | —                                           | Hash mật khẩu (bcrypt), tạo tài khoản trạng thái `pending` |
| 7    | —                                           | Hiển thị thông báo đăng ký thành công, nhắc chờ duyệt      |

**Luồng thay thế (Alternative Flow):**

- **3a.** Không phát hiện khuôn mặt trong ảnh → hệ thống thông báo lỗi, yêu cầu chụp lại.
- **5a.** Email đã tồn tại → hệ thống thông báo email trùng, không tạo tài khoản mới.

**Luồng ngoại lệ (Exception Flow):**

- Mất kết nối trong quá trình upload → thông báo lỗi mạng, giữ nguyên dữ liệu đã nhập trên form.

**Yêu cầu dữ liệu:**

- Email: định dạng hợp lệ, chưa được sử dụng.
- Mật khẩu: tối thiểu 6 ký tự.
- Ảnh khuôn mặt: phải phát hiện được ít nhất một khuôn mặt rõ nét.

---

#### UC-02: Đăng nhập bằng mật khẩu

| Trường             | Nội dung                                                          |
| ------------------ | ----------------------------------------------------------------- |
| **Mã UC**          | UC-02                                                             |
| **Tên**            | Đăng nhập bằng mật khẩu                                           |
| **Actor**          | Khách vãng lai                                                    |
| **Mục tiêu**       | Xác thực danh tính bằng email và mật khẩu, cấp phiên làm việc     |
| **Tiền điều kiện** | Tài khoản tồn tại, trạng thái `approved`                          |
| **Hậu điều kiện**  | Phiên làm việc được tạo trong session, chuyển hướng đến dashboard |

**Luồng chính:**

| Bước | Actor                  | Hệ thống                                         |
| ---- | ---------------------- | ------------------------------------------------ |
| 1    | Nhập email và mật khẩu | Hiển thị form đăng nhập                          |
| 2    | Nhấn "Đăng nhập"       | —                                                |
| 3    | —                      | Truy vấn tài khoản theo email                    |
| 4    | —                      | So khớp mật khẩu với hash đã lưu (bcrypt.verify) |
| 5    | —                      | Kiểm tra trạng thái tài khoản                    |
| 6    | —                      | Tạo session, ghi `user_id` vào session storage   |
| 7    | —                      | Chuyển hướng đến `/dashboard`                    |

**Luồng thay thế:**

- **4a.** Mật khẩu sai → trả về thông báo chung "Email hoặc mật khẩu không đúng" (không phân biệt lỗi).
- **5a.** Tài khoản `pending` → thông báo "Tài khoản đang chờ xét duyệt".
- **5b.** Tài khoản `locked` → thông báo "Tài khoản đã bị khóa, liên hệ quản trị viên".
- **5c.** Tài khoản `rejected` → thông báo "Đăng ký đã bị từ chối".

**Ghi chú bảo mật:**

Thông báo lỗi không phân biệt "sai email" và "sai mật khẩu" để tránh user enumeration attack.

---

#### UC-03: Đăng nhập bằng khuôn mặt

| Trường             | Nội dung                                                                     |
| ------------------ | ---------------------------------------------------------------------------- |
| **Mã UC**          | UC-03                                                                        |
| **Tên**            | Đăng nhập bằng khuôn mặt                                                     |
| **Actor**          | Khách vãng lai, Hệ thống AI                                                  |
| **Mục tiêu**       | Xác thực danh tính bằng sinh trắc học, cấp phiên làm việc                    |
| **Tiền điều kiện** | Tài khoản tồn tại, trạng thái `approved`, đã có `face_encoding` lưu trong DB |
| **Hậu điều kiện**  | Phiên làm việc được tạo, chuyển hướng đến dashboard                          |
| **Include**        | UC-03a: Kiểm tra liveness; UC-03b: Trích xuất và so khớp embedding           |

**Luồng chính:**

| Bước | Actor                                             | Hệ thống / AI                                                     |
| ---- | ------------------------------------------------- | ----------------------------------------------------------------- |
| 1    | Nhập email                                        | Kiểm tra email tồn tại và có face encoding                        |
| 2    | Cho phép truy cập camera                          | Hiển thị khung camera, hướng dẫn thực hiện liveness               |
| 3    | Thực hiện hành động liveness (nháy mắt / gật đầu) | —                                                                 |
| 4    | —                                                 | AI phân tích chuỗi frame, xác nhận người thật (anti-spoofing)     |
| 5    | —                                                 | AI trích xuất embedding từ frame khuôn mặt hiện tại               |
| 6    | —                                                 | So sánh embedding mới với embedding đã lưu theo ngưỡng similarity |
| 7    | —                                                 | Kết quả khớp (distance < threshold) → tạo session                 |
| 8    | —                                                 | Chuyển hướng đến dashboard                                        |

**Luồng thay thế:**

- **4a.** Liveness thất bại (phát hiện ảnh giả / video phát lại) → từ chối, không xử lý tiếp.
- **6a.** Embedding không khớp (similarity dưới ngưỡng) → từ chối đăng nhập, gợi ý dùng mật khẩu.

**Luồng ngoại lệ:**

- Camera không khả dụng hoặc bị chặn quyền → thông báo lỗi quyền truy cập camera, gợi ý đăng nhập bằng mật khẩu.

---

#### UC-05: Cập nhật KYC

| Trường             | Nội dung                                                          |
| ------------------ | ----------------------------------------------------------------- |
| **Mã UC**          | UC-05                                                             |
| **Tên**            | Cập nhật KYC (xác minh CCCD)                                      |
| **Actor**          | Người dùng đã đăng nhập, Hệ thống AI (OCR)                        |
| **Mục tiêu**       | Upload CCCD, trích xuất thông tin bằng OCR, lưu vào hồ sơ         |
| **Tiền điều kiện** | Người dùng đã đăng nhập, tài khoản trạng thái `approved`          |
| **Hậu điều kiện**  | Thông tin CCCD được lưu vào bảng `users`, trạng thái KYC cập nhật |
| **Include**        | UC-05a: OCR trích xuất thông tin CCCD                             |

**Luồng chính:**

| Bước | Actor                                 | Hệ thống / AI                                              |
| ---- | ------------------------------------- | ---------------------------------------------------------- |
| 1    | Vào trang hồ sơ, chọn mục KYC         | Hiển thị form upload CCCD                                  |
| 2    | Upload ảnh CCCD mặt trước             | —                                                          |
| 3    | Upload ảnh CCCD mặt sau               | —                                                          |
| 4    | —                                     | Gọi OCR engine phân tích ảnh mặt trước                     |
| 5    | —                                     | OCR trả về: số CCCD, họ tên, ngày sinh, địa chỉ thường trú |
| 6    | —                                     | Kiểm tra số CCCD chưa được đăng ký bởi tài khoản khác      |
| 7    | Xem thông tin đã trích xuất, xác nhận | —                                                          |
| 8    | —                                     | Lưu thông tin vào bảng `users`, đánh dấu KYC hoàn tất      |

**Luồng thay thế:**

- **4a.** OCR không nhận dạng được ảnh (mờ, lệch, thiếu ánh sáng) → thông báo ảnh không đạt yêu cầu, yêu cầu chụp lại.
- **6a.** Số CCCD đã tồn tại trong hệ thống ở tài khoản khác → thông báo lỗi trùng CCCD.

**Luồng ngoại lệ:**

- Ảnh quá lớn hoặc định dạng không hỗ trợ → thông báo yêu cầu ảnh JPG/PNG, dung lượng hợp lệ.

---

#### UC-08: Chuyển khoản nội địa

| Trường             | Nội dung                                                                    |
| ------------------ | --------------------------------------------------------------------------- |
| **Mã UC**          | UC-08                                                                       |
| **Tên**            | Chuyển khoản nội địa                                                        |
| **Actor**          | Người dùng đã đăng nhập                                                     |
| **Mục tiêu**       | Chuyển tiền giữa hai tài khoản trong cùng hệ thống                          |
| **Tiền điều kiện** | Người dùng đã đăng nhập, tài khoản `approved`, số dư > 0                    |
| **Hậu điều kiện**  | Số dư tài khoản nguồn giảm, tài khoản đích tăng, bản ghi giao dịch được lưu |
| **Include**        | UC-08a: Tra cứu tài khoản nhận                                              |

**Luồng chính:**

| Bước | Actor                                              | Hệ thống                                                |
| ---- | -------------------------------------------------- | ------------------------------------------------------- |
| 1    | Vào trang chuyển khoản                             | Hiển thị form chuyển khoản                              |
| 2    | Nhập số tài khoản hoặc email người nhận            | —                                                       |
| 3    | —                                                  | Tra cứu tài khoản nhận, trả về tên hiển thị để xác nhận |
| 4    | Xác nhận đúng người nhận, nhập số tiền và nội dung | —                                                       |
| 5    | Nhấn "Chuyển khoản"                                | —                                                       |
| 6    | —                                                  | Kiểm tra số dư tài khoản nguồn ≥ số tiền                |
| 7    | —                                                  | Thực hiện atomic transaction: trừ nguồn, cộng đích      |
| 8    | —                                                  | Lưu bản ghi vào bảng `bank_transactions`                |
| 9    | —                                                  | Hiển thị thông báo giao dịch thành công, cập nhật số dư |

**Luồng thay thế:**

- **3a.** Số tài khoản không tồn tại → thông báo "Không tìm thấy tài khoản".
- **3b.** Tài khoản nhận trùng tài khoản gửi → thông báo không hợp lệ.
- **3c.** Tài khoản nhận không ở trạng thái `approved` → thông báo tài khoản nhận không hoạt động.
- **6a.** Số dư không đủ → thông báo số dư không đủ để thực hiện giao dịch.

**Luồng ngoại lệ:**

- Lỗi database trong quá trình ghi → rollback toàn bộ, giữ nguyên số dư hai tài khoản, thông báo lỗi.

**Yêu cầu nghiệp vụ:**

- Số tiền chuyển khoản phải > 0.
- Số tiền không được vượt quá số dư hiện tại.

---

#### UC-12: Duyệt tài khoản (Admin)

| Trường             | Nội dung                                                                 |
| ------------------ | ------------------------------------------------------------------------ |
| **Mã UC**          | UC-12                                                                    |
| **Tên**            | Duyệt tài khoản chờ                                                      |
| **Actor**          | Quản trị viên                                                            |
| **Mục tiêu**       | Phê duyệt tài khoản đang ở trạng thái `pending` để kích hoạt             |
| **Tiền điều kiện** | Admin đã đăng nhập, có tài khoản `pending` trong hệ thống                |
| **Hậu điều kiện**  | Tài khoản chuyển sang trạng thái `approved`, người dùng có thể đăng nhập |

**Luồng chính:**

| Bước | Actor                                                  | Hệ thống                                     |
| ---- | ------------------------------------------------------ | -------------------------------------------- |
| 1    | Vào trang quản trị người dùng                          | Hiển thị danh sách người dùng                |
| 2    | Lọc theo trạng thái `pending`                          | Hiển thị danh sách chờ duyệt                 |
| 3    | Chọn tài khoản, xem thông tin đăng ký và ảnh khuôn mặt | —                                            |
| 4    | Nhấn "Duyệt"                                           | Cập nhật `approval_status = approved`        |
| 5    | —                                                      | Hiển thị cập nhật trạng thái trong danh sách |

**Luồng thay thế:**

- **4a.** Admin nhấn "Từ chối" → tài khoản chuyển sang `rejected` (UC-13).

---

#### UC-14: Khóa / Mở khóa tài khoản (Admin)

| Trường             | Nội dung                                                     |
| ------------------ | ------------------------------------------------------------ |
| **Mã UC**          | UC-14                                                        |
| **Tên**            | Khóa hoặc mở khóa tài khoản                                  |
| **Actor**          | Quản trị viên                                                |
| **Mục tiêu**       | Tạm thời vô hiệu hóa hoặc kích hoạt lại tài khoản người dùng |
| **Tiền điều kiện** | Admin đã đăng nhập                                           |
| **Hậu điều kiện**  | Trạng thái tài khoản chuyển đổi giữa `approved` ↔ `locked`   |

**Luồng chính:**

| Bước | Actor                       | Hệ thống                                              |
| ---- | --------------------------- | ----------------------------------------------------- |
| 1    | Tìm tài khoản cần can thiệp | —                                                     |
| 2    | Nhấn "Khóa"                 | Cập nhật `approval_status = locked`                   |
| 3    | —                           | Tài khoản không thể đăng nhập cho đến khi được mở lại |

**Luồng thay thế:**

- **2a.** Nhấn "Mở khóa" trên tài khoản `locked` → chuyển về `approved`.

**Yêu cầu nghiệp vụ:** Admin không thể tự khóa chính tài khoản admin đang sử dụng.

---

#### UC-15: Reset dữ liệu khuôn mặt (Admin)

| Trường             | Nội dung                                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **Mã UC**          | UC-15                                                                                                                |
| **Tên**            | Reset dữ liệu khuôn mặt                                                                                              |
| **Actor**          | Quản trị viên                                                                                                        |
| **Mục tiêu**       | Xóa embedding khuôn mặt của người dùng để họ có thể cập nhật lại                                                     |
| **Tiền điều kiện** | Admin đã đăng nhập, người dùng tồn tại trong hệ thống                                                                |
| **Hậu điều kiện**  | Trường `face_encoding` của người dùng bị xóa; người dùng không thể đăng nhập bằng khuôn mặt cho đến khi cập nhật lại |

**Luồng chính:**

| Bước | Actor                                     | Hệ thống                                       |
| ---- | ----------------------------------------- | ---------------------------------------------- |
| 1    | Chọn người dùng cần reset                 | —                                              |
| 2    | Nhấn "Reset khuôn mặt", xác nhận thao tác | —                                              |
| 3    | —                                         | Xóa giá trị `face_encoding` trong bảng `users` |
| 4    | —                                         | Thông báo reset thành công                     |

**Yêu cầu nghiệp vụ:** Cần bước xác nhận trước khi thực hiện để tránh thao tác nhầm.

---

### 2.5 Biểu đồ trạng thái tài khoản

Tài khoản người dùng trải qua các trạng thái sau trong vòng đời sử dụng.

```
                    +-----------+
        Đăng ký --> |  pending  |
                    +-----------+
                    /            \
          Admin duyệt        Admin từ chối
                /                  \
        +----------+           +----------+
        | approved |           | rejected |
        +----------+           +----------+
            |  ^
  Admin khóa|  |Admin mở khóa
            v  |
        +--------+
        | locked |
        +--------+
            |
       Admin xóa
            |
            v
    [deleted_profiles]
     (lưu archive)
```

**Bảng 2.3 – Mô tả các trạng thái tài khoản**

| Trạng thái | Mô tả                           | Đăng nhập | Giao dịch |
| ---------- | ------------------------------- | --------- | --------- |
| `pending`  | Chờ admin xét duyệt             | Không     | Không     |
| `approved` | Tài khoản hoạt động bình thường | Có        | Có        |
| `rejected` | Đăng ký bị từ chối              | Không     | Không     |
| `locked`   | Tạm thời bị vô hiệu hóa         | Không     | Không     |

---

### 2.6 Tiểu kết chương

Chương 2 đã phân tích toàn diện yêu cầu và nghiệp vụ của hệ thống thông qua các công cụ phân tích tiêu chuẩn.

Biểu đồ phân rã chức năng (BFD) trình bày cấu trúc chức năng từ tổng quan đến chi tiết, gồm 5 nhóm chức năng chính với 20 chức năng con.

Sơ đồ Use Case xác định ranh giới hệ thống, xác lập 17 use case phân bổ cho 3 actor chính (Khách vãng lai, Người dùng, Admin) và 1 actor hỗ trợ (Hệ thống AI).

Đặc tả Use Case chi tiết cho 8 use case trọng tâm cung cấp đủ thông tin về luồng chính, luồng thay thế, luồng ngoại lệ và yêu cầu nghiệp vụ để triển khai chính xác.

Biểu đồ trạng thái tài khoản làm rõ vòng đời và các điều kiện chuyển trạng thái, đảm bảo logic phân quyền nhất quán.

Trên cơ sở phân tích này, Chương 3 sẽ đi vào thiết kế kiến trúc tổng thể và cơ sở dữ liệu của hệ thống.

---

## Chuong 3. Phan tich va thiet ke he thong

### 3.1 Yeu cau chuc nang

- Dang ky tai khoan
- Duyet/tu choi tai khoan
- Dang nhap mat khau va khuon mat
- KYC, cap nhat khuon mat, doi mat khau
- Chuyen khoan noi dia va lich su giao dich
- Quan tri nguoi dung

### 3.2 Yeu cau phi chuc nang

- Hieu nang
- Bao mat
- Kha nang mo rong
- Kha nang bao tri

### 3.3 Kien truc tong the

[Mo ta mo hinh frontend + unified backend + SQLite + uploads.]

### 3.4 Thiet ke co so du lieu

[Mo ta cac bang users, bank_transactions, deleted_profiles va moi quan he.]

### 3.5 Luong nghiep vu chinh

- Dang ky + OCR + doi chieu khuon mat CCCD
- Dang nhap khuon mat + liveness
- Chuyen khoan noi dia
- Xoa user va luu archive

---

## Chuong 4. Trien khai he thong

### 4.1 Cong nghe su dung

- Frontend: HTML/CSS/JavaScript
- Backend: FastAPI (Python)
- AI/CV: face_recognition, OpenCV, EasyOCR/Google Vision
- Database: SQLite

### 4.2 Mo ta cac module chinh

- Module dang ky
- Module dang nhap
- Module KYC/OCR
- Module giao dich
- Module admin

### 4.3 Mo ta API tieu bieu

[Chon 5-8 API quan trong, ghi muc dich, input, output, validate.]

### 4.4 Huong dan chay thu he thong

1. Chay script khoi dong he thong.
2. Truy cap localhost cong 8001.
3. Kiem thu theo checklist nghiep vu.

---

## Chuong 5. Danh gia ket qua

### 5.1 Ket qua dat duoc

[Neu ro cac tinh nang da hoan thanh va do on dinh.]

### 5.2 Uu diem

- Tich hop AI vao xac thuc thanh cong
- Luong nghiep vu ro rang
- Co dashboard quan tri va duyet tai khoan

### 5.3 Han che

- Chua dat muc production banking
- Con phu thuoc vao chat luong camera/anh CCCD
- Chua co day du co che hardening bao mat nang cao

### 5.4 Huong phat trien

- Them MFA/OTP
- Ma hoa du lieu nhay cam
- Nang cap anti-spoofing
- Bo sung logging/audit va test bao mat

---

## Ket luan

[Tom tat dong gop chinh cua de tai va dinh huong tiep theo.]

## Tai lieu tham khao

1. [Tai lieu 1]
2. [Tai lieu 2]
3. [Tai lieu 3]

## Phu luc (neu co)

- Anh man hinh he thong
- Bang test case
- Ket qua test
