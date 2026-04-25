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

#### UC-04: Xem thông tin tài khoản

| Trường             | Nội dung                                                                  |
| ------------------ | ------------------------------------------------------------------------- |
| **Mã UC**          | UC-04                                                                     |
| **Tên**            | Xem thông tin tài khoản                                                   |
| **Actor**          | Người dùng đã đăng nhập                                                   |
| **Mục tiêu**       | Theo dõi hồ sơ cá nhân, số tài khoản, số dư và trạng thái tài khoản       |
| **Tiền điều kiện** | Người dùng đã đăng nhập hợp lệ                                            |
| **Hậu điều kiện**  | Thông tin hồ sơ được hiển thị đầy đủ, không làm thay đổi dữ liệu hệ thống |

**Luồng chính:**

| Bước | Actor                              | Hệ thống                                                |
| ---- | ---------------------------------- | ------------------------------------------------------- |
| 1    | Truy cập trang dashboard/hồ sơ     | Kiểm tra session đăng nhập                              |
| 2    | —                                  | Truy vấn dữ liệu hồ sơ theo `user_id` trong session     |
| 3    | —                                  | Hiển thị họ tên, email, số tài khoản, số dư, trạng thái |
| 4    | Xem và đối chiếu thông tin cá nhân | —                                                       |

**Luồng thay thế:**

- **2a.** Không tìm thấy hồ sơ theo session hiện tại → buộc đăng nhập lại.

**Luồng ngoại lệ:**

- Session hết hạn trong lúc thao tác → chuyển về trang đăng nhập, thông báo phiên đã hết hạn.

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

#### UC-06: Cập nhật khuôn mặt

| Trường             | Nội dung                                                                             |
| ------------------ | ------------------------------------------------------------------------------------ |
| **Mã UC**          | UC-06                                                                                |
| **Tên**            | Cập nhật khuôn mặt mới                                                               |
| **Actor**          | Người dùng đã đăng nhập, Hệ thống AI                                                 |
| **Mục tiêu**       | Thay thế dữ liệu khuôn mặt cũ bằng embedding mới để dùng cho đăng nhập sinh trắc học |
| **Tiền điều kiện** | Người dùng đã đăng nhập, camera hoạt động, tài khoản hợp lệ                          |
| **Hậu điều kiện**  | Trường `face_encoding` của người dùng được cập nhật bằng dữ liệu mới                 |

**Luồng chính:**

| Bước | Actor                      | Hệ thống / AI                                  |
| ---- | -------------------------- | ---------------------------------------------- |
| 1    | Vào mục cập nhật khuôn mặt | Hiển thị giao diện camera                      |
| 2    | Chụp ảnh khuôn mặt mới     | —                                              |
| 3    | —                          | AI phát hiện khuôn mặt và trích xuất embedding |
| 4    | —                          | Kiểm tra embedding hợp lệ                      |
| 5    | Xác nhận cập nhật          | —                                              |
| 6    | —                          | Ghi đè `face_encoding` mới vào bảng `users`    |
| 7    | —                          | Thông báo cập nhật thành công                  |

**Luồng thay thế:**

- **3a.** Không phát hiện khuôn mặt rõ nét → yêu cầu chụp lại.
- **4a.** Embedding không đạt chất lượng tối thiểu → yêu cầu chụp ở điều kiện ánh sáng tốt hơn.

**Luồng ngoại lệ:**

- Mất kết nối API AI trong quá trình xử lý → thông báo lỗi hệ thống, không cập nhật dữ liệu cũ.

---

#### UC-07: Đổi mật khẩu

| Trường             | Nội dung                                                      |
| ------------------ | ------------------------------------------------------------- |
| **Mã UC**          | UC-07                                                         |
| **Tên**            | Đổi mật khẩu                                                  |
| **Actor**          | Người dùng đã đăng nhập                                       |
| **Mục tiêu**       | Cập nhật mật khẩu mới để tăng an toàn tài khoản               |
| **Tiền điều kiện** | Người dùng đã đăng nhập, biết mật khẩu hiện tại               |
| **Hậu điều kiện**  | Mật khẩu mới được hash và lưu; mật khẩu cũ không còn hiệu lực |

**Luồng chính:**

| Bước | Actor                         | Hệ thống                                     |
| ---- | ----------------------------- | -------------------------------------------- |
| 1    | Vào mục đổi mật khẩu          | Hiển thị form mật khẩu cũ/mới/xác nhận       |
| 2    | Nhập thông tin và gửi yêu cầu | —                                            |
| 3    | —                             | Kiểm tra mật khẩu cũ đúng                    |
| 4    | —                             | Kiểm tra mật khẩu mới đạt chính sách độ mạnh |
| 5    | —                             | Hash mật khẩu mới và cập nhật DB             |
| 6    | —                             | Thông báo đổi mật khẩu thành công            |

**Luồng thay thế:**

- **3a.** Mật khẩu cũ sai → từ chối cập nhật, thông báo lỗi.
- **4a.** Mật khẩu mới yếu hoặc không khớp xác nhận → yêu cầu nhập lại.

**Luồng ngoại lệ:**

- Lỗi ghi dữ liệu → giữ nguyên mật khẩu cũ, thông báo thử lại sau.

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

#### UC-09: Xem lịch sử giao dịch

| Trường             | Nội dung                                                                           |
| ------------------ | ---------------------------------------------------------------------------------- |
| **Mã UC**          | UC-09                                                                              |
| **Tên**            | Xem lịch sử giao dịch                                                              |
| **Actor**          | Người dùng đã đăng nhập                                                            |
| **Mục tiêu**       | Theo dõi toàn bộ giao dịch vào/ra để kiểm soát tài chính cá nhân                   |
| **Tiền điều kiện** | Người dùng đã đăng nhập, có quyền truy cập lịch sử giao dịch của chính mình        |
| **Hậu điều kiện**  | Danh sách giao dịch được hiển thị theo thứ tự thời gian, không thay đổi dữ liệu DB |

**Luồng chính:**

| Bước | Actor                            | Hệ thống                                               |
| ---- | -------------------------------- | ------------------------------------------------------ |
| 1    | Vào mục lịch sử giao dịch        | Truy vấn `bank_transactions` theo tài khoản người dùng |
| 2    | —                                | Sắp xếp giao dịch mới nhất lên trước                   |
| 3    | —                                | Hiển thị số tiền, loại giao dịch, thời gian, nội dung  |
| 4    | Lọc hoặc tìm kiếm theo điều kiện | Hiển thị kết quả theo bộ lọc                           |

**Luồng thay thế:**

- **1a.** Chưa có giao dịch nào → hiển thị trạng thái rỗng và hướng dẫn sử dụng.

**Luồng ngoại lệ:**

- Truy vấn DB lỗi tạm thời → thông báo không tải được dữ liệu, cho phép tải lại.

---

#### UC-10: Đăng xuất

| Trường             | Nội dung                                           |
| ------------------ | -------------------------------------------------- |
| **Mã UC**          | UC-10                                              |
| **Tên**            | Đăng xuất                                          |
| **Actor**          | Người dùng đã đăng nhập                            |
| **Mục tiêu**       | Kết thúc phiên làm việc an toàn                    |
| **Tiền điều kiện** | Phiên đăng nhập còn hiệu lực                       |
| **Hậu điều kiện**  | Session bị hủy, người dùng quay về trang đăng nhập |

**Luồng chính:**

| Bước | Actor                | Hệ thống                                |
| ---- | -------------------- | --------------------------------------- |
| 1    | Nhấn nút "Đăng xuất" | Gửi yêu cầu logout                      |
| 2    | —                    | Xóa dữ liệu session phía server         |
| 3    | —                    | Thu hồi trạng thái đăng nhập ở frontend |
| 4    | —                    | Chuyển về trang đăng nhập               |

**Luồng thay thế:**

- **2a.** Session đã hết hạn trước đó → vẫn chuyển về trang đăng nhập bình thường.

**Luồng ngoại lệ:**

- Mất kết nối khi logout → frontend xóa phiên cục bộ và yêu cầu đăng nhập lại khi thao tác tiếp.

---

#### UC-11: Xem danh sách người dùng (Admin)

| Trường             | Nội dung                                                                     |
| ------------------ | ---------------------------------------------------------------------------- |
| **Mã UC**          | UC-11                                                                        |
| **Tên**            | Xem danh sách người dùng                                                     |
| **Actor**          | Quản trị viên                                                                |
| **Mục tiêu**       | Theo dõi toàn bộ tài khoản để phục vụ duyệt, khóa/mở khóa và bảo trì dữ liệu |
| **Tiền điều kiện** | Admin đã đăng nhập, có quyền truy cập trang quản trị                         |
| **Hậu điều kiện**  | Danh sách người dùng hiển thị đầy đủ theo bộ lọc yêu cầu                     |

**Luồng chính:**

| Bước | Actor                    | Hệ thống                                                |
| ---- | ------------------------ | ------------------------------------------------------- |
| 1    | Truy cập trang quản trị  | Kiểm tra quyền `role = admin`                           |
| 2    | —                        | Truy vấn danh sách người dùng từ bảng `users`           |
| 3    | —                        | Hiển thị thông tin chính: tên, email, trạng thái, số dư |
| 4    | Chọn lọc theo trạng thái | Cập nhật danh sách theo điều kiện lọc                   |

**Luồng thay thế:**

- **1a.** Người dùng thường truy cập URL admin → từ chối quyền và trả về 403.

**Luồng ngoại lệ:**

- Lỗi truy vấn dữ liệu quản trị → thông báo lỗi và ghi log hệ thống.

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

#### UC-13: Từ chối tài khoản (Admin)

| Trường             | Nội dung                                                             |
| ------------------ | -------------------------------------------------------------------- |
| **Mã UC**          | UC-13                                                                |
| **Tên**            | Từ chối tài khoản chờ                                                |
| **Actor**          | Quản trị viên                                                        |
| **Mục tiêu**       | Loại bỏ các hồ sơ đăng ký không hợp lệ trước khi kích hoạt tài khoản |
| **Tiền điều kiện** | Admin đã đăng nhập, tồn tại tài khoản trạng thái `pending`           |
| **Hậu điều kiện**  | Tài khoản chuyển sang trạng thái `rejected`, không thể đăng nhập     |

**Luồng chính:**

| Bước | Actor                              | Hệ thống                                |
| ---- | ---------------------------------- | --------------------------------------- |
| 1    | Lọc danh sách trạng thái `pending` | Hiển thị hồ sơ chờ duyệt                |
| 2    | Chọn tài khoản cần từ chối         | Hiển thị chi tiết hồ sơ                 |
| 3    | Nhấn "Từ chối"                     | Cập nhật `approval_status = rejected`   |
| 4    | —                                  | Hiển thị trạng thái mới trong danh sách |

**Luồng thay thế:**

- **2a.** Admin đổi ý, quay lại thao tác duyệt → chuyển sang UC-12.

**Luồng ngoại lệ:**

- Tài khoản đã đổi trạng thái do admin khác xử lý trước đó → thông báo dữ liệu đã thay đổi, yêu cầu tải lại danh sách.

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

#### UC-16: Xóa tài khoản (Admin)

| Trường             | Nội dung                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------- |
| **Mã UC**          | UC-16                                                                                     |
| **Tên**            | Xóa tài khoản người dùng                                                                  |
| **Actor**          | Quản trị viên                                                                             |
| **Mục tiêu**       | Xóa tài khoản khỏi hệ thống hoạt động, đồng thời lưu vết hồ sơ đã xóa để kiểm tra sau này |
| **Tiền điều kiện** | Admin đã đăng nhập, tài khoản mục tiêu tồn tại trong bảng `users`                         |
| **Hậu điều kiện**  | Dữ liệu người dùng chuyển sang `deleted_profiles`, bản ghi gốc bị xóa khỏi `users`        |

**Luồng chính:**

| Bước | Actor                  | Hệ thống                                            |
| ---- | ---------------------- | --------------------------------------------------- |
| 1    | Chọn tài khoản cần xóa | Hiển thị hộp thoại xác nhận                         |
| 2    | Xác nhận thao tác xóa  | —                                                   |
| 3    | —                      | Sao lưu thông tin hồ sơ vào bảng `deleted_profiles` |
| 4    | —                      | Xóa bản ghi khỏi bảng `users`                       |
| 5    | —                      | Thông báo xóa thành công và cập nhật danh sách      |

**Luồng thay thế:**

- **2a.** Admin hủy thao tác xác nhận → dừng xử lý, dữ liệu giữ nguyên.

**Luồng ngoại lệ:**

- Bước archive thất bại → hủy xóa để tránh mất dữ liệu, thông báo lỗi và ghi log.

**Yêu cầu nghiệp vụ:**

- Phải đảm bảo archive thành công trước khi xóa bản ghi gốc.
- Không cho phép xóa tài khoản admin mặc định của hệ thống.

---

#### UC-17: Xem hồ sơ đã xóa (Admin)

| Trường             | Nội dung                                                                 |
| ------------------ | ------------------------------------------------------------------------ |
| **Mã UC**          | UC-17                                                                    |
| **Tên**            | Xem danh sách hồ sơ đã xóa                                               |
| **Actor**          | Quản trị viên                                                            |
| **Mục tiêu**       | Theo dõi lịch sử xóa tài khoản để phục vụ truy vết, kiểm tra và đối soát |
| **Tiền điều kiện** | Admin đã đăng nhập, có quyền truy cập dữ liệu lưu trữ đã xóa             |
| **Hậu điều kiện**  | Danh sách hồ sơ trong `deleted_profiles` được hiển thị                   |

**Luồng chính:**

| Bước | Actor               | Hệ thống                                                    |
| ---- | ------------------- | ----------------------------------------------------------- |
| 1    | Mở mục hồ sơ đã xóa | Truy vấn bảng `deleted_profiles`                            |
| 2    | —                   | Hiển thị thông tin: email, họ tên, thời điểm xóa, người xóa |
| 3    | Lọc/tìm kiếm hồ sơ  | Cập nhật danh sách theo điều kiện                           |

**Luồng thay thế:**

- **1a.** Chưa có hồ sơ nào bị xóa → hiển thị danh sách rỗng.

**Luồng ngoại lệ:**

- Lỗi truy vấn archive → thông báo không thể tải dữ liệu, cho phép thử lại.

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
## Chuong 3. Phan tich va thiet ke he thong

### 3.1 Sơ đồ thực thể kết hợp (ERD) và chuyển ERD sang lược đồ cơ sở dữ liệu

#### 3.1.1 Cơ sở xây dựng ERD

ERD được xây dựng từ mô hình dữ liệu thực tế của hệ thống backend chạy trên SQLite.
Trong phiên bản hiện tại, dữ liệu nghiệp vụ tập trung vào 3 thực thể lõi:

1. users: lưu thông tin tài khoản, định danh, trạng thái duyệt và dữ liệu KYC.
2. bank_transactions: lưu lịch sử chuyển khoản nội địa giữa hai tài khoản.
3. deleted_profiles: lưu bản sao hồ sơ khi admin xóa người dùng (archive phục vụ truy vết).

Mặc dù SQLite cho phép không khai báo ràng buộc khóa ngoại vật lý, mô hình logic của hệ thống vẫn tồn tại các liên kết rõ ràng giữa các thực thể.
Vì vậy, trong phần phân tích ERD sẽ thể hiện cả:

- Ràng buộc logic (mức thiết kế nghiệp vụ).
- Ràng buộc vật lý đã có trong schema (PK, UNIQUE, chỉ mục).

#### 3.1.2 Xác định thực thể và thuộc tính

**Thực thể 1: USERS**

USERS là thực thể trung tâm của toàn hệ thống. Mỗi bản ghi biểu diễn một tài khoản có thể thuộc vai trò user hoặc admin.

**Bảng 3.1 - Thuộc tính chính của USERS**

| Thuộc tính            | Kiểu dữ liệu | Ràng buộc                           | Ý nghĩa                                                   |
| --------------------- | ------------ | ----------------------------------- | --------------------------------------------------------- |
| id                    | INTEGER      | PK, AUTOINCREMENT                   | Khóa chính định danh người dùng                           |
| username              | TEXT         | NOT NULL, UNIQUE                    | Tên đăng nhập duy nhất                                    |
| password_hash         | TEXT         | NOT NULL                            | Mật khẩu đã băm                                           |
| role                  | TEXT         | NOT NULL, DEFAULT 'user'            | Vai trò tài khoản (admin/user)                            |
| approval_status       | TEXT         | NOT NULL, DEFAULT 'approved'        | Trạng thái nghiệp vụ: pending, approved, rejected, locked |
| account_number        | TEXT         | UNIQUE                              | Số tài khoản ngân hàng nội bộ                             |
| balance               | INTEGER      | NOT NULL, DEFAULT 500000            | Số dư hiện tại                                            |
| full_name             | TEXT         | NULL                                | Họ tên người dùng                                         |
| face_encoding         | TEXT         | NULL                                | Dữ liệu embedding khuôn mặt                               |
| is_locked             | INTEGER      | NOT NULL, DEFAULT 0                 | Cờ khóa cũ (legacy), logic mới ưu tiên approval_status    |
| email                 | TEXT         | NULL                                | Email liên hệ                                             |
| phone                 | TEXT         | NULL                                | Số điện thoại                                             |
| birth_date            | TEXT         | NULL                                | Ngày sinh                                                 |
| cccd_number           | TEXT         | UNIQUE INDEX                        | Số CCCD (duy nhất toàn hệ thống)                          |
| gender                | TEXT         | NULL                                | Giới tính                                                 |
| hometown              | TEXT         | NULL                                | Quê quán                                                  |
| residence             | TEXT         | NULL                                | Nơi thường trú                                            |
| nationality           | TEXT         | NULL                                | Quốc tịch                                                 |
| valid_until           | TEXT         | NULL                                | Ngày hết hạn CCCD                                         |
| issued_date           | TEXT         | NULL                                | Ngày cấp CCCD                                             |
| issued_place          | TEXT         | NULL                                | Nơi cấp CCCD                                              |
| face_image_path       | TEXT         | NULL                                | Đường dẫn ảnh khuôn mặt                                   |
| cccd_front_image_path | TEXT         | NULL                                | Đường dẫn ảnh CCCD mặt trước                              |
| cccd_back_image_path  | TEXT         | NULL                                | Đường dẫn ảnh CCCD mặt sau                                |
| created_at            | TEXT         | NOT NULL, DEFAULT CURRENT_TIMESTAMP | Thời điểm tạo bản ghi                                     |
| updated_at            | TEXT         | NOT NULL, DEFAULT CURRENT_TIMESTAMP | Thời điểm cập nhật gần nhất                               |

**Thực thể 2: BANK_TRANSACTIONS**

BANK_TRANSACTIONS lưu mọi giao dịch chuyển khoản nội địa phát sinh trong hệ thống.

**Bảng 3.2 - Thuộc tính chính của BANK_TRANSACTIONS**

| Thuộc tính              | Kiểu dữ liệu | Ràng buộc                           | Ý nghĩa                                       |
| ----------------------- | ------------ | ----------------------------------- | --------------------------------------------- |
| id                      | INTEGER      | PK, AUTOINCREMENT                   | Khóa chính giao dịch                          |
| sender_user_id          | INTEGER      | NOT NULL                            | ID người gửi (tham chiếu logic đến users.id)  |
| receiver_user_id        | INTEGER      | NOT NULL                            | ID người nhận (tham chiếu logic đến users.id) |
| sender_account_number   | TEXT         | NOT NULL                            | Số tài khoản nguồn tại thời điểm giao dịch    |
| receiver_account_number | TEXT         | NOT NULL                            | Số tài khoản đích tại thời điểm giao dịch     |
| amount                  | INTEGER      | NOT NULL                            | Số tiền giao dịch                             |
| note                    | TEXT         | NULL                                | Nội dung chuyển khoản                         |
| created_at              | TEXT         | NOT NULL, DEFAULT CURRENT_TIMESTAMP | Thời điểm tạo giao dịch                       |

**Thực thể 3: DELETED_PROFILES**

DELETED_PROFILES đóng vai trò nhật ký lưu trữ hồ sơ đã xóa, hỗ trợ kiểm tra và đối soát sau này.

**Bảng 3.3 - Thuộc tính chính của DELETED_PROFILES**

| Thuộc tính       | Kiểu dữ liệu | Ràng buộc                           | Ý nghĩa                                                |
| ---------------- | ------------ | ----------------------------------- | ------------------------------------------------------ |
| id               | INTEGER      | PK, AUTOINCREMENT                   | Khóa chính bản ghi archive                             |
| original_user_id | INTEGER      | NULL                                | ID người dùng gốc trước khi bị xóa                     |
| username         | TEXT         | NULL                                | Username của tài khoản bị xóa                          |
| deleted_by       | INTEGER      | NULL                                | ID admin thực hiện xóa (tham chiếu logic đến users.id) |
| deleted_at       | TEXT         | NOT NULL, DEFAULT CURRENT_TIMESTAMP | Thời điểm xóa                                          |
| profile_json     | TEXT         | NOT NULL                            | Bản chụp JSON toàn bộ hồ sơ tại thời điểm xóa          |

#### 3.1.3 Xác định các mối quan hệ trong ERD

Từ ba thực thể trên, hệ thống có các quan hệ nghiệp vụ sau:

1. USERS (vai trò người gửi) - BANK_TRANSACTIONS:

- Một người dùng có thể phát sinh nhiều giao dịch gửi.
- Mỗi giao dịch gửi chỉ thuộc về đúng một người gửi.
- Bội số: USERS 1 - N BANK_TRANSACTIONS (theo sender_user_id).

2. USERS (vai trò người nhận) - BANK_TRANSACTIONS:

- Một người dùng có thể nhận nhiều giao dịch.
- Mỗi giao dịch nhận chỉ thuộc về đúng một người nhận.
- Bội số: USERS 1 - N BANK_TRANSACTIONS (theo receiver_user_id).

3. USERS (vai trò admin) - DELETED_PROFILES:

- Một admin có thể xóa nhiều hồ sơ.
- Mỗi bản ghi archive có tối đa một người xóa.
- Bội số: USERS 1 - N DELETED_PROFILES (theo deleted_by).

4. USERS (hồ sơ gốc) - DELETED_PROFILES:

- Một tài khoản người dùng khi bị xóa sẽ tạo một bản ghi archive tương ứng.
- Về logic có thể xem là 1 - 0..1 theo từng vòng đời tài khoản.
- Trong thiết kế lưu vết, biểu diễn thành USERS 1 - N DELETED_PROFILES (theo original_user_id) để vẫn tương thích trường hợp tái tạo tài khoản cùng username ở chu kỳ khác.

#### 3.1.4 Sơ đồ ERD mức logic

Sơ đồ dưới đây mô tả thực thể, khóa và liên kết ở mức logic nghiệp vụ.

```text
+-------------------------+
|          USERS          |
+-------------------------+
| PK id                   |
| UQ username             |
| password_hash           |
| role                    |
| approval_status         |
| UQ account_number       |
| balance                 |
| ... KYC fields ...      |
| UQ cccd_number (index)  |
| created_at, updated_at  |
+-------------------------+
    | 1                           1 |
    |                               |
    | N (sender_user_id)            | N (receiver_user_id)
    v                               v
+-------------------------+
|    BANK_TRANSACTIONS    |
+-------------------------+
| PK id                   |
| sender_user_id          | --> USERS.id (logic FK)
| receiver_user_id        | --> USERS.id (logic FK)
| sender_account_number   |
| receiver_account_number |
| amount, note            |
| created_at              |
+-------------------------+

USERS.id (admin) 1 -------- N DELETED_PROFILES.deleted_by
USERS.id (goc)   1 -------- N DELETED_PROFILES.original_user_id

+-------------------------+
|    DELETED_PROFILES     |
+-------------------------+
| PK id                   |
| original_user_id        | --> USERS.id (logic FK)
| username                |
| deleted_by              | --> USERS.id (logic FK)
| deleted_at              |
| profile_json            |
+-------------------------+
```

#### 3.1.5 Chuyển ERD sang lược đồ cơ sở dữ liệu quan hệ

Quy tắc chuyển đổi được áp dụng:

1. Mỗi thực thể mạnh chuyển thành một quan hệ (một bảng).
2. Khóa chính của thực thể giữ nguyên làm PK của bảng.
3. Quan hệ 1 - N chuyển bằng cách đưa khóa của phía 1 sang phía N làm FK logic.
4. Các thuộc tính đơn trị giữ nguyên thành cột.
5. Ràng buộc duy nhất nghiệp vụ chuyển thành UNIQUE hoặc UNIQUE INDEX.

Kết quả thu được lược đồ quan hệ như sau:

1. USERS(
   id PK,
   username UQ,
   password_hash,
   role,
   approval_status,
   account_number UQ,
   balance,
   full_name,
   face_encoding,
   is_locked,
   email,
   phone,
   birth_date,
   cccd_number UQ,
   gender,
   hometown,
   residence,
   nationality,
   valid_until,
   issued_date,
   issued_place,
   face_image_path,
   cccd_front_image_path,
   cccd_back_image_path,
   created_at,
   updated_at
   )

2. BANK_TRANSACTIONS(
   id PK,
   sender_user_id FK logic -> USERS.id,
   receiver_user_id FK logic -> USERS.id,
   sender_account_number,
   receiver_account_number,
   amount,
   note,
   created_at
   )

3. DELETED_PROFILES(
   id PK,
   original_user_id FK logic -> USERS.id,
   username,
   deleted_by FK logic -> USERS.id,
   deleted_at,
   profile_json
   )

#### 3.1.6 Ràng buộc toàn vẹn và chỉ mục quan trọng

Để đảm bảo dữ liệu nhất quán với nghiệp vụ ngân hàng mô phỏng, các ràng buộc chính gồm:

1. Toàn vẹn thực thể:

- Mọi bảng đều có khóa chính id tự tăng, không trùng lặp.

2. Toàn vẹn định danh tài khoản:

- username là duy nhất toàn hệ thống.
- account_number là duy nhất toàn hệ thống.
- cccd_number là duy nhất toàn hệ thống (qua unique index).

3. Toàn vẹn nghiệp vụ giao dịch:

- amount bắt buộc dương trong tầng xử lý nghiệp vụ.
- sender_user_id và receiver_user_id phải tồn tại ở thời điểm lập giao dịch (kiểm tra tại backend).

4. Toàn vẹn trạng thái tài khoản:

- approval_status phải nằm trong tập logic: pending, approved, rejected, locked.
- Admin luôn được cưỡng bức về trạng thái approved trong bước khởi tạo và migration.

5. Toàn vẹn lưu vết xóa:

- Trước khi xóa bản ghi ở USERS, hệ thống phải lưu snapshot vào DELETED_PROFILES.

#### 3.1.7 Nhận xét thiết kế ERD hiện tại

Ưu điểm:

1. Cấu trúc gọn, tập trung đúng 3 thực thể cốt lõi, phù hợp phạm vi prototype.
2. Đáp ứng đầy đủ luồng nghiệp vụ chính: đăng ký, xác thực, chuyển khoản, quản trị xóa hồ sơ.
3. Dễ mở rộng theo chiều ngang bằng cách bổ sung bảng mới (audit_logs, notifications, otp_sessions).

Hạn chế:

1. Khóa ngoại mới ở mức logic, chưa khai báo FK vật lý trong schema SQLite.
2. Chưa có CHECK constraint cho amount > 0 và miền giá trị approval_status.
3. Bảng USERS đang chứa nhiều thuộc tính KYC, có thể tách chuẩn hóa thành bảng user_kyc ở phiên bản sau.

Định hướng cải tiến:

1. Bổ sung FK vật lý và trigger đảm bảo toàn vẹn tham chiếu.
2. Tách bảng KYC và bảng ảnh tài liệu để giảm độ rộng của USERS.
3. Bổ sung bảng audit giao dịch quản trị để truy vết đầy đủ hơn.

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
