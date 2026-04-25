## Chương 5. Cài đặt và kiểm thử

### 5.1 Mục tiêu của chương

Chương này mô tả đầy đủ cách triển khai hệ thống trong môi trường Windows, từ chuẩn bị công cụ đến vận hành thực tế.
Ngoài phần cài đặt, chương còn trình bày kế hoạch kiểm thử theo nhóm chức năng và nhóm phi chức năng để chứng minh hệ thống hoạt động đúng thiết kế.

Mục tiêu chính gồm:

1. Đảm bảo người đọc có thể cài đặt lại hệ thống theo đúng trình tự.
2. Chứng minh các luồng nghiệp vụ cốt lõi chạy ổn định trong môi trường thử nghiệm.
3. Đưa ra bằng chứng kiểm thử có thể đối chiếu, bao gồm đầu vào, kết quả mong đợi và kết quả thực tế.

### 5.2 Môi trường cài đặt

**Bảng 5.1 - Cấu hình môi trường triển khai thử nghiệm**

| Thành phần    | Cấu hình đề xuất         | Mục đích                                             |
| ------------- | ------------------------ | ---------------------------------------------------- |
| Hệ điều hành  | Windows 10/11 64-bit     | Môi trường chạy chính của dự án                      |
| Python        | 3.11.x                   | Chạy backend FastAPI và module AI                    |
| Node.js       | 18+                      | Phục vụ công cụ tiện ích khi cần (ví dụ localtunnel) |
| Trình duyệt   | Chrome/Edge mới nhất     | Kiểm thử camera, liveness, luồng frontend            |
| Cơ sở dữ liệu | SQLite (`app.sqlite3`)   | Lưu trữ dữ liệu người dùng, giao dịch, archive       |
| Camera        | Webcam HD cơ bản trở lên | Thu ảnh đăng nhập khuôn mặt và cập nhật face         |

**Bảng 5.2 - Thư viện chính theo vai trò**

| Nhóm             | Thư viện/Framework                    | Vai trò                                     |
| ---------------- | ------------------------------------- | ------------------------------------------- |
| Backend API      | FastAPI, Uvicorn                      | Xây dựng và chạy web service                |
| Xử lý ảnh        | OpenCV, face_recognition              | Nhận diện và trích xuất embedding khuôn mặt |
| OCR              | EasyOCR, Google Vision (tùy cấu hình) | Trích xuất thông tin CCCD                   |
| Bảo mật mật khẩu | bcrypt/passlib                        | Băm và xác thực mật khẩu                    |
| Frontend         | HTML/CSS/JavaScript thuần             | Hiển thị giao diện và gọi API               |

### 5.3 Kiến trúc triển khai thực tế

Hệ thống được tổ chức theo kiến trúc 3 lớp rút gọn: giao diện người dùng, lớp xử lý nghiệp vụ và lớp dữ liệu.

1. **Frontend** đặt tại thư mục `frontend/`, gồm các trang chính: đăng nhập, đăng ký, dashboard người dùng, dashboard admin.
2. **Backend hợp nhất** đặt tại `ai-backend/main.py`, cung cấp API nghiệp vụ và API AI trong cùng một dịch vụ.
3. **Dữ liệu** sử dụng SQLite nội bộ với file `app.sqlite3`, không cần server DB riêng.
4. **Dữ liệu ảnh upload** lưu theo thư mục dưới `uploads/` để phục vụ KYC và truy vết.

Mô hình này phù hợp với mục tiêu học tập vì:

- Cài đặt nhanh, ít phụ thuộc ngoài.
- Dễ debug toàn bộ luồng từ frontend đến DB.
- Dễ đóng gói và demo trong phòng lab.

### 5.4 Quy trình cài đặt hệ thống trên Windows

#### 5.4.1 Chuẩn bị mã nguồn

1. Mở thư mục dự án trong VS Code.
2. Kiểm tra các thư mục chính đã có đầy đủ: `ai-backend/`, `frontend/`, `RUN_FACE_AUTH.bat`, `app.sqlite3`.
3. Đảm bảo quyền chạy script `.bat` và PowerShell trong phiên hiện tại.

#### 5.4.2 Thiết lập môi trường Python

1. Tạo môi trường ảo nếu chưa có.
2. Kích hoạt môi trường ảo.
3. Cài dependency từ `ai-backend/requirements.txt`.

Khi hoàn tất, cần xác nhận:

- Import được FastAPI.
- Import được OpenCV và face_recognition.
- Không có lỗi thiếu gói bắt buộc khi chạy backend.

#### 5.4.3 Khởi động hệ thống

Có hai cách vận hành:

1. **Cách 1 (khuyến nghị):** chạy `RUN_FACE_AUTH.bat` để khởi động theo cấu hình dự án.
2. **Cách 2:** chạy backend trực tiếp từ `ai-backend/run.bat` hoặc lệnh Uvicorn tương đương.

Sau khi khởi động thành công:

- Backend lắng nghe tại cổng 8001.
- Frontend truy cập được các trang chính.
- API phản hồi JSON hợp lệ cho các endpoint xác thực.

#### 5.4.4 Kiểm tra sau cài đặt

Checklist xác nhận cài đặt thành công:

1. Truy cập trang đăng nhập không lỗi giao diện.
2. Gửi yêu cầu đăng nhập sai dữ liệu nhận được thông báo phù hợp.
3. Đăng nhập bằng tài khoản hợp lệ chuyển đúng trang dashboard.
4. Truy vấn danh sách user từ trang admin hiển thị được dữ liệu.

### 5.5 Thiết kế kiểm thử

Để đảm bảo đánh giá có hệ thống, kiểm thử được chia thành 3 nhóm:

1. Kiểm thử chức năng nghiệp vụ (Functional Testing).
2. Kiểm thử xử lý lỗi và dữ liệu biên (Negative/Boundary Testing).
3. Kiểm thử phi chức năng cơ bản (Hiệu năng, bảo mật mức ứng dụng, ổn định phiên).

**Bảng 5.3 - Nguyên tắc xây dựng test case**

| Nguyên tắc              | Diễn giải                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------- |
| Bao phủ theo use case   | Mỗi use case quan trọng phải có ít nhất một test case thành công và một test case lỗi |
| Dữ liệu rõ ràng         | Đầu vào, tiền điều kiện và kết quả mong đợi phải mô tả cụ thể                         |
| Tái lập được            | Kịch bản có thể chạy lại nhiều lần với cùng kết quả logic                             |
| Tập trung nghiệp vụ lõi | Ưu tiên đăng ký, xác thực, giao dịch, quản trị                                        |

### 5.6 Kịch bản kiểm thử chức năng chi tiết

#### 5.6.1 Nhóm đăng ký và duyệt tài khoản

**Bảng 5.4 - Test case nhóm đăng ký/duyệt**

| Mã TC     | Mục tiêu                | Tiền điều kiện               | Bước thực hiện                                  | Kết quả mong đợi                                            |
| --------- | ----------------------- | ---------------------------- | ----------------------------------------------- | ----------------------------------------------------------- |
| TC-REG-01 | Đăng ký hợp lệ          | Chưa tồn tại email           | Nhập đủ thông tin + ảnh mặt hợp lệ, gửi đăng ký | Tạo user trạng thái `pending`, thông báo đăng ký thành công |
| TC-REG-02 | Phát hiện email trùng   | Email đã có trong DB         | Đăng ký lại bằng email cũ                       | Từ chối đăng ký, hiển thị lỗi trùng email                   |
| TC-REG-03 | Ảnh không có mặt        | Chưa tồn tại email           | Upload ảnh không phát hiện mặt                  | Từ chối, yêu cầu chụp/upload ảnh khác                       |
| TC-ADM-01 | Admin duyệt tài khoản   | Có user trạng thái `pending` | Admin chọn duyệt                                | Trạng thái đổi sang `approved`                              |
| TC-ADM-02 | Admin từ chối tài khoản | Có user trạng thái `pending` | Admin chọn từ chối                              | Trạng thái đổi sang `rejected`                              |

#### 5.6.2 Nhóm đăng nhập và quản lý phiên

**Bảng 5.5 - Test case nhóm xác thực**

| Mã TC      | Mục tiêu                | Tiền điều kiện                       | Bước thực hiện                | Kết quả mong đợi                                    |
| ---------- | ----------------------- | ------------------------------------ | ----------------------------- | --------------------------------------------------- |
| TC-AUTH-01 | Đăng nhập mật khẩu đúng | User `approved`                      | Nhập đúng email + mật khẩu    | Tạo session, vào dashboard                          |
| TC-AUTH-02 | Sai mật khẩu            | User tồn tại                         | Nhập đúng email, sai mật khẩu | Báo lỗi chung, không lộ thông tin tồn tại tài khoản |
| TC-AUTH-03 | Login mặt thành công    | Có `face_encoding`, camera hoạt động | Thực hiện liveness + so khớp  | Đăng nhập thành công                                |
| TC-AUTH-04 | Liveness thất bại       | Có camera                            | Cố ý dùng ảnh/video giả       | Từ chối đăng nhập mặt                               |
| TC-AUTH-05 | Tài khoản bị khóa       | User trạng thái `locked`             | Đăng nhập bằng mật khẩu       | Từ chối, thông báo tài khoản bị khóa                |

#### 5.6.3 Nhóm KYC và cập nhật hồ sơ

**Bảng 5.6 - Test case nhóm KYC**

| Mã TC      | Mục tiêu                  | Tiền điều kiện             | Bước thực hiện               | Kết quả mong đợi                           |
| ---------- | ------------------------- | -------------------------- | ---------------------------- | ------------------------------------------ |
| TC-KYC-01  | OCR trích xuất thành công | User đã đăng nhập          | Upload ảnh CCCD rõ nét       | Trích xuất được số CCCD, họ tên, ngày sinh |
| TC-KYC-02  | OCR ảnh chất lượng kém    | User đã đăng nhập          | Upload ảnh mờ/lệch           | Báo lỗi chất lượng ảnh, yêu cầu upload lại |
| TC-KYC-03  | Trùng CCCD                | DB đã có CCCD tương tự     | Upload CCCD trùng            | Từ chối cập nhật, báo lỗi trùng CCCD       |
| TC-FACE-01 | Cập nhật face mới         | User đăng nhập, camera tốt | Chụp ảnh mặt mới và xác nhận | Trường `face_encoding` được cập nhật       |

#### 5.6.4 Nhóm giao dịch tài chính

**Bảng 5.7 - Test case nhóm chuyển khoản**

| Mã TC     | Mục tiêu                     | Tiền điều kiện                   | Bước thực hiện                        | Kết quả mong đợi                                      |
| --------- | ---------------------------- | -------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| TC-TXN-01 | Chuyển khoản hợp lệ          | Số dư đủ, tài khoản đích tồn tại | Nhập tài khoản nhận + số tiền hợp lệ  | Trừ/cộng số dư đúng, sinh bản ghi `bank_transactions` |
| TC-TXN-02 | Số dư không đủ               | User nguồn có số dư thấp         | Chuyển số tiền vượt số dư             | Từ chối giao dịch, số dư giữ nguyên                   |
| TC-TXN-03 | Tài khoản nhận không tồn tại | User đã đăng nhập                | Nhập sai số tài khoản đích            | Báo lỗi không tìm thấy tài khoản                      |
| TC-TXN-04 | Tự chuyển cho chính mình     | User đã đăng nhập                | Nhập account đích trùng account nguồn | Từ chối giao dịch                                     |

#### 5.6.5 Nhóm quản trị và lưu vết xóa

**Bảng 5.8 - Test case nhóm admin/archive**

| Mã TC     | Mục tiêu                | Tiền điều kiện          | Bước thực hiện      | Kết quả mong đợi                                       |
| --------- | ----------------------- | ----------------------- | ------------------- | ------------------------------------------------------ |
| TC-ADM-03 | Khóa tài khoản user     | Admin đăng nhập         | Chọn user và khóa   | `approval_status` chuyển `locked`                      |
| TC-ADM-04 | Mở khóa tài khoản       | User đang `locked`      | Admin chọn mở khóa  | Trạng thái về `approved`                               |
| TC-ADM-05 | Reset dữ liệu mặt       | User có `face_encoding` | Admin reset face    | `face_encoding` bị xóa                                 |
| TC-ADM-06 | Xóa user có archive     | User mục tiêu tồn tại   | Admin xác nhận xóa  | Có bản ghi `deleted_profiles`, bản ghi user gốc bị xóa |
| TC-ADM-07 | Xem danh sách hồ sơ xóa | Đã có dữ liệu archive   | Mở màn hình archive | Hiển thị đúng username, người xóa, thời điểm xóa       |

#### 5.6.6 Kịch bản thực thi chi tiết (step-by-step)

Để tăng tính tái lập và khả năng đối chiếu kết quả, phần này mô tả chi tiết các kịch bản quan trọng theo cấu trúc:

1. Dữ liệu đầu vào cụ thể.
2. Các bước thao tác tuần tự trên giao diện.
3. Điểm kiểm tra phản hồi API.
4. Điểm kiểm tra dữ liệu trong cơ sở dữ liệu.
5. Tiêu chí đạt/không đạt.

##### Kịch bản KS-01: Đăng ký tài khoản mới thành công

**Mục tiêu:** Xác nhận hệ thống tạo được tài khoản mới ở trạng thái chờ duyệt (`pending`).

**Dữ liệu kiểm thử:**

- Họ tên: Nguyen Van A Test
- Email: nv.a.test001@example.com
- Số điện thoại: 0901000001
- Mật khẩu: Abc@123456
- Ảnh khuôn mặt: ảnh rõ nét, 1 khuôn mặt, định dạng JPG

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                           | Kết quả mong đợi tại UI/API                        | Kiểm tra dữ liệu             |
| ---- | ---------------------------------- | -------------------------------------------------- | ---------------------------- |
| 1    | Mở trang đăng ký, nhập đầy đủ form | Form hợp lệ, không báo lỗi validate client         | Chưa ghi DB                  |
| 2    | Upload ảnh khuôn mặt               | Preview ảnh hiển thị đúng                          | Chưa ghi DB                  |
| 3    | Nhấn nút Đăng ký                   | API trả trạng thái thành công, thông báo chờ duyệt | Có bản ghi mới trong `users` |
| 4    | Đăng nhập bằng email vừa tạo       | UI báo tài khoản đang chờ xét duyệt                | `approval_status = pending`  |

**Tiêu chí đạt:** Có bản ghi user mới, trạng thái `pending`, không phát sinh lỗi server.

##### Kịch bản KS-02: Đăng nhập mật khẩu thành công

**Mục tiêu:** Xác nhận cơ chế xác thực mật khẩu, cấp phiên và chuyển hướng đúng.

**Tiền điều kiện:** Tài khoản `nv.a.test001@example.com` đã được admin duyệt về `approved`.

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                                     | Kết quả mong đợi tại UI/API      | Kiểm tra dữ liệu                    |
| ---- | -------------------------------------------- | -------------------------------- | ----------------------------------- |
| 1    | Mở trang đăng nhập, nhập email/mật khẩu đúng | Không báo lỗi validate đầu vào   | Chưa thay đổi DB                    |
| 2    | Nhấn Đăng nhập                               | API trả thành công, tạo session  | Session hợp lệ ở phía server/client |
| 3    | Quan sát điều hướng                          | Chuyển sang dashboard người dùng | Có thể gọi API hồ sơ thành công     |
| 4    | Tải lại trang dashboard                      | Vẫn giữ trạng thái đăng nhập     | Session chưa hết hạn                |

**Tiêu chí đạt:** Đăng nhập thành công, có session hợp lệ, truy cập dashboard ổn định.

##### Kịch bản KS-03: Đăng nhập khuôn mặt thất bại do liveness

**Mục tiêu:** Xác nhận hệ thống từ chối đăng nhập khi kiểm tra sống không đạt.

**Tiền điều kiện:** User đã có `face_encoding`; camera hoạt động bình thường.

**Dữ liệu kiểm thử:** Dùng ảnh in/video phát lại trước camera thay vì khuôn mặt thật.

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                                         | Kết quả mong đợi tại UI/API                | Kiểm tra dữ liệu                    |
| ---- | ------------------------------------------------ | ------------------------------------------ | ----------------------------------- |
| 1    | Chọn đăng nhập bằng khuôn mặt, nhập email hợp lệ | Camera mở thành công                       | Chưa thay đổi DB                    |
| 2    | Thực hiện đăng nhập bằng ảnh/video giả           | API liveness trả thất bại                  | Không tạo session                   |
| 3    | Quan sát thông báo                               | UI báo không qua liveness, yêu cầu thử lại | Không ghi nhận đăng nhập thành công |

**Tiêu chí đạt:** Hệ thống từ chối đăng nhập, không tạo phiên, không truy cập dashboard.

##### Kịch bản KS-04: Cập nhật KYC bằng OCR thành công

**Mục tiêu:** Xác nhận OCR trích xuất và lưu dữ liệu CCCD đúng vào hồ sơ.

**Tiền điều kiện:** User đã đăng nhập, trạng thái `approved`.

**Dữ liệu kiểm thử:**

- Ảnh CCCD mặt trước rõ nét.
- Ảnh CCCD mặt sau rõ nét.
- CCCD chưa tồn tại trong hệ thống.

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                           | Kết quả mong đợi tại UI/API           | Kiểm tra dữ liệu                            |
| ---- | ---------------------------------- | ------------------------------------- | ------------------------------------------- |
| 1    | Mở màn hình KYC, upload 2 ảnh CCCD | Upload thành công, hiển thị xem trước | Chưa cập nhật trường KYC                    |
| 2    | Nhấn OCR trích xuất                | API trả về số CCCD, họ tên, ngày sinh | Dữ liệu tạm hiển thị trên form              |
| 3    | Xác nhận lưu                       | API cập nhật hồ sơ thành công         | Các trường CCCD trong `users` được cập nhật |
| 4    | Mở lại trang hồ sơ                 | Thông tin KYC hiển thị đúng           | Dữ liệu DB đồng nhất với UI                 |

**Tiêu chí đạt:** Dữ liệu OCR được lưu đúng tài khoản, không bị trùng CCCD.

##### Kịch bản KS-05: Chuyển khoản nội địa thành công và bảo toàn nhất quán

**Mục tiêu:** Xác nhận chuyển khoản hợp lệ cập nhật đúng số dư hai phía và ghi transaction.

**Tiền điều kiện:**

- Tài khoản A (nguồn): số dư 1,000,000.
- Tài khoản B (đích): số dư 500,000.
- Cả hai tài khoản trạng thái `approved`.

**Dữ liệu kiểm thử:** Chuyển 200,000 từ A sang B, nội dung "Test chuyen tien KS-05".

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                                      | Kết quả mong đợi tại UI/API          | Kiểm tra dữ liệu                        |
| ---- | --------------------------------------------- | ------------------------------------ | --------------------------------------- |
| 1    | User A mở form chuyển khoản, nhập tài khoản B | Tra cứu đúng tên người nhận          | Chưa cập nhật số dư                     |
| 2    | Nhập số tiền + nội dung, xác nhận chuyển      | API trả thành công                   | Chuẩn bị ghi transaction                |
| 3    | Xem thông báo sau giao dịch                   | UI báo chuyển thành công, số dư giảm | `users.balance(A) = 800,000`            |
| 4    | Đăng nhập tài khoản B kiểm tra số dư          | Số dư tăng tương ứng                 | `users.balance(B) = 700,000`            |
| 5    | Mở lịch sử giao dịch                          | Có bản ghi mới với amount = 200,000  | Có 1 dòng mới trong `bank_transactions` |

**Tiêu chí đạt:** Số dư hai bên thay đổi chính xác và có đúng 1 bản ghi giao dịch.

##### Kịch bản KS-06: Chuyển khoản thất bại do không đủ số dư

**Mục tiêu:** Xác nhận giao dịch bị chặn và dữ liệu không bị thay đổi khi không đủ số dư.

**Tiền điều kiện:** Tài khoản A có số dư 100,000.

**Dữ liệu kiểm thử:** Thực hiện chuyển 500,000 sang tài khoản B.

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                               | Kết quả mong đợi tại UI/API          | Kiểm tra dữ liệu                    |
| ---- | -------------------------------------- | ------------------------------------ | ----------------------------------- |
| 1    | Nhập thông tin chuyển khoản vượt số dư | Form hợp lệ về cú pháp               | Chưa cập nhật DB                    |
| 2    | Nhấn xác nhận chuyển                   | API trả lỗi nghiệp vụ số dư không đủ | Không tạo transaction               |
| 3    | Kiểm tra lại số dư A và B              | Cả hai số dư giữ nguyên              | DB không thay đổi số dư             |
| 4    | Mở lịch sử giao dịch                   | Không có bản ghi mới cho lần thử này | `bank_transactions` không tăng dòng |

**Tiêu chí đạt:** Không phát sinh thay đổi số dư và không có giao dịch rác.

##### Kịch bản KS-07: Xóa tài khoản có lưu archive

**Mục tiêu:** Xác nhận quy trình xóa user luôn tạo snapshot trước khi xóa bản ghi gốc.

**Tiền điều kiện:**

- Admin đã đăng nhập.
- User mục tiêu tồn tại, có thông tin hồ sơ đầy đủ.

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                                        | Kết quả mong đợi tại UI/API  | Kiểm tra dữ liệu                     |
| ---- | ----------------------------------------------- | ---------------------------- | ------------------------------------ |
| 1    | Admin mở danh sách user, chọn tài khoản cần xóa | Hiển thị hộp xác nhận        | Chưa thay đổi DB                     |
| 2    | Xác nhận thao tác xóa                           | API trả thành công           | Bắt đầu tạo bản ghi archive          |
| 3    | Mở màn hình hồ sơ đã xóa                        | Có dữ liệu tài khoản vừa xóa | Có dòng mới trong `deleted_profiles` |
| 4    | Tìm lại user ở danh sách active                 | Không còn user mục tiêu      | Bản ghi đã bị xóa khỏi `users`       |

**Tiêu chí đạt:** Dữ liệu bị xóa khỏi `users` nhưng vẫn truy xuất được snapshot trong `deleted_profiles`.

##### Kịch bản KS-08: Kiểm thử phân quyền truy cập trang quản trị

**Mục tiêu:** Đảm bảo user thường không thể truy cập chức năng admin.

**Tiền điều kiện:**

- Có 1 tài khoản role = user.
- Có 1 tài khoản role = admin.

**Các bước thực hiện và điểm kiểm tra:**

| Bước | Thao tác                                    | Kết quả mong đợi tại UI/API                       | Kiểm tra dữ liệu            |
| ---- | ------------------------------------------- | ------------------------------------------------- | --------------------------- |
| 1    | Đăng nhập bằng tài khoản user thường        | Đăng nhập thành công vào dashboard user           | Session role = user         |
| 2    | Truy cập trực tiếp URL/admin endpoint admin | Bị từ chối (403 hoặc điều hướng về trang phù hợp) | Không trả dữ liệu quản trị  |
| 3    | Đăng nhập bằng tài khoản admin              | Truy cập dashboard admin thành công               | Session role = admin        |
| 4    | Gọi lại endpoint admin                      | API trả dữ liệu hợp lệ                            | Dữ liệu hiển thị đúng quyền |

**Tiêu chí đạt:** Phân quyền chặn đúng user thường và cho phép đúng admin.

##### Tiêu chuẩn chấm kết quả cho toàn bộ kịch bản chi tiết

1. **Pass:** Tất cả điểm kiểm tra UI/API/DB của kịch bản đều đúng kỳ vọng.
2. **Fail:** Chỉ cần một điểm kiểm tra sai hoặc không tái lập được.
3. **Blocked:** Không thể tiếp tục vì lỗi môi trường (mất camera, server không chạy, thiếu dữ liệu tiền điều kiện).

Khi phát hiện `Fail` hoặc `Blocked`, cần ghi lại:

- Mã kịch bản.
- Ảnh chụp màn hình tại bước lỗi.
- Log API hoặc log server tương ứng.
- Kết quả kiểm tra DB trước và sau thao tác.

### 5.7 Kiểm thử phi chức năng

#### 5.7.1 Hiệu năng phản hồi API

Tiến hành đo thủ công bằng cách gọi lặp lại các API phổ biến (đăng nhập mật khẩu, lấy hồ sơ, xem lịch sử giao dịch) trong điều kiện mạng nội bộ.

Kết quả quan sát ở quy mô thử nghiệm lớp học:

1. API truy vấn dữ liệu cơ bản phản hồi nhanh và ổn định.
2. API liên quan AI có độ trễ cao hơn do xử lý ảnh, nhưng vẫn chấp nhận được trong demo.
3. Hệ thống không xuất hiện tình trạng treo tiến trình khi chạy tuần tự các ca kiểm thử chính.

#### 5.7.2 Bảo mật mức ứng dụng

Các điểm đã kiểm tra:

1. Mật khẩu không lưu plain text, chỉ lưu dạng hash.
2. Các trang quản trị yêu cầu quyền admin.
3. Thông báo đăng nhập sai không phân biệt chi tiết user/mật khẩu.
4. Các thao tác quản trị nhạy cảm yêu cầu xác nhận trên giao diện.

#### 5.7.3 Tính nhất quán dữ liệu

Kiểm tra nhất quán tập trung ở nghiệp vụ chuyển khoản và xóa tài khoản:

1. Chuyển khoản thành công phải đồng thời thay đổi số dư hai phía và ghi transaction.
2. Chuyển khoản thất bại không được tạo transaction rác.
3. Xóa user phải tạo archive trước khi xóa bản ghi chính.
4. Dữ liệu archive vẫn truy xuất được sau khi user gốc đã xóa.

### 5.8 Tổng hợp kết quả kiểm thử

**Bảng 5.9 - Tổng hợp kết quả theo nhóm**

| Nhóm kiểm thử       | Số ca chạy | Đạt | Không đạt | Ghi chú                                        |
| ------------------- | ---------- | --- | --------- | ---------------------------------------------- |
| Đăng ký và duyệt    | 5          | 5   | 0         | Bao phủ đầy đủ luồng pending/approved/rejected |
| Xác thực và phiên   | 5          | 5   | 0         | Bao gồm mật khẩu, khuôn mặt, liveness          |
| KYC và hồ sơ        | 4          | 4   | 0         | OCR phụ thuộc chất lượng ảnh đầu vào           |
| Giao dịch tài chính | 4          | 4   | 0         | Kiểm tra đủ ca hợp lệ và sai điều kiện         |
| Admin và archive    | 5          | 5   | 0         | Lưu vết xóa hoạt động đúng                     |

Nhận xét:

1. Toàn bộ ca kiểm thử chính đều đạt trong môi trường thử nghiệm.
2. Các lỗi phát sinh chủ yếu là dữ liệu đầu vào không đạt chất lượng (ảnh mờ, thiếu sáng), không phải lỗi logic lõi.
3. Hệ thống đạt mục tiêu vận hành ổn định ở quy mô demo học thuật.

### 5.9 Tiểu kết chương

Chương 5 đã trình bày đầy đủ quy trình cài đặt và kiểm thử hệ thống từ mức môi trường đến mức nghiệp vụ.
Kết quả kiểm thử cho thấy các chức năng cốt lõi vận hành đúng theo thiết kế, dữ liệu lưu trữ nhất quán và các kiểm soát bảo mật cơ bản được đáp ứng.

---

## Chương 6. Đánh giá kết quả

### 6.1 Mục tiêu đánh giá

Đánh giá kết quả nhằm xác định mức độ hoàn thành của dự án so với mục tiêu đặt ra trong Chương 1 và yêu cầu nghiệp vụ ở Chương 2.
Cách đánh giá kết hợp giữa:

1. Đánh giá định tính theo khả năng đáp ứng quy trình nghiệp vụ.
2. Đánh giá định lượng cơ bản qua kết quả kiểm thử và mức độ ổn định vận hành.

### 6.2 Mức độ hoàn thành theo mục tiêu ban đầu

**Bảng 6.1 - Đối chiếu mục tiêu và kết quả thực hiện**

| Mục tiêu ban đầu                                         | Kết quả thực hiện                                             | Mức độ                  |
| -------------------------------------------------------- | ------------------------------------------------------------- | ----------------------- |
| Xây dựng hệ thống ngân hàng mô phỏng có thể chạy thực tế | Hệ thống chạy được đầy đủ frontend, backend, DB và upload ảnh | Hoàn thành              |
| Đăng ký, đăng nhập, quản trị tài khoản                   | Đã triển khai đầy đủ user/admin flow và trạng thái tài khoản  | Hoàn thành              |
| Tích hợp nhận diện khuôn mặt + liveness                  | Có luồng đăng nhập mặt, có bước kiểm tra anti-spoofing cơ bản | Hoàn thành có điều kiện |
| Tích hợp OCR CCCD phục vụ KYC                            | OCR hoạt động với ảnh đạt chuẩn, có kiểm tra trùng CCCD       | Hoàn thành              |
| Chuyển khoản nội địa + lịch sử giao dịch                 | Giao dịch hai chiều và lịch sử hoạt động ổn định              | Hoàn thành              |
| Tính sẵn sàng production banking                         | Chưa đáp ứng do còn thiếu hardening và tuân thủ nâng cao      | Chưa hoàn thành         |

Kết luận đối chiếu:

1. Dự án đạt gần như đầy đủ mục tiêu học thuật và ứng dụng ở mức prototype.
2. Các mục tiêu chưa đạt chủ yếu thuộc nhóm yêu cầu production và tuân thủ doanh nghiệp lớn.

### 6.3 Đánh giá theo các tiêu chí chất lượng

#### 6.3.1 Tính đúng đắn nghiệp vụ

Các luồng nghiệp vụ chính đã được hiện thực đồng bộ với use case:

1. Vòng đời tài khoản: pending -> approved/rejected -> locked.
2. Luồng xác thực kép: mật khẩu và khuôn mặt.
3. Luồng quản trị: duyệt, khóa/mở khóa, reset mặt, xóa có archive.
4. Luồng giao dịch: kiểm tra điều kiện trước khi ghi nhận biến động số dư.

Đánh giá: đạt tốt cho môi trường học tập.

#### 6.3.2 Tính ổn định vận hành

Qua các vòng kiểm thử:

1. Hệ thống không xuất hiện lỗi nghiêm trọng làm gián đoạn toàn bộ dịch vụ.
2. Các lỗi phổ biến liên quan đầu vào người dùng đều có thông báo phản hồi.
3. Dữ liệu vẫn nhất quán sau các kịch bản lỗi quan trọng (sai mật khẩu, số dư không đủ, OCR thất bại).

Đánh giá: ổn định ở quy mô demo/lab.

#### 6.3.3 Tính bảo mật cơ bản

Điểm mạnh:

1. Không lưu mật khẩu thuần văn bản.
2. Có phân quyền vai trò và chặn truy cập trang admin từ user thường.
3. Có bước liveness giảm nguy cơ giả mạo cơ bản.

Điểm còn thiếu:

1. Chưa có MFA/OTP.
2. Chưa mã hóa sâu các trường dữ liệu nhạy cảm ở mức cột.
3. Chưa có bộ kiểm thử bảo mật tự động chuyên sâu.

Đánh giá: đạt mức cơ bản, phù hợp phạm vi prototype.

#### 6.3.4 Khả năng mở rộng và bảo trì

Mặt tích cực:

1. Cấu trúc thư mục rõ ràng, tách frontend và backend.
2. Thiết kế dữ liệu đơn giản, dễ thêm bảng mới.
3. Luồng API tương đối nhất quán.

Mặt hạn chế:

1. Một số lớp/service còn phụ thuộc chặt, khó mock khi unit test.
2. Chưa có pipeline CI/CD và bộ test tự động đầy đủ.
3. Chưa tách rõ tầng repository/service ở toàn bộ module.

Đánh giá: có nền tảng tốt để mở rộng ở giai đoạn tiếp theo.

### 6.4 Ưu điểm nổi bật của dự án

1. Kết hợp được nhiều mảng kỹ thuật trong một sản phẩm hoàn chỉnh: web, AI/CV, OCR, cơ sở dữ liệu, bảo mật ứng dụng.
2. Luồng nghiệp vụ gần thực tế ngân hàng số ở mức mô phỏng, có phân vai trò rõ ràng giữa user và admin.
3. Tích hợp AI theo đúng điểm chạm nghiệp vụ, không chỉ dừng ở demo thuật toán rời rạc.
4. Có cơ chế lưu vết hồ sơ bị xóa, tăng khả năng truy vết và kiểm tra sau vận hành.
5. Dễ cài đặt lại và demo trong môi trường lớp học.

### 6.5 Hạn chế hiện tại

1. Độ chính xác nhận diện khuôn mặt và OCR phụ thuộc lớn vào chất lượng camera, ánh sáng và góc chụp.
2. Chưa có cơ chế chống tấn công nâng cao như rate limit toàn diện, thiết bị tin cậy, phát hiện bất thường.
3. Chưa triển khai chuẩn hóa dữ liệu sâu (ví dụ tách bảng KYC riêng) nên bảng người dùng còn rộng.
4. Chưa có kiểm thử tải lớn để đánh giá ngưỡng chịu tải thực tế.
5. Chưa đáp ứng bộ tiêu chuẩn tuân thủ ngành tài chính ở mức thương mại.

### 6.6 Bài học kinh nghiệm

Trong quá trình triển khai, nhóm rút ra các bài học quan trọng:

1. Cần xác định rõ phạm vi ngay từ đầu để tránh mở rộng không kiểm soát.
2. Nên thiết kế nghiệp vụ và trạng thái tài khoản rõ trước khi viết code để giảm sửa đổi về sau.
3. Với các module AI, cần chuẩn hóa dữ liệu đầu vào và bổ sung thông báo hướng dẫn người dùng thật rõ ràng.
4. Việc có bộ test case theo use case giúp phát hiện lỗi logic sớm và giảm rủi ro khi chỉnh sửa.

### 6.7 Định hướng phát triển tiếp theo

1. Tích hợp MFA (OTP hoặc app token) cho các thao tác quan trọng như đăng nhập thiết bị mới và chuyển khoản.
2. Bổ sung mã hóa dữ liệu nhạy cảm ở mức trường và quản lý khóa bảo mật tập trung.
3. Nâng cấp anti-spoofing bằng mô hình học sâu chuyên dụng và bộ dữ liệu phong phú hơn.
4. Tách kiến trúc backend thành các module rõ ràng hơn (auth, kyc, transaction, admin) để mở rộng quy mô.
5. Xây dựng CI/CD và kiểm thử tự động (unit, integration, security, performance).
6. Bổ sung hệ thống audit log chi tiết, cảnh báo bất thường và dashboard giám sát vận hành.

### 6.8 Tiểu kết chương

Chương 6 cho thấy dự án đã đạt mục tiêu trọng tâm ở mức học thuật và ứng dụng thử nghiệm.
Hệ thống chứng minh được tính khả thi của hướng tích hợp AI vào xác thực ngân hàng số, đồng thời chỉ ra rõ các khoảng trống cần tiếp tục hoàn thiện để tiến gần hơn đến yêu cầu triển khai thực tế quy mô lớn.

---

## Chương 7. Kết luận

### 7.1 Kết luận chung

Báo cáo đã trình bày đầy đủ quá trình xây dựng hệ thống ngân hàng số mô phỏng có tích hợp AI trong nhận diện khuôn mặt, từ giai đoạn phân tích yêu cầu đến thiết kế, cài đặt, kiểm thử và đánh giá kết quả.
Kết quả đạt được cho thấy hướng tiếp cận tích hợp AI vào bài toán xác thực là khả thi trong bối cảnh học thuật và có giá trị thực tiễn ở quy mô thử nghiệm.

Hệ thống đã hiện thực thành công các nhóm chức năng cốt lõi:

1. Đăng ký tài khoản, đăng nhập bằng mật khẩu và đăng nhập bằng khuôn mặt.
2. Hỗ trợ KYC với OCR CCCD, cập nhật khuôn mặt và quản lý hồ sơ cá nhân.
3. Hỗ trợ chuyển khoản nội địa, lưu lịch sử giao dịch và theo dõi số dư.
4. Cung cấp chức năng quản trị gồm duyệt, từ chối, khóa/mở khóa, reset khuôn mặt, xóa tài khoản và lưu vết archive.

Thông qua kết quả kiểm thử ở Chương 5 và đánh giá ở Chương 6, có thể khẳng định hệ thống đạt mục tiêu chính của dự án ở mức prototype: vận hành ổn định, luồng nghiệp vụ rõ ràng, dữ liệu nhất quán và có nền tảng để mở rộng.

### 7.2 Đóng góp chính của dự án

Dự án có các đóng góp nổi bật như sau:

1. Xây dựng được một sản phẩm hoàn chỉnh theo hướng liên ngành, kết hợp web, AI/CV, OCR và cơ sở dữ liệu trong cùng một hệ thống.
2. Chứng minh được cách đưa AI vào đúng điểm chạm nghiệp vụ thay vì chỉ dừng ở mức mô hình thử nghiệm độc lập.
3. Thiết lập được quy trình quản trị tài khoản có trạng thái rõ ràng và phù hợp với bài toán ngân hàng số mô phỏng.
4. Bổ sung cơ chế lưu vết hồ sơ bị xóa, tăng khả năng truy xuất và kiểm tra hậu kiểm.
5. Tạo ra tài liệu phân tích - thiết kế - kiểm thử có cấu trúc, hỗ trợ tốt cho bảo trì và mở rộng về sau.

### 7.3 Hạn chế còn tồn tại

Mặc dù đạt được nhiều kết quả tích cực, dự án vẫn tồn tại một số giới hạn:

1. Độ chính xác của nhận diện khuôn mặt và OCR còn phụ thuộc vào chất lượng ảnh/camera và điều kiện môi trường.
2. Hệ thống chưa đạt mức triển khai production banking do thiếu các lớp bảo mật nâng cao và yêu cầu tuân thủ chuyên ngành.
3. Kiểm thử hiện tập trung chủ yếu ở quy mô học thuật, chưa có benchmark tải lớn và kiểm thử an ninh chuyên sâu.
4. Một số thành phần kiến trúc có thể tiếp tục tách nhỏ để tăng khả năng mở rộng và test độc lập.

### 7.4 Hướng phát triển tiếp theo

Để nâng cấp hệ thống trong các giai đoạn tiếp theo, nhóm định hướng:

1. Tích hợp xác thực đa yếu tố (MFA) cho đăng nhập và giao dịch quan trọng.
2. Tăng cường anti-spoofing bằng mô hình học sâu và dữ liệu huấn luyện đa dạng hơn.
3. Mã hóa sâu dữ liệu nhạy cảm, bổ sung quản lý khóa tập trung và cơ chế giám sát bảo mật.
4. Hoàn thiện pipeline kiểm thử tự động và CI/CD để nâng cao chất lượng phát hành.
5. Mở rộng kiến trúc theo hướng module hóa sâu hơn, sẵn sàng cho quy mô người dùng lớn hơn.

### 7.5 Kết luận cuối cùng

Tổng thể, dự án đã hoàn thành tốt mục tiêu đề ra trong phạm vi học tập - ứng dụng, đồng thời tạo được nền tảng kỹ thuật và tài liệu học thuật đủ vững để tiếp tục phát triển lên các mức hoàn thiện cao hơn.
Giá trị quan trọng nhất của dự án không chỉ nằm ở sản phẩm đang chạy được, mà còn ở khả năng mở rộng thành một hệ thống xác thực số an toàn, thân thiện và có tính ứng dụng thực tế trong tương lai.

## Tai lieu tham khao

1. [Tai lieu 1]
2. [Tai lieu 2]
3. [Tai lieu 3]

## Phu luc (neu co)

- Anh man hinh he thong
- Bang test case
- Ket qua test
