
[English Here](https://github.com/Qwekem482/Random_Forest_Rainfall-Estimation/blob/main/README_en.md)

# Mô hình vận hành

![RandomForest](https://github.com/Qwekem482/RandomForestRainfallEstimation/assets/80797630/f6a79cd7-7870-4bac-8346-12a26a818249)


# Dữ liệu huấn luyện và điều chỉnh (train/validate data)

Dữ liệu huấn luyện ban đầu sẽ được chia nhóm dựa theo lượng mưa để huấn luyện và điều chỉnh bộ phân loại (RF1): 

•	Nhóm 1: Không mưa (0 mm/h)

•	Nhóm 2: Mưa nhỏ (dưới 0.675 mm/h)

•	Nhóm 3: Mưa to (trên 0.675 mm/h)

Sau đó, những dữ liệu này sẽ được sao chép làm 2 bản. Bản thứ nhất sẽ được đưa vào huấn luyện bộ phân loại (RF1). Bản thứ 2 được chia làm 3 phần tương ứng với 3 nhóm mưa. Những dữ liệu nhóm 2 sẽ được dùng để huấn luyện và điều chỉnh bộ ước lượng mưa nhỏ (RF2), dữ liệu nhóm 3 sẽ được dùng để huấn luyện và điều chỉnh bộ ước lượng mưa to.

Bởi vì dữ liệu được cung cấp mất cân bằng nghiêm trọng (số lượng dữ liệu không mưa lớn hơn hẳn dữ liệu có mưa) nên để đảm bảo chất lượng mô hình, dữ liệu huấn luyện sẽ được cân bằng dựa trên kỹ thuật SMOTETomek-links

# Điều chỉnh mô hình

Đối với mô hình Random Forest, có các siêu tham số sau đây cần phải điều chỉnh để mô hình có thể hoạt động hiệu quả nhất:

•	n_estimators: từ 100 đến 3000, bước nhảy 100

•	max_features: từ 0.05 đến 1.0, bước nhảy 0.05

•	min_samples_split: từ 0.025 đến 0.5, bước nhảy 0.025

•	min_samples_leaf: từ 0.05 đến 1.0, bước nhảy 0.05

•	max_samples: từ 0.05 đến 1.0, bước nhảy 0.05

•	min_weight_fraction_leaf: từ 0.025 đến 0.5, bước nhảy 0.025

# Dữ liệu kiểm thử

Dữ liệu kiểm thử sẽ được đưa vào bộ phân loại (RF1) để phân loại. Các dữ liệu kiểm thử sau khi được phân loại thì lấy những dữ liệu thuộc nhóm 2, 3 để đưa vào các bộ ước lượng (RF2, RF3) tương ứng để ước tính lượng mưa. Kết quả sẽ được so sánh với lượng mưa mà IMERG đưa ra.

# Chạy mô hình
Build và chạy file RandomForest.py để chạy mô hình

•	Build và chạy file RandomForest.py để chạy mô hình

•	Sửa đường dẫn đến dataset trong file ImportData.py

•	Code trong file Jupyter Notebook giống hệt với code trong các file python

•	Thư mục Jupyter Notebook thiếu file ImportData.py
