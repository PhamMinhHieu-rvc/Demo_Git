from sklearn import tree

#buoc 1: Thu thập dữ liệu
#buoc 2: Xử lí dữ liệu
#buoc 3: Training model
#buoc 4: Dự đoán kết quản
# Bước đánh giá model

my_tree = tree.DecisionTreeClassifier()
# demo git pull

dactrung = [[1, 3, 3, 7],
            [5, 2, 4, 6],
            [1, 2, 4, 6],
            [5, 4, 4, 3],
            [1, 4, 4, 7],
            [3, 2, 3, 7],
            [3, 3, 3, 6],
            [5, 2, 2, 7],
            ]

nhan = [0 ,1, 1, 0, 0, 0, 0, 1]

result = my_tree.fit(dactrung, nhan)

ketqua =  result.predict([[1, 3 ,4, 7]])
print(ketqua)
