# name_method = ["own_class_center", "estimated_hyper_lin", "actual_hyper_lin", "distance_center_own_opposite_tam"]
# name_function = ["lin","lin_center_own", "exp", "func_own_opp_new"]
# # for namemethod in name_method:
# #     for namefunction in name_function:
# #         if namemethod == "distance_center_own_opposite_tam" and namefunction == "func_own_opp_new":
# #             print(namemethod, namefunction)
# #         elif namemethod == "distance_center_own_opposite_tam" and namefunction == "lin_center_own":
# #             continue
# #         elif namemethod == "distance_center_own_opposite_tam" and namefunction == "exp":
# #             continue
# #         elif namemethod == "own_class_center" and namefunction == "func_own_opp_new":
# #             continue
# #         elif namemethod == "estimated_hyper_lin" and namefunction == "func_own_opp_new":
# #             continue
# #         elif namemethod == "actual_hyper_lin" and namefunction == "func_own_opp_new":
# #             continue
# #         elif namemethod == "own_class_center" and namefunction == "lin":
# #             continue
# #         elif namemethod == "estimated_hyper_lin" and namefunction == "lin_center_own":
# #             continue
# #         elif namemethod == "actual_hyper_lin" and namefunction == "lin_center_own":
# #             continue
# #         elif namemethod == "distance_center_own_opposite_tam" and namefunction == "lin":
# #             continue
# #         else:
# #             print(namemethod, namefunction)
# # for namemethod in name_method:
# #         for namefunction in name_function:
# #             if namemethod =="distance_center_own_opposite_tam" and namefunction =="lin_center_own":
# #                 continue
# #             elif namemethod =="distance_center_own_opposite_tam" and namefunction =="exp":
# #                 continue
# #             elif namemethod == "own_class_center" and namefunction == "func_own_opp_new":
# #                 continue
# #             elif namemethod == "estimated_hyper_lin" and namefunction == "func_own_opp_new":
# #                 continue
# #             elif namemethod == "actual_hyper_lin" and namefunction == "func_own_opp_new":
# #                 continue
# #             elif namemethod == "distance_center_own_opposite_tam" and namefunction == "lin":
# #                 continue
# #             elif namemethod == "own_class_center" and namefunction == "lin":
# #                 continue
# #             elif namemethod == "estimated_hyper_lin" and namefunction == "lin":
# #                 continue
# #             elif namemethod == "actual_hyper_lin" and namefunction == "lin_center_own":
# #                 continue
# #             else:
# #                 print(namemethod, namefunction)

# # import csv
# #
# # header = ['SP', 'SE', 'Gmean', 'F1 Score']
# # data = [
# #     ['Albania', 28748, 'AL', 'ALB'],
# #     ['Algeria', 2381741, 'DZ', 'DZA'],
# #     ['American Samoa', 199, 'AS', 'ASM'],
# #     ['Andorra', 468, 'AD', 'AND'],
# #     ['Angola', 1246700, 'AO', 'AGO']
# # ]
# # data1 = []
# # # data1.append(['Albania', 28748, 'AL', 'ALB'])
# # # print(data1)
# #
# # with open('./Experiment/countries.csv', 'w', encoding='UTF8', newline='') as f:
# #     writer = csv.writer(f)
# #
# #     # write the header
# #     writer.writerow(header)
# #     data1.append(['Albania', 28748, 'AL', 'ALB'])
# #     # write multiple rows
# #     writer.writerows(data1)N = 5
# N = 5

# for i in range(0,5):
#     print(i)
#     print(3+2)

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

data = pd.read_csv("D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/Data/Test_Iris.csv")
data = np.array(data)
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
print(kmeans.cluster_centers_)