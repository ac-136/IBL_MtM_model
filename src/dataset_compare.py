from utils.dataset_utils import test_load_ibl_dataset, load_locally

eid = '03d9a098-07bf-4765-88b7-85f8d8f620cc_aligned'

dataset = load_locally(just_spikes=True, eid=eid)
val = dataset["val"]

print(val)

# dataset_hf = test_load_ibl_dataset(
#                             eid=eid,
#                             seed=42)

# print(dataset_hf)
# print()

# for i in range(5):
#     print(set(dataset_hf["cluster_regions"][i]))
#     print(len(set(dataset_hf["cluster_regions"][i])))
#     print()

# all_sets = [set(row) for row in dataset_hf["cluster_regions"]]
# first_set = all_sets[0]
# all_equal = all(s == first_set for s in all_sets)
# print(all_equal)

# for k in dataset_hf.features.keys():
#     print(f"{k}: ")
#     try:
#         print(len(dataset_hf[k][0]))
#         print(dataset_hf[k][0][0:10])
#         print(len(dataset_hf[k][1]))
#     except:
#         if not isinstance(dataset_hf[k][0], dict):
#             print(dataset_hf[k][0])
#         else:
#             print("is a dict")
#     print()