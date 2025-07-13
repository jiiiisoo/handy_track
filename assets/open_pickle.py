import pickle

with open('/workspace/ManipTrans/assets/scene_01__A001%2B%2Bseq__07bb164dc3d3873d6389__2023-04-27-20-45-29.pkl', 'rb') as f:
    data = pickle.load(f)

print(data["raw_smplx"][10446].keys())