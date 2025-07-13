import json

with open('/mnt/nas1/jisoo6687/affordance_dataset/gigahands/p001-folder/keypoints_3d_mano/000.json', 'r') as f:
    data = json.load(f)

print(len(data[0]))