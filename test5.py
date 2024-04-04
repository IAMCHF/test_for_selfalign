import os
import re


# 定义读取STAR文件函数
def read_star_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        data_start = lines.index('data_\n') + 1
        loop_start = lines.index('loop_\n') + 1

        subtomo_files = []
        for line in lines[loop_start:]:
            fields = line.strip().split()
            if len(fields) >= 5 and re.match(r"(.*)subtomo(.*).mrc", fields[4]):
                original_subtomo_file = os.path.basename(fields[4])
                modified_subtomo_file = os.path.splitext(original_subtomo_file)[0] + '_binned4' + \
                                        os.path.splitext(original_subtomo_file)[1]
                subtomo_files.append(modified_subtomo_file)

        return subtomo_files


# 读取STAR文件内容
star_filename = "/media/hao/Hard_disk_1T/datasets/10045/AnticipatedResults/particles_subtomo_good.star"  # 请替换为你的STAR文件路径
required_files = read_star_file(star_filename)

# 指定文件夹路径
folder_path = "/media/hao/Sata500g/my_dataset/subtomo_bin4/"  # 请替换为你要检查的文件夹路径

# 获取指定文件夹中所有文件名
existing_files = [os.path.basename(file) for file in os.listdir(folder_path)]

# 计算需要保留的文件名列表（交集）
files_to_keep = set(required_files) & set(existing_files)
# print(files_to_keep, len(files_to_keep))

# 删除不在列表中的文件
for file in existing_files:
    if file not in files_to_keep:
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            os.remove(full_path)
            print(f"Deleted: {full_path}")

print("Finished cleaning up folder.")