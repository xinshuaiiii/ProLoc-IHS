import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# 下载图片的函数
def download_image(protein_id, image_url, HPA,location, output_dir):
    # 从URL中提取图像文件名并去除最后的 .jpg 部分
    image_name_part = image_url.split('/')[-1].replace('.jpg', '')
    # 构建新的图像名称
    new_image_name = f"{protein_id}-{image_name_part}-{HPA}-{location}.jpg"
    # 构建图像保存路径
    image_path = os.path.join(output_dir, new_image_name)

    try:
        response = requests.get(image_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return f"下载成功: {new_image_name}"
        else:
            return f"下载失败: {protein_id}, URL: {image_url}"
    except Exception as e:
        return f"下载出错: {protein_id}, URL: {image_url}, 错误: {e}"

# 加载数据
data = pd.read_csv('test_img_URL.csv')

# 创建保存图像的主目录
output_dir = '../test_img'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用线程池加速下载
failed_downloads = []
successful_downloads = []  # 存储成功下载的记录
download_records = []  # 存储下载记录，用于生成新表格
max_workers = 8  # 设置最大线程数，根据系统资源进行调整

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for index, row in data.iterrows():
        protein_id = row['ProteinId']  # 使用Protein Id作为文件名的一部分
        # image_url = row.iloc[-1]  # 使用最后一列的URL
        image_url = row['URL']  # 使用最后一列的URL
        location = row['locations']  # locations列
        HPA = row['AntibodyId']

        # 提交下载任务
        futures.append(executor.submit(download_image, protein_id, image_url,HPA, location, output_dir))

    # 收集下载结果
    for future in as_completed(futures):
        result = future.result()
        print(result)
        if "下载成功" in result:
            successful_downloads.append(result)
            # 从结果提取文件名、locations和URL信息
            file_name = result.split(": ")[1]
            download_records.append([file_name, location, image_url])
        elif "下载失败" in result or "下载出错" in result:
            failed_downloads.append(result)

# 下载完成后，打印出下载失败的记录
if failed_downloads:
    print("\n以下图像下载失败:")
    for failure in failed_downloads:
        print(failure)
else:
    print("所有图像下载成功")

# 打印成功下载的图片数量
print(f"\n成功下载的图片数量: {len(successful_downloads)}")

download_df = pd.DataFrame(download_records, columns=['File Name', 'locations', 'URL'])
download_df.to_csv('../GmPLoc/all_data1.csv', index=False)
print("下载记录已保存到 'download_records.csv'")
