import os
import shutil
from PIL import Image

# set the folder paths
untest_folder = r"g:\\learning\\fyp\\visual_RFT\\Visual-RFT\\untest"
test_result_folder = r"g:\\learning\\fyp\\visual_RFT\\Visual-RFT\\test result"

# get all image files in the untest folder
images = [f for f in os.listdir(untest_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# set the starting folder number
start_folder_number = 404

# go through each image and create a folder for it
for i, image in enumerate(images):
    # 创建以编号命名的文件夹
    folder_name = str(start_folder_number + i)
    target_folder = os.path.join(test_result_folder, folder_name)
    os.makedirs(target_folder, exist_ok=True) 

    # move the image to the new folder
    source_path = os.path.join(untest_folder, image)
    target_path = os.path.join(target_folder, image)
    shutil.move(source_path, target_path)
    print(f"Moved {image} to {target_folder}")

    # show the image
    img_path = target_path
    img = Image.open(img_path)
    img.show()

    # getting user input for a.txt
    user_input = input("Enter content for a.txt: ")

    # close the image after user input
    img.close()

    # creating a.txt with user input
    txt_file_path = os.path.join(target_folder, "a.txt")
    with open(txt_file_path, "w") as txt_file:
        txt_file.write(user_input)
    print(f"Created a.txt in {target_folder} with content: {user_input}")