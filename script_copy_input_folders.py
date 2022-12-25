import common_utils

def main():
    input_main_folder_full_path = './input_data1/images'
    output_main_folder_full_path = './input_data/images'
    common_utils.copy_folders(input_main_folder_full_path, output_main_folder_full_path)

main()

