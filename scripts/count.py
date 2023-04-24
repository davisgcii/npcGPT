import os

FOLDER_PATH = "data"


def count_files(folder_path, extensions):
    count = {ext: 0 for ext in extensions}

    for entry in os.scandir(folder_path):
        if entry.is_file():
            for ext in extensions:
                if entry.name.endswith(ext):
                    count[ext] += 1
                    break  # exit the loop once the correct extension is found
        elif entry.is_dir():
            subfolder_count = count_files(entry.path, extensions)
            for ext in extensions:
                count[ext] += subfolder_count[ext]

    return count


def main():
    folder_path = FOLDER_PATH
    extensions = [".txt", ".png"]
    file_count = count_files(folder_path, extensions)

    for ext in extensions:
        print(f"File count for {ext}: {file_count[ext]}")


if __name__ == "__main__":
    main()
