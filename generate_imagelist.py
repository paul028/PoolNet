from os import listdir

def list_files(directory, extension):
    return (f for f in listdir(directory) if f.endswith('.' + extension))

def main():
    directory = 'C:/Users/paulvincentnonat/Documents/GitHub/Saliency_Dataset/DUTS/DUTS-TR/DUTS-TR-Image'
    files = list_files(directory, ".jpg")
    for f in files:
        print(f)

main()
