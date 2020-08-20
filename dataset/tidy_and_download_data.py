import urllib.request
import pandas as pd

#TIM: was videos/
viddir = "/media/tim/dl3storage/Datasets/asllrp/"

def find_class_occurance_and_download_video(excel_file):

    words = {}      #key = words, value = number of occurences
    signs = []
    firstline = True
    for i in excel_file.index:
        
        if firstline:    #skip first line
            firstline = False
            continue
    
        if excel_file['Main New Gloss'][i] == " " or str(excel_file['Main New Gloss'][i]) == "nan":
            if excel_file['Consultant'][i] != "------------":
                words[current] += 1 
        elif excel_file['Consultant'][i] != "------------":
            current = excel_file['Main New Gloss'][i]
            words[current] = 0

    with open("sign_class_and_occurance.csv", mode='w') as fout:
        for key, value in words.items():
            if value >= 10:
                fout.write("%s, %s\n" % (key, value))
                signs.append(key)
    
    return signs

def download_required(excel_file, signs):

    words = {}
    firstline = True

    for i in excel_file.index:
        
        if firstline:
            firstline = False
            continue
            
        if excel_file['Main New Gloss'][i] == " " or str(excel_file['Main New Gloss'][i]) == "nan":
            if excel_file['Consultant'][i] != "------------":
                if current in signs:
                    video_link = excel_file['Links seperate 1'][i][12:-9]
                    sign = current.replace("/", ".")
                    urllib.request.urlretrieve(video_link, viddir + sign + "__" + str(words[current] + 1) + ".mov") 
                    words[current] += 1 
        elif excel_file['Consultant'][i] != "------------":
                current = excel_file['Main New Gloss'][i]
                words[current] = 0

def main():
    
    excel_file = pd.read_excel("dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx", sheet_name="Sheet1")

    signs = find_class_occurance_and_download_video(excel_file) # Outputs a csv containing all the different signs and how many times they are signed in videos (in the dataset). Downloads and sanitisies video names and stores them in a video folder.
    download_required(excel_file, signs)
        
main()
