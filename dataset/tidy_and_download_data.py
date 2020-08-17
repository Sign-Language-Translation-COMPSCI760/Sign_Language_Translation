from openpyxl import load_workbook
import urllib.request
import pandas as pd
import math
import csv

def find_class_occurance_and_download_video(excel_file):

    words = {}      #key = words, value = number of occurences
    firstline = True
    for i in excel_file.index:
        
        if firstline:    #skip first line
            firstline = False
            continue
    
        if excel_file['Main New Gloss'][i] == " " or str(excel_file['Main New Gloss'][i]) == "nan":
            if excel_file['Consultant'][i] != "------------":
                video_link = excel_file['Links seperate 1'][i][12:-9]
                urllib.request.urlretrieve(video_link, current + "__" + str(words[current] + 1) + ".mov") 
                words[current] += 1 
        elif excel_file['Consultant'][i] != "------------":
            current = excel_file['Main New Gloss'][i]
            words[current] = 0

    with open("sign_class_and_occurance.csv", mode='w') as fout:
        for key, value in words.items():
            fout.write("%s, %s\n" % (key, value))

def main():
    
    excel_file = pd.read_excel("dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx", sheet_name="Sheet1")
    #print(excel_file)

    find_class_occurance_and_download_video(excel_file) # Outputs a csv containing all the different signs and how many times they are signed in videos (in the dataset). Downloads and sanitisies video names and stores them in a video folder.
        
main()
