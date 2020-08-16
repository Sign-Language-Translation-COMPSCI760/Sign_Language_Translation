import pandas as pd
import math

def main():
    
    file = pd.read_excel("dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx", sheet_name="Sheet1")
    words = {}      #key = words, value = number of occurences
    firstline = True
    for i in file.index:
        
        if firstline:    #skip first line
            firstline = False
            continue
    
        if file['Main New Gloss'][i] == " " or str(file['Main New Gloss'][i]) == "nan":
            words[current] += 1 
        else:
            current = file['Main New Gloss'][i]
            words[current] = 0

    with open("test examples.txt", mode='w') as fout:
        for key, value in words.items():
            if value >= 10:
                fout.write(key+"\n")
            
        
main()
