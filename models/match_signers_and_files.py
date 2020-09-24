import urllib.request
import pandas as pd
import random
import os
import shutil
import cs760

srcdir = '/media/tim/dl3storage/Datasets/asllrp_features_final2/train'
valdir = '/media/tim/dl3storage/Datasets/asllrp_features_final2/val'
rejdir = '/media/tim/dl3storage/Datasets/asllrp_features_final2/rejects'



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

    with open("sign_class_and_occurance_test.csv", mode='w') as fout:
        for key, value in words.items():
            if value >= 10:
                fout.write("%s, %s\n" % (key, value))
                signs.append(key)
    
    return signs

def calc_filename_and_signer(excel_file, signs):
    
    outputmatches = []
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
                    #urllib.request.urlretrieve(video_link, viddir + sign + "__" + str(words[current] + 1) + ".mov") 
                    outputmatches.append({'original_link': video_link, 'video_name': sign + "__" + str(words[current] + 1) + ".mov", 'signer': str(excel_file['Consultant'][i])})
                    words[current] += 1 
        elif excel_file['Consultant'][i] != "------------":
                current = excel_file['Main New Gloss'][i]
                words[current] = 0
    return outputmatches

def get_sign(video_name):
    idx = video_name.find('__')
    sign = video_name[:idx]
    return sign


def main():
    
    excel_file = pd.read_excel("../dataset/dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx", sheet_name="Sheet1")

    signs = find_class_occurance_and_download_video(excel_file) # Outputs a csv containing all the different signs and how many times they are signed in videos (in the dataset). Downloads and sanitisies video names and stores them in a video folder.
    outputmatches = calc_filename_and_signer(excel_file, signs)
    
    df = pd.DataFrame(outputmatches)
    
    signers = df['signer'].unique().tolist()
    df.shape
    df.to_csv('signers_vids.csv', index=False)
    
    gb = df.groupby('signer')['signer'].count()
    gb
    
    df['sign'] = df.apply(lambda row: get_sign(row['video_name']), axis=1)
    
    signs = df.groupby('sign')['sign'].count()
    signs
    signslist = signs.keys().tolist()
    
    signers_signs = df.groupby(['signer', 'sign'])['sign'].count()
    signers_signs
    
    sign_counts = signers_signs.to_dict()

    sign_counts_tuple = {}
    signs_set = set()
    for k in sign_counts:
        newk = (k[0], k[1], sign_counts[k])
        sign_counts_tuple[newk] = sign_counts[k]  # ('Tyler', 'TOUGH', 2)
        signs_set.add(k[1])
    print(len(signs_set))    

    random.seed(42)
    selected_signers = {}
    for sign in signs_set:
        random.shuffle(signers)
        got_signer = False
        for signer in signers:
            checkkey = (signer, sign, 1)
            if sign_counts_tuple.get(checkkey):
                got_signer = True
                selected_signers[sign] = signer
                print(checkkey)
                break
        if not got_signer:
            for signer in signers:
                checkkey = (signer, sign, 2)
                if sign_counts_tuple.get(checkkey):
                    got_signer = True
                    selected_signers[sign] = signer
                    print(checkkey)
                    break
        if not got_signer:
            for signer in signers:
                checkkey = (signer, sign, 3)
                if sign_counts_tuple.get(checkkey):
                    got_signer = True
                    selected_signers[sign] = signer
                    print(checkkey)
                    break
        if not got_signer:
            for signer in signers:
                checkkey = (signer, sign, 4)
                if sign_counts_tuple.get(checkkey):
                    got_signer = True
                    selected_signers[sign] = signer
                    print(checkkey)
                    break
        if not got_signer:
            print('No Signer selected for ', sign)
            
    
    val_list = []
    reject_list = []
    for sign in selected_signers:
        signer = selected_signers[sign]
        tmpdf = df[(df['sign'] == sign) & (df['signer'] == signer)]
        first = True
        for i, row in enumerate(tmpdf.iterrows()):
            name = row[1]['video_name']
            if first:
                first = False
                val_list.append(name)
            else:
                reject_list.append(name)
            
    print(val_list) #['CUTE__5.mov', 'DISAPPOINT__9.mov', 'FACE__3.mov', 'INCLUDE.INVOLVE__11.mov', 'COPY__7.mov', 'GUITAR__4.mov', 'APPOINTMENT__8.mov', 'ADVISE.INFLUENCE__9.mov', 'GROUND__3.mov', 'MACHINE__7.mov', 'WALK__8.mov', 'LOOK__14.mov', 'WEEKEND__9.mov', 'CITY.COMMUNITY__5.mov', 'EXCUSE__16.mov', 'COME__2.mov', 'DISCUSS__8.mov', 'GO__9.mov', 'GOVERNMENT__12.mov', 'ART.DESIGN__6.mov', 'DEVIL.MISCHIEVOUS__2.mov', 'ISLAND.INTEREST__10.mov', 'BOWTIE__4.mov', 'SILLY__14.mov', 'SHELF.FLOOR__5.mov', 'VACATION__4.mov', 'DEVELOP__4.mov', 'MAD__8.mov', 'CANCEL.CRITICIZE__7.mov', 'FIRE.BURN__2.mov', 'DATE.DESSERT__4.mov', 'EMPHASIZE__10.mov', 'COP__11.mov', 'GOLD.ns-CALIFORNIA__7.mov', 'SAME__6.mov', 'HAPPY__6.mov', 'AFRAID__11.mov', 'INFORM__8.mov', 'LIVE__7.mov', 'SHOW__4.mov', 'PAST__12.mov', 'COLLECT__3.mov', 'DRESS.CLOTHES__18.mov', 'REPLACE__9.mov', 'RUN__2.mov', 'FIFTH__6.mov', 'EXPERT__4.mov', 'INJECT__1.mov', 'FED-UP.FULL__4.mov', 'FINGERSPELL__9.mov', 'NICE.CLEAN__11.mov', 'BOSS__9.mov', 'ANSWER__11.mov', 'BIG__3.mov', 'STAND-UP__4.mov', 'TOUGH__4.mov', 'WORK-OUT__11.mov', 'CHAT__10.mov', 'DRIP__2.mov', 'AGAIN__11.mov', 'EAT__2.mov', 'MARRY__9.mov', 'BLAME__3.mov', 'DECREASE__5.mov', 'CAMP__10.mov', 'IN__9.mov', 'GET-TICKET__4.mov', 'DEPRESS__6.mov', 'DOCTOR__7.mov', 'DRINK__6.mov']
    print(reject_list) #['FACE__9.mov', 'GUITAR__5.mov', 'GUITAR__6.mov', 'GROUND__4.mov', 'WEEKEND__10.mov', 'COME__6.mov', 'ART.DESIGN__7.mov', 'ART.DESIGN__10.mov', 'ART.DESIGN__16.mov', 'SHELF.FLOOR__6.mov', 'DEVELOP__5.mov', 'CANCEL.CRITICIZE__8.mov', 'DATE.DESSERT__5.mov', 'COLLECT__4.mov', 'EXPERT__5.mov', 'INJECT__2.mov', 'FED-UP.FULL__5.mov', 'TOUGH__5.mov', 'DRIP__6.mov', 'EAT__11.mov', 'BLAME__4.mov']       

    for file in val_list:
        base = os.path.splitext(file)[0] + '__*'
        move_files = cs760.list_files_pattern(srcdir, base)
        if not move_files:
            print(f'NO files for {base}')
        for currfile in move_files:
            print(f"Moving {os.path.join(srcdir, currfile)} to {valdir}")
            shutil.move(os.path.join(srcdir, currfile), valdir)

    for file in reject_list:
        base = os.path.splitext(file)[0] + '__*'
        move_files = cs760.list_files_pattern(srcdir, base)
        if not move_files:
            print(f'NO files for {base}')
        for currfile in move_files:
            print(f"Moving {os.path.join(srcdir, currfile)} to {rejdir}")
            shutil.move(os.path.join(srcdir, currfile), rejdir)
    
        
main()
