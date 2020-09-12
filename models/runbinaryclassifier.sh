# run binary classifier
# "sign_classes" : ["WORK-OUT", "INFORM", "LOOK", "SHOW", "BOSS", "NICE.CLEAN", "DRESS.CLOTHES", "COLLECT", "FINGERSPELL", "FED-UP.FULL", "VACATION", "DECREASE", "GUITAR", "SHELF.FLOOR", "MACHINE", "DEVELOP", "SAME", "DRIP", "DEVIL.MISCHIEVOUS", "CUTE", "COME", "PAST", "CITY.COMMUNITY", "ADVISE.INFLUENCE", "FIRE.BURN", "ANSWER", "DISCUSS", "GET-TICKET", "AGAIN", "CHAT", "MAD", "BOWTIE", "EXCUSE", "FACE", "EMPHASIZE", "CAMP", "COPY", "BLAME", "WEEKEND", "BIG", "EXPERT", "DOCTOR", "DATE.DESSERT", "INJECT", "FIFTH", "RUN", "APPOINTMENT", "HAPPY", "MARRY", "DEPRESS", "SILLY", "GROUND", "GOLD.ns-CALIFORNIA", "DISAPPOINT", "REPLACE", "STAND-UP", "ART.DESIGN", "CANCEL.CRITICIZE", "AFRAID", "GO", "EAT", "WALK", "ISLAND.INTEREST", "IN", "GOVERNMENT", "COP", "DRINK", "INCLUDE.INVOLVE", "LIVE", "TOUGH"]


#"nzsl_signs" : ["FACE", "MACHINE", "CUTE", "FIRE.BURN", "FED-UP.FULL", "INCLUDE.INVOLVE", "VACATION", "BIG", "GROUND", "DRESS.CLOTHES", "LOOK", "GUITAR", "DISAPPOINT", "STAND-UP", "INJECT", "DRINK", "CANCEL.CRITICIZE", "EAT", "CAMP", "INFORM", "PAST", "NICE.CLEAN", "SHELF.FLOOR"]


python stage2model.py config_dirs.json config760_binaryclassifier.json FACE
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json MACHINE
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json CUTE
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json FIRE.BURN
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json FED-UP.FULL
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json INCLUDE.INVOLVE
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json VACATION
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json BIG
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json GROUND
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json DRESS.CLOTHES
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json LOOK
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json GUITAR
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json DISAPPOINT
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json STAND-UP
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json INJECT
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json DRINK
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json CANCEL.CRITICIZE
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json EAT
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json CAMP
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json INFORM
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json PAST
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json NICE.CLEAN
sleep 5
python stage2model.py config_dirs.json config760_binaryclassifier.json SHELF.FLOOR

