import pandas as pd
import csv
i_doc_path = "./inference_testing/answer_time_drop_202401-0118-0219-089635_0101_sliding.txt"
answer_df = pd.read_csv(i_doc_path, names =["file","class", "start","end","value", "regulation"], dtype = str, sep="\t", on_bad_lines='skip', quoting=csv.QUOTE_NONE)

print("ori len = {}" .format(len(answer_df)))
drop_row_list = []
for row in range(len(answer_df)):
    class_label = answer_df.at[row, 'class']
    if class_label in ["IDNUM", "DATE","TIME"]:
        value = answer_df.at[row, 'value']
        if len(value)<5:
            drop_row_list.append(row)
answer_df = answer_df.drop(drop_row_list)
print("drop len = {}" .format(len(answer_df)))
from datetime import datetime
timestr = datetime.now().strftime('%Y%m-%d%H-%M')
answer_df.to_csv("./inference_testing/answer_regulation_{}_0101_sliding.txt".format(timestr), sep = '\t', header=False ,index = None)