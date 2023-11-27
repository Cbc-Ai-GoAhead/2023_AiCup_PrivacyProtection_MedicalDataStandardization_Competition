import pandas as pd
i_doc_path = "./output/models/submission.txt"
answer_df = pd.read_csv(i_doc_path, names =["file","class", "start","end","value", "regulartime"], sep="\t")
print(answer_df.head())
ori_class_list = ["PATIENT", "DOCTOR", "USERNAME", "PROFESSION","ROOM", "DEPARTMENT", "HOSPITAL"\
,"ORGANIZATION","STREET","CITY","STATE","COUNTRY","ZIP", "LOCATION-OTHER", "AGE",\
 "DATE", "TIME", "DURATION", "SET", "PHONE", "FAX", "EMAIL", "URL","IPADDR",\
 "SSN", "MEDICALRECORD","HEALTHPLAN", "ACCOUNT","LICENSE", "VECHICLE","DEVICE",\
 "BIOID","IDNUM","OTHER"]

class_list = answer_df["class"].tolist()
print(set(class_list))
print("num of class = {}" .format(len(set(class_list))))

maintain_answer_df = answer_df[answer_df['class'].isin(ori_class_list)].reset_index(drop = True)

####
##  Time Regulariation
####

from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
# new_df.loc[new_df["class"].isin(["TIME"])]
# 取文字的位置
DATE_list = maintain_answer_df[maintain_answer_df["class"].isin(["DATE"])]["value"].tolist()
TIME_list = maintain_answer_df[maintain_answer_df["class"].isin(["TIME"])]["value"].tolist()
DURATION_list = maintain_answer_df[maintain_answer_df["class"].isin(["DURATION"])]["value"].tolist()
SET_list = maintain_answer_df[maintain_answer_df["class"].isin(["SET"])]["value"].tolist()

#取正規化的結果
regulartime_DATE_list = maintain_answer_df[maintain_answer_df["class"].isin(["SET"])]["regulartime"].tolist()
regulartime_TIME_list = maintain_answer_df[maintain_answer_df["class"].isin(["TIME"])]["regulartime"].tolist()
regulartime_DURATION_list = maintain_answer_df[maintain_answer_df["class"].isin(["DURATION"])]["regulartime"].tolist()
regulartime_SET_list = maintain_answer_df[maintain_answer_df["class"].isin(["SET"])]["regulartime"].tolist()
print(DATE_list)
print(TIME_list)
print(DURATION_list)
print(SET_list)#valid的資料裡面沒有SET
print("DATE_list len = {},TIME_list={}, DURATION_list={}, SET_list={} ".format(len(DATE_list), len(TIME_list),len(DURATION_list), len(SET_list)))
#DATE_list len = 1309,TIME_list=2, DURATION_list=1, SET_list=0

print(regulartime_DATE_list)
print(regulartime_TIME_list)
print(regulartime_DURATION_list)
print(regulartime_SET_list)
####
##  Time Regulariation Group
####
#Date
## '3/12/2062', '29.4.64','20010303'
#Time
#['2019', '27/5/64'] ->['2019-11-25T00:00', '2064-05-27T00:00']
#Duration
#['today']

#
iso_86_date_list = []
short_iso_86_date_list = []
iso_86_time_list = []
iso_86_duration_list = []
#
iso_86_regulartime_date_list = []
iso_86_regulartime_time_list = []
iso_86_regulartime_duration_list = []

for row in range(len(maintain_answer_df)):
    value_of_class = maintain_answer_df.at[row, 'class']
    if (value_of_class == 'DATE') :
        try:
            date_value = maintain_answer_df.at[row, 'value']
            if (date_value =='Now') or (date_value =="today"): #還有 .  /
                # print("date_value = {}".format(date_value))
                # print("row = {}, row-1={}".format(row, row-1))
                # print("len of iso_86_date_list = {}".format(len(iso_86_date_list)))
                # row 超出iso_86_date_list 長度
                idx_length = len(iso_86_date_list)
                # print("row = {}, row-1={}".format(iso_86_date_list[row], iso_86_date_list[row-1]))
                
                #要在印出大語言模型原本預測的結果
                # maintain_answer_df.at[row, 'regulartime'] =
                iso_86_date_list.append(iso_86_date_list[idx_length-1])
                iso_86_regulartime_date_list.append(maintain_answer_df.at[row, 'regulartime'])


                #應該保存上一次Date的值
                #maintain_answer_df.at[row, 'value'] = iso_86_date_list[idx_length-1]
                maintain_answer_df.at[row, 'regulartime'] = iso_86_date_list[idx_length-1]

                continue
                # iso_86_date_list[row] = iso_86_date_list[row-1]
            if len(date_value)<5:
                short_iso_86_date_list.append(date_value)
                now = parse(date_value)
                today = now.date()
                iso_86_date = today.strftime('%Y')
                #date_value =20., iso_86_date=2023 => 模型沒有取到正確位置
                print("date_value ={}, iso_86_date={}".format(date_value, iso_86_date))
                
                iso_86_date_list.append(iso_86_date)
                iso_86_regulartime_date_list.append(maintain_answer_df.at[row, 'regulartime'])
                # maintain_answer_df.at[row, 'value'] = iso_86_date
                maintain_answer_df.at[row, 'regulartime'] = iso_86_date
                continue
            now = parse(date_value)
            today = now.date()
            # year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
            # rdelta = relativedelta(easter(year), today)
            iso_86_date = today.strftime('%Y-%m-%d')
            # print(today)
            # print(t2)
            
            iso_86_date_list.append(iso_86_date)
            iso_86_regulartime_date_list.append(maintain_answer_df.at[row, 'regulartime'])

            # maintain_answer_df.at[row, 'value'] = iso_86_date
            maintain_answer_df.at[row, 'regulartime'] = iso_86_date
            # break
        except:
            iso_86_date_list.append(date_value)
            iso_86_regulartime_date_list.append(maintain_answer_df.at[row, 'regulartime'])
            # 維持一樣
            # maintain_answer_df.at[row, 'regulartime'] = iso_86_date
    elif value_of_class == 'TIME' :
        time_value=""
        try:
            time_value = maintain_answer_df.at[row, 'value']
            time_value_on = time_value.replace("the", "on")
            now = parse(time_value_on)
            # today = now.date()
            # year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
            # rdelta = relativedelta(easter(year), today)
            iso_86_time = now.strftime('%Y-%m-%dT%H:%M')
            # print(today)
            # print(t2)
            
            iso_86_time_list.append(iso_86_time)
            iso_86_regulartime_time_list.append(maintain_answer_df.at[row, 'regulartime'])

            # maintain_answer_df.at[row, 'value'] = iso_86_time
            maintain_answer_df.at[row, 'regulartime'] = iso_86_time
            # break
        except:
            # answer_df.at[row, 'value'] = iso_86_time
            iso_86_time_list.append(time_value)
            print(time_value)
            iso_86_regulartime_time_list.append(maintain_answer_df.at[row, 'regulartime'])
    elif value_of_class == 'DURATION' :
        time_value=""
        try:
            duration_value = maintain_answer_df.at[row, 'value']
            print("Duration ={}" .format(duration_value))
            now = ""
            if duration_value == "today":
                print("Duration today")
                now ="P1D"
            iso_86_duration = now
            # now = parse(time_value_on)
            # today = now.date()
            # year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
            # rdelta = relativedelta(easter(year), today)
            # iso_86_time = now.strftime('%Y-%m-%dT%H:%M')
            # print(today)
            # print(t2)
            
            iso_86_time_list.append(iso_86_time)
            iso_86_regulartime_duration_list.append(maintain_answer_df.at[row, 'regulartime'])
            # maintain_answer_df.at[row, 'value'] = iso_86_duration
            maintain_answer_df.at[row, 'regulartime'] = iso_86_duration
            # break
        except:
            # answer_df.at[row, 'value'] = iso_86_time
            iso_86_duration_list.append(duration_value)
            iso_86_regulartime_duration_list.append(maintain_answer_df.at[row, 'regulartime'])
            print(duration_value)


# print(iso_86_date_list)
# print(iso_86_time_list)
# print(iso_86_duration_list)

print(short_iso_86_date_list)
if "today" in iso_86_date_list  or "Now" in iso_86_date_list:
    print(iso_86_date_list.index("today"))
    print(iso_86_date_list.index("Now"))

# print(maintain_answer_df.head())
print("---Date")
print("#regular Date")
print(iso_86_regulartime_date_list)
print("#Date")
print(iso_86_date_list)
print("---Time")
print("#regular Time")
print(iso_86_regulartime_time_list)
print("#Time")
print(iso_86_time_list)
print("---Duration")
print("#regular Duration")
print(iso_86_regulartime_duration_list)
print("#Duration")
print(iso_86_duration_list)

new_df = pd.DataFrame(columns = ['ori_Date','regular_Date','ruled_Date'])
new_df['ori_Date'] = DATE_list
new_df['regular_Date'] = iso_86_regulartime_date_list
new_df['ruled_Date'] = iso_86_date_list
new_df.to_csv("Date.tsv", sep = '\t', header=True ,index = None)
####
##  Save File
####
# enroll_Df =  sre18eval_Df[sre18eval_Df['partition'].isin(['enrollment'])].reset_index(drop = True)
o_path = "./output/models/answer_regular.txt"
print("ori answer len = {} , maintain_answer_df len ={}".format(len(answer_df),len(maintain_answer_df)))
maintain_answer_df.to_csv(o_path, sep = '\t', header=False ,index = None)