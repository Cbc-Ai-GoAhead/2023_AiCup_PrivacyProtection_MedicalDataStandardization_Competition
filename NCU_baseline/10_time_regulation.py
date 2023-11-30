import pandas as pd
i_doc_path = "./submission/answer.txt"
answer_df = pd.read_csv(i_doc_path, names =["file","class", "start","end","value"], sep="\t")
print(answer_df.head())
some_values = ["DATE", "TIME", "DURATION", "SET"]
# 取出
new_df = answer_df.loc[answer_df["class"].isin(some_values)]
print(new_df)
print(len(new_df))
DATE_df = new_df.loc[new_df["class"].isin(["DATE"])]
print(DATE_df)
Date_value_list = DATE_df["value"].tolist()
import re
# from datetime import date
# date(2021, 10, 1).isoformat()
#2021-10-1
# time_re = re.compile(r'^(([01]\d|2[0-3]):([0-5]\d)|24:00)$')
# for value in Date_value_list:
####
##  Date
####
#https://dateutil.readthedocs.io/en/stable/index.html?fbclid=IwAR0DNqXquRn_6eGepjpHfuVkju7qeZdYgvFyG8Awo78KvLrFxfYZg5sHe08
tme_str= "24/4/1987"
val = re.findall(r'(\d{2}\/\d{1}\/\d{4})', tme_str)
print(val)

print(re.findall(r'\s(\d{2}\:\d{2}\s?(?:AM|PM|am|pm))', 'Time today is 10:30 PM'))

# print("Date_value_list set = {}" .format(set(Date_value_list)))
from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
####
##  DATE
####
now = parse("24/4/1987")
today = now.date()
year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
rdelta = relativedelta(easter(year), today)
print(today)
print("Today is: %s" % today)
iso_86_date_list = []
print(Date_value_list)
for date_value in Date_value_list:
    try:
        now = parse(date_value)
        today = now.date()
        # year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
        # rdelta = relativedelta(easter(year), today)
        iso_86_date = today.strftime('%Y-%m-%d')
        # print(today)
        # print(t2)
        iso_86_date_list.append(iso_86_date)
        # break
    except:
        iso_86_date_list.append(date_value)
    # print("Today is: %s" % today)
print(iso_86_date_list)
print("---------------")
for idx, date_value in enumerate(iso_86_date_list):
    if (date_value =='Now') or (date_value =="today"): #還有 .  /
        iso_86_date_list[idx] = iso_86_date_list[idx-1]
print("replace Now today")
print(iso_86_date_list)
# 後續合併成list 要drop 空白的列嗎
# if sta=="NOW"
# 斷掉空白 str.trim()

## 最後提交的answer有需要排序嗎？ 如果需要排序就需要兩個陣列執行
####
##  Time
####
# '10:20am on the 28.08.2018', '19/09/2012 at 13:56', 'at 6:15pm on 15.11.2016'
TIME_df = new_df.loc[new_df["class"].isin(["TIME"])]
print(TIME_df)
Time_value_list = TIME_df["value"].tolist()
# print("Time_value_list set = {}" .format(set(Time_value_list)))

now = parse("19/09/2012 at 13:56")
print(now)#2012-09-19 13:56:00

print(now.strftime('%Y-%m-%dT%H:%M'))
# iso_time = isoparse(now)
# print(iso_time)
today = now.time()
print(today)
iso_86_time_list = []
for time_value in Time_value_list:
    try:
        time_value
        now = parse(time_value)
        # today = now.date()
        # year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
        # rdelta = relativedelta(easter(year), today)
        iso_86_time = now.strftime('%Y-%m-%dT%H:%M')
        # print(today)
        # print(t2)
        iso_86_time_list.append(iso_86_time)
        # break
    except:
        time_value_on = time_value.replace("the", "on")
        iso_86_time_list.append(time_value_on)
    # print("Today is: %s" % today)
print(iso_86_time_list)

#####
# 9.50 on 04.09.13
# 11:27am on on 14.1.2014'
# on on 15.1.2014 at 11:12am
now = parse("at 6:15pm on 15.11.2016")
print(now)#2016-11-15 18:15:00

# now = parse("10:20am on the 28.08.2018") # 要把the 去掉
# print(now)#2018-08-28 10:20:00
# 會有例外字賺要處理 '26hr' =>就要查看原本文檔list的時間
# 'in 1' =>bert 預測錯誤的字元

####
##  Duration
####

# Duration_df = new_df.loc[new_df["class"].isin(["DURATION"])]
# print(DATE_df)
# Duration_value_list = DATE_df["value"].tolist()
# print("Duration_value_list set = {}" .format(set(Duration_value_list)))
# print(Duration_value_list)
# DURATION 資料比數太少 訓練不出來 找不到位置


####
##  Set 只有在第二個資料集有 13筆數 所以模型訓練不出來
####
# SET_df = new_df.loc[new_df["class"].isin(["SET"])]
# print(SET_df)
# Set_value_list = SET_df["value"].tolist()
# print("Duration_value_list set = {}" .format(set(Set_value_list)))