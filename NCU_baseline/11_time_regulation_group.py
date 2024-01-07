from dateutil.relativedelta import *
from dateutil.easter import *
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
import pandas as pd
import re
import csv
# i_doc_path = "./submission/answer_drop_n.txt"
# i_doc_path = "./11_26_submission_9_12/answer.txt"#delet 5939 MR
# i_doc_path = "./11_26_submission_9_12/answer_time_drop_202311-2710-2634-629846.txt"
# i_doc_path = "./submission/answer.txt"
#i_doc_path = "./11_26_submission_9_12/submission_8_12/answer_8.txt"
#i_doc_path = "./inference_testing/testingset_answer_dataset_3_12_0.54.txt"
i_doc_path = "./inference_testing/longformer.txt"
# i_doc_path = "./inference_testing/answer.txt"
# answer_df = pd.read_csv(i_doc_path, names =["file","class", "start","end","value"], dtype = str, sep="\t", quoting=csv.QUOTE_NONE)
# answer_df = pd.read_csv(i_doc_path, names =["file","class", "start","end","value"], dtype = str, sep="\t", on_bad_lines=False)# <1.4.0
answer_df = pd.read_csv(i_doc_path, names =["file","class", "start","end","value"], dtype = str, sep="\t", on_bad_lines='skip', quoting=csv.QUOTE_NONE)
# for chunk in pd.read_csv(i_doc_path, names =["file","class", "start","end","value"], dtype = str, sep="\t", chunksize=20, quoting=csv.QUOTE_NONE):
#     print(chunk)
#error: Expected 5 fields in line 12549, saw 6
#C error: EOF inside string starting at row 11959

print(answer_df.head())
#因為要新增第六個欄位 沒有的就給Nan

# drop特殊字元
# 使用「*str.strip()*」可以去除字串開頭或結尾的某些字元
# 直接給list , dataframe就可以drop
print("ori class len = {}" .format(len(answer_df)))
answer_df = answer_df.dropna().reset_index(drop=True)
print("dropna class len = {}" .format(len(answer_df)))
need_drop_row = []
for row in range(len(answer_df)):
    value_of_class = answer_df.at[row, 'value']
    # print("row ={}, class = {}, value_of_class={}".format(row,answer_df.at[row, 'class'], value_of_class))
    # if value_of_class =="Nan" or value_of_class =="nan":
    #     need_drop_row.append(row)
    #     continue
    if len(value_of_class) < 2:
        need_drop_row.append(row)
print(need_drop_row)
answer_df = answer_df.drop(need_drop_row).reset_index(drop=True)
print("after drop len = {}" .format(len(answer_df)))
print(answer_df[:50])
#answer_df.to_csv("./drop_answer.txt", sep = '\t', header=False ,index = None)


ori_date_list = []
ori_time_list = []

iso_86_date_list = []
iso_86_time_list = []
iso_86_duration_list = []
iso_86_set_list = []

##
time_drop_row = []
def hasNumber(stringVal):
    re_numbers = re.compile("\d")
    return False if (re_numbers.search(stringVal) == None) else True
for row in range(len(answer_df)):
    value_of_class = answer_df.at[row, 'class']
    if (value_of_class == 'DATE') :
        try:
            date_value = answer_df.at[row, 'value']
            ori_date_list.append(date_value)
            if (date_value =='now') or (date_value =="today"): #還有 .  /
                len_iso_86_date= len(iso_86_date_list)
                iso_86_date_list.append(iso_86_date_list[len_iso_86_date-1])
                # 取原本的date 或是取前一個  目前取當前的date
                answer_df.at[row, 'regulation_time'] = iso_86_date_list[len_iso_86_date-1]
                continue
            now = parse(date_value)
            today = now.date()
            # year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
            # rdelta = relativedelta(easter(year), today)
            iso_86_date = today.strftime('%Y-%m-%d')
            # print(today)
            # print(t2)
            answer_df.at[row, 'regulation_time'] = iso_86_date
            iso_86_date_list.append(iso_86_date)
            # break
        except:
            iso_86_date_list.append(date_value)
    elif value_of_class == 'TIME' :
        time_value=""
        try:
            time_value = answer_df.at[row, 'value']
            ori_time_list.append(time_value)
            time_value_on = time_value.replace("the", "on")
            # if time_value == "10:40 on the 10th of January 2014":
            #     print("83------time_value!!!!!!!10:40 on the 10th of January")
            # print("----72")
            #第一個部分 去掉文字
            if hasNumber(time_value_on):
                None
            else:
                time_drop_row.append(row)
                continue
            now = parse(time_value_on)
            # print("83------time_value!!!!!!!10:40 on the 10th of January now={}".format(now))
            # print("----74")
            #26/08/2018 at 17:10hr
            # today = now.date()
            # year = rrule(YEARLY,dtstart=now,bymonth=8,bymonthday=13,byweekday=FR)[0].year
            # rdelta = relativedelta(easter(year), today)
            iso_86_time = now.strftime('%Y-%m-%dT%H:%M')
            #1720hr on 8/1/13
            #10.20 on 16.07.13

            #0926hr
            # 1300 onthe 20th of January 2014
            # print(today)
            # print(t2)
            answer_df.at[row, 'regulation_time'] = iso_86_time
            iso_86_time_list.append(iso_86_time)
            # break
        except:
            # answer_df.at[row, 'value'] = iso_86_time
            

            # 要分為兩個部分
            """
            15.05 on 27/8/13
            1720hr on 8/1/13
            16:10pm on 21/4/15.
             3:20pm on 6/05/
             30am on 18/3/14
             15:50pm
             42 on 11/2/13.
             6/8/16 at 17:15pm)
             1300 onthe 20th of January 2014
             1010am on 26.12.18
             value = 10:00hr
             value = .00
            """
            """
            split_value = ['14.40', 'on', '12/.03.14.']
            Exception time value = 14.40 on 12/.03.14.
            
            split_value = ['on', '14/1/13', 'at', '1515hr']
            Exception time value = on 14/1/13 at 1515hr
            
            split_value = ['25/09/2015', 'at', '17:10hr']
            Exception time value = 25/09/2015 at 17:10hr

            """
            """
            split_value = ['145pm', 'on', '26.09.12']
            on day = 2012-09-26 00:00:00
            ['145pm']
            Exception time value = 145pm on 26.09.12
            split_value = ['0926hr']

            split_value = ['1320Hrs', 'on', '12.11.19']
            on day = 2019-12-11 00:00:00
            ['1320Hrs']
            Exception time value = 1320Hrs on 12.11.19


            split_value = ['14.40', 'on', '12/.03.14.']
            Exception time value = 14.40 on 12/.03.14.

            split_value = ['on', '14/1/13', 'at', '1515hr']
            Exception time value = on 14/1/13 at 1515hr
            
            split_value = ['0933', 'on', 'the', '24/1/14']
            Exception time value = 0933 on the 24/1/14
            
            split_value = ['1720hr', 'on', '8/1/13']
            on day = 2013-08-01 00:00:00
            ['1720hr']
            Exception time value = 1720hr on 8/1/13
            ----72
            split_value = ['1730hr', 'on', '8/1/13']
            on day = 2013-08-01 00:00:00
            ['1730hr']
            Exception time value = 1730hr on 8/1/13
            
            split_value = ['3:20pm', 'on', '6/05/']
            Exception time value = 3:20pm on 6/05/

            split_value = ['15:50pm']

            split_value = ['30am', 'on', '18/3/14']
            on day = 2014-03-18 00:00:00
            ['30am']
            Exception time value = 30am on 18/3/14

            split_value = [':42', 'on', '11/2/13.']
            on day = 2013-11-02 00:00:00
            ['', '42']
            Exception time value = :42 on 11/2/13.
            """
            try:
                if time_value == "10:40 on the 10th of January 2014":
                    print("time_value!!!!!!!10:40 on the 10th of January")
                split_value = time_value.split(" ")
                print("split_value = {}".format(split_value))
                if len(split_value)>2:
                    if split_value[1]=='on':
                        
                        day = parse(split_value[2])
                        print("on day = {}" .format(day))
                        tmp_time = split_value[0]
                        tmp_time = tmp_time.replace(".", ":")
                        
                        tmp_time = tmp_time.split(":")
                        print(tmp_time)
                        
                        #['145pm']
                        if len(tmp_time)>1:
                            # print("in len>1 tmp_time {}".format(tmp_time))
                            delta_time = relativedelta(hours=int(tmp_time[0]), minutes=int(tmp_time[1][:2]))
                        else:
                            # print("in len tmp_time {} type = {}".format(tmp_time, type(tmp_time)))
                            # for c in tmp_time[0]:
                            #     print("c = {}, type = {}".format(c, type(c)))
                            #     if c.isdigit():
                            #         print("YES")
                            # tmp_str_time = tmp_time[0]
                            # print("".join([c for c in tmp_time[0] if c.isdigit()]))
                            numbers = "".join([c for c in tmp_time[0] if c.isdigit()])
                            
                            # print("numbers = {}".format(numbers))
                            delta_time = relativedelta(hours=int(numbers[:2]), minutes=int(numbers[2:]))
                        
                            
                        print(delta_time)
                        # relativedelta(hours=17)
                        final_time = day+delta_time
                        
                        final_time = final_time.strftime('%Y-%m-%dT%H:%M')
                        print(final_time)
                        answer_df.at[row, 'regulation_time'] = final_time
                        iso_86_time_list.append(final_time)
                    else:#"at"
                        day = parse(split_value[0])
                        print("at day = {}" .format(day))
                        tmp_time = split_value[2]
                        tmp_time = tmp_time.replace(".", ":")
                        tmp_time = tmp_time.split(":")
                        print(tmp_time)
                        delta_time = relativedelta(hours=int(tmp_time[0]), minutes=int(tmp_time[1][:2]))
                        # print(delta_time)
                        # relativedelta(hours=17)
                        final_time = day+delta_time
                        # print(final_time)
                        final_time = final_time.strftime('%Y-%m-%dT%H:%M')
                        answer_df.at[row, 'regulation_time'] = final_time
                        iso_86_time_list.append(final_time)
                elif len(split_value)==2:
                    tmp_time = split_value
                    tmp_time = tmp_time.replace(".", ":")
                    # 取當前Date
                    len_iso_86_date= len(iso_86_date_list)
                    day = iso_86_date_list[len_iso_86_date-1]
                    day = parse(day)
                    tmp_time = tmp_time.split(":")
                    delta_time = relativedelta(hours=int(tmp_time[0]), minutes=int(tmp_time[1][:2]))

                    final_time = day+delta_time
                    final_time = final_time.strftime('%Y-%m-%dT%H:%M')
                    answer_df.at[row, 'regulation_time'] = final_time
                    iso_86_time_list.append(final_time)

                else:
                    #["0906hr"]
                    # split_value = ['.00']
                    # else number = 00
                    # split_value = ['10:00hr']
                    # else number = 1000
                    
                    numbers = "".join([c for c in split_value[0] if c.isdigit()])

                    
                    print("else number = {}".format(numbers))
                    if (len(numbers)>2):#len =3, 4
                        # 取當前Date
                        print("before date number = {},".format(numbers))
                        len_iso_86_date= len(iso_86_date_list)
                        print("len_iso_86_date = {}".format(len_iso_86_date))
                        day = parse(iso_86_date_list[len_iso_86_date-1])
                        print("number = {}".format(numbers))
                        delta_time = relativedelta(hours=int(numbers[:2]), minutes=int(numbers[2:]))

                        print("day = {}, delta_time={}".format(day, delta_time))
                        final_time = day+delta_time
                        print("final_time = {}".format(final_time))
                        final_time = final_time.strftime('%Y-%m-%dT%H:%M')
                        print("number = {}, final_time= {}".format(numbers,final_time))
                        
                        answer_df.at[row, 'regulation_time'] = final_time
                        iso_86_time_list.append(final_time)
                    else: ##00
                        answer_df.at[row, 'regulation_time'] = str(numbers)
                        iso_86_time_list.append(str(numbers))
                    
            except:
                # split_value = ['0933', 'on', 'the', '24/1/14']
                # Exception time value = 0933 on the 24/1/14

                # split_value = ['1300', 'onthe', '20th', 'of', 'January', '2014']
                # at day = 1300-11-27 00:00:00
                # ['20th']
                # Exception time value = 1300 onthe 20th of January 2014
                
                # iso_86_time_list.append(time_value)
                try:
                    split_time_value = time_value.split(" ")
                    if len(time_value)>4:
                        print("303  split_time_value={}".format(split_time_value))
                        tmp_day = " ".join([c for c in split_value[2:6]])
                        print("tmpday = {} type={}".format(tmp_day,type(tmp_day)))
                        print("306 parse= {}".format(parse(tmp_day)))
                        day = parse(tmp_day)
                        print("309 day = {} time {}".format(split_time_value[0][:2], split_time_value[0][2:]))
                        delta_time = relativedelta(hours=int(split_time_value[0][:2]), minutes=int(split_time_value[0][2:]))
                        print("309 delta_time = {}".format(delta_time))
                        final_time = day+delta_time
                        print("final_time ={}".format(final_time))
                        final_time = final_time.strftime('%Y-%m-%dT%H:%M')
                        print("314 final_time shift ={}".format(final_time))
                        answer_df.at[row, 'regulation_time'] = final_time
                        print(answer_df.at[row, 'regulation_time'])
                        print("316 after final_time shift ={}".format(final_time))
                        iso_86_time_list.append(time_value)
                        continue
                    val_t = ""
                    val_day = ""
                    print("299 time_value ={}".format(time_value))
                    for val in time_value:
                        numbers = "".join([c for c in val if c.isdigit()])
                        if len(numbers)<5:
                            val_t = numbers
                        else:
                            val_day = numbers
                    
                    day = parse(val_day)
                    # print("number = {}".format(numbers))
                    delta_time = relativedelta(hours=int(val_t[:2]), minutes=int(val_t[2:]))
    
                    print("day = {}, delta_time={}".format(day, delta_time))
                    final_time = day+delta_time
                    final_time = final_time.strftime('%Y-%m-%dT%H:%M')
                    answer_df.at[row, 'regulation_time'] = final_time
                    iso_86_time_list.append(time_value)
                except Exception as e:
                    print("error = {}".format(e))
                    print("Exception time value = {}".format(time_value))
                    answer_df.at[row, 'regulation_time'] = time_value #直接給原本的值
                    # time_drop_row.append(row)

            # #第二個部分 正規化
            # >>> relativedelta(hours=17)
            # relativedelta(hours=+17)
            # >>> relativedelta(hours=17, minutes=20)
            # relativedelta(hours=+17, minutes=+20)
            # >>> dt = parse("8/1/13")
            # >>> delta = relativedelta(hours=17, minutes=20)
            # >>> dt + delta
            # datetime.datetime(2013, 8, 1, 17, 20)
            # >>> final = dt + delta
            # >>> iso_86_time = final.strftime('%Y-%m-%dT%H:%M')
            # >>> iso_86_time


    elif value_of_class == 'DURATION' :
        time_value=""
        try:
            duration_value = answer_df.at[row, 'value']
            split_value_list=duration_value.split(" ")

            # year years
            if split_value_list[1] in ["year", "years"] :
                regular_duration_value="P"+split_value_list[0]+"Y"
            elif split_value_list[1] in ["month", "months"] :
                regular_duration_value="P"+split_value_list[0]+"M"
            elif split_value_list[1] in ["week", "weeks"] :
                regular_duration_value="P"+split_value_list[0]+"W"
            elif split_value_list[1] in ["day", "days"] :
                regular_duration_value="P"+split_value_list[0]+"D"
            print("EXXXXXX!!")
            # month months
            # week weeks
            # day days

            answer_df.at[row, 'regulation_time'] = regular_duration_value
            # print("EXXXXXX!! 384")
            # iso_86_duration_list.append(iso_86_time)
            # print("EXXXXXX!! 386")
            # break
        except:
            # answer_df.at[row, 'value'] = iso_86_time
            iso_86_duration_list.append(duration_value)
            duration_value = answer_df.at[row, 'value']
            split_value_list=duration_value.split(" ")
            print("duration_value = {}".format(duration_value))
            print("split_value_list = {}".format(split_value_list))
            print("Exception duration value = {}".format(duration_value))
            answer_df.at[row, 'regulation_time'] = duration_value
    elif value_of_class == 'SET' :
        time_value=""
        try:
            set_value = answer_df.at[row, 'value']
            if set_value =="years":
                regular_duration_value = "R2"
            elif set_value =="twice":
                regular_duration_value = "RP1D"
            # twice   R2
            # years   RP1D
           

            # month months
            # week weeks
            # day days

            answer_df.at[row, 'regulation_time'] = regular_duration_value
            iso_86_set_list.append(iso_86_time)
            # break
        except:
            # answer_df.at[row, 'value'] = iso_86_time
            time_drop_row.append(row)
            #iso_86_set_list.append(set_value)
            print("Exception set value = {}".format(set_value))
answer_df = answer_df.drop(time_drop_row).reset_index(drop = True)
# answer_df = answer_df.fillna("")
# import numpy as np
answer_df = answer_df.replace("nan",'')
# answer_df = answer_df[answer_df['class'].isin(ori_class_list)].reset_index(drop = True)
# split_value = ['.00']
# else number = 00
# ----74
    # O: 讀取DataFrame以後 直接在欄位新增值 後續存檔 沒有值的欄位會補上Nan
    # X:讀取DataFrame 新增欄位regulation_time 就會自動補值 Nan 
    # else:
    #     answer_df.at[row, 'regulation_time'] = 'Nan'
# print(answer_df.head())
# print(answer_df[:50])
from datetime import datetime
timestr = datetime.now().strftime('%Y%m-%d%H-%M%S-%f')
answer_df.to_csv("./inference_testing/longformer_time_drop_{}_0101_sliding.txt".format(timestr), sep = '\t', header=False ,index = None)
# date_df = pd.DataFrame(columns = ['bert_date','regular_date'])
# time_df = pd.DataFrame(columns = ['bert_time','regular_time'])
# date_df['bert_date'] = ori_date_list
# date_df['regular_date'] = iso_86_date_list
# print("ori_time_list len ={}, iso_86_time_list={}".format(len(ori_time_list), len(iso_86_time_list)))
# time_df['bert_time'] = ori_time_list
# time_df['regular_time'] = iso_86_time_list
# date_df.to_csv("./11_26_submission_9_12/answer_time_drop_date.txt", sep = '\t', header=True ,index = None)
# time_df.to_csv("./11_26_submission_9_12/answer_time_drop_time.txt", sep = '\t', header=True ,index = None)