from datetime import datetime
import re
import datetime as dt

####
##  Date 
####
"""

"""

"""
-- 9.txt
Last edited : 3/12/2063
--10.txt
D.O.B:  24/8/1993
--file14798
2:55pm on 19.9.16.
日、月、年
--file67900
(TO:DN/ta 27/5/71)

"""
stamp = "Fri Oct 11 15:09:30 GMT+01:00 2019"
fmt = "%a %b %d %H:%M:%S %Z%z %Y"
print(dt.datetime.strptime(stamp, fmt))
