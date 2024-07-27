import datetime
import os

#set training end date to 1 week ago
end_date=int(datetime.date.today().strftime('%Y%m%d'))-7

#go back one month because for some reason knmi website does not provide the current month data
#since this is just to try github action capabilities, and not make real time predcitions that will be used, it s ok
#better than causing data leakage 
#it will simulate how system would work one month ago
end_date=end_date-100

#start_date is 10 years before end_date
start_date=int(end_date)-100000

var_name='TRAIN_START_DATE'
var_value=str(start_date)
echo_statement="echo "+"\""+var_name+"="+var_value+"\""+" >> $GITHUB_ENV"
print("sending the following to os.system")
print(echo_statement)
os.system(echo_statement)

var_name='TRAIN_END_DATE'
var_value=str(end_date)
echo_statement="echo "+"\""+var_name+"="+var_value+"\""+" >> $GITHUB_ENV"
print("Sending the following to os.system")
print(echo_statement)
os.system(echo_statement)
