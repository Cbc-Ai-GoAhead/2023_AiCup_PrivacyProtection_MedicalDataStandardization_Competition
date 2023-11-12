#import datetime
import logging
import time
import os

class Log: 
# return 0 if success, otherwise error

    def __init__(self, dir, module_name):
        self.log_dir = dir
        if not os.path.isdir(dir):
            os.makedirs(dir)
        time_str = time.strftime("%Y%m%d", time.localtime()) 
        self.path = os.path.join(self.log_dir, "log_" + time_str + ".txt")
        self.module_name = module_name
        logging.basicConfig(level = logging.INFO, filename=self.path, \
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' \
        )
        self.loggingObj = logging.getLogger(module_name)
        
    
    def set_name(self, module_name):
        self.loggingObj = logging.getLogger(module_name)
        self.loggingObj.info('--------------------------------------------');
    
    def get_path(self):
        return self.path

    def info(self, message):
        self.loggingObj.info(message);
        
    def debug(self, message):
        self.loggingObj.debug(message);
        
    def warning(self, message):
        self.loggingObj.warning(message);
        
    def error(self, message):
        self.loggingObj.error(message);
# log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
# logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w',
# 	#format='[%(levelname).1s %(asctime)s] %(message)s',
# 	format='[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
# 	datefmt='%Y%m%d %H:%M:%S',
# 	)
# logger = logging.getLogger(' ')
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)

# logger.addHandler(handler)
########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
# import logging

# logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
# logger = logging.getLogger(' ')
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)