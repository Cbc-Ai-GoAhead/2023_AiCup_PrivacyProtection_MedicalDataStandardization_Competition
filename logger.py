import logging
import datetime

log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
logging.basicConfig(level=logging.INFO, filename=log_filename, filemode='w',
	#format='[%(levelname).1s %(asctime)s] %(message)s',
	format='[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)
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