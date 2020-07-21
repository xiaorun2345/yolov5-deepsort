import logging
import datetime
import os

def log():
    logOutPath = './LOG'
    if not os.path.exists(logOutPath):
        os.mkdir(logOutPath)

    today_date = datetime.date.today()
    logName = os.path.join(logOutPath, str(today_date) + '.log')
    print(logName)
    # logName = './LOG/%s.log' % (date)

    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(logName, encoding='utf-8')
        ch = logging.StreamHandler()
        lfrt1 = logging.Formatter(fmt='%(asctime)s-%(levelname)s: %(pathname)s, line %(lineno)d, %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        lfrt2 = logging.Formatter(fmt='%(message)s')
        fh.setFormatter(lfrt1)
        ch.setFormatter(lfrt2)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

logger = log()