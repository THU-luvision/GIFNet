#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import logging
import os.path
import time
 
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
 
time_line = time.strftime('%Y-%m%d-%H-%M', time.localtime(time.time()))
 

log_path = './logfile/'
logfile = log_path + time_line + '.log'
 
handler = logging.FileHandler(logfile, mode='w')
handler.setLevel(logging.INFO)
 
formatter = logging.Formatter(" %(message)s")
handler.setFormatter(formatter)
 
console = logging.StreamHandler()
console.setLevel(logging.INFO)
 
logger.addHandler(handler)
logger.addHandler(console)
 
# logger.info('This is an info message.')
# logger.warning('This is a warning message.')
# logger.error('This is an error message.')
# logger.critical('This is a critical message.')