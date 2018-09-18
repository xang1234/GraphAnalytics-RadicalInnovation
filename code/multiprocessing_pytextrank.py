# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 07:50:25 2018

@author: david
"""
#import os
#os.chdir(r'C:\Users\david\github\pytextrank\pytextrank')

import pandas as pd
import xang1234_pytextrank as pt
from multiprocessing import Pool
import time


if __name__== '__main__':

    data=pd.read_csv(r'D:/Social Project/ML articles expanded keywords.csv')
    p=Pool(4)
    start=time.time()
    results=p.map(pt.top_keywords_sentences,data['paperAbstract'][:100])
    p.terminate()
    p.join()
    out=pd.DataFrame(results)
    out.to_csv(r'C:\Users\david\Desktop\test.csv')
    print('time taken',round(time.time()-start,2))
