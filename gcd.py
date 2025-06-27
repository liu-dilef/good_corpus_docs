#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import re
from joblib import load
import os

clf = load('rf_model.joblib')

in_dir = "class1"
out_dir = "txtFromPdf_ok"

ds = pd.DataFrame(columns=['dir','fname','text','char','let','caps','num','blank','lines','underscore'])

def dir2text(fdir):
    path_list = Path(fdir).rglob('*.txt')
    for path in path_list:
        txt = path.read_text(encoding='utf-8', errors='ignore')
        ds.loc[len(ds)] = [fdir,os.path.basename(path),txt,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]

dir2text(in_dir)

for ind,row in ds.iterrows():
    txt = row['text']
    if len(txt) < 500:
        continue
    nchar = len(txt)
    nlines = len(re.findall("\n", txt))
    nblank = len(re.findall("[ \t]", txt))
    nlet = len(re.findall("[A-Za-zÈàèéòùì]", txt))
    ncaps = len(re.findall("[A-ZÈ]", txt))
    nnum = len(re.findall("[0-9]", txt))
    nuscore = len(re.findall("[_-]", txt))
    ds.at[ind, 'char'] = nchar
    ds.at[ind, 'let'] = nlet/nchar
    ds.at[ind, 'caps'] = ncaps/nchar
    ds.at[ind, 'num'] = nnum/nchar
    ds.at[ind, 'blank'] = nblank/nchar
    ds.at[ind, 'lines'] = nlines/nchar
    ds.at[ind, 'underscore'] = nuscore/nchar

dsx = ds[ds.char != -1]
dsx = dsx.reset_index(drop=True)
feature_cols = ['let','caps','num','blank','lines','underscore']
X = dsx[feature_cols]
dsx['class'] = pd.Series(clf.predict(X))

for ind,row in dsx.iterrows():
    ds.at[ind,'class'] = row['class']
for ind,row in ds.iterrows():
    if row['class'] == 1:
        with open(out_dir + "/" + row['fname'],"w+") as f:
            f.writelines(row['text'])