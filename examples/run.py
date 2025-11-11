#!/usr/bin/env python
import argparse
import jinh

funcDic = {"ndata": "sum_traindata"}

description ="""
Processing functions directly in which included jinh package\n
"""
description += jinh.__str__

par = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter, prog="jinh")
par.add_argument('-f', '--func', default="ndata", help="namespace of func included in jinh")
par.add_argument('-p', '--path', nargs="*", default=".", help="path")
args = par.parse_args()
 
d = vars(args)
kwargs = {k: v for k, v in d.items() if k != "func"}


if args.func not in funcDic.values():
    func = getattr(jinh, funcDic[args.func])
else:
    func = getattr(jinh, args.func)

func(**kwargs)
