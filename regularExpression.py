# coding:utf-8
import re

'''
############  正则表达式的主要功能函数：############

函数式用法(一次性)：rst = re.search(r'[1-9]\d{5}','BIT 100081')

面向对象用法：
pat = re.compile(r'[1-9]\d{5}')
rst = pat.search('BIT 100081')

'''

match = re.search(r'[1-9]\d{5}',"LiuJian19920517")
if match:
    print(match.group(0))
'''
如果正则表达式中定义了组，就可以在Match对象上用group()方法提取出子串来。

注意到group(0)永远是原始字符串，group(1)、group(2)……表示第1、2、……个子串。
'''

##################################################

match = re.match(r'[1-9]\d{5}','100081 BIT')
if match:
    print(match.group(0))

##################################################

ls = re.findall(r'[1-9]\d{5}',"LIU19920517 YE19921001")
print(ls)

#################################################
result = re.split(r'[1-9]\d{5}',
                  'BIT100081 TSU100084',
                  maxsplit=1)
print(result)

#####################################
'''
贪婪匹配/非贪婪匹配

'''
match  = re.search(r'PY.*?N','PYANBNCNDN')
if match:
    print(match.group(0))