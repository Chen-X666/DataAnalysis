# coding=utf-8
import re
# 1. 写一个正则表达式，使其能同时识别下面所有的字符串：'bat','bit', 'but', 'hat', 'hit', 'hut'

s ="bat ,bit ,but ,hat ,hit ,hut"
print(re.findall(r'[bh][aiu]t',s))

# 2.匹配由单个空格分隔的任意单词对，也就是姓和名

s = "Han meimei, Li lei, Zhan san, Li si"
print(re.findall(r'([A-Za-z]+) ([A-Za-z]+)',s))

# 3. 匹配由单个逗号和单个空白符分隔的任何单词和单个字母,如姓氏的首字母

s = "yu, Guan  bei, Liu  fei, Zhang"
print(re.findall(r'([a-zA-Z]+),\s([a-zA-Z])',s))

# 4.匹配所有的有效的Python标识符集合

s = "_hello , python_1 , 2world , Pra_ni , @dfa_ , ewq* "
print(re.findall(r'\b[a-zA-z_][\w]*(?!=\W) ',s))

# 5. 根据美国街道地址格式,匹配街道地址。美国接到地址使用如下格式:1180 Bordeaux Drive。使你的正则表达式足够灵活,以支持多单词的街道名称,如3120 De la Cruz Boulevard

s = """street 1: 1180  Bordeaux Drive,"
    street 1: 3120 De la Cruz Boulevard"""
print(re.search(r'\d+( +[a-zA-Z]+)+',s).group())

# 6. 匹配以“www”起始且以“.com”结尾的简单Web域名:例如,http://www.yahoo.com ，也支持其他域名，如.edu .net等

s = "http://www.yahoo.com        www.foothill.edu"
print(re.search(r'w{3}\.[a-zA-Z]+\.(com|edu|net)',s).group())

# 7. 匹配所有能够表示Python整数的字符串集

s = '520a1    20L 0  156   -8 -10a  A58'
ite = re.finditer(r'-?\d+',s)
for i in ite: print(i.group(),)

# 8. 匹配所有能够表示Python长整数的字符串集

s = '520a    20L 0  156   -8L  A58'
ite = re.finditer(r'-?\d+L',s)
for i in ite: print(i.group(),)

# 9. 匹配所有能够表示Python浮点数的字符串集

s = '80.2  fds2.1  0.003'
print(re.findall(r'\d+\.\d+',s))

# 10. 表示所有能够表示Python复数的字符串集

s = '12j  fds -4j  5-2j fdsa'
print(re.findall(r'\d*-?\d+j',s))

# 11、匹配一行文字中的所有开头的字母内容

s="Now, let's take a closer look at some iconic moments from the show's stage made by Chinese Angels."
print(re.findall(r'\b\w',s))

# 12、匹配一行文字中的所有开头的数字内容

s="Now, let's take a closer look at some iconic moments from the show's stage made by Chinese Angels."
print(re.findall(r'\b\d',s))

# 13、匹配一行文字中的所有开头的数字内容或字母内容

s = "577fsda3f you12daf f1s32dafffff"
print(re.findall(r'\b\d+|\b[A-Za-z]+',s))

# 14、 只匹配包含字母和数字的行

s = "nihao fsadf \n789! 3asfd 1\nfds12df e4 4564"
print(re.findall(r'^([a-zA-Z\d ]+)$',s,re.M))

# 15、提取每行中完整的年月日和时间字段

s="""time 1988-01-01 17:20:10 fsadf 2018-02-02 02:29:01"""
print(re.findall(r'[12]\d{3}\-[01]\d\-[0123]\d\s*[012]\d\:[012345]\d\:[012345]\d',s))

# 16、将每行中的电子邮件地址替换为你自己的电子邮件地址

s="""xss@qq.com, 465465@163.com, ppp@sina.com
    s121f@139.com, soifsdfj@134.com
    pfsadir423@123.com"""
print(re.sub(r'\w+?\@\w+?\.com','zeke@qq.com',s))

# 17、匹配\home关键字：

s ="fdsar \home   \homeer"
print(re.findall(r'\\home',s))