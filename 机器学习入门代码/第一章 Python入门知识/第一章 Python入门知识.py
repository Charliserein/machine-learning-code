#第一章 Python入门知识

#1.1 Python简介与本书教学理念
#1.2 Python下载与安装

#1.3 python注释、基本输出与输入
#1.3.1 python注释
#1.3.2 print函数
print(3+3-4)#运行结果为：2
print('对酒当歌，人生几何')#运行结果为：对酒当歌，人生几何
print("何以解忧，唯有杜康")#运行结果为：何以解忧，唯有杜康
print("""
对酒当歌，
人生几何，
……
何以解忧，
唯有杜康
""")
#1.3.3 input函数
a=input("请输入正方形的边长：")#输入正方形的边长
a= float(a)#由于返回值为字符串类型，所以需要转化为可计算的浮点数值型
s=a*a#计算正方形的面积s
print("正方形的面积为：",format(s,'.2f'))#输出正方形的面积s

#1.4 python变量和数据类型
#1.4.1 python保留字与标识符
import keyword#调用keyword模块
keyword.kwlist#输出python保留字
#1.4.2 python变量
a, b, c = "blue", "red", "green"#定义变量名称a, b, c并分别赋值为"blue", "red", "green"
print(a)#输出变量a的值
print(b)#输出变量b的值
print(c)#输出变量c的值

a=b=c="blue"#定义变量名称a, b, c，都赋值为"blue"
print(a)#输出变量a的值
print(b)#输出变量b的值
print(c)#输出变量c的值

a="100"#定义变量名称a,赋值为字符串"100"
type(a)#查看变量a的类型，运行结果为str，即字符串
a=100#定义变量名称a,赋值为数值100
type(a)#查看变量a的类型，运行结果为int，即整数型

#1.4.3 python基本数据类型
int(3.1415926)#对3.1415926取整数
float(3)#将数字3转换成浮点数
float('3.1415926')#将字符串'3.1415926'转换成浮点数
complex(1,3)#输出复数(1+3j)，函数括号内第1个数字1表示实部，第2个数字3表示虚部

print('对酒当歌\n人生几何')# \n 换行符，实现换行
print('对酒当歌\r人生几何')# \r 回车符，删掉本行之前的内容，将光标移到本行开头
print('对酒当歌\t人生几何')# \t 制表符，即Tab键，一般相当于四个空格
print('对酒当歌\b人生几何')# \b 退格符，将光标位置移到前一位
print('对酒当歌\\人生几何')# \\ 反斜杠，两个连续的反斜杠表示反斜杠本身
print('对酒当歌\'人生几何')# \' 单引号
print('对酒当歌\"人生几何')# \' 双引号
print('对酒当歌\人生几何')# \' 续行符
print(r"对酒当歌\n人生几何")# 原字符

type(True)#观察True的类型
type(False)#观察False的类型
True and True#逻辑运算中的“和”计算，True and True
True and False#逻辑运算中的“和”计算，True and False
False and True#逻辑运算中的“和”计算，False and True
False and False#逻辑运算中的“和”计算，False and False
True or True#逻辑运算中的“或”计算，True or True
True or False#逻辑运算中的“或”计算，True or False
False or True#逻辑运算中的“或”计算，False or True
False or False#逻辑运算中的“或”计算，False or False
not True#逻辑运算中的“非”计算，not True
not False#逻辑运算中的“非”计算，not False
6>=3#开展逻辑表达式运算
type(6>=3)#观察逻辑表达式运算的数据类型

#1.5 python序列
#1.5.1 索引（Indexing）
list = [1,3,5,7,9]
print('列表第一个元素',list[0])#访问列表第一个元素，索引值为0
print('列表第二个元素',list[1],list[-4])#访问列表第二个元素，正索引值为1，负索引值为-4
print('列表最后一个元素',list[4],list[-1])#访问列表最后一个元素，正索引值为4，负索引值为-1

list = ['对酒当歌','人生几何']
print('列表第一个元素:',list[0])#访问列表第一个元素，索引值为0

#1.5.2 切片（Slicing）
list = [1,3,5,7,9,11,13,15,17,19]
print('查看列表前5项：',list[0:5])#此处的0也可省略，即list[:5]
print('查看列表第2-4项：',list[1:4])#注意4是列表的第五项，但是在这里是不包含4的，所以没有第五项
print('查看列表所有项，设置步长为2：',list[::2])#设置步长为2
print('查看逆序排列的列表：',list[::-1])#设置步长为-1，即可实现逆序输出列表

#1.5.3 相加（adding）
list1 = [1,3,5,7,9]#生成列表list1，其中的元素均为数字
list2 = [2,4,6,8,10]#生成列表list2，其中的元素均为数字
list=list1 + list2#将list1与list2 相加，生成list
list#查看生成的序列list
list3=['对酒当歌','人生几何']#生成列表list3，其中的元素为字符串
list=list1+list2+list3#将list1与list2、list3相加，更新list
list#查看新的序列list

#1.5.4 相乘（Multiplying）
list1 = [1,3,5,7,9]#生成列表list1，其中的元素均为数字
list=list1*3
list

#1.5.5 元素检查
list1 = [1,3,5,7,9]#生成列表list1，其中的元素均为数字
print(3 in list1)#检查列表list1中是否包含数字3
print(4 not in list1)#检查列表list1中是否不包含数字4

#1.5.6 与序列相关的内置函数
list1 = [1,3,5,9,7]#生成列表list1，其中的元素均为数字
len(list1)#计算列表list的长度
max(list1)#查找序列中的最大元素
min(list1)#查找序列中的最小元素
str(list1)#将序列转化为字符串
sum(list1)#计算序列中的元素和，元素只能是数字
sorted(list1)#对元素进行排序，排序方式为升序
sorted(list1,reverse=True)#对元素进行排序，排序方式为降序

#1.6 python列表
#1.6.1 列表基本操作
list1=list(range(1,10,1))
list1

print('2022年重要新能源上市公司名单')#输出内容'2022年重要新能源上市公司名单'
company=['宁德时代 300750','比亚迪 002594','国轩高科 002074','亿纬锂能 300014','赣锋锂业 002460']#创建列表company
for company in company:
    print(company)#输出列表company中各个元素的值
    
print('2022年重要新能源上市公司名单')#输出内容'2022年重要新能源上市公司名单'
company=['宁德时代 300750','比亚迪 002594','国轩高科 002074','亿纬锂能 300014','赣锋锂业 002460']#创建列表company
for index,item in enumerate(company):
    print(index+1,item)#输出列表company中各个元素的值 
    
#1.6.2 列表元素基本操作
corporation=['宁德时代 300750','比亚迪 002594','国轩高科 002074','亿纬锂能 300014','赣锋锂业 002460']#创建列表corporation
len(corporation)#计算列表corporation的长度
corporation.append('欣旺达 300207')
len(corporation)#计算列表corporation的长度
print(corporation)#查看增加元素后的列表corporation
corporation[0]='宁德时代新能源科技股份有限公司'#将第一个元素修改为'宁德时代新能源科技股份有限公司'
print(corporation)#查看修改元素后的列表corporation
del corporation[0]#按照元素索引删除元素，删除列表corporation中的第一个元素
print(corporation)#查看删除元素后的列表corporation
corporation.remove('国轩高科 002074')#按照元素值删除元素，删除列表corporation中值为'国轩高科 002074'的元素
print(corporation)#查看删除元素后的列表corporation
print(corporation.count('亿纬锂能 300014'))#统计下元素'亿纬锂能 300014'出现的次数
print(corporation.index('亿纬锂能 300014'))#获取元素'亿纬锂能 300014'首先出现的下标

#1.6.3 列表推导式
list=[x for x in range(4)]#生成列表list，元素为range(4)中的元素
print(list)#查看list中的元素
listnew=[x*2 for x in list]#生成列表listnew,元素为list的各个元素乘以2
print(listnew)#查看listnew中的元素
listnew1=[x*2 for x in list if x>=2]#生成列表listnew1,元素为针对list的大于等于2的元素乘以2
print(listnew1)#查看listnew1中的元素

#1.7 python元组
#1.7.1 元组基本操作
tuple1 = (1,3,5,9,7) 
tuple2 =('宁德时代 300750','比亚迪 002594','国轩高科 002074','赣锋锂业 002460')
tuple3 =(1,3,5,7,'赣锋锂业 002460')

tuple2 ='宁德时代 300750','比亚迪 002594','国轩高科 002074','赣锋锂业 002460'
type(tuple2)

tuple2 ='宁德时代 300750',
type(tuple2)
tuple3 ='宁德时代 300750'
type(tuple3)

tuple1=tuple(range(1,10,1))#创建数值元组tuple1，数值为从1到10、按1步进
tuple1#查看数值元组tuple1

#1.7.2 元组元素基本操作
tuple2 =('宁德时代 300750','比亚迪 002594')#创建元组tuple2
print(tuple2)#查看元组tuple2中的元素
tuple2 =('宁德时代 300750','比亚迪 002594','国轩高科 002074')#对元组tuple2重新赋值
print(tuple2)#查看更新后的元组tuple2的元素
tuple2=tuple2+('赣锋锂业 002460',)#对元组tuple2进行组合连接
print(tuple2)#查看更新后的元组tuple2的元素

#1.7.3 元组推导式
tuple4=(i for i in range(4))#生成一个生成器对象，元素为range(4)中的元素
print(tuple4)#查看tuple4
tuple4=tuple(tuple4)#使用tuple（）函数将生成器对象转化为元组
print(tuple4)#查看tuple4

tuple5=(i for i in range(5))#生成一个生成器对象，元素为range(5)中的元素
for i in tuple5: #使用 for循环遍历生成器对象以获得各个元素
    print(i,end=',') # 输出元组元素在同一行显示，并且用“,”隔开
print(tuple(tuple5))# 输出新元组

#1.8 python字典
#1.8.1字典的创建与删除
# 创建方法1，通过给定的键值对以直接赋值的方式
dict1={'x':1,'y':2,'z':3}#通过给定的键值对以直接赋值的方式创建字典dict1
print(dict1)#查看字典dict1
dict={ }#空字典
print(dict)#查看空字典dict
# 创建方法2，通过映射函数的方式
list1=['x', 'y', 'z']#创建列表list1
list2=[1, 2, 3]#创建列表list2
dict2=dict(zip(list1,list2))#zip函数的作用是将多个列表或元组对应位置的元素组合为元组，并返回包含这些内容的zip对象，本例中先通过zip( )函数将列表list1、list2组合为元组，再使用dict( )函数生成字典dict2
print(dict2)#查看字典dict2
# 创建方法3，将列表序列转化为字典
list1=[('x',1), ('y',2), ('z',3)]#创建列表list1
dict3=dict(list1)#将列表list1转化为字典dict3
print(dict3)#查看字典dict3
# 创建方法4，通过已经存在的元组和列表创建字典
tuple1 = ('x', 'y', 'z')#创建元组tuple1
list1 = [1, 2, 3]#创建列表list1
dict4 = {tuple1:list1}#生成字典dict4，键为tuple1，值为list1
print(dict4)#查看字典dict4
# 创建方法5，定义字典另一种方式
dict5=dict(x=1,y=2,z=3)#创建字典dict5
print(dict5)#查看字典dict5

#del dict
dict1={'x':1,'y':2,'z':3}#创建字典dict1
del dict1#删除字典dict1

#dict.clear()
dict1={'x':1,'y':2,'z':3}#创建字典dict1
dict1.clear()#清除字典dict1内的所有元素
dict1#查看字典dict1

#dict.pop()
dict1={'x':1,'y':2,'z':3}#创建字典dict1
x=dict1.pop('x')#返回指定键'x'对应的值，并在原字典中删除这个键-值对
print(x)#查看指定键'x'对应的值
print(dict1)#查看字典dict1

#dict.popitem()
dict1={'x':1,'y':2,'z':3}#创建字典dict1
dict1.popitem()#删除字典dict1中的最后一个键-值对
print(dict1)#查看字典dict1

dict1={'x':1,'y':2,'z':3}#创建字典dict1
print(dict1['y'])#查看字典dict1中键'y'对应的值

dict1={'x':1,'y':2,'z':3}#创建字典dict1
x= dict1.get('x')#将x指定为字典dict1中键'x'对应的值
x#输出x的值
w= dict1.get('w')#将w指定为字典dict1中键'w'对应的值
w#输出w的值
w=dict1.get('w',4)#将字典dict1中键'w'对应的值指定为4
w#输出w的值

dict1={'x':1,'y':2,'z':3}#创建字典dict1
print(dict1.items())#获取并输出字典dict1中的所有键-值对

dict1={'x':1,'y':2,'z':3}#创建字典dict1
print(dict1.keys())#以列表返回字典dict1所有的键

dict1={'x':1,'y':2,'z':3}#创建字典dict1
print(dict1.values())#以列表返回字典dict1所有的值

#1.8.2字典元素基本操作
dict1={'x':1,'y':2,'z':3}#创建字典dict1
dict1['m']=4#向字典dict1增加一个元素
print(dict1)#查看更新后的字典dict1

dict1={'x':1,'y':2,'z':3}#创建字典dict1
dict1['x']=4#更新字典dict1中'x'键对应的值
print(dict1)#查看更新后的字典dict1

dict1={'x':1,'y':2,'z':3}#创建字典dict1
del dict1['x']#删除字典dict1中'x'键对应的键值对
print(dict1)#查看更新后的字典dict1

dict1={'x':1,'y':2,'z':3}#创建字典dict1
dict2={'x':4,'u':5,'n':7}#创建字典dict2
dict1.update(dict2)#将字典dict2中的键-值对更新到dict1里
print(dict1)#查看更新后的字典dict1

list1 = ['x', 'y', 'z']#创建列表list1
dict1 = dict.fromkeys(list1)#创建字典dict1，以列表list1中的元素作为字典dict1的键
dict2 = dict.fromkeys(list1, '6')#创建字典dict2，以6作为字典所有键对应的初始值
print(dict1)#查看更新后的字典dict1，运行结果为{'x': None, 'y': None, 'z': None}
print(dict2)#查看更新后的字典dict2，运行结果为 {'x': '6', 'y': '6', 'z': '6'}

#1.8.3 字典推导式
import random#调用random标准库
dict1={x:random.randint(1,10) for x in range(1,4)}#生成字典dict1，其中的键为1~4的整数，值为1~10之间的随机整数
print(dict1)#查看字典dict1

#1.9 python集合
set1 ={'宁德时代 300750','比亚迪 002594','赣锋锂业 002460'}#生成集合set1
set2 ={'国轩高科 002074','赣锋锂业 002460'}#生成集合set2
set3=set1|set2#生成set1与set2的并集set3
set3#查看集合set3
set4=set1&set2#生成set1与set2的交集set4
set4#查看集合set4

list1 =['宁德时代 300750','比亚迪 002594','赣锋锂业 002460','赣锋锂业 002460']#生成列表list1
type(list1)#查看list1的类型
set1=set(list1)#将列表list1转化为集合set1
type(set1)#查看set1的类型
set1#查看集合set1

#1.10 python字符串
#拼接字符串
str1='对酒当歌，'#生成字符串str1
str2='人生几何'#生成字符串str2
str3=str1+str2#将字符串str1和字符串str2拼接，生成字符串str3
str3#查看字符串str3
#计算字符串长度
len(str3)#计算字符串str3的长度
#截取字符串
str4=str3[2:7:2]# 从字符串str3截取生成字符串str4
str4#查看字符串str4
#分割字符串
str5='对酒当歌 人生几何 譬如朝露 去日苦多'#生成字符串str5
str5.split()#对字符串str5采用默认设置进行分割
#检索字符串
str5='对酒当歌 人生几何 …… 何以解忧 唯有杜康'#生成字符串str5
str5.count('何')#检索字符串str5中出现子字符串'何'的次数
str5.find('何')#检索字符串str5是否存在子字符串'何'
str5.find('短歌行')#检索字符串str5是否存在子字符串'短歌行'
str5.index('何')#检索字符串str5是否存在子字符串'何'
str5.index('短歌行')#检索字符串str5是否存在子字符串'短歌行'
str5.startswith('对')#检索字符串str5是否以指定的子字符串'对'开头
str5.startswith('康')#检索字符串str5是否以指定的子字符串'康'开头
str5.endswith('对')#检索字符串str5是否以指定的子字符串'对'结尾
str5.endswith('康')#检索字符串str5是否以指定的子字符串'康'结尾
#字母大小写转换
str6='ABCdefg'#生成字符串str6
str6.lower()#将字符串str6中的所有字母都降为小写
str6.upper()#将字符串str6中的所有字母都升为大写
#去除字符串左右两侧的空格或特殊字符
str7=' ABCdefg '#生成字符串str7
str7.strip()#去除字符串str7左右两侧的空格或特殊字符
str7.lstrip()#去除字符串str7左侧的空格或特殊字符
str7.rstrip()#去除字符串str7右侧的空格或特殊字符
#格式化字符串
company = '上市公司:{:s}\t 股票代码：{:d} \t'#制定模板
company1 = company.format('宁德时代',300750)#按照模板格式化字符串company1
company2 = company.format('派能科技',688063)#按照模板格式化字符串company2
print(company1)#查看字符串company1
print(company2)#查看字符串company2
