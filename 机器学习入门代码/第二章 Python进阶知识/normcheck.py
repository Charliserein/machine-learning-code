from scipy import stats
Ho = '数据服从正态分布'#定义原假设
Ha = '数据不服从正态分布'#定义备择假设
alpha = 0.05#定义显著性P值
def normality_check(data):#定义normality_check( )函数
    for columnName, columnData in data.iteritems():#针对数据集中的变量和数值使用for循环做以下操作
        print("Shapiro test for {columnName}".format(columnName=columnName))#输出“Shapiro test for 变量名称”
        res = stats.shapiro(columnData)#使用shapiro方法开展正态分布检验，生成统计量res
        pValue = round(res[1], 2)#计算统计量res的pValue
        if pValue > alpha:
            print("pvalue={pValue}>{alpha}.不能拒绝原假设.{Ho}".format(pValue=pValue, alpha=alpha, Ho=Ho))#如果pValue大于设定的alpha显著性P值，则输出“pvalue = ** > alpha. 不能拒绝原假设. 数据服从正态分布”
        else:
            print("pvalue={pValue}<={alpha}.拒绝原假设.{Ha}".format(pValue=pValue,alpha=alpha,Ha=Ha))#如果pValue小于等于设定的alpha显著性P值，则输出“pvalue = **<=alpha. 拒绝原假设. 数据不服从正态分布”