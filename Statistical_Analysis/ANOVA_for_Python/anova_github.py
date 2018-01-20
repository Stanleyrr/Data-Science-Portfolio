import numpy as np
import pandas as pd
import scipy
from scipy import stats
import scipy.stats
from scipy.stats import ttest_ind

datafile="/Users/stanleyruan/Downloads/anova_test.csv"
#from pyvttbl import DataFrame
 
data=pd.read_csv(datafile)

data.boxplot('avg_time',by='group')

#The following codes perform ANOVA:

#The following code converts the 'group' variable from dataframe to array:    
grps = pd.unique(data.group.values)


d_data={grp:data['avg_time'][data.group == grp] for grp in grps}

#The 3 lines below are optional and returns some basic statistics that the ANOVA formula uses:
k=len(pd.unique(data.group))
N=len(data.values)
n=data.groupby('group').size()[0]

#The few lines below derive the f score and p-value for the ANOVA test:

f1, p1 = stats.f_oneway(d_data['experienced_ctrl'],d_data['experienced_tst'])
print('The f score and p-value for the ANOVA test of imerit control group and imerit testing group are: ' + str(f1) + ' and ' + str(p1) + '. If p-value is smaller than 0.05, then the groups studied in this ANOVA are statistically different.')
print()

f2, p2 = stats.f_oneway(d_data['rookie_ctrl'],d_data['rookie_tst'])
print('The f score and p-value for the ANOVA test of opencrowd control group and opencrowd testing group are: ' + str(f2) + ' and ' + str(p2) + '. If p-value is smaller than 0.05, then the groups studied in this ANOVA are statistically different.')
print()


#The following codes perform 2-sample t-test:

ctrl_group_data=data[data['group']=='experienced_ctrl'].reset_index(drop=True)
trt1_data=data[data['group']=='imerit_tst'].reset_index(drop=True)

ctrl_group_data2=data[data['group']=='rookie_ctrl'].reset_index(drop=True)
trt2_data=data[data['group']=='rookie_tst'].reset_index(drop=True)

ctrl_array=ctrl_group_data['avg_time'].values
trt1_array=trt1_data['avg_time'].values

ctrl2_array=ctrl_group_data2['avg_time'].values
trt2_array=trt2_data['avg_time'].values

t1, p1 = ttest_ind(ctrl_array, trt1_array, equal_var=False)
print('The t-stat and p-value for the comparison of imerit control group and imerit testing group are: ' + str(t1) + ' and ' + str(p1) + ' .')
print()

t2, p2 = ttest_ind(ctrl2_array, trt2_array, equal_var=False)
print('The t-stat and p-value for the comparison of opencrowd control group and opencrowd testing group are: ' + str(t2) + ' and ' + str(p2) + ' .')


