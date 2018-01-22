import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

datafile="/Users/stanleyruan/Downloads/anova_test.csv"
 
data=pd.read_csv(datafile)

print('Number of experienced_ctrl observations: ' + str(len(data[(data['group']=='experienced_ctrl')])))
print('Number of experienced_tst observations: ' + str(len(data[(data['group']=='experienced_tst')])))
print('Number of rookie_ctrl observations: ' + str(len(data[(data['group']=='rookie_ctrl')])))
print('Number of rookie_tst observations: ' + str(len(data[(data['group']=='rookie_tst')])))
print()

experienced_ctrl_data = data[(data['group']=='experienced_ctrl')]
experienced_ctrl_time = sorted(experienced_ctrl_data['avg_time'].values)

experienced_ctrl_series=pd.Series(experienced_ctrl_time)
print('The five number summary for the experienced ctrl group: ')
print(experienced_ctrl_series.describe())

hmean = np.mean(experienced_ctrl_time)
hstd = np.std(experienced_ctrl_time)
fit_experienced_ctrl = stats.norm.pdf(experienced_ctrl_time, hmean, hstd)
plt.figure(1)
plt.plot(experienced_ctrl_time,fit_experienced_ctrl)
plt.hist(experienced_ctrl_time,normed=True)
plt.title('Distribution of experienced ctrl group average time')


experienced_tst_data = data[(data['group']=='experienced_tst')]
experienced_tst_time = sorted(experienced_tst_data['avg_time'].values)

experienced_tst_series=pd.Series(experienced_tst_time)
print('The five number summary for the experienced tst group: ')
print(experienced_tst_series.describe())

hmean = np.mean(experienced_tst_time)
hstd = np.std(experienced_tst_time)
fit_experienced_tst = stats.norm.pdf(experienced_tst_time, hmean, hstd)
plt.figure(2)
plt.plot(experienced_tst_time,fit_experienced_tst)
plt.hist(experienced_tst_time,normed=True)
plt.title('Distribution of experienced tst group average time')


rookie_ctrl_data = data[(data['group']=='rookie_ctrl')]
rookie_ctrl_time = sorted(rookie_ctrl_data['avg_time'].values)

rookie_ctrl_series=pd.Series(rookie_ctrl_time)
print('The five number summary for the rookie ctrl group: ')
print(rookie_ctrl_series.describe())

hmean = np.mean(rookie_ctrl_time)
hstd = np.std(rookie_ctrl_time)
fit_rookie_ctrl = stats.norm.pdf(rookie_ctrl_time, hmean, hstd)
plt.figure(3)
plt.plot(rookie_ctrl_time,fit_rookie_ctrl)
plt.hist(rookie_ctrl_time,normed=True)
plt.title('Distribution of rookie ctrl group average time')


rookie_tst_data = data[(data['group']=='rookie_tst')]
rookie_tst_time = sorted(rookie_tst_data['avg_time'].values)

rookie_tst_series=pd.Series(rookie_tst_time)
print('The five number summary for the rookie tst group: ')
print(rookie_tst_series.describe())

hmean = np.mean(rookie_tst_time)
hstd = np.std(rookie_tst_time)
fit_rookie_tst = stats.norm.pdf(rookie_tst_time, hmean, hstd)
plt.figure(4)
plt.plot(rookie_tst_time,fit_rookie_tst)
plt.hist(rookie_tst_time,normed=True)
plt.title('Distribution of rookie tst group average time')

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

f, p = stats.f_oneway(d_data['experienced_ctrl'], d_data['experienced_tst'], d_data['rookie_ctrl'],d_data['rookie_tst'])
print('The f score and p-value for the ANOVA test of all 4 groups are: ' + str(f) + ' and ' + str(p) + '. If p-value is smaller than 0.05, then the groups studied in this ANOVA are statistically different.')
print()
    
f1, p1 = stats.f_oneway(d_data['experienced_ctrl'],d_data['experienced_tst'])
print('The f score and p-value for the ANOVA test of experienced control group and experienced testing group are: ' + str(f1) + ' and ' + str(p1) + '. If p-value is smaller than 0.05, then the groups studied in this ANOVA are statistically different.')
print()

f2, p2 = stats.f_oneway(d_data['rookie_ctrl'],d_data['rookie_tst'])
print('The f score and p-value for the ANOVA test of rookie control group and rookie testing group are: ' + str(f2) + ' and ' + str(p2) + '. If p-value is smaller than 0.05, then the groups studied in this ANOVA are statistically different.')
print()


#The following codes perform 2-sample t-test:

ctrl_group_data=data[data['group']=='experienced_ctrl'].reset_index(drop=True)
trt1_data=data[data['group']=='experienced_tst'].reset_index(drop=True)

ctrl_group_data2=data[data['group']=='rookie_ctrl'].reset_index(drop=True)
trt2_data=data[data['group']=='rookie_tst'].reset_index(drop=True)

ctrl_array=ctrl_group_data['avg_time'].values
trt1_array=trt1_data['avg_time'].values

ctrl2_array=ctrl_group_data2['avg_time'].values
trt2_array=trt2_data['avg_time'].values

t1, p1 = ttest_ind(ctrl_array, trt1_array, equal_var=False)
print('The t-stat and p-value for the comparison of experienced control group and experienced testing group are: ' + str(t1) + ' and ' + str(p1) + ' .')
print()

t2, p2 = ttest_ind(ctrl2_array, trt2_array, equal_var=False)
print('The t-stat and p-value for the comparison of rookie control group and rookie testing group are: ' + str(t2) + ' and ' + str(p2) + ' .')
