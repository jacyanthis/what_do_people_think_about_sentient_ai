# %% 2021–2023 tests

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
import csv
import re
from scipy.stats import zscore
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import weightedstats as ws

with open('AIMS_Main_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

d21, d23 = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv')
d21['year'], d23['year'] = '2021', '2023'
d = pd.concat([d21, d23], ignore_index=True)

d = d.loc[:,~(d.columns.str.contains('(?i)nr|mr|[^f]exp') | d.columns.str.startswith('exp_'))]
for c in d.columns:
    if d.loc[0,c] in ['Yes', 'No', 'Not sure']:
        d[c] = (d[c] == 'Yes')
w = d['weight']
y = d['year']
m = d[['F2','F3']]

d = d.drop(columns=(['F2','F3','attention','exper','politics','religion','relelse','diet','feedback','minutes','progress',
                  'age','gender','region','raceethnicity','education','income','Pcode',
                  'education_recode','sex_age','income_recode','weight','age_recode','year']))

d['AS Caution'] = d[['PMC1', 'PMC3', 'PMC4']].mean(axis = 1)
d['Pro-AS Activism'] = d[['PMC2', 'PMC5', 'PMC6', 'PMC7', 'PMC8', 'PMC9', 'PMC10', 'PMC11']].mean(axis = 1)
d['AS Treatment'] = d[['MCE1', 'MCE2', 'MCE3', 'MCE4', 'MCE5', 'MCE6']].mean(axis = 1)
d['Malevolence Protection'] = d[['MCE7', 'MCE8', 'MCE9']].mean(axis = 1)
d['AI Moral Concern'] = d[['MCE21', 'MCE22', 'MCE23','MCE24','MCE25','MCE26','MCE27','MCE28','MCE29','MCE30','MCE31']].mean(axis = 1)
d['Mind Perception'] = d[['MP1', 'MP2', 'MP3', 'MP4']].mean(axis = 1)
d['Perceived Threat'] = d[['SI2', 'SI3', 'SI4']].mean(axis = 1)
d['Anthropomorphism'] = d[['Anth1', 'Anth2', 'Anth3', 'Anth4']].mean(axis = 1)

for i in ['AS Caution','Pro-AS Activism','AS Treatment','Malevolence Protection','AI Moral Concern',
          'Mind Perception', 'Perceived Threat', 'Anthropomorphism'
         ]:
    descriptions[i] = ''

results = []
for c in d.columns:
    wmean_21 = np.average(d[c][~pd.isna(d[c]) & (y == '2021')], weights = w[~pd.isna(d[c]) & (y == '2021')])
    wmean_23 = np.average(d[c][~pd.isna(d[c]) & (y == '2023')], weights = w[~pd.isna(d[c]) & (y == '2023')])
    _, p_value, df = ttest_ind(d[c][~pd.isna(d[c]) & (y == '2021')],
                               d[c][~pd.isna(d[c]) & (y == '2023')],
                               weights=(w[~pd.isna(d[c]) & (y == '2021')], w[~pd.isna(d[c]) & (y == '2023')]))
    results.append([c, p_value, np.nan, (wmean_23 - wmean_21), wmean_21, wmean_23, descriptions[c]])
r = pd.DataFrame(results,columns=['var','p_value','fdr_adj','change','2021','2023','description'])
r = r.sort_values('var', key=lambda col: col.str.lower())

r['fdr_adj'] = fdrcorrection(r['p_value'])[1]
display(r)

# %% MP in all AIs vs LLMs significance test

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import csv
import re
from scipy.stats import zscore
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import weightedstats as ws

with open('AIMS_Main_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

with open('AIMS_Supplement_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

m = pd.read_csv('AIMS_2023.csv')
s = pd.read_csv('AIMS_Supplement_2023.csv')
r = pd.DataFrame(columns=['var','description','p_value'])

for c in ['MP1', 'MP2', 'MP3', 'MP4']:
    _, p, _ = ttest_ind(m[c],
                        s[c],
                        weights=(m['weight'],
                                 s['weight'])
                    )
    r.loc[len(r)] = [c, descriptions[c], p]

display(r)


# Demographic predictors (main)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import csv
import re
from ast import literal_eval
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS
import weightedstats as ws

d21, d23 = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv')
d21['year'], d23['year'] = '2021', '2023'
d = pd.concat([d21, d23], ignore_index=True)

d = d.loc[:,~(d.columns.str.contains('(?i)nr|mr|[^f]exp') | d.columns.str.startswith('exp_'))]
for c in d.columns:
    if d.loc[0,c] in ['Yes', 'No', 'Not sure']:
        d[c] = (d[c] == 'Yes')
w = d['weight']
y = d['year']
m = d[['F2','F3']]

d['rel'] = d['religion'].apply(lambda x: 0 if x in ['None','Atheist','Agnostic'] else 1)
d['exper_count'] = d['exper'].apply(lambda l: len([e for e in literal_eval(l) if e != 'None of the above']))

demos = pd.get_dummies(d[['age','gender','region','raceethnicity','education','income',
                          'politics','rel','diet','own','work','smart','exper_count','fint','fexp','year']]
                      ).drop(columns=['gender_Female','region_South','raceethnicity_White','diet_Meat-eater','year_2021'])

demos['own'] = demos['own'].astype(int)

for c in demos.columns:
    demos[c] = zscore(demos[c])

demos = demos[['age','gender_Male','region_Northeast','region_Midwest','region_West',
               'raceethnicity_Asian','raceethnicity_Black','raceethnicity_Hispanic','raceethnicity_Indigenous','raceethnicity_Other (Non-Hispanic)',
               'education','income','politics','rel',
               'diet_Other restrictions','diet_Pescatarian','diet_Vegan','diet_Vegetarian',
               'smart','own','work','exper_count','fint','fexp','year_2023']]

d = d.drop(columns=(['F2','F3','attention','exper','politics','religion','relelse','diet','feedback','minutes','progress',
                  'age','gender','region','raceethnicity','education','income','Pcode',
                  'education_recode','sex_age','income_recode','weight','age_recode','year']))
for c in d.columns:
    d[c] = zscore(d[c])

d['AS Caution'] = d[['PMC1', 'PMC3', 'PMC4']].mean(axis = 1)
d['Pro-AS Activism'] = d[['PMC2', 'PMC5', 'PMC6', 'PMC7', 'PMC8', 'PMC9', 'PMC10', 'PMC11']].mean(axis = 1)
d['AS Treatment'] = d[['MCE1', 'MCE2', 'MCE3', 'MCE4', 'MCE5', 'MCE6']].mean(axis = 1)
d['Malevolence Protection'] = d[['MCE7', 'MCE8', 'MCE9']].mean(axis = 1)
d['AI Moral Concern'] = d[['MCE21', 'MCE22', 'MCE23','MCE24','MCE25','MCE26','MCE27','MCE28','MCE29','MCE30','MCE31']].mean(axis = 1)
d['Mind Perception'] = d[['MP1', 'MP2', 'MP3', 'MP4']].mean(axis = 1)
d['Perceived Threat'] = d[['SI2', 'SI3', 'SI4']].mean(axis = 1)
d['Anthropomorphism'] = d[['Anth1', 'Anth2', 'Anth3', 'Anth4']].mean(axis = 1)

for i in ['AS Caution','Pro-AS Activism','AS Treatment','Malevolence Protection','AI Moral Concern',
          'Mind Perception', 'Perceived Threat', 'Anthropomorphism']:
    descriptions[i] = ''
    
Y = d[['AS Caution','Pro-AS Activism','AS Treatment','Malevolence Protection','AI Moral Concern',
          'Mind Perception', 'Perceived Threat', 'Anthropomorphism','SI1','F1']]

for c in Y.columns:
    Y[c] = zscore(Y[c])

regression_results = pd.DataFrame(columns=['outcome','const'] + list(demos.columns))

for y in Y:
    model = WLS(Y[y], sm.add_constant(demos), weights=w).fit()
    row = list(zip(model.params, model.pvalues))
    for k, l in enumerate(row):
        if l[1] >= 0.05:
            row[k] = ''
        elif l[1] >= 0.01:
            row[k] = round(row[k][0],3)
        elif l[1] >= 0.001:
            row[k] = str(round(row[k][0],3)) + '*'
        else:
            row[k] = str(round(row[k][0],3)) + '**'
    regression_results.loc[len(regression_results)] = [y] + row
    
regression_results

# %% Demographic predictors (supplement)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import csv
import re
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS
import weightedstats as ws

d = pd.read_csv('AIMS_Supplement_2023.csv')

for c in d.columns:
    if (c[-5:] == 'aware') & ((c != 'selfaware') & (c != 'sitaware')):
        code = {'(1) definitely yes': 1, '(2) yes': 2, '(3) probably yes': 3, '(4) probably no': 4, '(5) no': 5, '(6) definitely no': 6, 'No opinion': np.nan}
        for i, x in d[c].items():
            d.loc[i, c] = code[x]
    elif d.loc[0,c] in ['Yes', 'No', 'Not sure']:
        d[c] = (d[c] == 'Yes')
    elif d.loc[0,c] in ["It's too fast.", "It's fine.", "It's too slow."]:
        d[c] = (d[c] == "It's too fast.")
    elif c == 'YG1':
        d[c] = (d[c] == 'Very concerned') | (d[c] == 'Somewhat concerned')
    elif c == 'YG2':
        d[c] = (d[c] == 'Very likely') | (d[c] == 'Somewhat likely')
    elif c in ['YG3','YG4']:
        d[c] = (d[c] == 'Strongly support') | (d[c] == 'Somewhat support')

d = d.loc[:,~(d.columns.str.contains('(?i)nr|mr|[^f]exp') | d.columns.str.startswith('exp_'))]
w = d['weight']
m = d[['emergence','HLAIemergence','SIemergence']]

d['rel'] = d['religion'].apply(lambda x: 0 if x in ['None','Atheist','Agnostic'] else 1)
d['exper_count'] = d['exper'].apply(lambda l: len([e for e in literal_eval(l) if e != 'None of the above']))

demos = pd.get_dummies(d[['age','gender','region','raceethnicity','education','income',
                          'politics','rel','diet','own','work','smart','exper_count','fint','fexp']]
                      ).drop(columns=['gender_Female','region_South','raceethnicity_White','diet_Meat-eater'])

demos['own'] = demos['own'].astype(int)

for c in demos.columns:
    demos[c] = zscore(demos[c])

demos = demos[['age','gender_Male','region_Northeast','region_Midwest','region_West',
               'raceethnicity_Asian','raceethnicity_Black','raceethnicity_Hispanic','raceethnicity_Indigenous','raceethnicity_Other (Non-Hispanic)',
               'education','income','politics','rel',
               'diet_Other restrictions','diet_Pescatarian','diet_Vegan','diet_Vegetarian',
               'smart','own','work','exper_count','fint','fexp']]

d = d.drop(columns=(['emergence','HLAIemergence','SIemergence','universe','attention','exper','politics','religion','relelse','diet','feedback','minutes',
                  'age','gender','region','raceethnicity','education','income','Pcode',
                  'education_recode','income_recode','weight','age_recode']))
for c in d.columns[~d.isin([0,1,True,False]).all()]:
    d[c] = zscore(d[c].astype(float))

d['AS Caution'] = d[['PMC1', 'PMC3', 'PMC4']].mean(axis = 1)
d['AI Risk'] = d[['RS1','RS2','RS3']].mean(axis = 1)
d['Perceived Threat'] = d[['SI2', 'SI3', 'SI4']].mean(axis = 1)
d['Positive Emotions'] = d[['respect','admiration','compassion','awe','pride','excitement']].mean(axis = 1)
d['AI Trust'] = d[['LLMtrust','chatbottrust', 'robottrust','gameAItrust']].mean(axis = 1)
d['LLM Mind Perception'] = d[['MP1', 'MP2', 'MP3', 'MP4','selfaware', 'sitaware', 'power', 'ownmotives', 'owngoals', 'selfcontrol', 'understanding', 'upholding', 'safegoals', 'friendliness']].mean(axis = 1)
d['LLM Suffering'] = d[['LLM1','LLM2','LLM3']].mean(axis = 1)
d['AS Treatment'] = d[['MCE1', 'MCE2', 'MCE3', 'MCE4', 'MCE5', 'MCE6']].mean(axis = 1)

for i in ['AS Caution','AI Risk','Perceived Threat','Positive Emotions','AI Trust','LLM Mind Perception','LLM Suffering','AS Treatment']:
    descriptions[i] = ''
    
Y = d[['AS Caution','AI Risk','Perceived Threat','Positive Emotions','AI Trust','LLM Mind Perception','LLM Suffering','AS Treatment','SI1','F1']]

for c in Y.columns:
    Y[c] = zscore(Y[c])

regression_results = pd.DataFrame(columns=['outcome','const'] + list(demos.columns))

for y in Y:
    model = WLS(Y[y], sm.add_constant(demos), weights=w).fit()
    row = list(zip(model.params, model.pvalues))
    for k, l in enumerate(row):
        if l[1] >= 0.05:
            row[k] = ''
        elif l[1] >= 0.01:
            row[k] = round(row[k][0],3)
        elif l[1] >= 0.001:
            row[k] = str(round(row[k][0],3)) + '*'
        else:
            row[k] = str(round(row[k][0],3)) + '**'
    regression_results.loc[len(regression_results)] = [y] + row
    
regression_results

# %% Median forecasts test

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import csv
import re
import statistics
import weightedstats as ws
from scipy.stats import median_test
from statsmodels.stats.proportion import proportions_ztest

d21, d23 = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv')
d21['year'], d23['year'] = '2021', '2023'
d = pd.concat([d21, d23], ignore_index=True)
d21, d23 = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv')
d21['year'], d23['year'] = '2021', '2023'
w21, w23 = d21['weight'], d23['weight']

y = d['year']
m = d[['F2','F3']]

def reject_outliers(data, m=2):
    data = data.apply(lambda x: x - 2023 if (x < 2100) & (x >= 2023) else x)
    return data

for c in m.columns:
    c21, c23 = m[c][y == '2021'], m[c][y =='2023'].reset_index(drop=True)
    c21.apply(lambda x: -1 if x == 0 else 0 if x == -1 else x for x in c21)

    print(f'{c} 2021')
    c21[c21 == -1] = np.nan
    print(f'mean without outliers: {np.mean(reject_outliers(c21))}')
    c21 = reject_outliers(c21)
    c21[c21.isna()] = 10e100
    print(f'unweighted mean: {statistics.mean(c21.dropna())}')
    print(f'unweighted median w/o already/nevers: {statistics.median(c21.dropna()[c21 > 0])}')
    print(f'weighted median w/o already/nevers: {ws.weighted_median(c21[~c21.isna()][c21 > 0], weights=w21[~c21.isna()][c21 > 0])}')
    print(f'unweighted median w/o nevers: {statistics.median(c21.dropna()[c21 >= 0])}')
    print(f'weighted median w/o nevers: {ws.weighted_median(c21[~c21.isna()][c21 >= 0], weights=w21[~c21.isna()][c21 >= 0])}')
    print(f'unweighted median: {statistics.median(c21.dropna())}')
    print(f'weighted median: {ws.weighted_median(c21[~c21.isna()], weights=w21[~c21.isna()])}')
    print(f'already happened: {round(100*w21[c21 == 0].sum()/w21.sum(),1)}%')
    print(f'never happen: {round(100*w21[c21 == 10e100].sum()/w21.sum(),1)}%\n')
    
    print(f'{c} 2023')
    print(f'mean without outliers: {np.mean(reject_outliers(c23))}')
    c23 = reject_outliers(c23)
    c23[c23 == -1] = np.nan
    print(f'unweighted mean: {statistics.mean(c23.dropna())}')
    print(f'unweighted median w/o already/nevers: {statistics.median(c23.dropna()[c23 > 0])}')
    print(f'weighted median w/o already/nevers: {ws.weighted_median(c23[~c23.isna()][c23 > 0], weights=w23[~c23.isna()][c23 > 0])}')
    print(f'unweighted median w/o nevers: {statistics.median(c23.dropna()[c23 >= 0])}')
    print(f'weighted median w/o nevers: {ws.weighted_median(c23[~c23.isna()][c23 >= 0], weights=w23[~c23.isna()][c23 >= 0])}')
    c23[c23.isna()] = 10e100
    print(f'unweighted median: {statistics.median(c23.dropna())}')
    print(f'weighted median: {ws.weighted_median(c23[~c23.isna()], weights=w23[~c23.isna()])}')
    print(f'already happened: {100*round(w23[c23 == 0].sum()/w23.sum(),1)}%')
    print(f'never happen: {100*round(w23[c23 == 10e100].sum()/w23.sum(),1)}%\n')
    
    
    print(f'{c} 2021 to 2023')
    mt_stat, mt_p_value, median, table = median_test(c21.dropna(), c23.dropna())
    print(f'{c} median test p_value: {round(mt_p_value,3)}')
    
    mt_stat_wo, mt_p_value_wo, median_wo, table_wo = median_test(c21.dropna()[(c21 != 0) & (c21 != 10e100)], c23.dropna()[(c23 != 0) & (c23 != 10e100)])
    print(f'{c} median test p_value w/o already/nevers: {round(mt_p_value_wo,3)}')
    zt_already_happened_stat, zt_already_happened_p_value = proportions_ztest([w21[c21 == 0].sum(),w23[c23 == 0].sum()],
                                                                               [w21.sum(),w23.sum()])
    print(f'{c} z-test already happened p_value: {round(zt_already_happened_p_value,10)}')
    zt_never_happen_stat, zt_never_happen_p_value = proportions_ztest([w21[c21 == 10e100].sum(),w23[c23 == 10e100].sum()],
                                                                      [w21.sum(),w23.sum()])
    print(f'{c} z-test never happen p_value: {round(zt_never_happen_p_value,10)}\n')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_rows', 20)

# %% Demographic table

import numpy as np
import pandas as pd
import inspect
import re

d21, d23, d23s = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv'), pd.read_csv('AIMS_Supplement_2023.csv')

r = pd.DataFrame(index = ['Age', '18–34', '35–54', '55–',
                          'Gender', 'Female', 'Male',
                          'Region', 'Midwest', 'Northeast', 'South', 'West',
                          'Household Income','–$24,999', '$25,000–$49,999', '$50,000–$74,999', '$75,000–$99,999', '$100,000–',
                          'Ethnicity/Race', 'Asian', 'Black', 'Native American', 'White', 'Hispanic (any race)', 'Other',
                          'Education', 'Less than high school', 'High school', 'Some college', 'Associate', 'Bachelor’s', 'Post-graduate'
                         ],
                 columns = ['AIMS Main 2021 (UW)', 'AIMS Main 2021 (W)',
                            'AIMS Main 2023 (UW)', 'AIMS Main 2023 (W)',
                            'AIMS Supp 2023 (UW)', 'AIMS Supp 2023 (W)'
                           ]
                )

for d, col in [(d21, 'AIMS Main 2021'), (d23, 'AIMS Main 2023'), (d23s, 'AIMS Supp 2023')]:
    r.loc['Age', :] = ' '
    r.loc['Gender', :] = ' '
    r.loc['Region', :] = ' '
    r.loc['Household Income', :] = ' '
    r.loc['Ethnicity/Race', :] = ' '
    r.loc['Education', :] = ' '
    
    r.loc['18–34', col + ' (UW)'] = round(len(d[d['age'] < 35])/len(d), 2)
    r.loc['18–34', col + ' (W)'] = round(d['weight'][d['age'] < 35].sum()/d['weight'].sum(), 2)
    
    r.loc['35–54', col + ' (UW)'] = round(len(d[(d['age'] > 34) & (d['age'] < 55)])/len(d), 2)
    r.loc['35–54', col + ' (W)'] = round(d['weight'][(d['age'] > 34) & (d['age'] < 55)].sum()/d['weight'].sum(), 2)
    
    r.loc['55–', col + ' (UW)'] = round(len(d[d['age'] > 54])/len(d), 2)
    r.loc['55–', col + ' (W)'] = round(d['weight'][d['age'] > 54].sum()/d['weight'].sum(), 2)
    
    r.loc['Female', col + ' (UW)'] = round(len(d[d['gender'] == 'Female'])/len(d), 2)
    r.loc['Female', col + ' (W)'] = round(d['weight'][d['gender'] == 'Female'].sum()/d['weight'].sum(), 2)
    
    r.loc['Male', col + ' (UW)'] = round(len(d[d['gender'] == 'Male'])/len(d), 2)
    r.loc['Male', col + ' (W)'] = round(d['weight'][d['gender'] == 'Male'].sum()/d['weight'].sum(), 2)
    
    r.loc['Midwest', col + ' (UW)'] = round(len(d[d['region'] == 'Midwest'])/len(d), 2)
    r.loc['Midwest', col + ' (W)'] = round(d['weight'][d['region'] == 'Midwest'].sum()/d['weight'].sum(), 2)
    
    r.loc['Northeast', col + ' (UW)'] = round(len(d[d['region'] == 'Northeast'])/len(d), 2)
    r.loc['Northeast', col + ' (W)'] = round(d['weight'][d['region'] == 'Northeast'].sum()/d['weight'].sum(), 2)
    
    r.loc['South', col + ' (UW)'] = round(len(d[d['region'] == 'South'])/len(d), 2)
    r.loc['South', col + ' (W)'] = round(d['weight'][d['region'] == 'South'].sum()/d['weight'].sum(), 2)
    
    r.loc['West', col + ' (UW)'] = round(len(d[d['region'] == 'West'])/len(d), 2)
    r.loc['West', col + ' (W)'] = round(d['weight'][d['region'] == 'West'].sum()/d['weight'].sum(), 2)
    
    r.loc['–$24,999', col + ' (UW)'] = round(len(d[d['income_recode'] == '_24999'])/len(d), 2)
    r.loc['–$24,999', col + ' (W)'] = round(d['weight'][d['income_recode'] == '_24999'].sum()/d['weight'].sum(), 2)
    
    r.loc['$25,000–$49,999', col + ' (UW)'] = round(len(d[d['income_recode'] == '25000_49999'])/len(d), 2)
    r.loc['$25,000–$49,999', col + ' (W)'] = round(d['weight'][d['income_recode'] == '25000_49999'].sum()/d['weight'].sum(), 2)
    
    r.loc['$50,000–$74,999', col + ' (UW)'] = round(len(d[d['income_recode'] == '50000_74999'])/len(d), 2)
    r.loc['$50,000–$74,999', col + ' (W)'] = round(d['weight'][d['income_recode'] == '50000_74999'].sum()/d['weight'].sum(), 2)
    
    r.loc['$75,000–$99,999', col + ' (UW)'] = round(len(d[d['income_recode'] == '75000_99999'])/len(d), 2)
    r.loc['$75,000–$99,999', col + ' (W)'] = round(d['weight'][d['income_recode'] == '75000_99999'].sum()/d['weight'].sum(), 2)
    
    r.loc['$100,000–', col + ' (UW)'] = round(len(d[d['income_recode'] == '100000_'])/len(d), 2)
    r.loc['$100,000–', col + ' (W)'] = round(d['weight'][d['income_recode'] == '100000_'].sum()/d['weight'].sum(), 2)
    
    r.loc['Asian', col + ' (UW)'] = round(len(d[d['raceethnicity'] == 'Asian'])/len(d), 2)
    r.loc['Asian', col + ' (W)'] = round(d['weight'][d['raceethnicity'] == 'Asian'].sum()/d['weight'].sum(), 2)
    
    r.loc['Black', col + ' (UW)'] = round(len(d[d['raceethnicity'] == 'Black'])/len(d), 2)
    r.loc['Black', col + ' (W)'] = round(d['weight'][d['raceethnicity'] == 'Black'].sum()/d['weight'].sum(), 2)
    
    r.loc['Native American', col + ' (UW)'] = round(len(d[d['raceethnicity'] == 'Indigenous'])/len(d), 2)
    r.loc['Native American', col + ' (W)'] = round(d['weight'][d['raceethnicity'] == 'Indigenous'].sum()/d['weight'].sum(), 2)
    
    r.loc['White', col + ' (UW)'] = round(len(d[d['raceethnicity'] == 'White'])/len(d), 2)
    r.loc['White', col + ' (W)'] = round(d['weight'][d['raceethnicity'] == 'White'].sum()/d['weight'].sum(), 2)
    
    r.loc['Hispanic (any race)', col + ' (UW)'] = round(len(d[d['raceethnicity'] == 'Hispanic'])/len(d), 2)
    r.loc['Hispanic (any race)', col + ' (W)'] = round(d['weight'][d['raceethnicity'] == 'Hispanic'].sum()/d['weight'].sum(), 2)
    
    r.loc['Other', col + ' (UW)'] = round(len(d[d['raceethnicity'] == 'Other (Non-Hispanic)'])/len(d), 2)
    r.loc['Other', col + ' (W)'] = round(d['weight'][d['raceethnicity'] == 'Other (Non-Hispanic)'].sum()/d['weight'].sum(), 2)
    
    r.loc['Less than high school', col + ' (UW)'] = round(len(d[d['education_recode'] == 'less_than_high'])/len(d), 2)
    r.loc['Less than high school', col + ' (W)'] = round(d['weight'][d['education_recode'] == 'less_than_high'].sum()/d['weight'].sum(), 2)
    
    r.loc['High school', col + ' (UW)'] = round(len(d[d['education_recode'] == 'high'])/len(d), 2)
    r.loc['High school', col + ' (W)'] = round(d['weight'][d['education_recode'] == 'high'].sum()/d['weight'].sum(), 2)
    
    r.loc['Some college', col + ' (UW)'] = round(len(d[d['education_recode'] == 'some_college'])/len(d), 2)
    r.loc['Some college', col + ' (W)'] = round(d['weight'][d['education_recode'] == 'some_college'].sum()/d['weight'].sum(), 2)
    
    r.loc['Associate', col + ' (UW)'] = round(len(d[d['education_recode'] == 'associates'])/len(d), 2)
    r.loc['Associate', col + ' (W)'] = round(d['weight'][d['education_recode'] == 'associates'].sum()/d['weight'].sum(), 2)
    
    r.loc['Bachelor’s', col + ' (UW)'] = round(len(d[d['education_recode'] == 'bachelors'])/len(d), 2)
    r.loc['Bachelor’s', col + ' (W)'] = round(d['weight'][d['education_recode'] == 'bachelors'].sum()/d['weight'].sum(), 2)
    
    r.loc['Post-graduate', col + ' (UW)'] = round(len(d[d['education_recode'] == 'post_grad'])/len(d), 2)
    r.loc['Post-graduate', col + ' (W)'] = round(d['weight'][d['education_recode'] == 'post_grad'].sum()/d['weight'].sum(), 2)

pd.set_option('display.max_rows', None)    
display(r)
pd.set_option('display.max_rows', 20)

# %% Supplement stats

import csv
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import weightedstats as ws

with open('AIMS_Supplement_Codebook.csv', mode='r', encoding="utf8") as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

d = pd.read_csv('AIMS_Supplement_2023.csv')
r = pd.DataFrame(columns=['variable','summary_statistic','meaning'])
display(d.loc[1100:1108])

for c in d.columns:
    if (c[-5:] == 'aware') & ((c != 'selfaware') & (c != 'sitaware')):
        code = {'(1) definitely yes': 1, '(2) yes': 2, '(3) probably yes': 3, '(4) probably no': 4, '(5) no': 5, '(6) definitely no': 6, 'No opinion': np.nan}
        for i, x in d[c].items():
            d.loc[i, c] = code[x]
        r.loc[len(r)] = [c, (d[d[c].isin([1,2,3])]['weight']).sum() / (d[d[c].isin([1,2,3,4,5,6])]['weight']).sum(), '% yes']
    elif (c in ['AID4','upload2','chatbottrust','robottrust','gameAItrust','relativeLLMchatbottrust',
                ''
              ]) | c.startswith(('RS','SI','PMC','MCE','LLM','MCA','MCEn','SF')) & (c != 'SIemergence'):
        r.loc[len(r)] = [c, round((d[d[c].isin([5,6,7])]['weight']).sum() / (d[d[c].isin([1,2,3,5,6,7])]['weight']).sum(), 4)
                         , 'agreement'] 
    elif (d[c][0] in ['Yes', 'No', 'Not sure']):
        r.loc[len(r)] = [c, [round((d[d[c].isin(['Yes'])]['weight']).sum() / (d[d[c].isin(['Yes','No','Not sure'])]['weight']).sum(), 4),
                             round((d[d[c].isin(['No'])]['weight']).sum() / (d[d[c].isin(['Yes','No','Not sure'])]['weight']).sum(), 4),
                             round((d[d[c].isin(['Not sure'])]['weight']).sum() / (d[d[c].isin(['Yes','No','Not sure'])]['weight']).sum(), 4)
                            ]
        , 'yes, no, not sure']
    elif (d[c][0] in ["It's too fast.", "It's fine.", "It's too slow."]):
        r.loc[len(r)] = [c, [round((d[d[c].isin(["It's too fast."])]['weight']).sum() / (d[d[c].isin(["It's too fast.", "It's fine.", "It's too slow.", "Not sure"])]['weight']).sum(), 4),
                             round((d[d[c].isin(["It's fine."])]['weight']).sum() / (d[d[c].isin(["It's too fast.", "It's fine.", "It's too slow.", "Not sure"])]['weight']).sum(), 4),
                             round((d[d[c].isin(["It's too slow."])]['weight']).sum() / (d[d[c].isin(["It's too fast.", "It's fine.", "It's too slow.", "Not sure"])]['weight']).sum(), 4),
                             round((d[d[c].isin(["Not sure"])]['weight']).sum() / (d[d[c].isin(["It's too fast.", "It's fine.", "It's too slow.", "Not sure"])]['weight']).sum(), 4)
                            ]
        
        , 'it"s too fast, it"s fine, it"s too slow, not sure']
    elif c == 'upload1':
        r.loc[len(r)] = [c, (d[d[c] >= 4]['weight']).sum() / (d['weight']).sum(), 'support (>4 out of 7)']
    elif c == 'YG1':
        r.loc[len(r)] = [c, (d[d[c].isin(['Very concerned','Somewhat concerned'])]['weight']).sum() / (d['weight']).sum(), 'very or somewhat concerned']
    elif c == 'YG2':
        r.loc[len(r)] = [c, (d[d[c].isin(['Very likely','Somewhat likely'])]['weight']).sum() / (d['weight']).sum(), 'very or somewhat likely']
    elif c in ['YG3','YG4']:
        r.loc[len(r)] = [c, (d[d[c].isin(['Strongly support','Somewhat support'])]['weight']).sum() / (d['weight']).sum(), 'strongly or somewhat support']
    elif c in ['universe']:
        r.loc[len(r)] = [c, (d[d[c].isin(['One with humans'])]['weight']).sum() / (d['weight']).sum(), 'One with humans']
    elif c.endswith('trust') | (c in ['respect','admiration','compassion','awe','pride','excitement']):
        r.loc[len(r)] = [c, ws.weighted_mean(d[c]), 'mean (1-7 slider)']
    elif c.startswith('MP') | (c in ['selfaware','sitaware','power','ownmotives','owngoals','selfcontrol','understanding','upholding','safegoals','friendliness']):
        r.loc[len(r)] = [c, ws.weighted_mean(d[c]), 'mean (1-100 slider)']
    elif c.endswith('emergence'):
        never = (d['weight'][d[c] == -1]).sum() / d['weight'].sum()
        max_years = max(d[c])
        s = d[c].apply(lambda x: max_years + 1 if x == -1 else x)
        med = ws.weighted_median(s, weights=d['weight'])
        r.loc[len(r)] = [c, [never, med], '% never, median']

r['description'] = r['variable'].map(descriptions)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 200)
display(r)
pd.set_option('display.max_rows', 20)

# %% Extra main stats

import csv
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import weightedstats as ws

with open('AIMS_Main_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

d = pd.read_csv('AIMS_2023.csv')
r = pd.DataFrame(columns=['variable','summary_statistic','meaning'])

for c in d.columns:
    if (c[-5:] == 'aware') & ((c != 'selfaware') & (c != 'sitaware')):
        code = {'(1) definitely yes': 1, '(2) yes': 2, '(3) probably yes': 3, '(4) probably no': 4, '(5) no': 5, '(6) definitely no': 6, 'No opinion': np.nan}
        for i, x in d[c].items():
            d.loc[i, c] = code[x]
        r.loc[len(r)] = [c, (d[d[c].isin([1,2,3])]['weight']).sum() / (d[d[c].isin([1,2,3,4,5,6])]['weight']).sum(), '_ yes']
    elif (c in ['AID4','upload2','chatbottrust','robottrust','gameAItrust','relativeLLMchatbottrust',
                'SI1','SI2','SI3','SI4',
                'MCE1','MCE2','MCE3','MCE4','MCE5','MCE6','MCE7','MCE8','MCE9',
                'Norm',
                ''
              ]) | c.startswith(('RS','PMC','LLM','MCA','MCEn','SF')) & (c != 'SIemergence'):
        r.loc[len(r)] = [c, round((d[d[c].isin([5,6,7])]['weight']).sum() / (d[d[c].isin([1,2,3,5,6,7])]['weight']).sum(), 4)
                         , 'agreement'] 
    elif (d[c][0] in ['Yes', 'No', 'Not sure']):
        r.loc[len(r)] = [c, [round((d[d[c].isin(['Yes'])]['weight']).sum() / (d[d[c].isin(['Yes','No','Not sure'])]['weight']).sum(), 4),
                             round((d[d[c].isin(['No'])]['weight']).sum() / (d[d[c].isin(['Yes','No','Not sure'])]['weight']).sum(), 4),
                             round((d[d[c].isin(['Not sure'])]['weight']).sum() / (d[d[c].isin(['Yes','No','Not sure'])]['weight']).sum(), 4)
                            ]
        , 'yes, no, not sure']
    elif (d[c][0] in ["It's too fast.", "It's fine.", "It's too slow."]):
        r.loc[len(r)] = [c, [round((d[d[c].isin(["It's too fast."])]['weight']).sum() / (d[d[c].isin(["It's too fast.", "It's fine.", "It's too slow."])]['weight']).sum(), 4),
                             round((d[d[c].isin(["It's fine."])]['weight']).sum() / (d[d[c].isin(["It's too fast.", "It's fine.", "It's too slow."])]['weight']).sum(), 4),
                             round((d[d[c].isin(["It's too slow."])]['weight']).sum() / (d[d[c].isin(["It's too fast.", "It's fine.", "It's too slow."])]['weight']).sum(), 4)
                            ]
        
        , 'it"s too fast, it"s fine, it"s too slow']
    elif c == 'upload1':
        r.loc[len(r)] = [c, (d[d[c] >= 4]['weight']).sum() / (d['weight']).sum(), 'support (>4 out of 7)']
    elif c == 'YG1':
        r.loc[len(r)] = [c, (d[d[c].isin(['Very concerned','Somewhat concerned'])]['weight']).sum() / (d['weight']).sum(), 'very or somewhat concerned']
    elif c == 'YG2':
        r.loc[len(r)] = [c, (d[d[c].isin(['Very likely','Somewhat likely'])]['weight']).sum() / (d['weight']).sum(), 'very or somewhat likely']
    elif c in ['YG3','YG4']:
        r.loc[len(r)] = [c, (d[d[c].isin(['Strongly support','Somewhat support'])]['weight']).sum() / (d['weight']).sum(), 'strongly or somewhat support']
    elif c.endswith('trust') | (c in ['respect','admiration','compassion','awe','pride','excitement']):
        r.loc[len(r)] = [c, ws.weighted_mean(d[c], d['weight']), 'mean (1-7 slider)']
    elif c.startswith('MCE'):
        r.loc[len(r)] = [c, ws.weighted_mean(d[c], d['weight']), 'mean (1-5 slider)']
    elif c.startswith('SI'):
        r.loc[len(r)] = [c, ws.weighted_mean(d[c], d['weight']), 'mean (1-7 slider)']
    elif c.startswith('MP') | (c in ['selfaware','sitaware','power','ownmotives','owngoals','selfcontrol','understanding','upholding','safegoals','friendliness']):
        r.loc[len(r)] = [c, ws.weighted_mean(d[c], d['weight']), 'mean (1-100 slider)']
    elif c.endswith('emergence'):
        never = (d['weight'][d[c] == -1]).sum() / d['weight'].sum()
        max_years = max(d[c])
        s = d[c].apply(lambda x: max_years + 1 if x == -1 else x)
        med = ws.weighted_median(s, weights=d['weight'])
        r.loc[len(r)] = [c, [never, med], '% never, median (without "never")']

r['description'] = r['variable'].map(descriptions)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 200)
display(r)
pd.set_option('display.max_rows', 20)

# %% Pairwise comparisons

import csv
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with open('AIMS_Main_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

d = pd.read_csv('AIMS_2023.csv')

w = d['weight']
tsmc = d[['MCE21','MCE22','MCE23','MCE24','MCE25','MCE26','MCE27','MCE28','MCE29','MCE30','MCE31']]
tssc = d[['SI6','SI7','SI8','SI9','SI10','SI11','SI12','SI13','SI14','SI15','SI16']]

tsmc = tsmc[(tsmc.mul(w,axis=0).sum() / w.sum()).sort_values(ascending=False).index]
tssc = tssc[(tssc.mul(w,axis=0).sum() / w.sum()).sort_values(ascending=False).index]

def paired_weighted_ttest(x1, x2, w):
    diffs = x1 - x2
    mean = np.sum(diffs * w) / np.sum(w)
    var = np.sum(w * (diffs - mean)**2) / np.sum(w)
    t_stat = mean / np.sqrt(var / len(diffs))
    df = len(diffs) - 1
    p_value = stats.t.sf(np.abs(t_stat), df) * 2
    return p_value

tsmc_p_value_matrix = np.full((len(tsmc.columns), len(tsmc.columns)), np.nan)
tssc_p_value_matrix = np.full((len(tssc.columns), len(tssc.columns)), np.nan)

tsmc_main_p_values = []

for i in range(len(tsmc.columns)):
    for j in range(i+1, len(tsmc.columns)):
        col1, col2 = tsmc.columns[i], tsmc.columns[j]
        p_value = paired_weighted_ttest(tsmc[col1], tsmc[col2], w)
        if j == (i + 1):
            tsmc_main_p_values.append(p_value)
        tsmc_p_value_matrix[i, j] = p_value

tsmc_main_p_values = fdrcorrection(tsmc_main_p_values)[1]

tssc_main_p_values = []

for i in range(len(tssc.columns)):
    for j in range(i+1, len(tssc.columns)):
        col1, col2 = tssc.columns[i], tssc.columns[j]
        p_value = paired_weighted_ttest(tssc[col1], tssc[col2], w)
        if j == (i + 1):
            tssc_main_p_values.append(p_value)
        tssc_p_value_matrix[i, j] = p_value

tssc_main_p_values = fdrcorrection(tssc_main_p_values)[1]

pd.set_option('display.float_format', lambda x: '%.4f' % x)

tsmc_p_value_table = pd.DataFrame(tsmc_p_value_matrix, index=[descriptions[x][15:-12] for x in tsmc.columns], columns=[descriptions[x][15:-12] for x in tsmc.columns])
for i in range(len(tsmc.columns) - 1):
    j = i + 1
    tsmc_p_value_table.loc[descriptions[tsmc.columns[i]][15:-12],
                           descriptions[tsmc.columns[j]][15:-12]] = f'{round(tsmc_p_value_matrix[i, j],4)}, ({round(tsmc_main_p_values[i], 4)})'
display(tsmc_p_value_table)

tssc_p_value_table = pd.DataFrame(tssc_p_value_matrix, index=[descriptions[x][15:-12] for x in tssc.columns], columns=[descriptions[x][15:-12] for x in tssc.columns])
for i in range(len(tssc.columns) - 1):
    j = i + 1
    tssc_p_value_table.loc[descriptions[tssc.columns[i]][15:-12],
                           descriptions[tssc.columns[j]][15:-12]] = f'{round(tssc_p_value_matrix[i, j],4)}, ({round(tssc_main_p_values[i], 4)})'
display(tssc_p_value_table)

# %% Sentience today and sentience possible

import csv
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import weightedstats as ws

with open('AIMS_Main_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

d = pd.read_csv('AIMS_2023.csv')
r = pd.DataFrame(columns=['name','value'])

r.loc[len(r)] = ['sentience today out of sentience possible', (d['weight'][(d['F11'] == 'Yes') & (d['F1'] == 'Yes')].sum() / d['weight'][d['F11'] == 'Yes'].sum())]
r.loc[len(r)] = ['sentience possible out of sentience today', (d['weight'][(d['F11'] == 'Yes') & (d['F1'] == 'Yes')].sum() / d['weight'][d['F1'] == 'Yes'].sum())]

display(r)

# %% Sentience pie chart data

import numpy as np
import pandas as pd

d21, d23m, d23s = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv'), pd.read_csv('AIMS_Supplement_2023.csv')

r = pd.DataFrame(columns=['variable','Yes','No','Not sure'])

r = pd.concat([r,pd.DataFrame({'variable':'d21_F1',
                               'Yes':sum(d21['weight'][d21['F1']=='Yes']) / sum(d21['weight']),
                               'No':sum(d21['weight'][d21['F1']=='No']) / sum(d21['weight']),
                               'Not sure':sum(d21['weight'][d21['F1']=='Not sure']) / sum(d21['weight']),
                              },index=[0])],ignore_index=True)

r = pd.concat([r,pd.DataFrame({'variable':'d21_F11',
                               'Yes':sum(d21['weight'][d21['F11']=='Yes']) / sum(d21['weight']),
                               'No':sum(d21['weight'][d21['F11']=='No']) / sum(d21['weight']),
                               'Not sure':sum(d21['weight'][d21['F11']=='Not sure']) / sum(d21['weight']),
                              },index=[0])],ignore_index=True)

r = pd.concat([r,pd.DataFrame({'variable':'d23m_F1',
                               'Yes':sum(d23m['weight'][d23m['F1']=='Yes']) / sum(d23m['weight']),
                               'No':sum(d23m['weight'][d23m['F1']=='No']) / sum(d23m['weight']),
                               'Not sure':sum(d23m['weight'][d23m['F1']=='Not sure']) / sum(d23m['weight']),
                              },index=[0])],ignore_index=True)

r = pd.concat([r,pd.DataFrame({'variable':'d23m_F11',
                               'Yes':sum(d23m['weight'][d23m['F11']=='Yes']) / sum(d23m['weight']),
                               'No':sum(d23m['weight'][d23m['F11']=='No']) / sum(d23m['weight']),
                               'Not sure':sum(d23m['weight'][d23m['F11']=='Not sure']) / sum(d23m['weight']),
                              },index=[0])],ignore_index=True)

r = pd.concat([r,pd.DataFrame({'variable':'d23s_F1',
                               'Yes':sum(d23s['weight'][d23s['F1']=='Yes']) / sum(d23s['weight']),
                               'No':sum(d23s['weight'][d23s['F1']=='No']) / sum(d23s['weight']),
                               'Not sure':sum(d23s['weight'][d23s['F1']=='Not sure']) / sum(d23s['weight']),
                              },index=[0])],ignore_index=True)

r = pd.concat([r,pd.DataFrame({'variable':'d23s_F11',
                               'Yes':sum(d23s['weight'][d23s['F11']=='Yes']) / sum(d23s['weight']),
                               'No':sum(d23s['weight'][d23s['F11']=='No']) / sum(d23s['weight']),
                               'Not sure':sum(d23s['weight'][d23s['F11']=='Not sure']) / sum(d23s['weight']),
                              },index=[0])],ignore_index=True)

r = pd.concat([r,pd.DataFrame({'variable':'d23s_chatGPTsentient',
                               'Yes':sum(d23s['weight'][d23s['chatGPTsentient']=='Yes']) / sum(d23s['weight']),
                               'No':sum(d23s['weight'][d23s['chatGPTsentient']=='No']) / sum(d23s['weight']),
                               'Not sure':sum(d23s['weight'][d23s['chatGPTsentient']=='Not sure']) / sum(d23s['weight']),
                              },index=[0])],ignore_index=True)

r.to_csv('AIMS_Pie_Chart_Data.csv',index=False)
r

# %% Forecast bar chart data

import numpy as np
import pandas as pd

d21, d23m, d23s = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv'), pd.read_csv('AIMS_Supplement_2023.csv')

r = pd.DataFrame(columns=['Forecast','s21','s23','agi','hlai','si'])

r.loc[len(r)] = ['Already happened',
                 (d21['weight'][d21['F2'] == 0]).sum() / d21['weight'].sum(),
                 (d23m['weight'][d23m['F2'] == 0]).sum() / d23m['weight'].sum(),
                 (d23s['weight'][d23s['emergence'] == 0]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][d23s['HLAIemergence'] == 0]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][d23s['SIemergence'] == 0]).sum() / d23s['weight'].sum()]
r.loc[len(r)] = ['1-4',
                 (d21['weight'][d21['F2'] <= 4]).sum() / d21['weight'].sum(),
                 (d23m['weight'][d23m['F2'] <= 4]).sum() / d23m['weight'].sum(),
                 (d23s['weight'][d23s['emergence'] <= 4]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][d23s['HLAIemergence'] <= 4]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][d23s['SIemergence'] <= 4]).sum() / d23s['weight'].sum()]
r.loc[len(r)] = ['5-9',
                 (d21['weight'][(d21['F2'] >= 5) & (d21['F2'] <= 9)]).sum() / d21['weight'].sum(),
                 (d23m['weight'][(d23m['F2'] >= 5) & (d23m['F2'] <= 9)]).sum() / d23m['weight'].sum(),
                 (d23s['weight'][(d23s['emergence'] >= 5) & (d23s['emergence'] <= 9)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['HLAIemergence'] >= 5) & (d23s['HLAIemergence'] <= 9)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['SIemergence'] >= 5) & (d23s['SIemergence'] <= 9)]).sum() / d23s['weight'].sum()]
r.loc[len(r)] = ['10-49',
                 (d21['weight'][(d21['F2'] >= 10) & (d21['F2'] <= 49)]).sum() / d21['weight'].sum(),
                 (d23m['weight'][(d23m['F2'] >= 10) & (d23m['F2'] <= 49)]).sum() / d23m['weight'].sum(),
                 (d23s['weight'][(d23s['emergence'] >= 10) & (d23s['emergence'] <= 49)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['HLAIemergence'] >= 10) & (d23s['HLAIemergence'] <= 49)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['SIemergence'] >= 10) & (d23s['SIemergence'] <= 49)]).sum() / d23s['weight'].sum()]
r.loc[len(r)] = ['50-99',
                 (d21['weight'][(d21['F2'] >= 50) & (d21['F2'] <= 99)]).sum() / d21['weight'].sum(),
                 (d23m['weight'][(d23m['F2'] >= 50) & (d23m['F2'] <= 99)]).sum() / d23m['weight'].sum(),
                 (d23s['weight'][(d23s['emergence'] >= 50) & (d23s['emergence'] <= 99)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['HLAIemergence'] >= 50) & (d23s['HLAIemergence'] <= 99)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['SIemergence'] >= 50) & (d23s['SIemergence'] <= 99)]).sum() / d23s['weight'].sum()]
r.loc[len(r)] = ['100+',
                 (d21['weight'][(d21['F2'] >= 100)]).sum() / d21['weight'].sum(),
                 (d23m['weight'][(d23m['F2'] >= 100)]).sum() / d23m['weight'].sum(),
                 (d23s['weight'][(d23s['emergence'] >= 100)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['HLAIemergence'] >= 100)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['SIemergence'] >= 100)]).sum() / d23s['weight'].sum()]
r.loc[len(r)] = ['Will never happen',
                 (d21['weight'][(d21['F2'] == -1)]).sum() / d21['weight'].sum(),
                 (d23m['weight'][(d23m['F2'] == -1)]).sum() / d23m['weight'].sum(),
                 (d23s['weight'][(d23s['emergence'] == -1)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['HLAIemergence'] == -1)]).sum() / d23s['weight'].sum(),
                 (d23s['weight'][(d23s['SIemergence'] == -1)]).sum() / d23s['weight'].sum()]

r.to_csv('AIMS_Forecasts_for_Chart.csv', index=False)
display(r)

# %% Policy agreement data

import numpy as np
import pandas as pd

d21, d23m, d23s = pd.read_csv('AIMS_2021.csv'), pd.read_csv('AIMS_2023.csv'), pd.read_csv('AIMS_Supplement_2023.csv')

r = pd.DataFrame(columns=['Statement','Strongly agree','Agree','Somewhat agree','No opinion','Somewhat disagree','Disagree','Strongly disagree'])

with open('AIMS_Main_Codebook.csv', mode='r', encoding="utf8") as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

for s in ['PMC1','PMC3','PMC4','PMC9']:
    r.loc[len(r)] = [descriptions[s]] + [((d23m['weight'][d23m[s] == i]).sum() / d23m['weight'].sum()) for i in reversed(range(1,8))]

with open('AIMS_Supplement_Codebook.csv', mode='r', encoding="utf8") as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

for s in ['PMC9','RS5','RS7','RS8']:
    r.loc[len(r)] = [descriptions[s]] + [((d23s['weight'][d23s[s] == i]).sum() / d23s['weight'].sum()) for i in reversed(range(1,8))]

r = r.sort_values(by=['Strongly agree'])
r.to_csv('AIMS_Policy_Agreement_Data_for_Chart.csv', index=False)
display(r)

# %% Confidence intervals

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import csv
import re
from scipy.stats import zscore
from scipy.stats import norm
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import weightedstats as ws

with open('AIMS_Main_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

with open('AIMS_Supplement_Codebook.csv', mode='r', encoding="utf8") as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

dm = pd.read_csv('AIMS_2023.csv')
ds = pd.read_csv('AIMS_Supplement_2023.csv')

r = pd.DataFrame(columns=['variable','description','general','sentient','diff','lower_ci','upper_ci'])

for c in ['MCE5','MCE1','PMC10','MCE3','PMC8','PMC11',
          'MCE2','MCE6','MCE4','PMC12','PMC9']:
    
    pos_m = dm[dm[c].isin([5,6,7])]['weight'].sum()
    pos_s = ds[ds[c].isin([5,6,7])]['weight'].sum()
    tot_m = dm[dm[c].isin([1,2,3,5,6,7])]['weight'].sum()
    tot_s = ds[ds[c].isin([1,2,3,5,6,7])]['weight'].sum()

    agree_m = pos_m / tot_m
    agree_s = pos_s / tot_s

    var_m = agree_m * (1 - agree_m) / tot_m
    var_s = agree_s * (1 - agree_s) / tot_s
    var_diff = var_m + var_s

    moe = norm.ppf(0.975) * np.sqrt(var_diff)
    lower_ci = agree_m - agree_s - moe
    upper_ci = agree_m - agree_s + moe

    r.loc[len(r)] = [c,
                     descriptions[c],
                     round(agree_s,3),
                     round(agree_m,3),
                     round(agree_m - agree_s,3),
                     round(lower_ci,3),
                     round(upper_ci,3)]

display(r)

# %% Example participant variation

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import csv
import re
from scipy.stats import zscore
from scipy.stats import norm
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import weightedstats as ws

with open('AIMS_Main_Codebook.csv', mode='r') as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

with open('AIMS_Supplement_Codebook.csv', mode='r', encoding="utf8") as infile:
    reader = csv.reader(infile)
    descriptions = {rows[0]:rows[1] for rows in reader}

dm = pd.read_csv('AIMS_2023.csv')
ds = pd.read_csv('AIMS_Supplement_2023.csv')

# Test for the difference in agreement with the MCE questions between old (50+) and young (18-35) participants

r = pd.DataFrame(columns=['variable','description','overall','young','old'])

r.loc[len(r)] = ['F1',
                 descriptions['F1'],
                 [
                  round((dm[dm['F1'].isin(['Yes'])]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure'])]['weight']).sum(), 4),
                  round((dm[dm['F1'].isin(['No'])]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure'])]['weight']).sum(), 4),
                  round((dm[dm['F1'].isin(['Not sure'])]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure'])]['weight']).sum(), 4)
                 ],
                 [
                  round((dm[dm['F1'].isin(['Yes']) & (dm['age'] < 36)]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure']) & (dm['age'] < 36)]['weight']).sum(), 4),
                  round((dm[dm['F1'].isin(['No']) & (dm['age'] < 36)]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure']) & (dm['age'] < 36)]['weight']).sum(), 4),
                  round((dm[dm['F1'].isin(['Not sure']) & (dm['age'] < 36)]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure']) & (dm['age'] < 36)]['weight']).sum(), 4)
                 ],
                 [
                  round((dm[dm['F1'].isin(['Yes']) & (dm['age'] > 54)]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure']) & (dm['age'] > 54)]['weight']).sum(), 4),
                  round((dm[dm['F1'].isin(['No']) & (dm['age'] > 54)]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure']) & (dm['age'] > 54)]['weight']).sum(), 4),
                  round((dm[dm['F1'].isin(['Not sure']) & (dm['age'] > 54)]['weight']).sum() / (dm[dm['F1'].isin(['Yes','No','Not sure']) & (dm['age'] > 54)]['weight']).sum(), 4)
                 ]
                ]

display(r)