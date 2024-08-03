import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# tmp=pd.DataFrame({'a':range(0,10),'b':range(10,20)})
# fig=px.scatter(tmp,x='a',y='b')
# path='images/test.png'
# print('Path being used: ',path)
# fig.write_image(path)
# plotly.offline.plot(fig,filename=path,auto_open=False,image='png',output_type='file')


#Plot a heatmap from consuion matrix

# #create y_test and y_pred from a list
# data_tmp = [1, 2, 3,1,1,2,3,1,1,2,3,2,2,2,1,1,1,1]
# y_test_example = pd.Series(data_tmp, copy=False)
# data_tmp = [1, 2, 3,1,2,2,3,1,1,1,3,2,3,2,3,1,1,1]
# y_pred_example = pd.Series(data_tmp, copy=False)

#plot heatmap
df=pd.read_csv('train_test_set_comprehensive.csv')
kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}
df.columns


#CONFUSION MATRIX TEST
preds=df[df.sample_type=='test']['pred_rain_occurrence'].copy()
actuals=df[df.sample_type=='test']['rain_occurrence'].copy()

cf_matrix = confusion_matrix(actuals, preds)
tmp = pd.DataFrame(cf_matrix).transpose()

loc_labels=np.unique(actuals.to_list())
fig=sns.heatmap(cf_matrix, cmap='Reds', xticklabels=loc_labels, yticklabels=loc_labels, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('Confusion matrix TEST set\n model')
print("fig created")
path='images/confusion_matrix_test.png'
plt.savefig(path)
print("fig saved to: ", path)
plt.close('all')

#CONFUSION MATRIX TRAIN
preds=df[df.sample_type=='train']['pred_rain_occurrence'].copy()
actuals=df[df.sample_type=='train']['rain_occurrence'].copy()


cf_matrix = confusion_matrix(actuals, preds)
tmp = pd.DataFrame(cf_matrix).transpose()

loc_labels=np.unique(actuals.to_list())
fig=sns.heatmap(cf_matrix, cmap='Blues', xticklabels=loc_labels, yticklabels=loc_labels, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('Confusion matrix TRAIN set\n model')
print("fig created")
path='images/confusion_matrix_train.png'
plt.savefig(path)
print("fig saved to: ", path)
plt.close('all')


