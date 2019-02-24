import pandas as pd
from sklearn.utils import shuffle
import import_data

def output_test():
    data = import_data.import_data('Align_Pixel_RGB1.csv')
    data = shuffle(data)
    train_size=100000
    test_size=3000

    print('--------------start split data----------------')

    data_test = data[-test_size:]
    data_test.to_csv('Align_Pixel_test.csv',index=None)


def combine_RGB():
    R_df=pd.read_table('input/Align_Pixel_R.txt')
    G_df=pd.read_table('input/Align_Pixel_G.txt')
    B_df=pd.read_table('input/Align_Pixel_B.txt')
    label_df=pd.read_table('input/行列label.txt')

    print ('删除R中的angle ,G,B中的row,col,angle')
    R_df.drop(['row', 'col','angle'],axis=1,inplace=True)
    G_df.drop(['row', 'col','angle'],axis=1,inplace=True)
    B_df.drop(['row', 'col','angle'],axis=1,inplace=True)
    R_df.rename_axis(dict(zip(R_df.columns.tolist(),['R'+col_name for col_name in R_df.columns])),axis=1,inplace=True)
    G_df.rename_axis(dict(zip(G_df.columns.tolist(),['G'+col_name for col_name in G_df.columns])),axis=1,inplace=True)
    B_df.rename_axis(dict(zip(B_df.columns.tolist(),['B'+col_name for col_name in B_df.columns])),axis=1,inplace=True)
    for df in [R_df,G_df,B_df]:
        for col_name in df.columns:
            print('process',col_name)
            df[col_name].astype('int16')
    labels=label_df['o_label']
    print ('将RGB三种颜色合并后到一起并保存...')
    concat_data=pd.concat([R_df,G_df,B_df,labels],axis=1)

    concat_data.to_csv('Align_Pixel_RGB1.csv',index=None)

def combine_Gray():

    Gray_df=pd.read_table('input/Align_Pixel_GRAY.txt')

    label_df = pd.read_table('input/行列label.txt')

    Gray_df.drop(['row', 'col', 'angle'], axis=1, inplace=True)

    labels = label_df['o_label']

    concat_data = pd.concat([Gray_df, labels], axis=1)

    concat_data.to_csv('Align_Pixel_Gray_.csv', index=None)

output_test()