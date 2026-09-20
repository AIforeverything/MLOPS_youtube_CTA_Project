import pandas as pd

def remove_null(df:pd.DataFrame)->pd.DataFrame:
    df1=df.copy()
    df1.dropna(inplace=True)
    print("Total null values removed: ",(len(df)-len(df1)))
    print("percent of null values removed: ",((len(df)-len(df1))*100/len(df)))
    return df1

if __name__=="__main__":
    remove_null(df=pd.DataFrame())