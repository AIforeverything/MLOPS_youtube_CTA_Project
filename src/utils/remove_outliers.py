import pandas as pd
# removing the outliers using IQR method
def remove_outliers(df:pd.DataFrame,columns: list[str])->pd.DataFrame:
    """ Removing outliers using IQR method."""
    df_cleaned= df.copy()
    for column in columns:
        q1= df_cleaned[column].quantile(0.25)
        q3= df_cleaned[column].quantile(0.75)
        
        iqr= q3-q1
        
        lower_bound= q1-1.5*iqr
        upper_bound= q3+1.5*iqr
        
        df_cleaned=df_cleaned[df_cleaned[column].between(lower_bound,upper_bound)]
    print("Rows removed: ",len(df)-len(df_cleaned)) 
    print("percent of Rows removed: ",(len(df)-len(df_cleaned))*100/len(df)) 
    
    # return df_cleaned.reset_index(drop=True)
    return df_cleaned.reset_index(drop=True)