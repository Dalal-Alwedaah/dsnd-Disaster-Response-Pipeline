import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Load messages and categories dataset '''
    messages =  pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')
    return df


def clean_data(df):
    ''' cleaning data '''
    # split categories into separate category columns
    categories = df.categories.str.split(";", expand=True)
    row = categories.loc[0,:]
    category_colnames = row.str.split("-").str[0]
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
    df = df.drop('categories', 1)
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])
    
    # drop duplicates
    df=df.drop_duplicates(keep='first')
    
    return df
    
   


def save_data(df, database_filename):
    ''' save the clean dataset into an sqlite database '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()