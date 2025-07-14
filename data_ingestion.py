import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
     
    df.drop(columns=['tweet_id'], inplace=True)
    df = df[df['sentiment'].isin(['happiness', 'sadness'])]
    df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs("data/raw", exist_ok=True)
    train.to_csv("data/raw/train.csv", index=False)
    test.to_csv("data/raw/test.csv", index=False)
    print("✅ Done")

if __name__ == '__main__':
    main()


