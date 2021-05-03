
import pandas as pd
import model

# ========================================
#             Import Dataset
# ========================================
user = 'postgres'
password = 'Komaeda'

from sqlalchemy import create_engine
engine = create_engine('postgresql://'+user+':'+password+'@localhost/news')

Query = "SELECT * FROM redditcomment"
comment_table = pd.read_sql_query(Query, con=engine)
Query = "SELECT * FROM redditnews"
news_table = pd.read_sql_query(Query, con=engine)
Query = "SELECT * FROM factcheck"
factcheck = pd.read_sql_query(Query, con=engine)


# ========================================
#                 Main
# ========================================

def main():
    sample = news_table.sample(n=100).reset_index(drop=True)

    text = sample.text.values
    label = sample.label.values

    attention_masks, input_ids = model.vectorize(text) # tokenization + vectorization

    model.train(attention_masks, input_ids,label) # train modle


if __name__ == '__main__':
    main()
