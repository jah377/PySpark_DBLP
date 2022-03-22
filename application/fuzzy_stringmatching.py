from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer, NGram, HashingTF, MinHashLSH, RegexTokenizer, SQLTransformer



def calc_jaccard(key_df, data_df, column:str):
    """
    model, transform, and calculate jaccard distance

    Args:
        key_df (pyspark.sql.dataframe.DataFrame):  dataframe containing keys used to identify rows to be compared
        data_df (pyspark.sql.dataframe.DataFrame): dataframe containing dblp data
        column (str): name of feature to be compared
        
    Output:
        pyspark.sql.dataframe.DataFrame containing _c1 (pkey1), _c2 (pkey2), and jaccard distance of a given column
    """
    
    # create model
    model = Pipeline(stages=[
        SQLTransformer(statement=f"SELECT *, lower({column}) lower FROM __THIS__"),
        Tokenizer(inputCol="lower", outputCol="token"),
        StopWordsRemover(inputCol="token", outputCol="stop"),
        SQLTransformer(statement="SELECT *, concat_ws(' ', stop) concat FROM __THIS__"),
        RegexTokenizer(pattern="", inputCol="concat", outputCol="char", minTokenLength=2), # len(token)>2
        NGram(n=2, inputCol="char", outputCol="ngram"),
        HashingTF(inputCol="ngram", outputCol="vector"),
        MinHashLSH(inputCol="vector", outputCol="lsh", numHashTables=3)
    ]).fit(data_df)
    
    # transform data 
    transformed = model\
                    .transform(data_df.select(['pkey',column]))\
                    .select('pkey', column, 'concat', 'char', 'ngram', 'vector', 'lsh')
                    
    # create tables to compare based on keys in key_df
    table1 = key_df.select('_c1').join(transformed, transformed.pkey==key_df._c1, 'left').drop('pkey')
    table2 = key_df.select('_c2').join(transformed, transformed.pkey==key_df._c2, 'left').drop('pkey')

    # calculate jaccardDist
    ## calculates distance from entire row of Tokens, NGram, HashingTF, and MinHashLSH transformations
    results = model.stages[-1].approxSimilarityJoin(table1, table2, 0.5, 'jaccardDist')
    
    return results.select('datasetA._c1', 'datasetB._c2', 'jaccardDist')
