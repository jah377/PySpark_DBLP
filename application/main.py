import duckdb
from data import (create_train_table, get_and_upload_clean_train_data,
                  get_train_with_extra, load_extra_information_from_jsons,
                  upload_duckdb, upload_extra_information)
from utils import start_spark

if __name__ == "__main__":
    con = duckdb.connect(database=":memory:")
    table_names = ["BOOKTITLE", "BOOKTITLEFULL",
                   "JOURNAL", "JOURNALFULL", "TYPE"]
    upload_extra_information(load_extra_information_from_jsons,
                             upload_duckdb, table_names, con)
    create_train_table(con)

    spark = start_spark()

    data = get_and_upload_clean_train_data(con, spark)

    # This query retrieves the extra data from the JSONs
    print(get_train_with_extra(con))
