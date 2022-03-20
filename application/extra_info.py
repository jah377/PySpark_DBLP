import duckdb
import json
import pandas as pd
import unidecode
from typing import Any, Callable, Dict, List, Union
from constants import DATA_PATH


def get_json_dict(path: str) -> Dict:
    """Loads the json information at the given patha and returns
    the dictionary with the data.

    Args:
        path (str): path of the JSON to laod

    Returns:
        Dict: dictionary with JSON information
    """
    with open(path) as file:
        data = json.load(file)

    return data


def load_extra_information_from_jsons() -> List[Union[Dict, List[Dict]]]:
    """Loads the additional information for the book title, full book title,
    journal, full journal, and type.

    Returns:
        List[Union(Dict, List[Dict])]: List of extra data in dictionaries
    """
    jsons_data = []

    # Load book title info
    title_info = get_json_dict(f"{DATA_PATH}/pbooktitle.json")
    title_info["name"] = {int(k): unidecode.unidecode(v).strip()
                          if v is not None else v
                          for k, v in title_info["name"].items()}
    title_info["name"][0] = None
    jsons_data.append(title_info["name"])

    # Load book title full info
    title_full_info = get_json_dict(f"{DATA_PATH}/pbooktitlefull.json")
    title_full_info = [{**data,
                        "name": unidecode.unidecode(data["name"]).strip()}
                       if data["name"] is not None else data
                       for data in title_full_info]
    jsons_data.append(title_full_info)

    # Load journal info
    journal_info = get_json_dict(f"{DATA_PATH}/pjournal.json")
    journal_info = [{**data,
                     "name": unidecode.unidecode(data["name"]).strip()}
                    if data["name"] is not None else data
                    for data in journal_info]
    jsons_data.append(journal_info)

    # Load journal full info
    journal_full_info = get_json_dict(f"{DATA_PATH}/pjournalfull.json")
    journal_full_info = [{**data,
                          "name": unidecode.unidecode(data["name"]).strip()}
                         if data["name"] is not None else data
                         for data in journal_full_info]
    jsons_data.append(journal_full_info)

    # Load type info
    type_info = get_json_dict(f"{DATA_PATH}/ptype.json")
    type_info["name"] = {int(k): unidecode.unidecode(v).strip()
                         for k, v in type_info["name"].items()}
    jsons_data.append(type_info["name"])

    return jsons_data


def upload_duckdb(data_list: List[Union[Dict, List[Dict]]],
                  table_names: List[str],
                  con: duckdb.DuckDBPyConnection) -> None:
    """Uploads the list of data to DuckDB with the given table names and
    connection.

    Args:
        data_list (List[Union): List with data to upload
        table_names (List[str]): List of table names
        con (duckdb.DuckDBPyConnection): DuckDB connection
    """
    for name, data in zip(table_names, data_list):
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        else:
            df = pd.DataFrame(data)

        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {name}(
                id int,
                name varchar,
                PRIMARY KEY(id))
        """)
        con.append(name, df)


def upload_extra_information(extra_data_function: Callable[[Any], Any],
                             upload_function: Callable[[Any], None], *args) -> None:
    """Uploads the returned extra data with the given upload function.

    Args:
        extra_data_function (Callable[[Any], Any]): Function retrieving extra data
        upload_function (Callable[[Any], None]): Function uploading extra data
    """
    data = extra_data_function()
    upload_function(data, *args)
