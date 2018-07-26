"""
Purpose of this script is to show that the eventually we will be 
checking aginst a database. 
"""

import sqlite3
import logging
from typing import Union

models_to_image = [("mnist", "simonmok/scalabel-mnist")]


def get_image_from_model(model_name: str) -> Union[None, str]:
    conn = sqlite3.connect("model.db")
    c.execute(
        """
    SELECT image FROM models WHERE name=?""",
        (model_name,),
    )
    result = c.fetchall()
    if len(result) == 0:
        return None
    else:
        return result[0][0]
    conn.close()


if __name__ == "__main__":

    conn = sqlite3.connect("model.db")

    logging.info("Creating models table.")
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE models
    (name text, image text)"""
    )

    logging.info("Start inserting values.")
    for model_name, image_name in models_to_image:
        c.execute(
            f"""
        INSERT INTO models VALUES
        ('{model_name}', '{image_name}')"""
        )

    conn.commit()
    conn.close()

