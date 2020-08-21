import uuid
import logging
import psycopg2
import numpy as np
import psycopg2.extras


def db_connect(host, port, name, user, password):
    """
    db_connect
    The function is used to connect to Postgresql database
    :param host: str
    :param port: int
    :param name: str
    :param user: str
    :param password: str
    :return: connection, db
    """
    try:
        connection = psycopg2.connect(user=user,
                                      password=password,
                                      host=host,
                                      port=port,
                                      database=name)
        db = connection.cursor()

        # for debugging, printn DB version
        db.execute("SELECT version();")
        record = db.fetchone()
        #print("DB connection OK\n")

        # enable uuid extension
        psycopg2.extras.register_uuid()

    except (Exception, psycopg2.Error) as error :
        logging.info("Error while connecting to PostgreSQL" + str(error))
        print("Error while connecting to PostgreSQL", error)
        return None, None
    return connection, db


def read_descriptors(db):
    """
    read_descriptors
    The function for reading all face descriptors and ids from database
    :param db: cursor
    :return: {} or [], [], []
    """
    # check connection
    try:
        db
    except NameError:
        print('Problem with the database connection')  # db cursor does not exist at all
        return -1

    query = 'SELECT f."ID", f.descriptor, f.personid FROM public.faces f'
    db.execute(query)
    records = db.fetchall()

    ids = []
    descriptors = []
    persons_ids = []

    if records:

        for r in records:
            ids.append(r[0])
            descriptors.append(r[1][0])
            persons_ids.append(r[2])

        return ids, descriptors, persons_ids
    else:
        return {'status': 'ERROR'}


def receive_descriptors(db, db_conn, name, embeds) -> dict:
    """
    receive_descriptors
    The function for writing records in database
    :param db: cursor
    :param db_conn: connection
    :param name: str
    :param embeds: np.array([])
    :return: dict
    """
    query = 'INSERT INTO public.persons (name) VALUES (\'' + str(name) + '\') RETURNING "ID";'
    db.execute(query)
    records = db.fetchall()
    if records[0][0] != '':
        person_id = records[0][0]

    for emb in embeds:
        emb = np.array(emb).tolist()
        query = 'INSERT INTO public.faces (descriptor, personid) VALUES (%s, %s);'
        db.execute(query, (emb, (person_id,)))
    db_conn.commit()
    return {'status': 'SUCCESS'}


def find_person_by_id(db, person_id) -> dict:
    """
    find_person_by_id
    The function for search the database for name
    :param db: cursor
    :param person_id: str
    :return: dict
    """
    query = 'SELECT p.name FROM public.persons p WHERE "ID" = %s;'
    db.execute(query, (person_id,))
    records = db.fetchall()
    if records[0][0] is not None:
        return {
            'status': 'SUCCESS',
            'name': str(records[0][0])
        }
    else:
        return{'status': 'ERROR'}
