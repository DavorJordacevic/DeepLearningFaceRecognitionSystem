import uuid
import logging
import psycopg2
import numpy as np
import psycopg2.extras

# dbConnect function is used to connect to Postgresql database
def dbConnect(host, port, name, user, password):
    try:
        # user, password, host, port, name are from the config file
        connection = psycopg2.connect(user = user,
                                      password = password,
                                      host = host,
                                      port = port,
                                      database = name)
        db = connection.cursor()
        # Print PostgreSQL Connection properties
        #print(connection.get_dsn_parameters(),"\n")

        # Print PostgreSQL version
        db.execute("SELECT version();")
        record = db.fetchone()
        #print("You are connected to - ", record,"\n")
        #print("DB connection OK\n")

        psycopg2.extras.register_uuid()

    except (Exception, psycopg2.Error) as error :
        logging.info("Error while connecting to PostgreSQL" + str(error))
        print("Error while connecting to PostgreSQL", error)
    '''
    finally:
        # used only for testing purposs
        # closing database connection.
        if(connection):
            db.close()
            connection.close()
            print("PostgreSQL connection is closed")
    '''
    return connection, db

# function for reading all face descriptors and ids from database
def readDescriptors(db):
    # check connection
    try:
        db
    except NameError:
        print('Problem with the database connection')  # db cursor does not exist at all
        return -1

    query = 'SELECT f."ID", f.descriptor, f.personid FROM public.faces f'
    db.execute(query)
    records = db.fetchall()
    #records = np.array(records, dtype='object')

    ids = []
    descriptors = []
    personsids = []

    if records != []:

        for r in records:
            ids.append(r[0])
            descriptors.append(r[1][0])
            personsids.append(r[2])

        return ids, descriptors, personsids
    else:
        return {'status': 'ERROR'}



# function for writing records in database
def receiveDescriptors(db, db_conn, name, embeds: np.array([])) -> str:

    query = 'INSERT INTO public.persons (name) VALUES (\'' + str(name) + '\') RETURNING "ID";'
    db.execute(query)
    records = db.fetchall()
    if records[0][0] != '':
        personid = records[0][0]

    for emb in embeds:
        emb = np.array(emb).tolist()
        query = 'INSERT INTO public.faces (descriptor, personid) VALUES (%s, %s);'
        db.execute(query, (emb, (personid,)))
    db_conn.commit()
    return {'status': 'SUCCESS'}


def findPersonByID(db, db_conn, personid: str) -> dict:
    query = 'SELECT p.name FROM public.persons p WHERE "ID" = %s;'
    db.execute(query, (personid,))
    records = db.fetchall()
    if records[0][0] is not None:
        return {
            'status': 'SUCCESS',
            'name': str(records[0][0])
        }
    else:
        return{'status': 'ERROR'}