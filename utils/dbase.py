import psycopg2
import numpy as np

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
        print("DB connection OK\n")

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
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

    query = 'SELECT f."ID", f.descriptor FROM public.faces f';
    db.execute(query)
    records = db.fetchall()
    records = np.array(records, dtype='object')
    ids, descriptors = np.array(records[:, 0]), np.array(records[:, 1])

    return ids, descriptors

# function for writing records in database
def receiveDescriptors(db, db_conn, embeds: np.array([])) -> dict:

    query = 'INSERT INTO public.persons ("ID", name) VALUES (%s, %s);'
    db.execute(query, embeds)
    db_conn.commit()

    embeds = np.array(embeds).tolist()
    query = 'INSERT INTO public.faces (descriptor) VALUES (%s) RETURNING "ID"'
    db.execute(query, embeds)
    db_conn.commit()
    records = db.fetchall()
    return 'SUCCESS'