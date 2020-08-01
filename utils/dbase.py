import psycopg2
import numpy as np

def dbConnect(host, port, name, user, password):
    try:
        connection = psycopg2.connect(user = user,
                                      password = password,
                                      host = host,
                                      port = port,
                                      database = name)

        db = connection.cursor()
        # Print PostgreSQL Connection properties
        #print ( connection.get_dsn_parameters(),"\n")

        # Print PostgreSQL version
        db.execute("SELECT version();")
        record = db.fetchone()
        #print("You are connected to - ", record,"\n")
        print("DB connection OK\n")

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
    '''
    finally:
        #closing database connection.
            if(connection):
                db.close()
                connection.close()
                print("PostgreSQL connection is closed")
    '''
    return connection, db

def readDescriptors(db):
    try:
        db
    except NameError:
        pass  # db cursor does not exist at all

    query = 'SELECT f."ID", f.descriptor FROM public.faces f';
    db.execute(query)
    records = db.fetchall()
    records = np.array(records, dtype='object')
    ids, descriptors = np.array(records[:, 0]), np.array(records[:, 1])

    return ids, descriptors

def receiveDescriptors():
    pass