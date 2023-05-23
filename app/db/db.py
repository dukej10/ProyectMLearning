from pymongo import MongoClient
import configparser
class MongoConnection:
    
    def __init__(self):
        
        self.config = configparser.ConfigParser()
        self.config.read('./app/db/config.ini')
        self.user = self.config.get('DATABASE', 'DB_USER')
        self.password = self.config.get('DATABASE', 'DB_PASS')
        self.db = self.config.get('DATABASE', 'DB_NAME')


    def get_mongo_connection(self):
        client = MongoClient('mongodb+srv://'+self.user+':'+self.password+'@'+self.db +'.5qtjdyg.mongodb.net/?retryWrites=true&w=majority')
        return client['MLearning']
