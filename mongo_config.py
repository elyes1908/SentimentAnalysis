from pymongo import MongoClient


class MongoConfig:
    CONNECTION_STRING = "mongodb://dxcUser:dxc@localhost:27017/?authMechanism=DEFAULT&authSource=dxc&replicaSet=rs1"
    DB_NAME = "dxc"
    COLLECTION_NAME = "DXCnew"

    def __init__(self):
        self.client = MongoClient(self.CONNECTION_STRING)
        self.db1 = self.client[self.DB_NAME]

    def get_documents_collection(self):
        return self.db1[self.COLLECTION_NAME].find()


documents = MongoConfig().get_documents_collection()
