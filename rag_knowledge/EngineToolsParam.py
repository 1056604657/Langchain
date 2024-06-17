class EngineToolsParam:

    def __init__(self, collection_name, metadata_name, description):
        self.collection_name = collection_name
        self.metadata_name = metadata_name
        self.description = description

    def get_collection_name(self):
        return self.collection_name

    def get_metadata_name(self):
        return self.metadata_name

    def get_description(self):
        return self.description

