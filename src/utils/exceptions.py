class MNISTProjectError(Exception):
    pass

class DataLoadError(MNISTProjectError):
    pass

class ModelBuildError(MNISTProjectError):    
    pass