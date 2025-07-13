from .factory import ManipDataFactory


def register_manipdata(manipdata_type):
    def decorator(cls):
        ManipDataFactory.register(manipdata_type, cls)
        return cls

    return decorator
