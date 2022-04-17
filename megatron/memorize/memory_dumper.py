import os
import pickle
import warnings

class MemoryDumper:
    """
    Utility to dump memory records into a file.
    """

    def __init__(self, file_name, header={}, file_bytes_limit=1000*1024*1024):
        self.file_bytes_limit = file_bytes_limit

        self.file = open(file_name, 'wb')
        pickle.dump(header, self.file)

    def dump(self, record):
        # if the current batch is full, close it
        if self.file.tell() > self.file_bytes_limit:
            raise ValueError("MemoryDumper reached the limit of {} bytes.".format(self.file_bytes_limit))

        # pickle the record into the current batch
        pickle.dump(record, self.file)

    def sync(self):
        os.fsync(self.file)
    
    def close(self):
        self.file.close()
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
