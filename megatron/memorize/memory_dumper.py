import os
import pickle
import warnings

class MemoryDumper:
    """
    Utility to dump memory records into a file.
    """

    def __init__(self, file_name, dim, heads, file_bytes_limit=1000*1024*1024):
        self.file_bytes_limit = file_bytes_limit

        self.file = open(file_name, 'wb')
        self.records_count = 0

        header = {
            "dim": dim,
            "heads": heads,
        }
        pickle.dump(header, self.file)

    def dump(self, record):
        # if the current batch is full, close it
        if self.file.tell() > self.file_bytes_limit:
            raise ValueError("MemoryDumper reached the limit of {} bytes.".format(self.file_bytes_limit))

        # write the record into the file
        pickle.dump(record, self.file)
        self.records_count += 1

    def sync(self):
        os.fsync(self.file)
    
    def close(self):
        footer = {
            "records_count": self.records_count,
        }
        pickle.dump(footer, self.file)

        self.file.close()
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
