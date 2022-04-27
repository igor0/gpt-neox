import os
import pickle
import warnings

class MemoryDumper:
    """
    Utility to dump memory records into a file.
    """

    def __init__(self, layer_number, file_path, index_memory_func, dim, heads, file_bytes_limit=1000*1024*1024):
        self.layer_number = layer_number
        self.file_path = file_path
        self.index_memory_func = index_memory_func
        self.file_bytes_limit = file_bytes_limit

        self.file = open(file_path, 'wb')
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
        self.records_count += record[0].shape[0]

    def sync(self):
        os.fsync(self.file)
    
    def close(self):
        print(f'MemoryDumper wrote {self.records_count} records into "{self.file_path}".')

        # write a None value to mark the end
        pickle.dump(None, self.file)

        # write a footer
        footer = {
            "records_count": self.records_count,
        }
        pickle.dump(footer, self.file)

        # close the file
        self.file.close()
        self.file = None

        # build an index
        self.index_memory_func(os.path.dirname(self.file_path), self.layer_number)

        # remove the pickled file
        os.remove(self.file_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()